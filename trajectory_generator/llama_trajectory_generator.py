import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, String
import numpy as np
from scipy.linalg import logm, expm
import time
import sys
import json
from scipy.spatial.transform import Rotation as R
from custom_interface.msg import ObjectArray

class TrajectoryGenerator(Node):
    """
    output: x, x_dot
    """ 
    def __init__(self, T_init = None, T_final=None, traj_time=10, sampling_frequency=100, traj_type="polynomial"):
        super().__init__('trajectory_generator_node')
        self.publisher_pose = self.create_publisher(Float64MultiArray, "/ur_trajectory_generator/pose_ref", 10)
        self.publisher_posedot = self.create_publisher(Float64MultiArray, "/ur_trajectory_generator/posedot_ref", 10)
        
        self.sampling_frequency = sampling_frequency
        # Timer to publish at 125Hz (every 8ms)
        self.timer = self.create_timer(1/self.sampling_frequency, self.publish_trajectory)
        
        self.traj_time = traj_time
        self.total_samples = self.sampling_frequency*traj_time
        l1,l2, l3, l4, l5, l6, l7, l8 = np.array([0.1273, 0.220491, 0.612, 0.1719, 0.5723, 0.1149, 0.1157, 0.0922])

        self.T_init = np.array([[0, 1, 0, l5+l7], 
                                 [1, 0, 0, l2-l4+l6], 
                                 [0, 0, -1, l1+l3-l8], 
                                 [0, 0, 0, 1]]) if T_init is None else T_init
        
        self.T_final = np.array([[0, 1, 0, -l2+l4-l6], 
                                 [1, 0, 0, l3+l5-0.2843], 
                                 [0, 0, -1, l1+l8], 
                                 [0, 0, 0, 1]]) if T_final is None else T_final
       
        self.start_time = time.time()
        self.plot_pose = []
        self.traj_type = traj_type
        self.start_trajectory = False
        self.llama_actions_f = False
        self.llama_actions = {}
        self.create_subscription(
            ObjectArray,
            '/detected_objects',
            self.listener_callback,
            10
        )
        self.subscription = self.create_subscription(
            String,
            '/llama_output',
            self.llama_listener_callback,
            10
        )
        self.final_pose_f = True
        self.get_logger().info("Trajectory generator node started...")
        self.get_logger().info("Waiting for target position to start trajectory...")

    def listener_callback(self, msg):
        self.objs = []
        self.lbs = []
        for obj in msg.objects:
            self.objs.append([obj.x, obj.y, obj.z])
            self.lbs.append(obj.label)
            #self.get_logger().info(f"Label: {obj.label}, x: {obj.x}, y: {obj.y}, z: {obj.z}")
        self.start_trajectory = True
        self.time_counter = 0.0
        self.sample_counter = 0
        self.prev_position = None
        self.prev_euler = None
        self.get_logger().info(f"Received target objs and positions: {self.objs}")

    def llama_listener_callback(self, msg):
        try:
            self.llama_actions_f = True
            data = json.loads(msg.data)
            self.get_logger().info("Received llama actions")
            for key, value in data.items():
                self.llama_actions[key] = value
            self.get_logger().info(f"llama calback: {self.llama_actions}")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode JSON: " + msg.data)

    def rot2eul(self,rot_matrix, seq='XYZ'):
        """
        Convert rotation matrix to Euler angles.
        Uses scipy's spatial transform for high efficiency.
        """
        rotation = R.from_matrix(rot_matrix)
        eul = rotation.as_euler(seq, degrees=False)
        return eul
        
    def eul2angular_vel(self,euler):
        alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot = euler

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        omega_x = ((ca * sg + cg * sa * sb) * 
                   (ca * sg * alpha_dot + cg * sa * gamma_dot - ca * cb * cg * beta_dot + 
                    cg * sa * sb * alpha_dot + ca * sb * sg * gamma_dot) +(ca * cg - sa * sb * sg) * 
                    (ca * cg * alpha_dot - sa * sg * gamma_dot + ca * cb * sg * beta_dot +
                    ca * cg * sb * gamma_dot - sa * sb * sg * alpha_dot) +
                    cb * sa * (cb * sa * alpha_dot + ca * sb * beta_dot))

        omega_y = ca * beta_dot - cb * sa * gamma_dot
        omega_z = sa * beta_dot + ca * cb * gamma_dot

        return np.array([omega_x, omega_y, omega_z])

    def compute_differentiation(self, prev_pose, curr_pose, dt):
        """
        Compute linear velocity given previous pose, current pose, and time increment.

        Args:
            prev_pose (np.array): Previous position (x, y, z)
            curr_pose (np.array): Current position (x, y, z)
            dt (float): Time increment (should be > 0)

        Returns:
            np.array: Linear velocity (vx, vy, vz)
        """
        if dt <= 0:
            raise ValueError("Time increment must be positive.")
    
        velocity = (curr_pose - prev_pose) / dt
        return velocity
    
    def trajectory_smooth(self):
        """ Generate smooth trajectory using the exponential matrix method. """
        R_init = self.T_init[:3, :3]
        R_final = self.T_final[:3, :3]
        position_init = self.T_init[:3, 3]
        position_final = self.T_final[:3, 3]
        s = self.time_counter/self.traj_time
        # Compute interpolated rotation using exponential map
        R_instant = R_init @ expm(logm(R_init.T @ R_final) * s)
        
        if self.traj_type=="polynomial":
            a0 = position_init
            a1=0.0
            a2=0.0
            a3=10*(position_final-position_init)
            a4=15*(position_final-position_init)
            a5=6*(position_final-position_init)
            position_instant = a0 + a1*s+a2*pow(s,2)+a3*pow(s,3)-a4*pow(s,4)+a5*pow(s,5)
        elif self.traj_type =="straight":
            position_instant = position_init + s * (position_final - position_init)
        elif self.traj_type == "ellipsoid":
            a = 0.3 #major
            b = 0.1 #minor
            x = position_init[0] + b*np.sin(2*np.pi*s)
            y = position_init[1]-a + a*np.cos(2*np.pi*s)
            z = position_init[2]
            position_instant = np.array([x,y,z])

        euler_instant = self.rot2eul(R_instant)

        if self.prev_position is None:
            self.prev_position = position_init
        position_dot = self.compute_differentiation(self.prev_position, position_instant, 1/self.sampling_frequency)
        
        if self.prev_euler is None:
            self.prev_euler = self.rot2eul(R_init)
        euler_dot = self.compute_differentiation(self.prev_euler, euler_instant, 1/self.sampling_frequency)
        orientation_dot = self.eul2angular_vel(np.hstack([euler_instant, euler_dot]))
        self.prev_position, self.prev_euler = position_instant, euler_instant
        self.time_counter += 1/self.sampling_frequency
        return np.hstack([euler_instant, position_instant]), np.hstack([orientation_dot, position_dot])

    def publish_trajectory(self):
        if not self.start_trajectory or not self.llama_actions_f:
            self.get_logger().info(f"Didn't receive llama actions or detected objects.")
            return 
        if self.llama_actions and self.final_pose_f:
            for key, value in self.llama_actions.items():
                print(f"key: {key}, value: {value}")
                if key == "object":
                    print(f"lbs, objs: {self.lbs, self.objs}")
                    for dlbs, dobj in zip(self.lbs, self.objs):
                        print(f"dlbs: {dlbs}, dobjs: {dobj}")
                        if dlbs == value:
                            self.T_final[:3, 3] = [dobj[0], dobj[1], dobj[2]]
                            self.final_pose_f = False
                        else:
                            self.get_logger().info(f"Didn't found {dlbs} in the scene...please check scene.")
                            return
        else:
            self.get_logger().info(f"Didn't receive valid llama actions. ask again")
            return
    
        elapsed_samples = self.sample_counter
        if elapsed_samples > self.total_samples and elapsed_samples < self.total_samples+1:
            self.publisher_pose.publish(self.msg_pose)
            self.publisher_posedot.publish(self.msg_pose_dot)
            self.get_logger().info(f"Published {self.traj_time} sec trajectory last point {self.msg_pose.data}")
 
        elif elapsed_samples<=self.total_samples:
            pose, pose_dot = self.trajectory_smooth()
            self.plot_pose.append(pose)
            if not pose.any():
                print("No trajectory point received")
                return
            self.msg_pose = Float64MultiArray()
            self.msg_pose.data = [i for i in pose]  # List of six float values
            self.msg_pose_dot = Float64MultiArray()
            self.msg_pose_dot.data = [i for i in pose_dot]
        
            self.publisher_pose.publish(self.msg_pose)
            self.publisher_posedot.publish(self.msg_pose_dot)
            if elapsed_samples == 0 or elapsed_samples==1000:
                self.get_logger().info(f'elapsed sample: {elapsed_samples}: {self.msg_pose.data}')
        self.sample_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
