#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
from math import sin, cos, pi, atan2

# ============================================================================
# DATA STRUCTURES
# ============================================================================
class KinematicState:
    def __init__(self):
        # World Frame Variables (w_)
        self.w_X = np.zeros(4)      # Current pose: [x, y, z, yaw]
        self.w_Xd = np.zeros(4)     # Desired pose: [xd, yd, zd, yawd]
        self.w_dXd = np.zeros(4)    # Desired velocity: [dxd, dyd, dzd, dyawd]
        
        # Body Frame Variables (b_)
        self.b_Xtil = np.zeros(4)   # Position error projected to Body
        self.b_dXd = np.zeros(4)    # Desired velocity projected to Body
        self.b_Ucw = np.zeros(4)    # Final control command in Body

# ============================================================================
# BEBOP KINEMATIC CONTROL CLASS
# ============================================================================
class BebopKinematic:
    def __init__(self, node: Node):
        self.node = node
        self.state = KinematicState()

        self.ref_received = False
        self.is_flying = False
        self.b_u_sat = np.array([1.0, 1.0, 1.0, 1.0]) 

        # Control Gains
        self.node.declare_parameter('ksp_x', 1.0)
        self.node.declare_parameter('ksp_y', 1.0)
        self.node.declare_parameter('ksp_z', 1.0)
        self.node.declare_parameter('ksp_yaw', 1.0)
        
        self.node.declare_parameter('kp_x', 0.3)
        self.node.declare_parameter('kp_y', 0.3)
        self.node.declare_parameter('kp_z', 0.3)
        self.node.declare_parameter('kp_yaw', 1.2)

        # ROS Communications
        self.sub_odom = node.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.sub_ref = node.create_subscription(Float64MultiArray, "/bebop/ref_vec", self.ref_callback, 10)
        self.sub_is_flying = node.create_subscription(Bool, "/bebop/is_flying", self.is_flying_callback, 10)
        self.pub_cmd = node.create_publisher(Twist, "/safe_bebop/cmd_vel", 10)
        
        self.current_odom = None

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg
        p = msg.pose.pose
        q = p.orientation
        
        # Extract only Yaw from Quaternion
        yaw = atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))

        # Store only necessary World state
        self.state.w_X = np.array([p.position.x, p.position.y, p.position.z, yaw])

    def ref_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= 8:
            self.ref_received = True
            self.state.w_Xd = np.array(msg.data[0:4])
            self.state.w_dXd = np.array(msg.data[4:8])

    def is_flying_callback(self, msg: Bool):
        self.is_flying = msg.data

    def compute_control(self):
        if not self.ref_received or self.current_odom is None:
            return

        # 1. Gain Matrices
        Ksp = np.diag([
            self.node.get_parameter('ksp_x').value,
            self.node.get_parameter('ksp_y').value,
            self.node.get_parameter('ksp_z').value,
            self.node.get_parameter('ksp_yaw').value
        ])
        Kp = np.diag([
            self.node.get_parameter('kp_x').value,
            self.node.get_parameter('kp_y').value,
            self.node.get_parameter('kp_z').value,
            self.node.get_parameter('kp_yaw').value
        ])

        # 2. Rotation Matrix World to Body (R^T)
        w_psi = self.state.w_X[3]
        RT = np.array([
            [ cos(w_psi), sin(w_psi), 0, 0],
            [-sin(w_psi), cos(w_psi), 0, 0],
            [ 0,          0,          1, 0],
            [ 0,          0,          0, 1]
        ])

        # 3. Transform Reference Velocity to Body Frame
        self.state.b_dXd = RT @ self.state.w_dXd

        # 4. Calculate World Position Error and Transform to Body
        w_Xtil = self.state.w_Xd - self.state.w_X
        w_Xtil[3] = atan2(sin(w_Xtil[3]), cos(w_Xtil[3])) # Yaw Error Normalization
        
        self.state.b_Xtil = RT @ w_Xtil

        # 5. Control Law: b_Ucw = b_dXd + Ksp * tanh(Kp * b_Xtil)
        control_vec = self.state.b_dXd + Ksp @ np.tanh(Kp @ self.state.b_Xtil)

        # 6. Apply Saturation
        self.state.b_Ucw[0] = np.clip(control_vec[0], -self.b_u_sat[0], self.b_u_sat[0])
        self.state.b_Ucw[1] = np.clip(control_vec[1], -self.b_u_sat[1], self.b_u_sat[1])
        self.state.b_Ucw[2] = np.clip(control_vec[2], -self.b_u_sat[2], self.b_u_sat[2])
        self.state.b_Ucw[3] = np.clip(control_vec[3], -self.b_u_sat[3], self.b_u_sat[3])

    def send_commands(self):
        if not self.is_flying:
            return
        
        cmd = Twist()
        cmd.linear.x = float(self.state.b_Ucw[0])
        cmd.linear.y = float(self.state.b_Ucw[1])
        cmd.linear.z = float(self.state.b_Ucw[2])
        cmd.angular.z = float(self.state.b_Ucw[3])
        self.pub_cmd.publish(cmd)

# ============================================================================
# EXECUTION NODE
# ============================================================================
class KinematicNode(Node):
    def __init__(self):
        super().__init__("kinematic_controller_node")
        self.drone = BebopKinematic(self)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Kinematic Node initialized (Yaw-only transformation)")

    def control_loop(self):
        self.drone.compute_control()
        self.drone.send_commands()

def main(args=None):
    rclpy.init(args=args)
    node = KinematicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()