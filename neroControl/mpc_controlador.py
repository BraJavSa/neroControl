#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
from rcl_interfaces.msg import SetParametersResult
import numpy as np
from math import sin, cos, pi, atan2, asin

class Position:
    """Mathematical representation of the vehicle states in the Body Frame."""
    def __init__(self):
        self.X = np.zeros(12)        
        self.dX_body = np.zeros(4) # [vx_b, vy_b, vz_b, vyaw_b]
        self.dXd_body = np.zeros(4) # Velocity references in Body Frame
        self.Xa = np.zeros(12)

class Parameters:
    """Dynamic model coefficients and physical constraints."""
    def __init__(self):
        # Coefficients identified for the dynamic model
        self.Model_simp = np.array([0.8417, 0.18227, 0.8354, 0.17095,
                                    3.966, 4.001, 9.8524, 4.7295])
        self.uSat = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

class SC:
    """Control signals and internal command states."""
    def __init__(self):
        self.Ud = np.zeros(6)

class BebopBodySMC:
    """
    Implementation of a Sliding Mode Controller for Body-Frame Velocity Tracking.
    """
    def __init__(self, node: Node):
        self.node = node
        self.dt = 1/10 
        self.pPos = Position()
        self.pPar = Parameters()
        self.pSC = SC()

        self.ref_received = False
        self.is_flying = False
        self.pOdom = None

        self.declare_ros_parameters()

        # Communication interfaces
        self.node.add_on_set_parameters_callback(self.parameters_callback)
        self.sub_odom = node.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.sub_ref = node.create_subscription(Float64MultiArray, "/bebop/ref_vec", self.ref_callback, 10)
        self.sub_is_flying = node.create_subscription(Bool, "/bebop/is_flying", self.is_flying_callback, 10)
        self.pub_cmd = node.create_publisher(Twist, "/safe_bebop/cmd_vel", 10)

    def declare_ros_parameters(self):
        """Initializes gains for the Body-Frame SMC."""
        self.node.declare_parameter('ksmc_x', 0.4)
        self.node.declare_parameter('ksmc_y', 0.4)
        self.node.declare_parameter('ksmc_z', 2.0)
        self.node.declare_parameter('ksmc_psi', 5.5)
        self.node.declare_parameter('phi', 0.4) # Chattering boundary layer

    def parameters_callback(self, params):
        for param in params:
            self.node.get_logger().info(f"Updated: {param.name} = {param.value}")
        return SetParametersResult(successful=True)

    def odom_callback(self, msg: Odometry):
        self.pOdom = msg

    def ref_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 8:
            return
        self.ref_received = True
        # Reference assumes velocity commands are provided in the Body Frame
        # Mapping: [_, _, _, _, vx_body, vy_body, vz_body, vyaw_body]
        self.pPos.dXd_body[0:4] = np.array(msg.data[4:8])

    def is_flying_callback(self, msg: Bool):
        self.is_flying = msg.data

    def rGetSensorData(self):
        """
        Extracts odometry and computes the current velocity in the Body Frame.
        """
        if self.pOdom is None:
            return
        
        self.pPos.Xa = self.pPos.X.copy()
        twist = self.pOdom.twist.twist
        
        # In ROS2 Odometry msg, twist is typically already in the child_frame_id (Body Frame)
        # If not, a rotation matrix based on current orientation would be required.
        self.pPos.dX_body[0] = twist.linear.x
        self.pPos.dX_body[1] = twist.linear.y
        self.pPos.dX_body[2] = twist.linear.z
        self.pPos.dX_body[3] = twist.angular.z

    def cBodySlidingModeController(self):
        """
        Control law formulated directly in the Body Frame.
        This approach eliminates the need for the global rotation matrix F in the control loop.
        """
        if not self.ref_received:
            return

        # Gain acquisition
        K_smc = np.diag([
            self.node.get_parameter('ksmc_x').value,
            self.node.get_parameter('ksmc_y').value,
            self.node.get_parameter('ksmc_z').value,
            self.node.get_parameter('ksmc_psi').value
        ])
        phi = self.node.get_parameter('phi').value

        # Dynamic Model Matrices (Body Frame simplified dynamics)
        Ku = np.diag([self.pPar.Model_simp[0], self.pPar.Model_simp[2], self.pPar.Model_simp[4], self.pPar.Model_simp[6]])
        Kv = np.diag([self.pPar.Model_simp[1], self.pPar.Model_simp[3], self.pPar.Model_simp[5], self.pPar.Model_simp[7]])
        
        # Error Surface S in Body Frame
        S = self.pPos.dXd_body - self.pPos.dX_body

        # Feed-forward / Acceleration term (assumed zero for set-point tracking)
        ddXd_body = np.zeros(4)

        # Robust Control Term (Hyperbolic tangent for smoothness)
        u_robust = K_smc @ np.tanh(S / phi)

        # Control Law: u = Ku^-1 * (ddXd_body + Kv * dX_body + u_robust)
        # Note: F matrix is omitted as the control is now local to the Body Frame.
        try:
            invKu = np.linalg.inv(Ku)
            Udw = invKu @ (ddXd_body + Kv @ self.pPos.dX_body + u_robust)
        except np.linalg.LinAlgError:
            return

        self.pSC.Ud[0] = Udw[0] # vx
        self.pSC.Ud[1] = Udw[1] # vy
        self.pSC.Ud[2] = Udw[2] # vz
        self.pSC.Ud[5] = Udw[3] # vpsi

    def rSendControlSignals(self):
        if not self.ref_received or not self.is_flying:
            return
        
        cmd = Twist()
        cmd.linear.x = float(np.clip(self.pSC.Ud[0], -self.pPar.uSat[0], self.pPar.uSat[0]))
        cmd.linear.y = float(np.clip(self.pSC.Ud[1], -self.pPar.uSat[1], self.pPar.uSat[1]))
        cmd.linear.z = float(np.clip(self.pSC.Ud[2], -self.pPar.uSat[2], self.pPar.uSat[2]))
        cmd.angular.z = float(np.clip(self.pSC.Ud[5], -self.pPar.uSat[5], self.pPar.uSat[5]))
        self.pub_cmd.publish(cmd)

class NeroDroneNode(Node):
    def __init__(self):
        super().__init__("neroControl_node")
        self.drone = BebopBodySMC(self)
        self.timer = self.create_timer(0.1, self.control_loop) 
        self.get_logger().info("Body-Frame Sliding Mode Controller Initialized.")

    def control_loop(self):
        self.drone.rGetSensorData()
        self.drone.cBodySlidingModeController()
        self.drone.rSendControlSignals()

def main(args=None):
    rclpy.init(args=args)
    node = NeroDroneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()