#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
from rcl_interfaces.msg import SetParametersResult
import numpy as np
from math import sin, cos, pi, atan2, asin

# ============================================================================
# DATA STRUCTURES
# ============================================================================
class Position:
    def __init__(self):
        self.X = np.zeros(12)        
        self.dX = np.zeros(6)
        self.ddX = np.zeros(6)
        self.Xd = np.zeros(4)
        self.dXd = np.zeros(4)
        self.ddXd = np.zeros(4)
        self.Xtil = np.zeros(12)
        self.dXtil = np.zeros(12)
        self.Xr = np.zeros(12)
        self.Xa = np.zeros(12)

class Parameters:
    def __init__(self):
        # Simplificated dynamic model coefficients
        self.Model_simp = np.array([0.8417, 0.18227, 0.8354, 0.17095,
                                    3.966, 4.001, 9.8524, 4.7295])
        self.g = 9.8
        self.uSat = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

class SC:
    def __init__(self):
        self.Ud = np.zeros(6)
        self.Ur = np.zeros(4)
        self.tcontrol = None

# ============================================================================
# BEBOP CONTROLLER CLASS
# ============================================================================
class Bebop:
    def __init__(self, node: Node):
        self.node = node
        self.dt = 1/5  # Control loop at 10Hz
        self.pPos = Position()
        self.pPar = Parameters()
        self.pSC = SC()

        self.ref_received = False
        self.is_flying = False
        self.first_ref = True
        self.last_ref = np.zeros(8)

        # Dynamic Parameters Declaration
        self.declare_ros_parameters()

        # Callbacks and Communications
        self.node.add_on_set_parameters_callback(self.parameters_callback)
        self.sub_odom = node.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.sub_ref = node.create_subscription(Float64MultiArray, "/bebop/ref_vec", self.ref_callback, 10)
        self.sub_is_flying = node.create_subscription(Bool, "/bebop/is_flying", self.is_flying_callback, 10)
        self.pub_cmd = node.create_publisher(Twist, "/safe_bebop/cmd_vel", 10)
        self.pOdom = None

    def declare_ros_parameters(self):
        self.node.declare_parameter('ksp_x', 0.7)
        self.node.declare_parameter('ksp_y', 0.9)
        self.node.declare_parameter('ksp_z', 2.0)
        self.node.declare_parameter('ksp_psi', 2.8)
        self.node.declare_parameter('ksd_x', 0.55)
        self.node.declare_parameter('ksd_y', 0.55)
        self.node.declare_parameter('ksd_z', 1.5)
        self.node.declare_parameter('ksd_psi', 2.0)
        self.node.declare_parameter('kp_x', 0.8)
        self.node.declare_parameter('kp_y', 0.8)
        self.node.declare_parameter('kp_z', 1.5)
        self.node.declare_parameter('kp_psi', 1.8)

    def parameters_callback(self, params):
        for param in params:
            self.node.get_logger().info(f"Updated: {param.name} = {param.value}")
        return SetParametersResult(successful=True)

    def odom_callback(self, msg: Odometry):
        self.pOdom = msg

    def ref_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 8:
            return
        self.ref_received = True
        data = np.array(msg.data)
        self.pPos.Xd[0:3] = data[0:3]
        self.pPos.Xr[5] = data[3]      # Desired Yaw
        self.pPos.dXd[0:3] = data[4:7]  # Desired Velocities
        self.pPos.dXd[3] = data[7]      # Desired Yaw rate
        self.last_ref = data

    def is_flying_callback(self, msg: Bool):
        self.is_flying = msg.data

    def rGetSensorData(self):
        if self.pOdom is None:
            return
        self.pPos.Xa = self.pPos.X.copy()
        pose = self.pOdom.pose.pose
        twist = self.pOdom.twist.twist

        # Quaternion to Euler
        qw, qx, qy, qz = pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
        roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.sign(sinp) * (pi / 2) if abs(sinp) >= 1 else asin(sinp)
        yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        self.pPos.X[0:6] = [pose.position.x, pose.position.y, pose.position.z, roll, pitch, yaw]

        # Velocity in Odom Frame
        dXc = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        psi = self.pPos.X[5]
        F = np.array([[cos(psi), -sin(psi), 0, 0],
                      [sin(psi),  cos(psi), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        dX = F @ dXc
        
        # Numerical derivative for Yaw rate
        auxdX12 = (self.pPos.X[5] - self.pPos.Xa[5]) / self.dt

        self.pPos.X[6:9] = dX[0:3]
        self.pPos.X[11] = auxdX12
        self.pPos.dX[:] = self.pPos.X[6:12]

    def cInverseDynamicController_Compensador(self):
        if not self.ref_received:
            return

        # Fetch Reconfigurable Parameters
        Ksp = np.diag([self.node.get_parameter(n).value for n in ['ksp_x', 'ksp_y', 'ksp_z', 'ksp_psi']])
        Ksd = np.diag([self.node.get_parameter(n).value for n in ['ksd_x', 'ksd_y', 'ksd_z', 'ksd_psi']])
        Kp  = np.diag([self.node.get_parameter(n).value for n in ['kp_x', 'kp_y', 'kp_z', 'kp_psi']])

        # Model Matrices
        Ku = np.diag([self.pPar.Model_simp[0], self.pPar.Model_simp[2], self.pPar.Model_simp[4], self.pPar.Model_simp[6]])
        Kv = np.diag([self.pPar.Model_simp[1], self.pPar.Model_simp[3], self.pPar.Model_simp[5], self.pPar.Model_simp[7]])
        
        X   = np.array([self.pPos.X[0], self.pPos.X[1], self.pPos.X[2], self.pPos.X[5]])
        dX  = np.array([self.pPos.X[6], self.pPos.X[7], self.pPos.X[8], self.pPos.X[11]])
        Xd  = np.array([self.pPos.Xd[0], self.pPos.Xd[1], self.pPos.Xd[2], self.pPos.Xr[5]])
        dXd = np.array([self.pPos.dXd[0], self.pPos.dXd[1], self.pPos.dXd[2], self.pPos.dXd[3]])

        # --- SHORTEST PATH CALCULATION ---
        Xtil = Xd - X
        # Normalize Yaw error to [-pi, pi]
        Xtil[3] = (Xtil[3] + pi) % (2 * pi) - pi
        
        # Position Saturation for kinematic safety
        sat_limit = 1.0 
        for i in range(3):
            Xtil[i] = np.clip(Xtil[i], -sat_limit, sat_limit)

        # Kinematic Compensator
        Ucw_ant = np.copy(self.pSC.Ur)
        Ucw = dXd + Ksp @ np.tanh(Kp @ Xtil)
        dUcw = (Ucw - Ucw_ant) / self.dt
        self.pSC.Ur = np.copy(Ucw)

        # Dynamic Linearization
        psi = X[3]
        F = np.array([[cos(psi), -sin(psi), 0, 0],
                      [sin(psi),  cos(psi), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Control Law: u = (F*Ku)^-1 * (dUcw + Ksd*(Ucw - dX) + Kv*dX)
        Udw = np.linalg.inv(F @ Ku) @ (dUcw + Ksd @ (Ucw - dX) + Kv @ dX)

        self.pSC.Ud[0:3] = Udw[0:3]
        self.pSC.Ud[5] = Udw[3]

    def rSendControlSignals(self):
        if not self.ref_received or not self.is_flying:
            return
        cmd = Twist()
        cmd.linear.x = float(np.clip(self.pSC.Ud[0], -self.pPar.uSat[0], self.pPar.uSat[0]))
        cmd.linear.y = float(np.clip(self.pSC.Ud[1], -self.pPar.uSat[1], self.pPar.uSat[1]))
        cmd.linear.z = float(np.clip(self.pSC.Ud[2], -self.pPar.uSat[2], self.pPar.uSat[2]))
        cmd.angular.z = float(np.clip(self.pSC.Ud[5], -self.pPar.uSat[5], self.pPar.uSat[5]))
        self.pub_cmd.publish(cmd)

# ============================================================================
# MAIN NODE
# ============================================================================
class NeroDroneNode(Node):
    def __init__(self):
        super().__init__("nero_drone_node")
        self.drone = Bebop(self)
        self.create_timer(0.1, self.control_loop) 
        self.get_logger().info("Nero Drone Node: Geodesic Controller Active.")

    def control_loop(self):
        self.drone.rGetSensorData()
        self.drone.cInverseDynamicController_Compensador()
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