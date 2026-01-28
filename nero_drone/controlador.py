#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
from math import sin, cos, pi, atan2, asin

# ============================================================================
# ESTRUCTURAS
# ============================================================================
class Position:
    def __init__(self):
        self.X = np.zeros(12)        # [x y z roll pitch yaw vx vy vz wx wy wz]
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
        self.Model_simp = np.array([0.8417, 0.18227, 0.8354, 0.17095,
                                    3.966, 4.001, 9.8524, 4.7295])
        self.g = 9.8
        self.Altmax = 2000.0
        self.uSat = np.ones(6)

class SC:
    def __init__(self):
        self.Ud = np.zeros(6)
        self.Ur = np.zeros(4)
        self.Kinematics_control = 1
        self.tcontrol = None

# ============================================================================
# CLASE BEBOP
# ============================================================================
class Bebop:
    def __init__(self, node: Node):
        self.node = node
        self.dt = 0.1  
        self.pPos = Position()
        self.pPar = Parameters()
        self.pSC = SC()

        self.ref_received = False
        self.is_flying = False
        self.first_ref = True
        self.last_ref = np.zeros(8)

        self.sub_odom = node.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)
        #self.sub_odom = node.create_subscription(Odometry, "/bebop/odom", self.odom_callback, 10)
        self.sub_ref = node.create_subscription(Float64MultiArray, "/bebop/ref_vec", self.ref_callback, 10)
        self.sub_is_flying = node.create_subscription(Bool, "/bebop/is_flying", self.is_flying_callback, 10)
        self.pub_cmd = node.create_publisher(Twist, "/safe_bebop/cmd_vel", 10)
        self.pOdom = None

    # ---------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        self.pOdom = msg

    def ref_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 8:
            self.node.get_logger().warn("Referencia incorrecta: se esperaban 8 elementos [Xd Yd Zd Psid dXd dYd dZd dPsid]")
            return

        self.ref_received = True
        data = np.array(msg.data)
        self.pPos.Xd[0:3] = data[0:3]        # X, Y, Z
        self.pPos.Xr[5] = data[3]            # Psi (Yaw)
        self.pPos.dXd[0:3] = data[4:7]       # dX, dY, dZ
        self.pPos.dXd[3] = data[7]           # dPsi

        if not self.first_ref:
            self.pPos.ddXd = (data[4:8] - self.last_ref[4:8]) / self.dt
        else:
            self.pPos.ddXd = np.zeros(4)
            self.first_ref = False
        self.last_ref = data

    def is_flying_callback(self, msg: Bool):
        self.is_flying = msg.data

    # ---------------------------------------------------------------------
    def rGetSensorData(self):
        if self.pOdom is None:
            return

        # Guardar última posición
        self.pPos.Xa = self.pPos.X.copy()

        pose = self.pOdom.pose.pose
        twist = self.pOdom.twist.twist

        # Conversión de cuaternión a Euler (roll, pitch, yaw)
        qw, qx, qy, qz = pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.sign(sinp) * (pi / 2) if abs(sinp) >= 1 else asin(sinp)

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)

        # Asignar posición y orientación
        self.pPos.X[0:6] = [pose.position.x, pose.position.y, pose.position.z,
                             roll, pitch, yaw]

        # Velocidades (cuerpo → global)
        dXc = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        psi = self.pPos.X[5]
        F = np.array([[cos(psi), -sin(psi), 0, 0],
                      [sin(psi),  cos(psi), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        dX = F @ dXc
        auxdX12 = (self.pPos.X[5] - self.pPos.Xa[5]) / self.dt

        # Velocidades globales (igual al MATLAB)
        self.pPos.X[6]  = dX[0] * self.pPar.uSat[3]
        self.pPos.X[7]  = dX[1] * self.pPar.uSat[4]
        self.pPos.X[8]  = dX[2] * self.pPar.uSat[5]
        self.pPos.X[9:11] = 0.0
        self.pPos.X[11] = auxdX12 * self.pPar.uSat[2]

        self.pPos.dX[:] = [self.pPos.X[6], self.pPos.X[7], self.pPos.X[8],
                           self.pPos.X[9], self.pPos.X[10], self.pPos.X[11]]

    # ---------------------------------------------------------------------
    def cInverseDynamicController_Compensador(self, gains=None):
        if not self.ref_received:
            return


        gains = [1.2, 1.2, 3, 1.5,
                    1.0, 1.0, 1.8, 1.2,
                    1.7, 1.7, 1, 1.5]
        g = np.array(gains)

        # Modelo
        Ku = np.diag([self.pPar.Model_simp[0], self.pPar.Model_simp[2],
                      self.pPar.Model_simp[4], self.pPar.Model_simp[6]])
        Kv = np.diag([self.pPar.Model_simp[1], self.pPar.Model_simp[3],
                      self.pPar.Model_simp[5], self.pPar.Model_simp[7]])
        Ksp = np.diag(g[0:4])
        Ksd = np.diag(g[4:8])
        Kp  = np.diag(g[8:12])
        # Estados
        X   = np.array([self.pPos.X[0], self.pPos.X[1], self.pPos.X[2], self.pPos.X[5]])
        dX  = np.array([self.pPos.X[6], self.pPos.X[7], self.pPos.X[8], self.pPos.X[11]])
        Xd  = np.array([self.pPos.Xd[0], self.pPos.Xd[1], self.pPos.Xd[2], self.pPos.Xr[5]])
        dXd = np.array([self.pPos.dXd[0], self.pPos.dXd[1], self.pPos.dXd[2], self.pPos.dXd[3]])

        # Erros (protección)
        if np.linalg.norm(self.pPos.Xtil) == 0:
            Xtil = Xd - X
        else:
            Xtil = np.array([self.pPos.Xtil[0], self.pPos.Xtil[1],
                             self.pPos.Xtil[2], self.pPos.Xtil[5]])


        if abs(Xtil[3]) > pi:
            Xtil[3] = Xtil[3] - 2 * pi * np.sign(Xtil[3])

        # Controle cinemático
        Ucw_ant = np.copy(self.pSC.Ur)
        Ucw = dXd + Ksp @ np.tanh(Kp @ Xtil)
        dUcw = (Ucw - Ucw_ant) / max(self.dt, 1e-3)
        self.pSC.Ur = np.copy(Ucw)

        # Cinemática direta
        F = np.array([[cos(X[3]), -sin(X[3]), 0, 0],
                      [sin(X[3]),  cos(X[3]), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Compensador dinâmico
        Udw = np.linalg.inv(F @ Ku) @ (dUcw + Ksd @ (Ucw - dX) + Kv @ dX)

        # Comandos enviados ao Bebop 2
        self.pSC.Ud[0:3] = Udw[0:3]
        self.pSC.Ud[3:5] = 0.0
        self.pSC.Ud[5] = Udw[3]
        self.pSC.tcontrol = self.node.get_clock().now()

    # ---------------------------------------------------------------------
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
class NeroDroneNode(Node):
    def __init__(self):
        super().__init__("nero_drone_node")
        self.get_logger().info("Nero Drone Node iniciado (modo cinemático, 10 Hz)")
        self.drone = Bebop(self)
        self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        self.drone.rGetSensorData()
        self.drone.cInverseDynamicController_Compensador([])
        self.drone.rSendControlSignals()

# ============================================================================
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