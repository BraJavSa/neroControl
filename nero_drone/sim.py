#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped, PointStamped
from std_msgs.msg import Empty, Float64
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


class DroneSimulator(Node):
    def __init__(self):
        super().__init__("bebop_sim")

        # Timer maestro 150 Hz
        self.dt = 1.0 / 150.0
        self.tick = 0

        # Estado del dron: [x, y, z, yaw]
        self.x = np.zeros(4)
        self.xdot = np.zeros(4)
        self.u = np.zeros(4)

        # Máquina de estados
        # IDLE: en el suelo, sin dinámica
        # TAKING_OFF: subiendo directo hasta z_target_takeoff
        # FLYING: dinámica activa, acepta cmd_vel
        # LANDING: bajando directo hasta z = 0
        # EMERGENCY: bajando directo rápido hasta z = 0
        self.mode = "IDLE"
        self.z_target_takeoff = 1.2
        self.v_takeoff = 0.5      # m/s
        self.v_land = -0.5        # m/s
        self.v_emergency = -1.0   # m/s

        # Control
        self.last_cmd = None
        self.cmd_counter = 0

        # Publishers
        self.pub_xy = self.create_publisher(PointStamped, "/bebop/position", 10)
        self.pub_z = self.create_publisher(Float64, "/bebop/altitude", 10)
        self.pub_imu = self.create_publisher(Imu, "/bebop/imu", 10)
        self.pub_odom = self.create_publisher(Odometry, "/bebop/odom", 10)

        # Subscribers
        self.sub_cmd = self.create_subscription(Twist, "/bebop/cmd_vel", self.cmd_cb, 10)
        self.sub_takeoff = self.create_subscription(Empty, "/bebop/takeoff", self.takeoff_cb, 10)
        self.sub_land = self.create_subscription(Empty, "/bebop/land", self.land_cb, 10)
        self.sub_emergency = self.create_subscription(Empty, "/bebop/emergency", self.emergency_cb, 10)
        self.sub_reset = self.create_subscription(Empty, "/bebop/reset", self.reset_cb, 10)

        # TF broadcaster
        self.tf_br = TransformBroadcaster(self)

        # Timer principal
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.get_logger().info(
            "Sim Bebop running at 150 Hz "
            "(IMU 30 Hz, XY 15 Hz, Z 10 Hz, dynamics 50 Hz, odom 5 Hz)"
        )

    # ------------------------------
    # CALLBACKS
    # ------------------------------

    def cmd_cb(self, msg: Twist):
        # Solo aceptar comandos a 10 Hz → cada 15 ticks y solo en FLYING
        if self.mode == "FLYING" and self.cmd_counter >= 15:
            self.last_cmd = msg
            self.cmd_counter = 0

    def takeoff_cb(self, _):
        if self.mode in ["IDLE", "LANDING"]:
            self.get_logger().info("TAKEOFF received → TAKING_OFF")
            self.mode = "TAKING_OFF"
            self.last_cmd = None
            self.u[:] = 0.0

    def land_cb(self, _):
        if self.mode in ["FLYING", "TAKING_OFF"]:
            self.get_logger().info("LAND received → LANDING")
            self.mode = "LANDING"
            self.last_cmd = None
            self.u[:] = 0.0

    def emergency_cb(self, _):
        self.get_logger().warn("EMERGENCY received → EMERGENCY")
        self.mode = "EMERGENCY"
        self.last_cmd = None
        self.u[:] = 0.0

    def reset_cb(self, _):
        self.get_logger().warn("RESET received → EMERGENCY-like descent")
        self.mode = "EMERGENCY"
        self.last_cmd = None
        self.u[:] = 0.0

    # ------------------------------
    # LÓGICA DE MODO / CONTROL VERTICAL
    # ------------------------------

    def update_vertical_control(self):
        # Movimiento directo en z SIN dinámica en modos automáticos
        if self.mode == "TAKING_OFF":
            self.x[2] += self.v_takeoff * self.dt
            if self.x[2] >= self.z_target_takeoff:
                self.x[2] = self.z_target_takeoff
                self.xdot[:] = 0.0
                self.mode = "FLYING"
                self.get_logger().info("Reached 1.2 m → FLYING")

        elif self.mode == "LANDING":
            self.x[2] += self.v_land * self.dt
            if self.x[2] <= 0.0:
                self.x[2] = 0.0
                self.xdot[:] = 0.0
                self.mode = "IDLE"
                self.get_logger().info("Landed → IDLE")

        elif self.mode == "EMERGENCY":
            self.x[2] += self.v_emergency * self.dt
            if self.x[2] <= 0.0:
                self.x[2] = 0.0
                self.xdot[:] = 0.0
                self.mode = "IDLE"
                self.get_logger().warn("Emergency descent complete → IDLE")

        elif self.mode == "IDLE":
            # Asegurar que no haya z negativa
            if self.x[2] < 0.0:
                self.x[2] = 0.0
                self.xdot[:] = 0.0

        # En FLYING, z se maneja por dinámica (u[2] puede venir de cmd_vel)

    # ------------------------------
    # DINÁMICA 50 Hz (solo en FLYING)
    # ------------------------------

    def dynamics_step(self):
        yaw = self.x[3]
        c, s = math.cos(yaw), math.sin(yaw)

        F = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])

        xddot = F @ self.u - 0.1 * self.xdot
        self.xdot += xddot * (1.0 / 50.0)
        self.x += self.xdot * (1.0 / 50.0)

        if self.x[2] < 0.0:
            self.x[2] = 0.0
            self.xdot[2] = 0.0

    # ------------------------------
    # PUBLICADORES DE SENSORES
    # ------------------------------

    def publish_xy(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.point.x = float(self.x[0])
        msg.point.y = float(self.x[1])
        self.pub_xy.publish(msg)

    def publish_z(self):
        msg = Float64()
        msg.data = float(self.x[2])
        self.pub_z.publish(msg)

    def publish_imu(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        roll = 0.0
        pitch = 0.0
        yaw = self.x[3]

        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        self.pub_imu.publish(msg)

    def publish_tf(self):
        x, y, z, yaw = self.x
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_br.sendTransform(t)

    def publish_odom(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(self.x[0])
        msg.pose.pose.position.y = float(self.x[1])
        msg.pose.pose.position.z = float(self.x[2])

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.x[3])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(self.xdot[0])
        msg.twist.twist.linear.y = float(self.xdot[1])
        msg.twist.twist.linear.z = float(self.xdot[2])
        msg.twist.twist.angular.z = float(self.xdot[3])

        self.pub_odom.publish(msg)

    # ------------------------------
    # TIMER PRINCIPAL 150 Hz
    # ------------------------------

    def timer_cb(self):
        self.tick += 1
        self.cmd_counter += 1

        # 1) Actualizar control vertical (modos automáticos mueven z directo)
        self.update_vertical_control()

        # 2) Control horizontal/yaw (solo en FLYING)
        if self.mode == "FLYING" and self.last_cmd is not None:
            self.u[0] = self.last_cmd.linear.x
            self.u[1] = self.last_cmd.linear.y
            self.u[2] = self.last_cmd.linear.z  # opcional: control vertical en vuelo
            self.u[3] = self.last_cmd.angular.z
        else:
            self.u[0] = 0.0
            self.u[1] = 0.0
            self.u[3] = 0.0
            if self.mode != "FLYING":
                self.u[2] = 0.0

        # 3) Dinámica 50 Hz solo en FLYING
        if self.tick % 3 == 0 and self.mode == "FLYING":
            self.dynamics_step()

        # 4) Publicadores
        if self.tick % 5 == 0:
            self.publish_imu()
            self.publish_tf()

        if self.tick % 10 == 0:
            self.publish_xy()

        if self.tick % 15 == 0:
            self.publish_z()

        if self.tick % 30 == 0:
            self.publish_odom()


def main(args=None):
    rclpy.init(args=args)
    node = DroneSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
