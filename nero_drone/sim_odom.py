#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


class DroneSimulator(Node):
    def __init__(self):
        super().__init__("bebop_sim")

        # Tiempos
        self.dt = 1.0 / 150.0   # timer
        self.dt_dyn = 1.0 / 50.0  # dinámica
        self.tick = 0

        # Estado inicial [x, y, z, yaw]
        self.x = np.array([0.0, 0.0, 1.2, 0.0])
        self.xdot = np.zeros(4)
        self.u = np.zeros(4)

        # Modelo identificado (Ku, Kv)
        model_simp = np.array([
            0.8417, 0.18227,
            0.8354, 0.17095,
            3.966,  4.001,
            9.8524, 4.7295
        ])

        self.Ku = np.diag([model_simp[0], model_simp[2], model_simp[4], model_simp[6]])
        self.Kv = np.diag([model_simp[1], model_simp[3], model_simp[5], model_simp[7]])

        # Último cmd_vel
        self.last_cmd = None
        self.cmd_counter = 0

        # Publisher
        self.pub_odom = self.create_publisher(Odometry, "/odometry/filtered", 10)

        # Subscriber de cmd_vel
        self.create_subscription(Twist, "/safe_bebop/cmd_vel", self.cmd_cb, 10)

        # TF
        self.tf_br = TransformBroadcaster(self)

        # Timer
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.get_logger().info("Bebop sim → solo dinámica 50Hz, odom 30Hz")

    def cmd_cb(self, msg):
        self.last_cmd = msg
        self.cmd_counter = 0

    def dynamics_step(self):
        yaw = self.x[3]
        c, s = math.cos(yaw), math.sin(yaw)

        F = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])

        xddot = F @ (self.Ku @ self.u) - self.Kv @ self.xdot
        self.xdot += xddot * self.dt_dyn
        self.x += self.xdot * self.dt_dyn

        # No bajar de z=0
        if self.x[2] < 0:
            self.x[2] = 0
            self.xdot[2] = 0

    def publish_odom(self):
        now = self.get_clock().now()
        x, y, z, yaw = self.x
        vx, vy, vz, wz = self.xdot

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)

        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(vx)
        msg.twist.twist.linear.y = float(vy)
        msg.twist.twist.linear.z = float(vz)
        msg.twist.twist.angular.z = float(wz)

        self.pub_odom.publish(msg)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_br.sendTransform(t)

    def timer_cb(self):
        self.tick += 1
        self.cmd_counter += 1

        if self.last_cmd is not None:
            self.u[0] = self.last_cmd.linear.x
            self.u[1] = self.last_cmd.linear.y
            self.u[2] = self.last_cmd.linear.z
            self.u[3] = self.last_cmd.angular.z
        else:
            self.u[:] = 0.0

        if self.tick % 3 == 0:
            self.dynamics_step()

        if self.tick % 5 == 0:  # 150/5 = 30Hz
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
