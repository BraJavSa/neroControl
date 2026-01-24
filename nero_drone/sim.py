#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


def yaw_to_quaternion(yaw: float):
    qz = math.sin(yaw/2.0)
    qw = math.cos(yaw/2.0)
    return (0.0, 0.0, qz, qw)


class Parameters:
    def __init__(self):
        self.Model_simp = np.array([
            0.8417, 0.18227,
            0.8354, 0.17095,
            3.966,  4.001,
            9.8524, 4.7295
        ])
        self.uSat = np.ones(6)


class DroneSimulator(Node):
    def __init__(self):
        super().__init__("bebop_sim")

        self.dt = 1.0/30.0          # simu 30 Hz
        self.odom_div = 6           # odom 30/6 = 5 Hz
        self.odom_cnt = 0

        self.pPar = Parameters()

        self.Ku = np.diag([
            self.pPar.Model_simp[0],
            self.pPar.Model_simp[2],
            self.pPar.Model_simp[4],
            self.pPar.Model_simp[6]
        ])
        self.Kv = np.diag([
            self.pPar.Model_simp[1],
            self.pPar.Model_simp[3],
            self.pPar.Model_simp[5],
            self.pPar.Model_simp[7]
        ])

        # x = [x,y,z,yaw]
        self.x = np.array([0.0, 0.0, 0.0, 0.0], float)
        self.xdot = np.zeros(4, float)
        self.u = np.zeros(4, float)

        self.mode = "IDLE"
        self.takeoff_alt = 1.0
        self.vert_speed = 0.6

        # referencia para derivar odom a 5 Hz
        self.last_odom_x = self.x.copy()
        self.last_odom_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, "/bebop/odom", 10)
        self.sub_cmd = self.create_subscription(Twist, "/bebop/cmd_vel", self.cmd_cb, 10)
        self.sub_takeoff = self.create_subscription(Empty, "/bebop/takeoff", self.takeoff_cb, 10)
        self.sub_land = self.create_subscription(Empty, "/bebop/land", self.land_cb, 10)

        self.tf_br = TransformBroadcaster(self)
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.get_logger().info("Sim Bebop running at 30 Hz, odom @ 5 Hz (vel en base_link)")

    def compute_F(self, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])

    def base_step(self):
        yaw = self.x[3]
        F = self.compute_F(yaw)
        xddot = F @ (self.Ku @ self.u) - self.Kv @ self.xdot
        self.xdot += xddot*self.dt
        self.x += self.xdot*self.dt

    def takeoff_cb(self, _):
        if self.mode in ["IDLE","LANDING"]:
            self.get_logger().info("TAKEOFF")
            self.mode = "TAKING_OFF"
            self.xdot[:] = 0
            self.u[:] = 0

    def land_cb(self, _):
        if self.mode in ["FLYING","TAKING_OFF"]:
            self.get_logger().info("LAND")
            self.mode = "LANDING"
            self.xdot[:] = 0
            self.u[:] = 0

    def cmd_cb(self, msg: Twist):
        if self.mode == "FLYING":
            self.u[:] = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z]
            self.u = np.clip(self.u, -self.pPar.uSat[:4], self.pPar.uSat[:4])

    def timer_cb(self):
        if self.mode == "IDLE":
            self.u[:] = 0
            self.xdot[:] = 0

        elif self.mode == "TAKING_OFF":
            z = self.x[2]
            if z < self.takeoff_alt:
                dz = min(self.vert_speed*self.dt, self.takeoff_alt - z)
                self.x[2] += dz
            else:
                self.x[2] = self.takeoff_alt
                self.mode = "FLYING"
                self.get_logger().info("FLYING")

        elif self.mode == "FLYING":
            self.base_step()

        elif self.mode == "LANDING":
            z = self.x[2]
            if z > 0:
                dz = min(self.vert_speed*self.dt, z)
                self.x[2] -= dz
            else:
                self.x[2] = 0
                self.xdot[:] = 0
                self.mode = "IDLE"
                self.get_logger().info("IDLE")

        self.publish_tf()

        self.odom_cnt += 1
        if self.odom_cnt >= self.odom_div:
            self.publish_odom()
            self.odom_cnt = 0

    def publish_tf(self):
        x,y,z,yaw = self.x

        max_ang = math.radians(5.0)
        pitch = -self.u[0] * max_ang
        roll  =  self.u[1] * max_ang

        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

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
        x,y,z,yaw = self.x
        _,_,qz,qw = yaw_to_quaternion(yaw)

        now = self.get_clock().now()
        dt = (now.nanoseconds - self.last_odom_time.nanoseconds)*1e-9
        if dt <= 0:
            dt = self.odom_div*self.dt

        # derivadas en odom
        vx = (x - self.last_odom_x[0]) / dt
        vy = (y - self.last_odom_x[1]) / dt
        vz = (z - self.last_odom_x[2]) / dt

        dyaw = yaw - self.last_odom_x[3]
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        wz = dyaw / dt

        # convertir a base_link
        c = math.cos(yaw)
        s = math.sin(yaw)
        vx_b =  c*vx + s*vy
        vy_b = -s*vx + c*vy
        vz_b = vz   # simplificado: eje z alinea cuerpo/mundo

        od = Odometry()
        od.header.stamp = now.to_msg()
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"

        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = z
        od.pose.pose.orientation.z = qz
        od.pose.pose.orientation.w = qw

        od.twist.twist.linear.x = vx_b
        od.twist.twist.linear.y = vy_b
        od.twist.twist.linear.z = vz_b
        od.twist.twist.angular.z = wz

        self.odom_pub.publish(od)

        self.last_odom_x = self.x.copy()
        self.last_odom_time = now


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
