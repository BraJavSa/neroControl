#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
import numpy as np
import time


class RefPublisher(Node):
    def __init__(self):
        super().__init__("trajectory_ref_publisher")

        self.pub_ref = self.create_publisher(Float64MultiArray, "/bebop/ref_vec", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.dt = 1.0 / 30.0
        self.T_total = 40.0
        self.omega = 2 * np.pi / self.T_total
        self.t0 = time.time()

        self.first_sample = True
        self.last_yaw = 0.0

        # main timer
        self.timer_ref = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info("Publishing reference trajectory (20 Hz).")

    def timer_callback(self):
        t = time.time() - self.t0
        if t > self.T_total:
            self.get_logger().info("Trajectory completed.")
            self.timer_ref.cancel()
            self.destroy_node()
            return

        w = self.omega

        # position
        x = 1.5 * np.sin(w * t)
        y = 1.5 * np.sin(w * t) * np.cos(w * t)
        z = 1.0 + 0.3 * np.sin(0.5 * w * t) - 0.5

        # previous sample
        x_prev = 1.5 * np.sin(w * (t - self.dt))
        y_prev = 1.5 * np.sin(w * (t - self.dt)) * np.cos(w * (t - self.dt))
        z_prev = 1.0 + 0.3 * np.sin(0.5 * w * (t - self.dt)) - 0.5

        dx = (x - x_prev) / self.dt
        dy = (y - y_prev) / self.dt
        dz = (z - z_prev) / self.dt

        yaw = np.arctan2(dy, dx)
        if self.first_sample:
            self.last_yaw = yaw
            self.first_sample = False

        yaw = np.unwrap([self.last_yaw, yaw])[1]
        wyaw = (yaw - self.last_yaw) / self.dt
        self.last_yaw = yaw

        # publish array ref
        msg = Float64MultiArray()
        msg.data = [x, y, z, yaw, dx, dy, dz, wyaw]
        self.pub_ref.publish(msg)

        # publish TF
        q = quaternion_from_euler(0.0, 0.0, yaw)
        tmsg = TransformStamped()
        tmsg.header.stamp = self.get_clock().now().to_msg()
        tmsg.header.frame_id = "odom"
        tmsg.child_frame_id = "ref"
        tmsg.transform.translation.x = float(x)
        tmsg.transform.translation.y = float(y)
        tmsg.transform.translation.z = float(z)
        tmsg.transform.rotation.x = q[0]
        tmsg.transform.rotation.y = q[1]
        tmsg.transform.rotation.z = q[2]
        tmsg.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(tmsg)


def main():
    rclpy.init()
    node = RefPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
