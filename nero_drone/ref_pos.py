#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import time


class RefPublisher(Node):
    def __init__(self):
        super().__init__("trajectory_ref_publisher")

        # publishers
        self.pub_ref = self.create_publisher(Float64MultiArray, "/bebop/ref_vec", 10)
        self.tf_br = TransformBroadcaster(self)

        # parameters
        self.dt = 1.0 / 30.0     # 30 Hz
        self.hold_time = 12.0    # seconds per pose
        self.t0 = time.time()
        self.idx = 0

        # cube points
        L = 1.5
        self.points = np.array([
            [0.0, 0.0, 1.5],
            [ L/2,  L/2, 1.7],
            [-L/2,  L/2, 1.4],
            [-L/2, -L/2, 1.8],
            [ L/2, -L/2, 1.2],
        ])

        # yaw angles (rad)
        self.yaws = np.deg2rad([0, 50, 150, 180, 210])

        # timer
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.get_logger().info("Publishing 5 static poses reference (30 Hz).")

    def timer_cb(self):
        elapsed = time.time() - self.t0

        # advance to next pose
        if elapsed > (self.idx + 1) * self.hold_time:
            self.idx += 1
            if self.idx >= len(self.points):
                self.get_logger().info("Reference sequence completed.")
                self.timer.cancel()
                self.destroy_node()
                return

        # current pose
        pos = self.points[self.idx]
        yaw = self.yaws[self.idx]

        # publish ref vector (all velocities = 0)
        msg = Float64MultiArray()
        msg.data = [
            float(pos[0]), float(pos[1]), float(pos[2]), float(yaw),
            0.0, 0.0, 0.0, 0.0
        ]
        self.pub_ref.publish(msg)

        # publish TF odom -> ref
        qz = np.sin(yaw * 0.5)
        qw = np.cos(yaw * 0.5)

        tmsg = TransformStamped()
        tmsg.header.stamp = self.get_clock().now().to_msg()
        tmsg.header.frame_id = "odom"
        tmsg.child_frame_id = "ref"
        tmsg.transform.translation.x = float(pos[0])
        tmsg.transform.translation.y = float(pos[1])
        tmsg.transform.translation.z = float(pos[2])
        tmsg.transform.rotation.z = qz
        tmsg.transform.rotation.w = qw

        self.tf_br.sendTransform(tmsg)


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
