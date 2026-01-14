#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
import numpy as np
import time


class RefPublisher(Node):
    def __init__(self):
        super().__init__("trajectory_ref_publisher")

        self.pub_ref = self.create_publisher(Float64MultiArray, "/bebop/ref_vec", 10)
        self.pub_line = self.create_publisher(Marker, "/ref_line", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.dt = 1.0 / 20.0
        self.T_total = 40.0
        self.omega = 2 * np.pi / self.T_total
        self.t0 = time.time()

        self.first_sample = True
        self.last_yaw = 0.0

        # Trayectoria completa precalculada
        self.traj_points = self.generate_full_trajectory()

        # Temporizadores
        self.timer_ref = self.create_timer(self.dt, self.timer_callback)
        self.timer_line = self.create_timer(1.0, self.publish_full_trajectory_marker)

        self.get_logger().info(
            "Publicando trayectoria tipo 8 en 'odom' a 20 Hz:\n"
            "- Frame TF dinámico 'ref'\n"
            "- Línea amarilla fina republicada cada 1 s en /ref_line"
        )

    def generate_full_trajectory(self):
        w = self.omega
        times = np.arange(0, self.T_total, self.dt)
        points = []
        for t in times:
            x = 1.5 * np.sin(w * t)
            y = 1.5 * np.sin(w * t) * np.cos(w * t)
            z = 1.0 + 0.3 * np.sin(0.5 * w * t) - 0.5
            points.append(Point(x=float(x), y=float(y), z=float(z)))
        return points

    def publish_full_trajectory_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Línea más fina
        marker.scale.x = 0.01

        # Color amarillo (R=1, G=1, B=0)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = self.traj_points
        marker.lifetime.sec = 0
        self.pub_line.publish(marker)

    def timer_callback(self):
        t = time.time() - self.t0
        if t > self.T_total:
            self.get_logger().info("Trayectoria completada.")
            self.timer_ref.cancel()
            self.timer_line.cancel()
            self.destroy_node()
            return

        w = self.omega

        # Posición deseada actual
        x = 1.5 * np.sin(w * t)
        y = 1.5 * np.sin(w * t) * np.cos(w * t)
        z = 1.0 + 0.3 * np.sin(0.5 * w * t) - 0.5

        # Posición anterior
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

        # Publicar vector de referencia
        msg = Float64MultiArray()
        msg.data = [x, y, z, yaw, dx, dy, dz, wyaw]
        self.pub_ref.publish(msg)

        # Publicar TF dinámico
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
            node.get_logger().info("Apagando contexto ROS2.")
            rclpy.shutdown()


if __name__ == "__main__":
    main()
