#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


class InitialFramePublisher(Node):
    def __init__(self):
        super().__init__('initial_frame_publisher')

        # TF buffer y listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.br = TransformBroadcaster(self)

        self.initial_transform = None
        self.initialized = False

        # Timer para publicar TF a 5 Hz
        self.timer = self.create_timer(1.0 / 5.0, self.publish_tf)

        # Tiempo de inicio
        self.start_time = self.get_clock().now()

    def publish_tf(self):
        now = self.get_clock().now()

        # Si aún no fijamos initial_frame
        if not self.initialized:

            # Esperar 2 segundos SOLO UNA VEZ
            if (now - self.start_time).nanoseconds < 2e9:
                return

            # Intentar obtener TF odom -> base_link_ekf
            try:
                tf_bl = self.tf_buffer.lookup_transform(
                    "odom", "base_link_ekf", rclpy.time.Time()
                )
            except Exception:
                self.get_logger().warn("Esperando TF odom -> base_link_ekf")
                return

            # Guardar la transformación EXACTA
            self.initial_transform = tf_bl
            self.initialized = True
            self.get_logger().info("initial_frame fijado desde TF base_link_ekf.")

        # Publicar TF odom -> initial_frame (estático)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now.to_msg()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "initial_frame"

        tf_msg.transform = self.initial_transform.transform

        self.br.sendTransform(tf_msg)


def main():
    rclpy.init()
    node = InitialFramePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
