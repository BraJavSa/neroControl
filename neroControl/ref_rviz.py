#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker

class ReferenceMarker(Node):
    def __init__(self):
        super().__init__('reference_marker_node')

        # Suscripción a las referencias
        self.sub_ref = self.create_subscription(
            Float64MultiArray, '/bebop/ref_vec', self.ref_callback, 10)

        # Publicador de marcador para RViz
        self.marker_pub = self.create_publisher(Marker, '/reference_marker', 10)

        # Crear objeto marcador base
        self.marker = Marker()
        self.marker.header.frame_id = 'odom'  # o el frame de tu odometría
        self.marker.ns = 'reference'
        self.marker.id = 0
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD

        # Tamaño de la esfera (5 cm de radio)
        self.marker.scale.x = 0.05
        self.marker.scale.y = 0.05
        self.marker.scale.z = 0.05

        # Color verde con opacidad total
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

        self.marker.pose.orientation.w = 1.0

        self.get_logger().info("Reference marker node started — green sphere, 5 cm radius")

    def ref_callback(self, msg):
        if len(msg.data) >= 3:
            # Actualizar posición del marcador
            self.marker.header.stamp = self.get_clock().now().to_msg()
            self.marker.pose.position.x = msg.data[0]
            self.marker.pose.position.y = msg.data[1]
            self.marker.pose.position.z = msg.data[2]

            self.marker_pub.publish(self.marker)

def main(args=None):
    rclpy.init(args=args)
    node = ReferenceMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
