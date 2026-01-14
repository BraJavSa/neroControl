#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np
import time


class RefPublisher(Node):
    def __init__(self):
        super().__init__("trajectory_ref_publisher")

        # --------------------- Publicadores ---------------------
        self.pub_ref = self.create_publisher(Float64MultiArray, "/bebop/ref_vec", 10)
        self.pub_marker = self.create_publisher(Marker, "/ref_marker", 10)

        # --------------------- Parámetros ---------------------
        self.dt = 1.0 / 30.0   # 30 Hz
        self.hold_time = 12.0   # 8 segundos por pose
        self.t0 = time.time()
        self.idx = 0           # Índice de punto actual

        # --------------------- Puntos del cubo (1.5 m de lado) ---------------------
        L = 1.5
        self.points = np.array([
            [0.0, 0.0, 1.5],        # Centro
            [L/2, L/2, 1.7],        # Esquina superior derecha
            [-L/2, L/2, 1.4],       # Esquina superior izquierda
            [-L/2, -L/2, 1.8],      # Inferior izquierda
            [L/2, -L/2, 1.2]        # Inferior derecha
        ])

        # --------------------- Yaw asociados (en radianes) ---------------------
        self.yaws = np.deg2rad([0, 50, 150, 180, 210])

        self.get_logger().info(
            f"Publicando 5 poses fijas (8 s cada una) en cubo de {L} m con diferentes yaw"
        )

        # --------------------- Temporizador ---------------------
        self.timer = self.create_timer(self.dt, self.timer_callback)

    # ===========================================================
    # Callback periódico
    # ===========================================================
    def timer_callback(self):
        elapsed = time.time() - self.t0

        # Pasar al siguiente punto cada hold_time segundos
        if elapsed > (self.idx + 1) * self.hold_time:
            self.idx += 1
            if self.idx >= len(self.points):
                self.get_logger().info("Secuencia de 5 poses completada.")
                self.timer.cancel()
                self.destroy_node()
                return

        # Pose actual
        pos = self.points[self.idx]
        yaw = self.yaws[self.idx]

        # --------------------- Publicar vector de referencia ---------------------
        msg = Float64MultiArray()
        # [x, y, z, yaw, dx, dy, dz, wyaw] (todas velocidades = 0)
        msg.data = [float(pos[0]), float(pos[1]), float(pos[2]), float(yaw),
                    0.0, 0.0, 0.0, 0.0]
        self.pub_ref.publish(msg)

        # --------------------- Publicar marcador en RViz ---------------------
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ref_point"
        marker.id = self.idx
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        # Convertir yaw a orientación (cuaternión)
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        marker.pose.orientation.z = float(qz)
        marker.pose.orientation.w = float(qw)
        marker.scale.x = 0.3  # largo de la flecha
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 0
        self.pub_marker.publish(marker)


# ===============================================================
# Punto de entrada
# ===============================================================
def main():
    rclpy.init()
    node = RefPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupción por teclado: cerrando nodo.")
    finally:
        if rclpy.ok():
            node.get_logger().info("Apagando contexto ROS2.")
            rclpy.shutdown()


if __name__ == "__main__":
    main()
