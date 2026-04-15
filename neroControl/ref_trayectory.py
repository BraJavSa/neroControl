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

        # Estado persistente para continuidad de ángulo
        self.first_sample = True
        self.last_unwrapped_yaw = 0.0
        self.last_raw_yaw = 0.0

        self.timer_ref = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("Publishing infinite continuous trajectory (30 Hz).")

    def timer_callback(self):
        t_raw = time.time() - self.t0
        t = t_raw % self.T_total 
        w = self.omega

        # Ecuaciones de posición
        x = 0.8 * np.sin(w * t)
        y = 0.8 * np.sin(w * t) * np.cos(w * t)
        z = 1.0 + 0.3 * np.sin(0.5 * w * t) 

        # Diferenciación para velocidad tangencial
        t_prev = (t - self.dt) % self.T_total
        x_prev = 0.8 * np.sin(w * t_prev)
        y_prev = 0.8 * np.sin(w * t_prev) * np.cos(w * t_prev)
        z_prev = 1.0 + 0.3 * np.sin(0.5 * w * t_prev) 

        dx = (x - x_prev) / self.dt
        dy = (y - y_prev) / self.dt
        dz = (z - z_prev) / self.dt

        # --- Lógica de Yaw Continuo ---
        current_raw_yaw = np.arctan2(dy, dx)
        
        if self.first_sample:
            self.last_raw_yaw = current_raw_yaw
            self.last_unwrapped_yaw = current_raw_yaw
            self.first_sample = False

        # Calcular diferencia de ángulo y normalizar al rango [-pi, pi]
        delta_yaw = current_raw_yaw - self.last_raw_yaw
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # Acumular el cambio para obtener un valor continuo (unwrapped)
        current_unwrapped_yaw = self.last_unwrapped_yaw + delta_yaw
        
        # Velocidad angular
        wyaw = (current_unwrapped_yaw - self.last_unwrapped_yaw) / self.dt

        # Actualizar estados previos
        self.last_raw_yaw = current_raw_yaw
        self.last_unwrapped_yaw = current_unwrapped_yaw

        # Publicación
        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(z), float(current_unwrapped_yaw), 
                    float(dx), float(dy), float(dz), float(wyaw)]
        self.pub_ref.publish(msg)

        # TF Broadcast
        q = quaternion_from_euler(0.0, 0.0, current_unwrapped_yaw)
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