#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


class TfToGazebo(Node):
    def __init__(self):
        super().__init__('tf_to_gazebo')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Servicio para mover entidades en Gazebo
        # OJO: si tu mundo no se llama "empty", cambia aquí el nombre
        self.world_name = 'empty'
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)

        self.entity_name = 'sphere_tf'   # nombre con el que la spawneaste en Gazebo

        self.get_logger().info(
            f'Esperando servicio Gazebo: {self.service_name}'
        )

        # Espera (no bloqueante) a que el servicio esté
        self.timer_wait = self.create_timer(0.5, self._wait_for_service)

        # Timer principal (lo activamos cuando el servicio esté listo)
        self.timer_main = None

        # Frames TF
        self.source_frame = 'odom'
        self.target_frame = 'base_link'

    def _wait_for_service(self):
        if self.client.wait_for_service(timeout_sec=0.1):
            self.get_logger().info('Servicio SetEntityPose disponible.')
            self.timer_wait.cancel()
            # 20 Hz es suficiente
            self.timer_main = self.create_timer(0.05, self._update_pose)

    def _update_pose(self):
        # 1) Leemos TF odom -> base_link
        try:
            tf = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            # Solo logueamos de vez en cuando para no spamear
            self.get_logger().throttle_logger(
                self.get_clock(), 2000,
                f'No TF {self.source_frame}->{self.target_frame}: {e}'
            )
            return

        # 2) Construimos la pose de ROS
        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation

        # 3) Llamamos al servicio SetEntityPose
        req = SetEntityPose.Request()
        req.entity = Entity()
        req.entity.name = self.entity_name
        req.entity.type = Entity.MODEL  # importante: es un modelo
        req.pose = pose

        future = self.client.call_async(req)

        # No esperamos la respuesta cada vez; solo logueamos si falla
        def _cb(fut):
            if fut.result() is not None:
                if not fut.result().success:
                    self.get_logger().warn('SetEntityPose devolvió success = False')
            else:
                self.get_logger().warn(f'Error en SetEntityPose: {fut.exception()}')

        future.add_done_callback(_cb)


def main(args=None):
    rclpy.init(args=args)
    node = TfToGazebo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
