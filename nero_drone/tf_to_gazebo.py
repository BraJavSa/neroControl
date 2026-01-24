#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Pose, TransformStamped

class TFtoIgn(Node):
    def __init__(self):
        super().__init__('tf_to_ign')

        # Pose hacia Gazebo/Ignition
        self.pose_pub = self.create_publisher(Pose, '/model/visual_sphere/pose', 10)

        # TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Broadcaster sphere_link
        self.br = TransformBroadcaster(self)

        # 30Hz update
        self.timer = self.create_timer(0.033, self.update)

    def update(self):
        try:
            # base_link en odom
            tf = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())

            # === 1) sphere_link sigue a base_link ===
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = 'sphere_link'
            t.transform = tf.transform
            self.br.sendTransform(t)

            # === 2) publicar pose a ignition ===
            pose = Pose()
            pose.position.x = tf.transform.translation.x
            pose.position.y = tf.transform.translation.y
            pose.position.z = tf.transform.translation.z
            pose.orientation = tf.transform.rotation
            self.pose_pub.publish(pose)

        except Exception:
            return

def main(args=None):
    rclpy.init(args=args)
    node = TFtoIgn()
    rclpy.spin(node)
    rclpy.shutdown()
