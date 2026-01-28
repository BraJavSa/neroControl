#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

class MapOdomIdentity : public rclcpp::Node
{
public:
  MapOdomIdentity()
  : Node("map_odom_identity")
  {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    // Publicación a 20 Hz (~50 ms)
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&MapOdomIdentity::broadcast_tf, this)
    );
  }

private:
  void broadcast_tf()
  {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = this->now();
    tf.header.frame_id = "map";
    tf.child_frame_id = "odom";

    // Traslación identidad
    tf.transform.translation.x = 0.0;
    tf.transform.translation.y = 0.0;
    tf.transform.translation.z = 0.0;

    // Rotación identidad (quaternion)
    tf.transform.rotation.x = 0.0;
    tf.transform.rotation.y = 0.0;
    tf.transform.rotation.z = 0.0;
    tf.transform.rotation.w = 1.0;

    tf_broadcaster_->sendTransform(tf);
  }

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapOdomIdentity>());
  rclcpp::shutdown();
  return 0;
}
