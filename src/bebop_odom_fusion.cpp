#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/imu.hpp>

class BebopOdomFusion : public rclcpp::Node
{
public:
  BebopOdomFusion() : Node("bebop_odom_fusion")
  {
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/bebop/imu", 10,
      [this](sensor_msgs::msg::Imu::SharedPtr msg){
        last_imu_ = *msg;
        imu_ready_ = true;
      });

    sub_pos_ = create_subscription<geometry_msgs::msg::Point>(
      "/bebop/position", 10,
      [this](geometry_msgs::msg::Point::SharedPtr msg){
        last_pos_ = *msg;
        pos_ready_ = true;
      });

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(
      "/bebop/odom_fast", 10);

    timer_ = create_wall_timer(
      std::chrono::milliseconds(33),
      std::bind(&BebopOdomFusion::tick, this));
  }

private:
  void tick()
  {
    if(!imu_ready_ || !pos_ready_)
      return;

    auto nowt = this->now();

    double vx=0, vy=0, vz=0;
    if(has_last_pos_)
    {
      double dt = (nowt - last_stamp_).seconds();
      if(dt > 0.0)
      {
        vx = (last_pos_.x - prev_pos_.x) / dt;
        vy = (last_pos_.y - prev_pos_.y) / dt;
        vz = (last_pos_.z - prev_pos_.z) / dt;
      }
    }
    prev_pos_ = last_pos_;
    has_last_pos_ = true;
    last_stamp_ = nowt;

    nav_msgs::msg::Odometry od;
    od.header.stamp = nowt;
    od.header.frame_id = "odom";
    od.child_frame_id = "bebop_link";

    od.pose.pose.position.x = last_pos_.x;
    od.pose.pose.position.y = last_pos_.y;
    od.pose.pose.position.z = last_pos_.z;
    od.pose.pose.orientation = last_imu_.orientation;

    od.twist.twist.linear.x = vx;
    od.twist.twist.linear.y = vy;
    od.twist.twist.linear.z = vz;

    pub_odom_->publish(od);
  }

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr sub_pos_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::TimerBase::SharedPtr timer_;

  sensor_msgs::msg::Imu last_imu_;
  geometry_msgs::msg::Point last_pos_;
  geometry_msgs::msg::Point prev_pos_;

  bool imu_ready_ = false;
  bool pos_ready_ = false;
  bool has_last_pos_ = false;

  rclcpp::Time last_stamp_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BebopOdomFusion>());
  rclcpp::shutdown();
  return 0;
}
