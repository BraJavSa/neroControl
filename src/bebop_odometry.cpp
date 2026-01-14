/***************************************************************************************
 * File: tf_to_odom_node.cpp
 * Author: Brayan Saldarriaga-Mesa
 * Description:
 *   This ROS 2 node samples the TF transform between "odom" and "base_link" at 30 Hz
 *   and publishes a stable nav_msgs::msg::Odometry message. The pose is expressed in
 *   the world frame ("odom"), while linear and angular velocities are expressed in the
 *   body frame ("base_link"). Velocities are computed numerically from TF differences.
 *
 *   This node is useful when TF is published at irregular frequencies and a controller
 *   requires a fixed-rate odometry stream with consistent timestamps.
 *
 * ROS Version: ROS 2 Jazzy
 ***************************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

class TFToOdomNode : public rclcpp::Node {
public:
    TFToOdomNode()
    : Node("tf_to_odom_node"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
    {
        // Publisher for the generated odometry
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/bebop/odom_tf", 10);

        // Timer at 30 Hz (33 ms)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&TFToOdomNode::timer_callback, this)
        );

        first_ = true;

        RCLCPP_INFO(this->get_logger(), "TF→Odometry node initialized at 30 Hz.");
    }

private:
    void timer_callback() {
        geometry_msgs::msg::TransformStamped tf;

        // Try to read the transform odom → base_link
        try {
            tf = tf_buffer_.lookupTransform(
                "odom", "base_link", tf2::TimePointZero
            );
        }
        catch (const tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 3000,
                "Failed to lookup TF odom→base_link: %s", ex.what()
            );
            return;
        }

        rclcpp::Time now = this->now();

        // -----------------------------
        // Extract pose from TF
        // -----------------------------
        double x = tf.transform.translation.x;
        double y = tf.transform.translation.y;
        double z = tf.transform.translation.z;

        double qx = tf.transform.rotation.x;
        double qy = tf.transform.rotation.y;
        double qz = tf.transform.rotation.z;
        double qw = tf.transform.rotation.w;

        // -----------------------------
        // Numerical velocity estimation
        // -----------------------------
        double vx = 0, vy = 0, vz = 0;
        double wx = 0, wy = 0, wz = 0;

        if (!first_) {
            double dt = (now - last_time_).seconds();

            if (dt > 0.0001) {
                // Linear velocity in odom frame
                double dx = x - last_x_;
                double dy = y - last_y_;
                double dz = z - last_z_;

                // Convert linear velocity to body frame
                tf2::Quaternion q(qx, qy, qz, qw);
                tf2::Matrix3x3 R(q);

                tf2::Vector3 vel_odom(dx/dt, dy/dt, dz/dt);
                tf2::Vector3 vel_body = R.transpose() * vel_odom;

                vx = vel_body.x();
                vy = vel_body.y();
                vz = vel_body.z();

                // Angular velocity from quaternion difference
                tf2::Quaternion q_prev(last_qx_, last_qy_, last_qz_, last_qw_);
                tf2::Quaternion dq = q_prev.inverse() * q;
                dq.normalize();

                double angle = 2 * acos(dq.w());
                double sin_half = sqrt(1 - dq.w()*dq.w());
                tf2::Vector3 axis(0,0,0);

                if (sin_half > 1e-6) {
                    axis.setX(dq.x() / sin_half);
                    axis.setY(dq.y() / sin_half);
                    axis.setZ(dq.z() / sin_half);
                }

                wx = axis.x() * angle / dt;
                wy = axis.y() * angle / dt;
                wz = axis.z() * angle / dt;
            }
        }

        // -----------------------------
        // Publish odometry message
        // -----------------------------
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = now;
        odom.header.frame_id = "odom";        // world frame
        odom.child_frame_id = "base_link";    // body frame

        // Pose in odom
        odom.pose.pose.position.x = x;
        odom.pose.pose.position.y = y;
        odom.pose.pose.position.z = z;

        odom.pose.pose.orientation.x = qx;
        odom.pose.pose.orientation.y = qy;
        odom.pose.pose.orientation.z = qz;
        odom.pose.pose.orientation.w = qw;

        // Twist in base_link
        odom.twist.twist.linear.x = vx;
        odom.twist.twist.linear.y = vy;
        odom.twist.twist.linear.z = vz;

        odom.twist.twist.angular.x = wx;
        odom.twist.twist.angular.y = wy;
        odom.twist.twist.angular.z = wz;

        odom_pub_->publish(odom);

        // Save previous state
        last_x_ = x;
        last_y_ = y;
        last_z_ = z;

        last_qx_ = qx;
        last_qy_ = qy;
        last_qz_ = qz;
        last_qw_ = qw;

        last_time_ = now;
        first_ = false;
    }

    // ROS interfaces
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // TF listener
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Previous state for velocity estimation
    bool first_;
    double last_x_, last_y_, last_z_;
    double last_qx_, last_qy_, last_qz_, last_qw_;
    rclcpp::Time last_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TFToOdomNode>());
    rclcpp::shutdown();
    return 0;
}
