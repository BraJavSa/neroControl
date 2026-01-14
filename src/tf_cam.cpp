// CameraTFWithGimbal: ROS2 node that publishes three TFs (base_link→absolute_cam_link, base_link→camera_link, camera_link→camera_gimbal)
// and a boolean topic /bebop/camera_moving that indicates when the camera is moving, waiting, or within 2 seconds after completing movement.
// Author: Brayan Saldarriaga-Mesa (bsaldarriaga@inaut.unsj.edu.ar), in collaboration with UFV.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

class CameraTFWithGimbal : public rclcpp::Node {
public:
    CameraTFWithGimbal() : Node("camera_tf_with_gimbal") {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        sub_move_camera_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/bebop/move_camera", 10,
            std::bind(&CameraTFWithGimbal::move_camera_callback, this, std::placeholders::_1));

        pub_camera_moving_ = this->create_publisher<std_msgs::msg::Bool>("/bebop/camera_moving", 10);

        timer_ = this->create_wall_timer(20ms, std::bind(&CameraTFWithGimbal::update_camera_pitch, this));

        // Configuración inicial
        min_pitch_deg_ = -90.0;
        max_pitch_deg_ = 15.0;
        min_pitch_ = min_pitch_deg_ * M_PI / 180.0;
        max_pitch_ = max_pitch_deg_ * M_PI / 180.0;
        angular_speed_ = 22.0 * M_PI / 180.0;
        delay_before_move_ = 0.5;
        tilt_offset_ = -10.0 * M_PI / 180.0;
        gimbal_abs_max_ = 20.0 * M_PI / 180.0;
        gimbal_abs_min_ = -100.0 * M_PI / 180.0;
        current_pitch_ = 0.0;
        target_pitch_ = 0.0;
        waiting_ = false;
        moving_ = false;
        extra_publish_duration_ = 2.0;
        movement_end_time_ = this->now();
        last_update_time_ = this->now();
        wait_start_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "CameraTFWithGimbal started using TF from base_link.");
    }

private:
    void move_camera_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        double target_deg = std::clamp(msg->x, min_pitch_deg_, max_pitch_deg_);
        target_pitch_ = target_deg * M_PI / 180.0;
        waiting_ = true;
        moving_ = true;
        wait_start_time_ = this->now();
    }

    void update_camera_pitch() {
        auto now = this->now();
        double dt = (now - last_update_time_).seconds();
        last_update_time_ = now;

        // Fase de espera antes del movimiento
        if (waiting_) {
            if ((now - wait_start_time_).seconds() >= delay_before_move_) {
                waiting_ = false;
            } else {
                publish_absolute_tf();
                publish_camera_tf(current_pitch_);
                publish_camera_moving(true);
                return;
            }
        }

        // Movimiento de la cámara
        if (moving_) {
            double error = target_pitch_ - current_pitch_;
            double direction = (error > 0.0) ? 1.0 : -1.0;
            double delta = direction * angular_speed_ * dt;
            if (std::fabs(error) <= std::fabs(delta)) {
                current_pitch_ = target_pitch_;
                moving_ = false;
                movement_end_time_ = now;
            } else {
                current_pitch_ += delta;
            }
        }

        // Obtener orientación del dron desde TF
        geometry_msgs::msg::TransformStamped tf_base;
        try {
            tf_base = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed: %s", ex.what());
            return;
        }

        // Calcular compensación del gimbal
        tf2::Quaternion q_drone(
            tf_base.transform.rotation.x,
            tf_base.transform.rotation.y,
            tf_base.transform.rotation.z,
            tf_base.transform.rotation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_drone).getRPY(roll, pitch, yaw);

        double gimbal_max = gimbal_abs_max_ - current_pitch_;
        double gimbal_min = gimbal_abs_min_ - current_pitch_;
        double limited_pitch = std::clamp(pitch, gimbal_min, gimbal_max);

        tf2::Quaternion q_pitch_comp, q_roll_comp, q_total_comp;
        q_pitch_comp.setRPY(limited_pitch, 0.0, 0.0);
        q_roll_comp.setRPY(0.0, 0.0, -roll);
        q_total_comp = q_roll_comp * q_pitch_comp;
        q_total_comp.normalize();

        // Publicar TFs
        publish_absolute_tf();
        publish_camera_tf(current_pitch_);
        publish_gimbal_tf(q_total_comp);

        // Publicar estado de movimiento
        bool is_moving = waiting_ || moving_ || (now - movement_end_time_).seconds() < extra_publish_duration_;
        publish_camera_moving(is_moving);
    }

    void publish_camera_moving(bool is_moving) {
        std_msgs::msg::Bool msg;
        msg.data = is_moving;
        pub_camera_moving_->publish(msg);
    }

    void publish_absolute_tf() {
        geometry_msgs::msg::TransformStamped tf_abs;
        tf_abs.header.stamp = this->get_clock()->now();
        tf_abs.header.frame_id = "base_link";
        tf_abs.child_frame_id = "absolute_cam_link";
        tf_abs.transform.translation.x = 0.09;
        tf_abs.transform.translation.y = 0.0;
        tf_abs.transform.translation.z = 0.02;

        tf2::Quaternion q_z, q_x, q_base, q_tilt;
        q_z.setRPY(0.0, 0.0, -M_PI_2);
        q_x.setRPY(0.0, M_PI_2, 0.0);
        q_base = q_x * q_z;
        q_tilt.setRPY(tilt_offset_, 0.0, 0.0);
        tf2::Quaternion q_total = q_base * q_tilt;
        q_total.normalize();

        tf_abs.transform.rotation.x = q_total.x();
        tf_abs.transform.rotation.y = q_total.y();
        tf_abs.transform.rotation.z = q_total.z();
        tf_abs.transform.rotation.w = q_total.w();
        tf_broadcaster_->sendTransform(tf_abs);
    }

    void publish_camera_tf(double pitch_angle) {
        geometry_msgs::msg::TransformStamped tf_cam;
        tf_cam.header.stamp = this->get_clock()->now();
        tf_cam.header.frame_id = "base_link";
        tf_cam.child_frame_id = "camera_link";
        tf_cam.transform.translation.x = 0.09;
        tf_cam.transform.translation.y = 0.0;
        tf_cam.transform.translation.z = 0.02;

        tf2::Quaternion q_z, q_x, q_base;
        q_z.setRPY(0.0, 0.0, -M_PI_2);
        q_x.setRPY(0.0, M_PI_2, 0.0);
        q_base = q_x * q_z;

        tf2::Quaternion q_pitch;
        q_pitch.setRPY(pitch_angle, 0.0, 0.0);
        tf2::Quaternion q_total = q_base * q_pitch;
        q_total.normalize();

        tf_cam.transform.rotation.x = q_total.x();
        tf_cam.transform.rotation.y = q_total.y();
        tf_cam.transform.rotation.z = q_total.z();
        tf_cam.transform.rotation.w = q_total.w();
        tf_broadcaster_->sendTransform(tf_cam);
    }

    void publish_gimbal_tf(const tf2::Quaternion &q_total_comp) {
        geometry_msgs::msg::TransformStamped tf_gimbal;
        tf_gimbal.header.stamp = this->get_clock()->now();
        tf_gimbal.header.frame_id = "camera_link";
        tf_gimbal.child_frame_id = "camera_gimbal";
        tf_gimbal.transform.rotation.x = q_total_comp.x();
        tf_gimbal.transform.rotation.y = q_total_comp.y();
        tf_gimbal.transform.rotation.z = q_total_comp.z();
        tf_gimbal.transform.rotation.w = q_total_comp.w();
        tf_broadcaster_->sendTransform(tf_gimbal);
    }

    // Variables
    double min_pitch_, max_pitch_;
    double min_pitch_deg_, max_pitch_deg_;
    double current_pitch_, target_pitch_;
    double angular_speed_, delay_before_move_;
    double tilt_offset_;
    double gimbal_abs_max_, gimbal_abs_min_;
    bool waiting_, moving_;
    double extra_publish_duration_;
    rclcpp::Time movement_end_time_;
    rclcpp::Time last_update_time_, wait_start_time_;

    // ROS interfaces
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr sub_move_camera_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_camera_moving_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraTFWithGimbal>());
    rclcpp::shutdown();
    return 0;
}
