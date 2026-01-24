#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <cmath>

class BebopTagNode : public rclcpp::Node {
public:
    BebopTagNode() : Node("bebop_tag_node") {
        tf_ = tag36h11_create();
        td_ = apriltag_detector_create();
        apriltag_detector_add_family(td_, tf_);

        sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/bebop/camera/camera_info", 10,
            std::bind(&BebopTagNode::camera_info_callback, this, std::placeholders::_1));

        sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/bebop/camera/image_raw", 10,
            std::bind(&BebopTagNode::image_callback, this, std::placeholders::_1));

        tag_size_ = 0.165;
        has_prev_pose_ = false;
    }

    ~BebopTagNode() {
        apriltag_detector_destroy(td_);
        tag36h11_destroy(tf_);
    }

private:
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (has_camera_info_) return;

        cameraMatrix_ = (cv::Mat1d(3,3) <<
            msg->k[0], msg->k[1], msg->k[2],
            msg->k[3], msg->k[4], msg->k[5],
            msg->k[6], msg->k[7], msg->k[8]);

        distCoeffs_ = cv::Mat(msg->d).clone();
        has_camera_info_ = true;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!has_camera_info_) return;

        cv::Mat frame;
        try { frame = cv_bridge::toCvCopy(msg, "bgr8")->image; }
        catch (...) { return; }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        image_u8_t im = { gray.cols, gray.rows, gray.cols, gray.data };
        zarray_t* detections = apriltag_detector_detect(td_, &im);

        for (int i = 0; i < zarray_size(detections); i++) {

            apriltag_detection_t* det;
            zarray_get(detections, i, &det);

            // 3D tag corners (meters)
            double s = tag_size_ / 2.0;
            std::vector<cv::Point3f> objectPoints = {
                {-s, -s, 0}, { s, -s, 0}, { s,  s, 0}, { -s,  s, 0}
            };

            // 2D detected corners (pixels)
            std::vector<cv::Point2f> imagePoints = {
                {float(det->p[0][0]), float(det->p[0][1])},
                {float(det->p[1][0]), float(det->p[1][1])},
                {float(det->p[2][0]), float(det->p[2][1])},
                {float(det->p[3][0]), float(det->p[3][1])}
            };

            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, distCoeffs_, rvec, tvec);

            cv::Mat R;
            cv::Rodrigues(rvec, R);

            // Draw tag outline
            for (int j = 0; j < 4; j++)
                cv::line(frame, imagePoints[j], imagePoints[(j+1)%4], cv::Scalar(0,255,0), 2);

            cv::putText(frame, std::to_string(det->id),
                        imagePoints[0], cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0,0,255), 2);

            // Draw coordinate axes
            std::vector<cv::Point3f> axisPoints = {
                {0,0,0}, {0.1f,0,0}, {0,0.1f,0}, {0,0,0.1f}
            };
            std::vector<cv::Point2f> imgpts;
            cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix_, distCoeffs_, imgpts);

            cv::line(frame, imgpts[0], imgpts[1], cv::Scalar(255,0,0), 3);
            cv::line(frame, imgpts[0], imgpts[2], cv::Scalar(0,255,0), 3);
            cv::line(frame, imgpts[0], imgpts[3], cv::Scalar(0,0,255), 3);

            // Compute tag center in pixels
            float u = (imagePoints[0].x + imagePoints[1].x +
                       imagePoints[2].x + imagePoints[3].x) / 4.0f;

            float v = (imagePoints[0].y + imagePoints[1].y +
                       imagePoints[2].y + imagePoints[3].y) / 4.0f;

            float u_des = frame.cols / 2.0f;
            float v_des = frame.rows / 2.0f;

            // Pixel error
            float e_u = u - u_des;
            float e_v = v - v_des;

            cv::putText(frame,
                        "Error (u,v): [" + std::to_string(e_u) + ", " + std::to_string(e_v) + "]",
                        cv::Point(30, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);

            cv::circle(frame, cv::Point(u, v), 5, cv::Scalar(0, 255, 255), -1);
            cv::circle(frame, cv::Point(u_des, v_des), 5, cv::Scalar(255, 0, 0), -1);

            // Tag orientation in the image
            float dx = imagePoints[2].x - imagePoints[1].x;
            float dy = imagePoints[2].y - imagePoints[1].y;

            float theta = std::atan2(dx, -dy);
            float theta_des = 0.0f;

            float e_yaw = theta - theta_des;

            cv::putText(frame,
                        "Error yaw: " + std::to_string(e_yaw),
                        cv::Point(30, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 200, 255), 2);

            // Draw current orientation arrow
            cv::arrowedLine(frame,
                cv::Point(u, v),
                cv::Point(u + 40 * std::sin(theta),
                          v - 40 * std::cos(theta)),
                cv::Scalar(0, 200, 255), 2);

            // Draw desired orientation arrow
            cv::arrowedLine(frame,
                cv::Point(u_des, v_des),
                cv::Point(u_des + 40 * std::sin(theta_des),
                          v_des - 40 * std::cos(theta_des)),
                cv::Scalar(255, 255, 0), 2);

            // Depth error (meters)
            double Z = tvec.at<double>(2);
            double Z_des = 0.30;
            double e_z = Z - Z_des;

            cv::putText(frame,
                        "Error Z: " + std::to_string(e_z) + " m",
                        cv::Point(30, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2);

            // ---------------------------------------------------
            // Pixel error derivatives (pixels/second)
            // ---------------------------------------------------
            double du_dt = 0.0;
            double dv_dt = 0.0;
            double dyaw_dt = 0.0;
            double dt = 0.0;

            rclcpp::Time now = this->now();
            if (has_prev_pose_) {
                dt = (now - prev_time_).seconds();
                if (dt > 1e-4) {
                    du_dt   = (static_cast<double>(e_u)   - prev_e_u_)   / dt;
                    dv_dt   = (static_cast<double>(e_v)   - prev_e_v_)   / dt;
                    dyaw_dt = (static_cast<double>(e_yaw) - prev_e_yaw_) / dt;
                }
            }

            // Display dt
            cv::putText(frame,
                        "dt: " + std::to_string(dt) + " s",
                        cv::Point(30, 210),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 150, 255), 2);

            // Display derivatives
            cv::putText(frame,
                        "de_u/dt: " + std::to_string(du_dt) + " pix/s",
                        cv::Point(30, 240),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2);

            cv::putText(frame,
                        "de_v/dt: " + std::to_string(dv_dt) + " pix/s",
                        cv::Point(30, 270),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2);

            cv::putText(frame,
                        "de_yaw/dt: " + std::to_string(dyaw_dt) + " rad/s",
                        cv::Point(30, 300),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2);

            // Update previous values
            prev_e_u_   = static_cast<double>(e_u);
            prev_e_v_   = static_cast<double>(e_v);
            prev_e_yaw_ = static_cast<double>(e_yaw);
            prev_time_  = now;
            has_prev_pose_ = true;
        }

        apriltag_detections_destroy(detections);
        cv::imshow("Bebop Camera", frame);
        if (cv::waitKey(1) == 'q') rclcpp::shutdown();
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
    apriltag_family_t *tf_;
    apriltag_detector_t *td_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    double tag_size_;
    bool has_camera_info_ = false;

    // Previous error values for derivative computation
    double prev_e_u_   = 0.0;
    double prev_e_v_   = 0.0;
    double prev_e_yaw_ = 0.0;
    rclcpp::Time prev_time_;
    bool has_prev_pose_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BebopTagNode>());
    rclcpp::shutdown();
    return 0;
}
