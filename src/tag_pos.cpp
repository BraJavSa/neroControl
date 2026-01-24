// tag_pos.cpp
// Nodo ROS2 para detectar AprilTags en la cámara del Bebop,
// estimar posición y velocidad de cada tag en el mundo (odom)
// usando un filtro alpha-beta.
//
// Author: Brayan Saldarriaga-Mesa

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <unordered_map>
#include <unordered_set>
#include <cmath>

class BebopTagWorldTrackerNode : public rclcpp::Node {
public:
    BebopTagWorldTrackerNode()
    : Node("bebop_tag_world_tracker_node")
    {
        // Parámetros del filtro alpha-beta
        alpha_ = this->declare_parameter<double>("alpha", 0.7);
        beta_  = this->declare_parameter<double>("beta", 0.2);

        // Tamaño físico del tag en metros (lado)
        tag_size_ = this->declare_parameter<double>("tag_size", 0.12);

        // Parámetros para manejar pérdida de detecciones
        vel_lambda_ = this->declare_parameter<double>("vel_lambda", 1.0);         // 1/s
        max_missing_frames_ = this->declare_parameter<int>("max_missing_frames", 10);

        // Frame de mundo y cámara
        world_frame_  = this->declare_parameter<std::string>("world_frame",  "odom");
        camera_frame_ = this->declare_parameter<std::string>("camera_frame", "camera_link");

        // Inicializar AprilTag
        tag_family_ = tag36h11_create();
        tag_detector_ = apriltag_detector_create();
        apriltag_detector_add_family(tag_detector_, tag_family_);

        // TF2 buffer + listener
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Suscripciones
        sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/bebop/camera/camera_info", 10,
            std::bind(&BebopTagWorldTrackerNode::camera_info_callback, this, std::placeholders::_1));

        sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/bebop/camera/image_raw", 10,
            std::bind(&BebopTagWorldTrackerNode::image_callback, this, std::placeholders::_1));

        // Publicador de odometría de tags en odom
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("tag_odom", 10);

        RCLCPP_INFO(this->get_logger(),
                    "BebopTagWorldTrackerNode started. alpha=%.3f, beta=%.3f, tag_size=%.3f m, "
                    "vel_lambda=%.3f, max_missing_frames=%d, world_frame=%s, camera_frame=%s",
                    alpha_, beta_, tag_size_, vel_lambda_, max_missing_frames_,
                    world_frame_.c_str(), camera_frame_.c_str());
    }

    ~BebopTagWorldTrackerNode() override
    {
        apriltag_detector_destroy(tag_detector_);
        tag36h11_destroy(tag_family_);
    }

private:
    struct AlphaBetaState {
        cv::Vec3d x;          // posición estimada en mundo (odom) [m]
        cv::Vec3d v;          // velocidad estimada en mundo (odom) [m/s]
        geometry_msgs::msg::Quaternion orientation;  // última orientación medida en mundo
        rclcpp::Time last_time;
        int frames_missing = 0;
        bool initialized = false;
        bool valid = false;
    };

    // ===== Callbacks =====

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        // Solo usar la primera vez: la intrínseca de la cámara no cambia
        if (has_camera_info_) return;

        RCLCPP_INFO(this->get_logger(), "Camera info received. K and D set once.");

        camera_matrix_ = (cv::Mat1d(3,3) <<
            msg->k[0], msg->k[1], msg->k[2],
            msg->k[3], msg->k[4], msg->k[5],
            msg->k[6], msg->k[7], msg->k[8]);

        dist_coeffs_ = cv::Mat(msg->d).clone();

        has_camera_info_ = true;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!has_camera_info_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Waiting for camera info...");
            return;
        }

        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        rclcpp::Time stamp(msg->header.stamp);

        // Detectar tags
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        image_u8_t im = { gray.cols, gray.rows, gray.cols, gray.data };
        zarray_t* detections = apriltag_detector_detect(tag_detector_, &im);

        // Para saber qué tags se vieron en este frame
        std::unordered_set<int> seen_ids;

        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);

            int id = det->id;
            seen_ids.insert(id);

            // Geometría del tag: 4 esquinas en 3D (plano z=0, tamaño tag_size_)
            double s = tag_size_ / 2.0;
            std::vector<cv::Point3f> object_points = {
                {-s, -s, 0}, { s, -s, 0}, { s,  s, 0}, { -s,  s, 0}
            };

            // Esquinas detectadas en imagen
            std::vector<cv::Point2f> image_points = {
                {static_cast<float>(det->p[0][0]), static_cast<float>(det->p[0][1])},
                {static_cast<float>(det->p[1][0]), static_cast<float>(det->p[1][1])},
                {static_cast<float>(det->p[2][0]), static_cast<float>(det->p[2][1])},
                {static_cast<float>(det->p[3][0]), static_cast<float>(det->p[3][1])}
            };

            // Pose del tag respecto a la cámara (frame = camera_frame_)
            cv::Mat rvec, tvec;
            cv::solvePnP(object_points, image_points,
                         camera_matrix_, dist_coeffs_,
                         rvec, tvec);

            // Matriz de rotación en cámara
            cv::Mat R_cam_tag;
            cv::Rodrigues(rvec, R_cam_tag);

            // Posición medida en cámara
            cv::Vec3d pos_cam(
                tvec.at<double>(0),
                tvec.at<double>(1),
                tvec.at<double>(2)
            );

            // Convertir R_cam_tag a quaternion
            tf2::Matrix3x3 m_ct(
                R_cam_tag.at<double>(0,0), R_cam_tag.at<double>(0,1), R_cam_tag.at<double>(0,2),
                R_cam_tag.at<double>(1,0), R_cam_tag.at<double>(1,1), R_cam_tag.at<double>(1,2),
                R_cam_tag.at<double>(2,0), R_cam_tag.at<double>(2,1), R_cam_tag.at<double>(2,2)
            );
            tf2::Quaternion q_ct;
            m_ct.getRotation(q_ct);

            // Pose del tag en el frame de la cámara (camera_link)
            geometry_msgs::msg::PoseStamped pose_cam;
            pose_cam.header.stamp = stamp;
            pose_cam.header.frame_id = camera_frame_;
            pose_cam.pose.position.x = pos_cam[0];
            pose_cam.pose.position.y = pos_cam[1];
            pose_cam.pose.position.z = pos_cam[2];
            pose_cam.pose.orientation = tf2::toMsg(q_ct);

            // Transformar pose a mundo (world_frame_)
            geometry_msgs::msg::PoseStamped pose_world;
            try {
                geometry_msgs::msg::TransformStamped tf_cam_to_world =
                    tf_buffer_->lookupTransform(world_frame_, camera_frame_, stamp);

                tf2::doTransform(pose_cam, pose_world, tf_cam_to_world);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "TF transform error %s -> %s: %s",
                            camera_frame_.c_str(), world_frame_.c_str(), ex.what());
                // No podemos usar esta detección en mundo, saltamos al siguiente tag
                continue;
            }

            // Posición medida en mundo (odom)
            cv::Vec3d pos_world(
                pose_world.pose.position.x,
                pose_world.pose.position.y,
                pose_world.pose.position.z
            );

            // === Alpha-Beta filter por tag EN MUNDO (odom) ===
            auto &state = states_[id];

            if (!state.initialized) {
                // Primera vez: inicializar con la medida en mundo
                state.x = pos_world;
                state.v = cv::Vec3d(0.0, 0.0, 0.0);
                state.orientation = pose_world.pose.orientation;
                state.last_time = stamp;
                state.frames_missing = 0;
                state.initialized = true;
                state.valid = true;
            } else {
                double dt = (stamp - state.last_time).seconds();

                if (dt <= 0.0) {
                    // Si algo raro pasa con el tiempo, solo actualizar medida cruda
                    state.x = pos_world;
                    state.orientation = pose_world.pose.orientation;
                } else {
                    // Predicción
                    cv::Vec3d x_pred = state.x + state.v * dt;
                    cv::Vec3d v_pred = state.v;

                    // Residuo
                    cv::Vec3d r = pos_world - x_pred;

                    // Corrección (componente a componente)
                    cv::Vec3d x_new = x_pred + alpha_ * r;
                    cv::Vec3d v_new = v_pred + (beta_ / dt) * r;

                    state.x = x_new;
                    state.v = v_new;
                    state.orientation = pose_world.pose.orientation;  // orientación medida
                }

                state.last_time = stamp;
                state.frames_missing = 0;
                state.valid = true;
            }

            double dist_meas_world = cv::norm(pos_world);
            double dist_filt_world = cv::norm(state.x);

            // Loguear en marco de mundo
            RCLCPP_INFO(this->get_logger(),
                "Tag %d (SEEN, world=%s): "
                "MeasWorld[%.3f, %.3f, %.3f] m, DistMeas=%.2f m | "
                "FiltWorld[%.3f, %.3f, %.3f] m, VelWorld[%.3f, %.3f, %.3f] m/s, DistFilt=%.2f m | "
                "valid=%d missing=%d",
                id, world_frame_.c_str(),
                pos_world[0], pos_world[1], pos_world[2], dist_meas_world,
                state.x[0], state.x[1], state.x[2],
                state.v[0], state.v[1], state.v[2], dist_filt_world,
                state.valid ? 1 : 0,
                state.frames_missing
            );

            // Publicar odometría en mundo
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = stamp;
            odom_msg.header.frame_id = world_frame_;                 // marco de referencia: odom
            odom_msg.child_frame_id = "tag_" + std::to_string(id);   // identificador del tag

            odom_msg.pose.pose.position.x = state.x[0];
            odom_msg.pose.pose.position.y = state.x[1];
            odom_msg.pose.pose.position.z = state.x[2];
            odom_msg.pose.pose.orientation = state.orientation;

            odom_msg.twist.twist.linear.x = state.v[0];
            odom_msg.twist.twist.linear.y = state.v[1];
            odom_msg.twist.twist.linear.z = state.v[2];

            odom_pub_->publish(odom_msg);

            // Dibujar el tag en la imagen para debug
            for (int j = 0; j < 4; j++) {
                cv::line(frame, image_points[j], image_points[(j+1)%4],
                         cv::Scalar(0, 255, 0), 2);
            }
            cv::putText(frame, std::to_string(id),
                        image_points[0], cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Ya procesamos detecciones.
        // Para los tags que NO se vieron en este frame:
        // solo predicción + decay de velocidad EN MUNDO.
        for (auto &kv : states_) {
            int id = kv.first;
            auto &state = kv.second;

            if (!state.initialized)
                continue;

            // Si se vio en este frame, ya está actualizado arriba
            if (seen_ids.find(id) != seen_ids.end())
                continue;

            double dt = (stamp - state.last_time).seconds();
            if (dt <= 0.0)
                continue;

            // Predicción pura en mundo
            state.x = state.x + state.v * dt;

            // Decaimiento exponencial de velocidad: v <- v * exp(-lambda * dt)
            double decay = std::exp(-vel_lambda_ * dt);
            state.v *= decay;

            state.last_time = stamp;
            state.frames_missing++;

            if (state.frames_missing > max_missing_frames_) {
                state.valid = false;
            }

            RCLCPP_DEBUG(this->get_logger(),
                "Tag %d (NOT SEEN, world=%s): PredOnly FiltWorld[%.3f, %.3f, %.3f] m, "
                "VelWorld[%.3f, %.3f, %.3f] m/s | valid=%d missing=%d",
                id, world_frame_.c_str(),
                state.x[0], state.x[1], state.x[2],
                state.v[0], state.v[1], state.v[2],
                state.valid ? 1 : 0,
                state.frames_missing
            );

            // Si quieres, puedes comentar este bloque si NO quieres publicar
            // cuando el tag no se ve (solo estimación predicha).
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = stamp;
            odom_msg.header.frame_id = world_frame_;
            odom_msg.child_frame_id = "tag_" + std::to_string(id);

            odom_msg.pose.pose.position.x = state.x[0];
            odom_msg.pose.pose.position.y = state.x[1];
            odom_msg.pose.pose.position.z = state.x[2];
            odom_msg.pose.pose.orientation = state.orientation;

            odom_msg.twist.twist.linear.x = state.v[0];
            odom_msg.twist.twist.linear.y = state.v[1];
            odom_msg.twist.twist.linear.z = state.v[2];

            odom_pub_->publish(odom_msg);
        }

        apriltag_detections_destroy(detections);

        cv::imshow("Bebop Tag World Tracker", frame);
        if (cv::waitKey(1) == 'q') {
            rclcpp::shutdown();
        }
    }

    // ===== Miembros =====

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       sub_image_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          odom_pub_;

    apriltag_family_t  *tag_family_ = nullptr;
    apriltag_detector_t *tag_detector_ = nullptr;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool has_camera_info_ = false;

    double tag_size_;
    double alpha_;
    double beta_;

    double vel_lambda_;          // tasa de decaimiento de velocidad (1/s)
    int    max_missing_frames_;  // máximo número de frames sin ver el tag antes de invalidar

    std::string world_frame_;    // normalmente "odom"
    std::string camera_frame_;   // aquí "camera_link"

    std::shared_ptr<tf2_ros::Buffer>          tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::unordered_map<int, AlphaBetaState> states_;  // clave: id del tag
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BebopTagWorldTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
