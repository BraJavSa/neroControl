// TopicLogger: Nodo ROS2 para registrar posiciones, velocidades, referencias y comandos del Bebop
// con manejo continuo (sin saltos) de yaw y yawd, y cálculo estable de yaw_rate.
// Ahora también filtra suavemente los comandos cmd_linx, cmd_liny, cmd_linz y cmd_angz.
// Guarda automáticamente los logs en ~/ros2_ws/src/neroControl/data
// Autor: Brayan Saldarriaga-Mesa (bsaldarriaga@inaut.unsj.edu.ar)

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <vector>
#include <chrono>
#include <ctime>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class TopicLogger : public rclcpp::Node
{
public:
    TopicLogger()
        : Node("topic_logger_node"),
          first_sample_(true),
          ref_first_sample_(true),
          ref_received_(false),
          odom_received_(false),
          cmd_first_sample_(true),
          yaw_cont_(0.0),
          yawd_cont_(0.0),
          yaw_rate_(0.0),
          cmd_linx_filt_(0.0),
          cmd_liny_filt_(0.0),
          cmd_linz_filt_(0.0),
          cmd_angz_filt_(0.0)
    {
        RCLCPP_INFO(this->get_logger(), "Nodo de registro iniciado (30 Hz, yaw continuo, cálculo estable de yaw_rate, comandos suavizados)");

        // ------------------- Suscripciones -------------------
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10,
            std::bind(&TopicLogger::odomCallback, this, std::placeholders::_1));

        sub_cmd_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/safe_bebop/cmd_vel", 10,
            std::bind(&TopicLogger::cmdCallback, this, std::placeholders::_1));

        sub_ref_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/bebop/ref_vec", 10,
            std::bind(&TopicLogger::refCallback, this, std::placeholders::_1));

        // ------------------- Directorio base ROS2 -------------------
        std::string base_path = std::string(std::getenv("HOME")) + "/ros2_ws/src/neroControl/data";
        fs::create_directories(base_path);  // asegúrate de que exista

        // ------------------- Archivo CSV con fecha y hora -------------------
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm = *std::localtime(&now_time);

        std::ostringstream filename;
        filename << base_path << "/bebop_log_"
                 << std::put_time(&local_tm, "%Y%m%d_%H%M%S")
                 << ".csv";

        logfile_.open(filename.str(), std::ios::out);
        if (!logfile_.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "No se pudo abrir el archivo de log: %s", filename.str().c_str());
            throw std::runtime_error("Error al abrir archivo CSV");
        }

        RCLCPP_INFO(this->get_logger(), "Guardando log en: %s", filename.str().c_str());

        logfile_ << "time,"
                 << "x,y,z,yaw,"
                 << "linx,liny,linz,yaw_rate,"
                 << "xd,yd,zd,yawd,"
                 << "vxd,vyd,vzd,wyawd,"
                 << "cmd_linx,cmd_liny,cmd_linz,cmd_angz\n";

        // ------------------- Temporizador 30 Hz -------------------
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / 30.0),
            std::bind(&TopicLogger::logData, this));
    }

    ~TopicLogger()
    {
        logfile_.close();
    }

private:
    std::mutex mtx_;

    // ------------------- ODOMETRÍA -------------------
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        last_odom_ = *msg;
        odom_received_ = true;

        const auto &pose = msg->pose.pose;
        tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        rclcpp::Time msg_time = msg->header.stamp;
        double t = msg_time.seconds();

        if (!first_sample_)
        {
            double dyaw = std::atan2(std::sin(yaw - last_yaw_), std::cos(yaw - last_yaw_));
            yaw_cont_ += dyaw;

            double dt = t - last_time_;
            if (dt > 1e-3)
            {
                double raw_rate = dyaw / dt;
                yaw_rate_ = 0.9 * yaw_rate_ + 0.1 * raw_rate;
            }
        }
        else
        {
            first_sample_ = false;
            yaw_cont_ = yaw;
            yaw_rate_ = 0.0;
        }

        last_yaw_ = yaw;
        last_time_ = t;
    }

    // ------------------- COMANDOS -------------------
    void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        double alpha = 0.9; // suavizado

        if (!cmd_first_sample_)
        {
            cmd_linx_filt_ = alpha * cmd_linx_filt_ + (1.0 - alpha) * msg->linear.x;
            cmd_liny_filt_ = alpha * cmd_liny_filt_ + (1.0 - alpha) * msg->linear.y;
            cmd_linz_filt_ = alpha * cmd_linz_filt_ + (1.0 - alpha) * msg->linear.z;
            cmd_angz_filt_ = alpha * cmd_angz_filt_ + (1.0 - alpha) * msg->angular.z;
        }
        else
        {
            cmd_first_sample_ = false;
            cmd_linx_filt_ = msg->linear.x;
            cmd_liny_filt_ = msg->linear.y;
            cmd_linz_filt_ = msg->linear.z;
            cmd_angz_filt_ = msg->angular.z;
        }

        last_cmd_ = *msg;
    }

    // ------------------- REFERENCIAS -------------------
    void refCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() == 8)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            double yawd = msg->data[3];
            if (!ref_first_sample_)
            {
                double dyawd = std::atan2(std::sin(yawd - last_yawd_), std::cos(yawd - last_yawd_));
                yawd_cont_ += dyawd;
            }
            else
            {
                ref_first_sample_ = false;
                yawd_cont_ = yawd;
            }

            last_yawd_ = yawd;
            last_ref_ = msg->data;
            ref_received_ = true;
        }
    }

    // ------------------- REGISTRO PERIÓDICO -------------------
    void logData()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        double t = this->now().seconds();

        if (!odom_received_)
            return;

        logfile_ << std::fixed << std::setprecision(6) << t << ",";

        logfile_ << last_odom_.pose.pose.position.x << ","
                 << last_odom_.pose.pose.position.y << ","
                 << last_odom_.pose.pose.position.z << ","
                 << yaw_cont_ << ",";

        logfile_ << last_odom_.twist.twist.linear.x << ","
                 << last_odom_.twist.twist.linear.y << ","
                 << last_odom_.twist.twist.linear.z << ","

                 << yaw_rate_ << ",";

        if (ref_received_)
        {
            logfile_ << last_ref_[0] << ","
                     << last_ref_[1] << ","
                     << last_ref_[2] << ","
                     << yawd_cont_ << ","
                     << last_ref_[4] << ","
                     << last_ref_[5] << ","
                     << last_ref_[6] << ","
                     << last_ref_[7] << ",";
        }
        else
        {
            logfile_ << "0,0,0,0,0,0,0,0,";
        }

        logfile_ << cmd_linx_filt_ << ","
                 << cmd_liny_filt_ << ","
                 << cmd_linz_filt_ << ","
                 << cmd_angz_filt_ << "\n";

        logfile_.flush();
    }

    // ------------------- VARIABLES -------------------
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_ref_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::ofstream logfile_;

    nav_msgs::msg::Odometry last_odom_;
    geometry_msgs::msg::Twist last_cmd_;
    std::vector<double> last_ref_;

    bool first_sample_;
    bool ref_first_sample_;
    bool ref_received_;
    bool odom_received_;
    bool cmd_first_sample_;

    double last_time_;
    double last_yaw_;
    double yaw_cont_;
    double yaw_rate_;
    double last_yawd_;
    double yawd_cont_;

    double cmd_linx_filt_;
    double cmd_liny_filt_;
    double cmd_linz_filt_;
    double cmd_angz_filt_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TopicLogger>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
