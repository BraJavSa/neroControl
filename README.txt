# Nero Drone – ROS 2 Control Framework for Parrot Bebop

This repository provides a ROS 2 control framework for the Parrot Bebop drone. It is designed for controlled test environments where the objective is to command the drone, document its behavior, and visualize experimental results. The system includes high‑rate odometry generation, control modules, reference publishing, data logging, and visualization tools for academic research in autonomous aerial robotics.

## Launch Instructions

### 1. Start the Parrot Bebop Driver
```bash
ros2 launch ros2_bebop_driver bebop_node_launch.xml ip:=192.168.42.1
```