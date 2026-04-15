# Nero Drone – ROS 2 Control Framework for Parrot Bebop

This repository provides a ROS 2 control framework for the Parrot Bebop drone. It is designed for controlled test environments where the objective is to command the drone, document its behavior, and visualize experimental results. The system includes high‑rate odometry generation, control modules, reference publishing, data logging, and visualization tools for academic research in autonomous aerial robotics.

## Installation

To install the required Python dependencies, run:

```bash
pip install -r requirements.txt
```

## Launch Instructions

1. Start the Parrot Bebop Driver  

```
ros2 launch ros2_bebop_driver bebop_node_launch.xml ip:=192.168.42.1
```

2. Launch the GUI:  
```
ros2 launch neroControl full_bebop.launch.py
```

3. Run the Velocity Logger  
```
ros2 run neroControl velocity_logger
```

4. Publish Reference Trajectory  
Reference format: (x, y, z, yaw, dx, dy, dz, dyaw)  
```
ros2 run neroControl ref_pos.py
```

5. Offline Plotting (inside neroControl/neroControl)  
```
python3 graphical.py
```

6. Real‑Time Plotting (inside neroControl/neroControl)  
```
python3 graphs.py
```

## Launch Files

The package includes several `.launch.py` files to start different configurations of the system:

| Launch File | Description |
|---|---|
| `full_bebop.launch.py` | Full launch of the Bebop control stack, including tracking nodes and controllers. |
| `joy_full_bebop.launch.py` | Full launch configured to accept manual inputs via a joystick / gamepad. |
| `sim_bebop.launch.py` | Launches the simulated environment for test and validation before real flights. |
| `rviz_control.launch.py` | Launches control nodes alongside RViz visualization tools. |
| `only_control.launch.py` | Starts only the controller nodes (useful if the driver is already running on a separate machine). |
| `cam_control.launch.py` | Specific launch file for initializing camera tracking and movement controls. |

## ROS 2 Topics

### Core Publishers

| Topic | Message Type | Description |
|---|---|---|
| `/safe_bebop/cmd_vel` | `geometry_msgs/msg/Twist` | Computed, bounded velocity commands sent from the active controller. |
| `/bebop/ref_vec` | `std_msgs/msg/Float64MultiArray` | The current reference trajectory array values for the drone to follow. |
| `/bebop/is_flying` | `std_msgs/msg/Bool` | Indicates whether the drone is currently airborne. |
| `/bebop/detected` | `std_msgs/msg/Bool` | Indicates whether the visual target (e.g., AprilTag) is detected in the camera view. |
| `/bebop/camera_moving` | `std_msgs/msg/Bool` | Publishes the state indicating if the camera is currently moving. |
| `/bebop/move_camera` | `geometry_msgs/msg/Vector3` | Commands for moving the onboard camera (tilt/pan). |
| `/reference_marker` | `visualization_msgs/msg/Marker` | Visual marker of the reference position, visible in RViz. |
| `/tag_odom` | `nav_msgs/msg/Odometry` | Computed odometry relative to the detected visual target. |

*(Note: The internal simulator scripts also publish raw topics like `/bebop/position`, `/bebop/altitude`, `/bebop/imu`, and `/bebop/odom`).*

### Core Subscribers

| Topic | Message Type | Description |
|---|---|---|
| `/odometry/filtered` | `nav_msgs/msg/Odometry` | State data (pose/twist) from the EKF filter, consumed by the control modules. |
| `/bebop/ref_vec` | `std_msgs/msg/Float64MultiArray` | Control nodes read the target reference from this topic. |
| `/joy` | `sensor_msgs/msg/Joy` | Reads joystick inputs for manual override or commanding. |
| `/bebop/cmd_vel` | `geometry_msgs/msg/Twist` | Used by the simulator or bridge nodes to capture base commands. |
| `/bebop/takeoff` | `std_msgs/msg/Empty` | Subscribed by the driver and simulator nodes to take off. |
| `/bebop/land` | `std_msgs/msg/Empty` | Subscribed by the driver and simulator nodes to land. |
| `/bebop/emergency` | `std_msgs/msg/Empty` | Subscribed by the driver and simulator nodes to instantly cut motors. |

## Notes

These commands represent typical usage for experiments. Additional modules or launch files may be required depending on the configuration.

For questions or academic inquiries, please contact: bsaldarriaga@inaut.unsj.edu.ar