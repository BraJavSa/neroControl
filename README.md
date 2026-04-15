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
ros2 launch nero_drone full_bebop.launch.py
```

3. Run the Velocity Logger  
```
ros2 run nero_drone velocity_logger
```

4. Publish Reference Trajectory  
Reference format: (x, y, z, yaw, dx, dy, dz, dyaw)  
```
ros2 run nero_drone ref_pos.py
```

5. Offline Plotting (inside nero_drone/nero_drone)  
```
python3 graphical.py
```

6. Real‑Time Plotting (inside nero_drone/nero_drone)  
```
python3 graphs.py
```

## Notes

These commands represent typical usage for experiments. Additional modules or launch files may be required depending on the configuration.

For questions or academic inquiries, please contact: bsaldarriaga@inaut.unsj.edu.ar