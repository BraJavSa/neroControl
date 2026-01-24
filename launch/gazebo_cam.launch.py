from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('nero_drone')

    # URDF dron + RVIZ config
    bebop_urdf = os.path.join(pkg_share, 'urdf', 'bebop2.urdf')
    rviz_config = os.path.join(pkg_share, 'others', 'bebop2_sim.rviz')

    # URDF esfera sólo para Ignition
    sphere_urdf = os.path.join(pkg_share, 'urdf', 'sphere.urdf')

    return LaunchDescription([

        # ================================
        # === TUS NODOS DE SIMULACIÓN ===
        # ================================
        Node(package='nero_drone', executable='sim.py', output='screen'),
        Node(package='nero_drone', executable='safety_watchdog', output='screen'),
        Node(package='nero_drone', executable='safe_bebop_republisher', output='screen'),
        Node(package='nero_drone', executable='isfly', output='screen'),
        Node(package='nero_drone', executable='bebop_control_gui.py', output='screen'),
        Node(package='nero_drone', executable='tf_cam', output='screen'),
        Node(package='nero_drone', executable='ref_vec_filter', output='screen'),

        # ======================
        # === IGNITION (GZ) ===
        # ======================
        ExecuteProcess(
            cmd=[
                'gz', 'sim',
                '-v4',
                '-r',
                'empty.sdf'
            ],
            output='screen'
        ),

        # ===========================
        # === CREA ESFERA VISUAL ===
        # ===========================

        # =====================================================
        # === TF FOLLOWER (sphere_link → base_link via TF)  ===
        # =====================================================
        Node(
            package='nero_drone',
            executable='tf_to_gazebo.py',
            name='sphere_tf_follower',
            output='screen'
        ),

        # =========================================
        # === IMPORTANTE: SIN BRIDGE DE VUELTA ===
        # =========================================
        # Si deseas mover esfera en GZ, sólo ROS->GZ:
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/model/visual_sphere/pose@geometry_msgs/msg/Pose@gz.msgs.Pose[ROSToGZ]'
            ],
            output='screen'
        ),

        # ==========================
        # === TF DEL DRON (URDF) ===
        # ==========================
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': open(bebop_urdf).read()}],
            output='screen'
        ),

        # ==========
        # === RVIZ =
        # ==========
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])
