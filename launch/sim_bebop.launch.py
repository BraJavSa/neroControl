from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('nero_drone')

    urdf_file = os.path.join(pkg_share, 'urdf', 'bebop2.urdf')
    rviz_config = os.path.join(pkg_share, 'others', 'bebop2_sim.rviz')

    return LaunchDescription([

        # === SIMULADOR BEBOP ===
        Node(
            package='nero_drone',
            executable='sim.py',
            name='bebop_sim',
            output='screen'
        ),

        # === NODOS TUYOS ===
        Node(
            package='nero_drone',
            executable='safety_watchdog',
            name='safety_watchdog',
            output='screen'
        ),

        Node(
            package='nero_drone',
            executable='safe_bebop_republisher',
            name='safe_bebop_republisher',
            output='screen'
        ),

        Node(
            package='nero_drone',
            executable='isfly',
            name='isfly',
            output='screen',
        ),

        Node(
            package='nero_drone',
            executable='bebop_control_gui.py',
            name='bebop_control_gui',
            output='screen'
        ),

        Node(
            package='nero_drone',
            executable='tf_cam',
            name='tf_cam',
            output='screen'
        ),

        # === URDF ===
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_file).read()}]
        ),

        # === RVIZ ===
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),

    ])
