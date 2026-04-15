from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('neroControl')
    urdf_file = os.path.join(pkg_share, 'urdf', 'bebop2.urdf')
    rviz_config = os.path.join(pkg_share, 'others', 'bebop2.rviz')

    return LaunchDescription([

        # --- Nodo del joystick ---
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),

        # --- Nodo de traducción del joystick a comandos ---
        Node(
            package='neroControl',
            executable='Joy2Cmd',
            name='joy2cmd',
            output='screen'
        ),

        # --- Nodo TF de la cámara ---
        Node(
            package='neroControl',
            executable='tf_cam',
            name='tf_cam',
            output='screen'
        ),

        # --- Nodo TF del tag ---
        Node(
            package='neroControl',
            executable='tf_tag_bebop',
            name='tf_tag_bebop',
            output='screen'
        ),

        # --- Robot State Publisher ---
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_file).read()}]
        ),

        # --- RViz para visualización ---
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),
    ])
