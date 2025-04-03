import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import xacro
from launch.actions import TimerAction
from launch_ros.actions import Node
from time import sleep
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_cart_description = get_package_share_directory("cart_description")
    pkg_cart_display = get_package_share_directory("cart_display")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    # urdf_file = os.path.join(cart_description_pkg, 'urdf', 'invpend.urdf.xacro')
    world_file_path = os.path.join(pkg_cart_display, "worlds", "invpend.world")
    rviz_config_file = os.path.join(pkg_cart_display, 'rviz', 'real.rviz')

    controller_config_path = os.path.join(pkg_cart_description, "config", "my_controllers.yaml")

    urdf_file_path = os.path.join(pkg_cart_description, "urdf",  "invpend.urdf.xacro")
    robot_description = {"robot_description": xacro.process_file(urdf_file_path).toxml()}

    # Launch Gazebo
    gazebo_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")),
    launch_arguments={"world": world_file_path}.items(),
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "cart_pendulum", "-topic", "robot_description"],
        output="screen"
    )

    # Start controller manager
    control_node = TimerAction(
        period = 5.0,
        actions = [
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[robot_description, controller_config_path],
                output="both",
            )
        ]
    )
    # Joint State Publisher
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )


    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    # # Launch RViz with the provided configuration
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_entity,
        control_node,
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz,
    ])

