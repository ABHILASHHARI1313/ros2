import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration

def generate_launch_description():
    # Get package paths
    cart_description_pkg = get_package_share_directory('cart_description')
    cart_display_pkg = get_package_share_directory('cart_display')
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')

    # Define file paths
    urdf_file = os.path.join(cart_description_pkg, 'urdf', 'invpend.urdf.xacro')
    world_file = os.path.join(cart_display_pkg, 'worlds', 'invpend.world')
    rviz_config_file = os.path.join(cart_display_pkg, 'rviz', 'real.rviz')
    controllers_config = os.path.join(cart_description_pkg, 'config', 'my_controllers.yaml')

    # Generate URDF using xacro
    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([


        # Launch Gazebo with the specified world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'world': world_file}.items(),
        ),

         # Start Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),

        # Wait for robot state publisher to be fully ready
        ExecuteProcess(
            cmd=['sleep', '2'],
            output='screen',
            shell=True
        ),

        # Spawn the robot entity in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'my_robot']
        ),

        # Start ros2_control node with simulation time and robot description
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'robot_description': robot_description},
                controllers_config
            ]
        ),

        # Wait for controller manager to be ready and start controllers
        ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
            output='screen',
            shell=True
        ),

        # Delay to avoid conflicts before loading effort_controller
        ExecuteProcess(
            cmd=['sleep', '3'],  # Wait 3 seconds before executing the next command
            output='screen',
            shell=True
        ),
        ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'effort_controller'],
            output='screen',
            shell=True
        ),

        # Delay to avoid conflicts before loading effort_controller
        ExecuteProcess(
            cmd=['sleep', '3'],  # Wait 3 seconds before executing the next command
            output='screen',
            shell=True
        ),

        ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', ' joint_effort_controller'],
            output='screen',
            shell=True
        ),


        # Launch RViz with the provided configuration
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        ),
    ])
