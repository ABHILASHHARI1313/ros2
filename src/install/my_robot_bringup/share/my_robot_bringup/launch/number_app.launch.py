from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    remap_number_topic = ("number","my_number")
    number_publisher_node = Node(
    package="my_py_pkg",
    executable="number_publisher",
    name="my_number_publisher", #remapping nodes
    remappings=[
        remap_number_topic
        ], #remapping topics
    parameters=[{
        "number_to_publish": 2,
        "publish_frequency": 1
    }])


    number_counter_node = Node(
        package="my_py_pkg",
        name="my_number_counter", #remapping nodes
        remappings=[
            remap_number_topic,
            ("number_count","my_number_count")
        ], #remapping topics
        executable="number_counter"
    )

    ld.add_action(number_publisher_node)
    ld.add_action(number_counter_node)
    return ld 