#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from example_interfaces.msg import Int64

class NumberPublisher(Node):
    def __init__(self):
        super().__init__("number_publisher")
        self.declare_parameter("number_to_publish",2)
        self.declare_parameter("publish_frequency",1)
        self.number = self.get_parameter("number_to_publish").value
        self.publish_frequency = self.get_parameter("publish_frequency").value
        self.publisher = self.create_publisher(Int64,"number",10)
        self.timer = self.create_timer(1/self.publish_frequency,self.publish_numbers)
        self.get_logger().info("Publishing numbers initiated")
    
    def publish_numbers(self):
        msg = Int64()
        msg.data = self.number
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init()
    node = NumberPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

