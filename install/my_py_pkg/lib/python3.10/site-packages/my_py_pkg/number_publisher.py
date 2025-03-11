#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from example_interfaces.msg import Int64

class NumberPublisher(Node):
    def __init__(self):
        super().__init__("number_publisher")
        self.number = 2
        self.publisher = self.create_publisher(Int64,"number",10)
        self.timer = self.create_timer(1.0,self.publish_numbers)
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

