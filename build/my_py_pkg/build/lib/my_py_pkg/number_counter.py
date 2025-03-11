#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from example_interfaces.msg import Int64
from example_interfaces.srv import SetBool


class NumberReciever(Node):
    def __init__(self):
        super().__init__("number_counter")
        self.counter = 2
        self.num_publisher = self.create_publisher(Int64, "number_count", 10)
        self.subscriber = self.create_subscription(
            Int64, "number", self.callback_number, 10
        )
        self.get_logger().info("Receiving numbers started")
        self.server = self.create_service(SetBool,"reset_counter",self.callback_reset_counter)

    def callback_number(self, msg):
        self.counter += msg.data
        new_msg = Int64()
        new_msg.data += self.counter
        self.num_publisher.publish(new_msg)

    def callback_reset_counter(self,request,response):
        if request.data == True:
            self.counter = 0
            response.success = True
            response.message = "Counter has been reset."
        else:
            response.success = False
            response.message = "Counter has not been reset"
        return response

def main(args=None):
    rclpy.init()
    node = NumberReciever()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
