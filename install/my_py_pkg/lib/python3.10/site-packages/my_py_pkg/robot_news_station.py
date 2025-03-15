#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.msg import String


class RobotNewsStationNode(Node):
    def __init__(self):
        self.counter = 0
        super().__init__("robot_news_station")
        self.declare_parameter("robot_name")
        self.robot_name = self.get_parameter("robot_name").value
        self.publisher = self.create_publisher(String,"robot_news",10)
        self.timer = self.create_timer(0.5,self.publish_news)
        self.get_logger().info("Robot news station has been started")


    def publish_news(self):
        msg = String()
        msg.data = "Hello this is "+str(self.robot_name)+" from robot news station."
        self.publisher.publish(msg)        

def main(args=None):
    rclpy.init()    
    node = RobotNewsStationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()



