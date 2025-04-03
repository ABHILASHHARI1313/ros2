#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class ForcePublisherNode(Node):
    def __init__(self):
        super().__init__('force_publisher_node')
        # Publisher for the effort command (force)
        self.publisher_ = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        # Publish at 10 Hz (every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.timer_callback)
        # Set the desired force value (adjust as needed)
        self.force = 10.0  
        self.get_logger().info('Force Publisher Node has started.')

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = [self.force]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing force: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = ForcePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
