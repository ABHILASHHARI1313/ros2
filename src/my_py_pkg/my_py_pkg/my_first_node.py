import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init()
    node = Node("py_test")
    node.get_logger().info("Hello Abhilash")
    rclpy.spin(node)    
    rclpy.shutdown()


if __name__ == "__main__":
    main()