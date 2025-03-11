import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        self.counter = 0
        super().__init__("py_test")
        self.create_timer(0.9,self.term_callback)

    
    def term_callback(self):
        self.counter += 1
        self.get_logger().info("Hello Friend !!"+str(self.counter))

def main(args=None):
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()