import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateListener(Node):
    def __init__(self):
        super().__init__('joint_state_listener')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def joint_state_callback(self, msg):
        # Specify the joint you are interested in
        joint_name = "cart_to_pole"  # Replace with your joint name

        if joint_name in msg.name:
            joint_index = msg.name.index(joint_name)
            joint_position = msg.position[joint_index]
            self.get_logger().info(f"Position (Angle) of {joint_name}: {joint_position} radians")

def main(args=None):
    rclpy.init(args=args)
    node = JointStateListener()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
