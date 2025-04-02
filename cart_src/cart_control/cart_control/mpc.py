from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control 
import math 
import numpy as np
from std_msgs.msg import String


class CartPendulumBalancer(Node):
    def __init__(self):
        super().__init__('read_node')
        # self.publisher = self.create_publisher(String,'/joint_states',10)

        self.state_variable_sub = self.create_subscription(JointState,'/joint_states',self.joint_state_callback,10)


    def joint_state_callback(self,msg):
        
        position = msg.position
        velocity = msg.velocity
        pendulum_angle = position[1]
        pendulum_velocity = velocity[1]

        cart_position = position[0]
        cart_velocity = velocity[0]

        print(
                f"Pendulum Angle: {position[1]} rad, "
                f"Pendulum Velocity: {velocity[1]} rad/s"
                f"Cart Position: {position[0]} rad, "
                f"Cart Velocity : {velocity[0]} rad/s"
            )





def main():
    rclpy.init(args=None)
    node = CartPendulumBalancer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
   



if __name__ == "__main__":
    main()

