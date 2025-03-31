import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench, Vector3
from builtin_interfaces.msg import Duration

class ApplyForceToBody(Node):
    def __init__(self):
        super().__init__('apply_force_to_body')
        self.client = self.create_client(ApplyBodyWrench, '/gazebo/apply_body_wrench')

        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /gazebo/apply_body_wrench service...')

        self.send_wrench()

    def send_wrench(self):
        req = ApplyBodyWrench.Request()
        # Replace "cart_body" with the actual link name from your URDF
        req.body_name = "cart_body"
        req.reference_frame = "world"  # Use "world" to apply force in the global frame
        
        # Set the force to be applied (e.g., 100 N along x)
        req.wrench.force = Vector3(x=100.0, y=0.0, z=0.0)
        # No torque needed for just moving the cart
        req.wrench.torque = Vector3(x=0.0, y=0.0, z=0.0)
        
        # Set duration for which the force is applied
        req.duration = Duration(sec=1, nanosec=0)

        self.get_logger().info(f'Applying force to {req.body_name}')
        future = self.client.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info("Force applied successfully!")
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

def main(args=None):
    rclpy.init(args=args)
    node = ApplyForceToBody()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
