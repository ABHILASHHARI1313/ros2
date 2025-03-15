import rclpy
from rclpy.node import Node
from my_robot_interfaces.msg import LEDPanelState
from my_robot_interfaces.srv import SetLEDState
class MyNode(Node):
    def __init__(self):
        self.counter = 0
        super().__init__("led_panel")
        self.declare_parameter("led_states")
        self.led_states = self.get_parameter("led_states").value
        self.led_state_publisher = self.create_publisher(LEDPanelState,"led_states",10)
        self.led_states_timer = self.create_timer(4,self.publish_led_states)
        self.set_led_service = self.create_service(SetLEDState,"set_led",self.callback_set_led)
        self.get_logger().info("Set LED has been started")
    def publish_led_states(self):
        msg = LEDPanelState()
        msg.led_states = self.led_states
        self.led_state_publisher.publish(msg)
    def callback_set_led(self,request,response):
        led_number = request.led_number
        state = request.state
        if led_number>len(self.led_states) or led_number<=0:
            response.success = False
            return response
        if state not in [0,1]:
            response.success = False
            return response

        self.led_states[led_number-1] = state
        response.success = True
        self.publish_led_states()
        return response


        
def main(args=None):
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()