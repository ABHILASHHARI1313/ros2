#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import time
import control
from datetime import datetime
from pyMPC.mpc import MPCController





class InvertedPendulumController(Node):
    def __init__(self,x_ref=None,rate = 100,**kwargs):

        super().__init__(**kwargs)

        self.command_pub = self.create_publisher(Float64MultiArray,'/effort_controller/commands',1)

        self.create_subscription(JointState,'/joint_states',self.joint_state_callback,1)

        self.current_state = np.zeros(4)
        self.x_ref = np.zeros(4) if x_ref is None else x_ref

        self.freq = rate
        self.timer =self.create_timer(1.0/self.freq,self.balance)

        time.sleep(1.5)
        self.counter = 0

    def joint_state_callback(self,data):

        self.current_state[0] = data.position[0]       # x
        self.current_state[1] = data.velocity[0]       # x_dot
        self.current_state[2] = -data.position[1]      # theta (inverted)
        self.current_state[3] = -data.velocity[1]      # theta_dot (inverted)
        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.get_logger().info(f"The current time is {current_time}")

    def balance(self):
        try:
            output = self.get_output()
        except NotImplementedError:
            self.get_logger().warn('get_output() not implemented!')
            return

        msg = Float64MultiArray()
        msg.data = [float(output)]
        publish_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.get_logger().info(f"The publish time is {publish_time}")
        self.command_pub.publish(msg)
        self.counter += 1
        self.get_logger().info(f"The count number is {self.counter}")
        # self.get_logger().info(f"The force applied is {str(msg)}")

    def get_output(self):
        raise NotImplementedError("get_output() must be implemented by subclass")

    @property
    def desired_state(self):
        return self.x_ref





class MPCInvertedPendulumController(InvertedPendulumController):
    def __init__(self, A_cont, B_cont, Q, R, dt, N=10, **kwargs):
        super().__init__(**kwargs) 

        C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

        D = np.array([[0],
                    [0]])

        sys = control.StateSpace(A_cont, B_cont, C, D)
        sys_discrete = control.c2d(sys, dt, method='zoh')

        A_d = np.array(sys_discrete.A)
        B_d  = np.array(sys_discrete.B)

        self.controller = MPCController(A_d, B_d, Np=N, Qx=Q, Qu=R, xref=self.x_ref)
        self.controller.setup()

    def get_output(self):
        """
        Compute the control output using MPC.
        """
        u = self.controller.output()
        self.controller.update(self.current_state)
        return u.item()
    
def main():
    rclpy.init()
    M = 20  # Cart mass
    m = 2  # Pendulum mass
    b = 0.1  # Coefficient of friction for cart
    l = 0.5  # Length to pendulum center of mass
    I = (m*l*l)*(1/3)  # Mass moment of inertia of the pendulum
    g = 9.8  # Gravity


    P = (M + m) * I + M * m * l * l

     # Continuous-time A and B matrices
    A = np.array([[0, 1, 0, 0],
                  [0, -b * (I + m * l * l) / P, m * m * g * l * l / P, 0],
                  [0, 0, 0, 1],
                  [0, -b * m * l / P, m * g * l * (M + m) / P, 0]])
    B = np.array([[0],
                  [(I + m * l * l) / P],
                  [0],
                  [m * l / P]])

    # Weights
    Q = np.diag([10.0, 10.0, 10.0, 10.0])
    R = np.array([[0.1]])
    x_ref = np.array([2.0,0.0,0.0,0.0])

    dt = 0.02
    N = 200
    controller = MPCInvertedPendulumController(A, B, Q, R, dt, N, x_ref=x_ref, node_name='mpc_controller')


    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

    
if __name__ == '__main__':
    main()