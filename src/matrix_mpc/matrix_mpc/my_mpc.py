#!/usr/bin/env python3
from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control 
import numpy as np
from scipy import sparse
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from datetime import datetime
import time
from matrix_mpc.matrix_mpc import MPCController
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

''' Pendulum Dynamics in States Space Equation '''
    
M = 20  # Cart mass
m = 2  # Pendulum mass
b = 0.1  # Coefficient of friction for cart
l = 0.5  # Length to pendulum center of mass
I = (m*l**2)*(1/3)  # Mass moment of inertia of the pendulum
g = 9.8  # Gravity
dt = 0.1



p = I*(M+m)+M*m*l**2

A = np.array([[0,      1,              0,            0],
    [0, -(I+m*l**2)*b/p,  (m**2*g*l**2)/p, 0],
    [0,      0,              0,            1],
    [0, -(m*l*b)/p,       m*g*l*(M+m)/p,   0]])

B = np.array([[0],
    [(I+m*l**2)/p],
    [0],
    [m*l/p]])

C = np.array([[1, 0, 0, 0],
        [0, 0, 1, 0]])

D = np.array([[0],
        [0]])

sys = control.StateSpace(A, B, C, D)
sys_discrete = control.c2d(sys, dt, method='zoh')

A_zoh = np.array(sys_discrete.A)
B_zoh = np.array(sys_discrete.B)

''' Model Predictive Control implementation using State Space Equation '''

nx, nu = B_zoh.shape
Q = sparse.diags([100.0, 5.0, 100.0, 5.0]).toarray()
R = np.array([[0.1]])

xr = np.array([2.0, 0.0, 0.0, 0.0]).astype(float)  # Desired states
# xr *= -1.0
N = 50 # length of horizon

nsim = 1000

class CartPendulumBalancer(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.publisher = self.create_publisher(Float64MultiArray,'/effort_controller/commands',1)
        self.qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.state_variable_sub = self.create_subscription(JointState,'/joint_states',self.joint_state_callback,self.qos)

        self.x0 = np.array([0.0, 0.0, 0.0, 0.0]).astype(float) # Current states
        self.controller = MPCController(A_zoh, B_zoh, Np=N, Qx=Q, Qu=R, xref=xr)
        self.controller.setup()
        self.count = 0
        self.sim_reached = 0
        # time.sleep(0.5)
        ''' For visualization purpose '''
        self.cart_pos = []
        self.pend_ang = []
        self.ctrl_effort = []


    def joint_state_callback(self,msg):
        self.count += 1
        if self.count == nsim:
            self.sim_reached = True
        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.get_logger().info(f"The current time is {current_time}")
        position = msg.position
        velocity = msg.velocity
        self.x0[0] = position[0]
        self.x0[2] = -position[1]
        self.x0[1] = velocity[0]
        self.x0[3] = -velocity[1]

        ''' For visualization purpose '''
        self.cart_pos.append(position[0])
        self.pend_ang.append(velocity[0])

        self.controller.update(self.x0)
        u = self.controller.output()
        try:
            effort = u.item()
            self.ctrl_effort.append(effort)
        except Exception as e:
            self.get_logger().info(f"The error is {e}")
            return
        msg = Float64MultiArray()
        msg.data = [float(effort)]
        publish_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.get_logger().info(f"The publish time is {publish_time}")
        self.publisher.publish(msg)

def plot_results(cart_pos,pend_ang,ctrl_effort):
   plt.figure(figsize=(8, 4))
   plt.plot(cart_pos, label='Cart Position', color='blue')
   plt.ylabel("Position (m)")
   plt.xlabel("Time Step")
   plt.title("Cart position over time")
   plt.legend()
   plt.grid()
   plt.show()

   plt.figure(figsize=(8, 4))
   plt.plot(pend_ang, label='Pendulum Angle', color='green')
   plt.ylabel("Angle (m)")
   plt.xlabel("Time Step")
   plt.title("Pendulum angle over time")
   plt.legend()
   plt.grid()
   plt.show()

   plt.figure(figsize=(8, 4))
   plt.plot(ctrl_effort, label='Control Effort', color='red')
   plt.ylabel("Control Effort(m)")
   plt.xlabel("Time Step")
   plt.title("Control effort over time")
   plt.legend()
   plt.grid()
   plt.show()




def main():
    rclpy.init(domain_id=26)
    node = CartPendulumBalancer()
    try :
        while (node.sim_reached == False):
            rclpy.spin_once(node)
    except Exception as e : 
        print(f"The exception is {e}")
    finally:
        cart_pos  = node.cart_pos
        pend_ang = node.pend_ang
        ctrl_effort = node.ctrl_effort
        plot_results(cart_pos,pend_ang,ctrl_effort) 
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()

