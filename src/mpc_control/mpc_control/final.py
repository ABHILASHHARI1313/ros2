from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control 
import numpy as np
import threading
from scipy import sparse
from std_msgs.msg import Float64MultiArray
import cvxpy as cp
from datetime import datetime
import time
from mpc_control.controller import MPCController
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

''' Pendulum Dynamics in States Space Equation '''
    
M = 20  # Cart mass
m = 2  # Pendulum mass
b = 0.1  # Coefficient of friction for cart
l = 0.5  # Length to pendulum center of mass
I = (m*l**2)*(1/3)  # Mass moment of inertia of the pendulum
g = 9.8  # Gravity
dt = 0.2 # Time step


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
Q = sparse.diags([10.0, 10.0, 10.0, 10.0]).toarray()
R = np.array([[0.1]])

xr = np.array([2.0, 0.0, 0.0, 0.0]).astype(float)  # Desired states
# xr *= -1.0
N = 20 # length of horizon
dt = 0.01 # time step


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
        # time.sleep(0.5)
        self.timer = self.create_timer(dt,self.balance)
        self.lock  = threading.Lock()



    def joint_state_callback(self,msg):
        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.get_logger().info(f"The current time is {current_time}")
        with self.lock:
            position = msg.position
            velocity = msg.velocity
            self.x0[0] = position[0]
            self.x0[2] = -position[1]
            self.x0[1] = velocity[0]
            self.x0[3] = -velocity[1]

    def solver(self):
        u = self.controller.output()
        with self.lock:
            self.controller.update(self.x0)
        return u.item()
    
    def balance(self):
            try:
                effort = self.solver()
            except Exception as e:
                self.get_logger().info(f"The error is {e}")
                return
            msg = Float64MultiArray()
            msg.data = [float(effort)]
            publish_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            self.get_logger().info(f"The publish time is {publish_time}")
            self.publisher.publish(msg)






def main():
    rclpy.init(domain_id=26)
    node = CartPendulumBalancer()
    try :
        rclpy.spin(node)
    except Exception as e : 
        print(f"The exception is {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
   



if __name__ == "__main__":
    main()

