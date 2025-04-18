from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control 
import numpy as np
from rclpy.executors import SingleThreadedExecutor
from scipy import sparse
from std_msgs.msg import Float64MultiArray
import cvxpy as cp
from datetime import datetime
import time

''' Pendulum Dynamics in States Space Equation '''
    
M = 20  # Cart mass
m = 2  # Pendulum mass
b = 0.1  # Coefficient of friction for cart
l = 0.5  # Length to pendulum center of mass
I = (m*l**2)*(1/3)  # Mass moment of inertia of the pendulum
g = 9.8  # Gravity
dt = 0.1  # Time step


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
Q = sparse.diags([10.0, 5.0, 100.0, 5.0]).toarray()
R = np.array([[0.1]])

xr = np.array([2.0, 0.0, 0.0, 0.0]).astype(float)  # Desired states
xr *= -1

N = 10 # length of horizon
dt = 0.1 # time step

nsim = 1000 # number of simulation steps

class CartPendulumBalancer(Node):
    def __init__(self):
        super().__init__('mpc_node')
       
        self.publisher = self.create_publisher(Float64MultiArray,'/effort_controller/commands',10)
        self.state_variable_sub = self.create_subscription(JointState,'/joint_states',self.joint_state_callback,10)

        self.x0 = np.array([0.0, 0.0, 0.0, 0.0]).astype(float) # Current states
        self.timer = self.create_timer(dt,self.balance)
        self.angle_change = []
        self.count = 0
        self.is_shutdown_requested = False


    def joint_state_callback(self,msg):
        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.get_logger().info(f"The current time is {current_time}")
        position = msg.position
        velocity = msg.velocity
        self.x0[0] = position[0]
        self.x0[2] = position[1]
        self.x0[1] = velocity[0]
        self.x0[3] = velocity[1]
        # self.angle_change.append(position[1])


    
    def balance(self):
        x = cp.Variable((nx, N+1))
        u = cp.Variable((nu, N))
        cost = 0.0
        # self.get_logger().info(str(x_init.value))
        constr = [x[:, 0] == self.x0]
        for t in range(N):
            cost += cp.quad_form(xr - x[:, t], Q) + cp.quad_form(u[:, t], R)
            constr += [cp.norm(u[:, t], 'inf') <= 10.0]
            constr += [x[:, t + 1] == A_zoh @ x[:, t] + B_zoh @ u[:, t]]
        cost += cp.quad_form(x[:, N] - xr, Q)

        problem = cp.Problem(cp.Minimize(cost), constr)
        # self.get_logger().info(f"Attempting MPC solve with x0: {[f'{s:.4f}' for s in self.x0]}")
        try :
            problem.solve(solver=cp.OSQP, warm_start=True,verbose=False)
        except Exception as e:
            self.get_logger().info(f"MPC error : {e}")
            return
        
        # self.get_logger().info(str(x.value))

        if u[:,0].value is not None:
            control_command = u[:,0].value
            msg = Float64MultiArray()
            msg.data = [float(control_command)]
            publish_time_before = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            self.get_logger().info(f"The published time before is, {publish_time_before}")
            if self.count == nsim:
                self.is_shutdown_requested = True
            else:
                self.publisher.publish(msg)
            self.count += 1
            # publish_time_after = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            # self.get_logger().info(f"The difference between published time and the time of computation is {float(publish_time[-8:-1])-float(current_time[-8:-1])}")
            # self.get_logger().info(f"The published time after is, {publish_time_after}")
            self.get_logger().info(f"The published control at {self.count}th  is output is, {str(msg)}")
        else:
            self.get_logger().info("MPC didn't return a solution")





def main():
    rclpy.init(args=None)
    node = CartPendulumBalancer()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try :
        while rclpy.ok() and not node.is_shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
        if node.is_shutdown_requested:
            node.get_logger().info("Node has requested shutdown...")
            rclpy.shutdown()
        # rclpy.spin(node)
    except Exception as e : 
        print(f"The exception is {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
   



if __name__ == "__main__":
    main()

