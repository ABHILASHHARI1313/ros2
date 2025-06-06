from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control 
import numpy as np
import threading
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
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
Q_x = np.diag([10.0, 1.0, 100.0, 1.0])     # Strongly penalize θ
Q_u = np.diag([0.1])                       # Less penalty on effort
Q_delta_u = np.diag([1.0])                  # Smooth input changes
Q_terminal = Q_x.copy()                    # Same as Q_x

x_ref = np.array([1.0, 0.0, 0.0, 0.0]).astype(float)  # Desired states
# xr *= -1.0
N = 30 # length of horizon
dt = 0.01 # time step
u_ref = np.array([0.0])
u_prev = np.array([0.0])

nsim = 20 # number of simulation steps

class CartPendulumBalancer(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.publisher = self.create_publisher(Float64MultiArray,'/effort_controller/commands',10)
        self.state_variable_sub = self.create_subscription(JointState,'/joint_states',self.joint_state_callback,10)

        self.x0 = np.array([0.0, 0.0, 0.0, 0.0]).astype(float) # Current states
        self.timer = self.create_timer(dt,self.balance)
        self.lock  = threading.Lock()
        self.angle_change = []


    def joint_state_callback(self,msg):
        current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        with self.lock:
            position = msg.position
            velocity = msg.velocity
            self.x0[0] = position[0]
            self.x0[2] = -position[1]
            self.x0[1] = velocity[0]
            self.x0[3] = -velocity[1]
        self.get_logger().info(f"The current state @  time {current_time} is {str(self.x0)}")
        # self.get_logger().info(f"The state at time {current_time} is {str(self.x0)}")
        # self.angle_change.append(position[1])
    
    def balance(self):
        x = cp.Variable((nx, N+1))
        u = cp.Variable((nu, N))
        with self.lock:
            current_state = self.x0.copy()
        cost = 0.0
        # self.get_logger().info(str(x_init.value))
        # publish_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
        # self.get_logger().info(f"The state at time {publish_time} is {str(self.x0)}")
        constraints = [x[:, 0] == current_state]
        for k in range(N):
            # error penalty
            cost += cp.quad_form(x[:,k]-x_ref,Q_x)

             # control effort penalty
            cost += cp.quad_form(u[:,k]-u_ref,Q_u)

            # change in u penalty
            if k == 0:
                delta_u = u[:,k] - u_prev
            else:
                delta_u = u[:,k] - u[:,k-1]
            cost += cp.quad_form(delta_u,Q_delta_u)

            constraints += [x[:, k+1] == A_zoh @ x[:, k] + B_zoh @ u[:, k]]

            constraints += [cp.norm(u[:, k], 'inf') <= 10.0]
        
        cost += cp.quad_form(x[:, N]-x_ref, Q_terminal)


        problem = cp.Problem(cp.Minimize(cost), constraints)
        try :
            solve_start_time = time.perf_counter()
            problem.solve(solver=cp.OSQP, warm_start=True)
            solve_duration = time.perf_counter() - solve_start_time
            # self.get_logger().info(f"The time taken to solve the problem is, {str(solve_duration)}")

        except Exception as e:
            self.get_logger().info(f"MPC error : {e}")
            return
    
        
        # self.get_logger().info(str(x.value))

        if u[:,0].value is not None:
            control_command = u[:,0].value
            msg = Float64MultiArray()
            msg.data = [float(control_command)]
            publish_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            self.get_logger().info(f"The state @ publish time  {publish_time} is {current_state}")
            self.publisher.publish(msg)
            # self.get_logger().info(f"The difference between published time and the time of computation is {float(publish_time[-8:-1])-float(current_time[-8:-1])}")
            # self.get_logger().info(str(msg))
        else:
            self.get_logger().info("MPC didn't return a solution")





def main():
    rclpy.init(args=None)
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

