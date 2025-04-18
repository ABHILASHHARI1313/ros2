from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control 
import numpy as np
from std_msgs.msg import Float64MultiArray
import cvxpy as cp
import time

# Pendulum Dynamics in State Space
M = 20.0    # Cart mass [kg]
m = 2.0     # Pendulum mass [kg]
b = 0.1     # Coefficient of friction for cart [N/m/s]
l = 0.5     # Length to pendulum center of mass [m]
I = (m*l**2)/3.0  # Mass moment of inertia [kg·m²]
g = 9.8     # Gravity [m/s²]
dt = 0.1    # Time step [s]

# State space matrices
p = I*(M+m) + M*m*l**2
A = np.array([
    [0, 1, 0, 0],
    [0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
    [0, 0, 0, 1],
    [0, -(m*l*b)/p, m*g*l*(M+m)/p, 0]
])
B = np.array([[0], [(I+m*l**2)/p], [0], [m*l/p]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
D = np.array([[0], [0]])

# Discrete system
sys = control.StateSpace(A, B, C, D)
sys_discrete = control.c2d(sys, dt, method='zoh')
A_zoh = np.array(sys_discrete.A)
B_zoh = np.array(sys_discrete.B)

# MPC Parameters
nx, nu = B_zoh.shape  # Number of states and inputs
N = 20                # Prediction horizon
Q_x = np.diag([10.0, 1.0, 100.0, 1.0])  # State weights
Q_u = np.diag([0.1])                    # Control effort weights
Q_du = np.diag([1.0])                   # Control change weights
Q_terminal = 10 * Q_x                   # Terminal cost

# Reference states (desired position)
x_ref = np.array([2.0, 0.0, 0.0, 0.0])  # [position, velocity, angle, angular velocity]
u_ref = np.array([0.0])                  # Zero force reference

class CartPendulumMPC(Node):
    def __init__(self):
        super().__init__('cart_pendulum_mpc')
        
        # Publishers and Subscribers
        self.publisher = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        
        # MPC Variables
        self.x = cp.Variable((nx, N+1))  # State trajectory
        self.u = cp.Variable((nu, N))     # Control trajectory
        
        # Current state and previous control
        self.current_state = np.zeros(nx)
        self.u_prev = np.zeros(nu)
        self.last_solution = None
        
        # Fixed constraints (dynamics)
        self.constraints = []
        for k in range(N):
            self.constraints += [
                self.x[:, k+1] == A_zoh @ self.x[:, k] + B_zoh @ self.u[:, k],
                cp.abs(self.u[:, k]) <= 10.0  # Force limits
            ]
        
        # Timer for control loop (runs at dt seconds)
        self.control_timer = self.create_timer(dt, self.mpc_control_loop)
        self.new_state = False
        
        self.get_logger().info("MPC Node Initialized")

    def state_callback(self, msg):
        """Update current state from joint states"""
        self.current_state[0] = msg.position[0]      # Cart position
        self.current_state[1] = msg.velocity[0]      # Cart velocity
        self.current_state[2] = -msg.position[1]      # Pendulum angle (inverted)
        self.current_state[3] = -msg.velocity[1]      # Pendulum angular velocity
        self.new_state = True

    def mpc_control_loop(self):
        if not self.new_state:
            return
            
        start_time = time.time()
        
        # Build cost function
        cost = 0.0
        for k in range(N):
            cost += cp.quad_form(self.x[:, k] - x_ref, Q_x)
            cost += cp.quad_form(self.u[:, k] - u_ref, Q_u)
            
            # Control change penalty
            if k == 0:
                cost += cp.quad_form(self.u[:, k] - self.u_prev, Q_du)
            else:
                cost += cp.quad_form(self.u[:, k] - self.u[:, k-1], Q_du)
        
        # Terminal cost
        cost += cp.quad_form(self.x[:, N] - x_ref, Q_terminal)
        
        # Complete constraints with current state
        constraints = [self.x[:, 0] == self.current_state] + self.constraints
        
        # Warm start if available
        if self.last_solution is not None:
            last_x, last_u = self.last_solution
            self.x.value = last_x
            self.u.value = last_u
        
        # Solve MPC problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                control_force = float(self.u[:, 0].value)
                self.u_prev[:] = control_force
                
                # Publish control command
                cmd_msg = Float64MultiArray()
                cmd_msg.data = [control_force]
                self.publisher.publish(cmd_msg)
                
                # Store solution for warm start (shift forward)
                self.last_solution = (
                    np.hstack([self.x.value[:, 1:], self.x.value[:, -1:]]),  # Shift states
                    np.hstack([self.u.value[:, 1:], self.u.value[:, -1:]])   # Shift controls
                )
            else:
                self.get_logger().warn(f"MPC suboptimal: {problem.status}")
                self.publish_zero_force()
                self.last_solution = None
                
        except Exception as e:
            self.get_logger().error(f"MPC solve failed: {e}")
            self.publish_zero_force()
            self.last_solution = None
            
        self.new_state = False
        solve_time = time.time() - start_time
        self.get_logger().info(f"MPC cycle: {solve_time:.4f}s")

    def publish_zero_force(self):
        """Safety fallback to zero force"""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [0.0]
        self.publisher.publish(cmd_msg)

def main():
    rclpy.init()
    node = CartPendulumMPC()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()