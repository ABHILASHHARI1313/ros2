from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import control
import math
import numpy as np
from std_msgs.msg import String
from scipy import sparse
from std_msgs.msg import Float64MultiArray
import cvxpy as cp

''' Pendulum Dynamics in States Space Equation '''
M = 20    # Cart mass
m = 2     # Pendulum mass
b = 0.1   # Coefficient of friction for cart
l = 0.5   # Length to pendulum center of mass
I = (m*l**2)*(1/12)  # Mass moment of inertia of the pendulum
g = 9.8   # Gravity
dt = 0.1  # Time step

p = I*(M+m)+M*m*l**2
A = np.array([
    [0, 1, 0, 0],
    [0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
    [0, 0, 0, 1],
    [0, -(m*l*b)/p, m*g*l*(M+m)/p, 0]
])

B = np.array([
    [0],
    [(I+m*l**2)/p],
    [0],
    [m*l/p]
])

C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

D = np.array([
    [0],
    [0]
])

sys = control.StateSpace(A, B, C, D)
sys_discrete = control.c2d(sys, dt, method='zoh')
A_zoh = np.array(sys_discrete.A)
B_zoh = np.array(sys_discrete.B)

''' Model Predictive Control implementation using State Space Equation '''
nx, nu = B_zoh.shape
# Higher weights on angle for upright stabilization
Q = sparse.diags([1.0, 0.1, 50.0, 5.0]).toarray()
R = np.array([[0.01]])  # Lower control penalty for more aggressive control
# Target is upright position (angle = pi)
xr = np.array([0.0, 0.0, np.pi, 0.0]).astype(float)
N = 10    # Prediction horizon

class CartPendulumBalancer(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.publisher = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 10)
        self.state_variable_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Initialize the state
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0]).astype(float)
        
        # Backup PD controller gains
        self.kp_angle = 30.0  # Higher gain for upright position
        self.kd_angle = 5.0
        self.last_control = 0.0
        
        # For debugging
        self.debug_publisher = self.create_publisher(String, '/mpc_debug', 10)

    def joint_state_callback(self, msg):
        """Process sensor data and compute control in a single callback"""
        # Update state from sensor data
        if len(msg.position) < 2 or len(msg.velocity) < 2:
            self.get_logger().warn("Incomplete joint state message")
            return
            
        position = msg.position
        velocity = msg.velocity
        
        # Get current state and normalize angle to be near Pi for upright position
        cart_pos = position[0]
        pendulum_angle = position[1]
        
        # Normalize angle to be near pi if it's in the upper half
        # This helps MPC target the correct upright position
        pendulum_angle = pendulum_angle % (2*np.pi)
        if abs(pendulum_angle - np.pi) > np.pi:
            if pendulum_angle < np.pi:
                pendulum_angle += 2*np.pi
            else:
                pendulum_angle -= 2*np.pi
        
        self.x0[0] = cart_pos           # Cart position
        self.x0[2] = pendulum_angle     # Pendulum angle (normalized near pi)
        self.x0[1] = velocity[0]        # Cart velocity
        self.x0[3] = velocity[1]        # Pendulum angular velocity
        
        # Debug current state
        debug_msg = String()
        debug_msg.data = f"State: pos={self.x0[0]:.2f}, angle={self.x0[2]:.2f}, target={np.pi:.2f}"
        self.debug_publisher.publish(debug_msg)
        
        # Compute MPC control
        control_input = self.compute_mpc_control()
        
        # If MPC fails, use simple PD control as backup
        if control_input is None:
            # Error relative to upright position (pi)
            angle_error = np.pi - self.x0[2]
            angle_rate = self.x0[3]
            control_input = self.kp_angle * angle_error - self.kd_angle * angle_rate
            
            # Apply smoothing to avoid jerky control
            control_input = 0.7 * control_input + 0.3 * self.last_control
            control_input = np.clip(control_input, -10, 10)
            self.get_logger().info(f"Using PD control: {control_input:.2f}")
        else:
            self.get_logger().info(f"Using MPC control: {control_input:.2f}")
        
        # Remember last control for smoothing
        self.last_control = control_input
        
        # Publish control command
        msg = Float64MultiArray()
        msg.data = [float(control_input)]
        self.publisher.publish(msg)

    def compute_mpc_control(self):
        """Compute control using MPC"""
        try:
            # Create optimization variables
            x = cp.Variable((nx, N+1))
            u = cp.Variable((nu, N))
            
            # Initialize cost
            cost = 0.0
            
            # Initial state constraint
            constraints = [x[:, 0] == self.x0]
            
            # Build cost and constraints over horizon
            for t in range(N):
                # State cost (targeting upright position)
                cost += cp.quad_form(x[:, t] - xr, Q)
                # Control cost
                cost += cp.quad_form(u[:, t], R)
                # System dynamics
                constraints += [x[:, t+1] == A_zoh @ x[:, t] + B_zoh @ u[:, t]]
                # Control limits
                constraints += [u[:, t] <= 10.0, u[:, t] >= -10.0]
            
            # Terminal cost
            cost += cp.quad_form(x[:, N] - xr, Q*5)
            
            # Set up and solve the problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            result = problem.solve(solver=cp.OSQP, 
                                  eps_abs=1e-2,
                                  eps_rel=1e-2, 
                                  max_iter=2000,
                                  verbose=False,
                                  warm_start=True)
            
            # Check if we have a valid solution
            if problem.status in ["optimal", "optimal_inaccurate"]:
                return float(u.value[0, 0])
            else:
                self.get_logger().warn(f"MPC Status: {problem.status}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"MPC error: {e}")
            return None


def main():
    rclpy.init(args=None)
    node = CartPendulumBalancer()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"The exception is {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()