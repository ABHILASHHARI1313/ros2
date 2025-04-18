import numpy as np
import scipy.sparse as sparse
import osqp
import warnings


def __is_vector__(vec):
    if vec.ndim == 1:
        return True
    else:
        if vec.ndim == 2:
            if vec.shape[0] == 1 or vec.shape[1] == 0:
                return True
        else:
            return False
        return False


def __is_matrix__(mat):
    if mat.ndim == 2:
        return True
    else:
        return False


class MPCController:
    
    def __init__(self, Ad, Bd, Np=20, Nc=None,
                 x0=None, xref=None, uref=None,
                 Qx=None, QxN=None, Qu=None,
                 umin=None, umax=None, Dumin=None, Dumax=None,
                 eps_rel=1e-3, eps_abs=1e-3):

        if __is_matrix__(Ad) and (Ad.shape[0] == Ad.shape[1]):
            self.Ad = Ad
            self.nx = Ad.shape[0]  # number of states
        else:
            raise ValueError("Ad should be a square matrix of dimension (nx,nx)!")

        if __is_matrix__(Bd) and Bd.shape[0] == self.nx:
            self.Bd = Bd
            self.nu = Bd.shape[1]  # number of inputs
        else:
            raise ValueError("Bd should be a matrix of dimension (nx, nu)!")

        if Np > 1:
            self.Np = Np  # assert
        else:
            raise ValueError("Np should be > 1!")

        if Nc is not None:
            if Nc <= Np:
                self.Nc = Nc
            else:
                raise ValueError("Nc should be <= Np!")
        else:
            self.Nc = self.Np

        # x0 handling
        if x0 is not None:
            if __is_vector__(x0) and x0.size == self.nx:
                self.x0 = x0.ravel()
            else:
                raise ValueError("x0 should be an array of dimension (nx,)!")
        else:
            self.x0 = np.zeros(self.nx)

        # reference handing
        if xref is not None:
            if __is_vector__(xref) and xref.size == self.nx:
                self.xref = xref.ravel()
            elif __is_matrix__(xref) and xref.shape[1] == self.nx and xref.shape[0] >= self.Np:
                self.xref = xref
            else:
                raise ValueError("xref should be either a vector of shape (nx,) or a matrix of shape (Np+1, nx)!")
        else:
            self.xref = np.zeros(self.nx)

        if uref is not None:
            if __is_vector__(uref) and uref.size == self.nu:
                self.uref = uref.ravel()  # assert...
            else:
                raise ValueError("uref should be a vector of shape (nu,)!")
        else:
            self.uref = np.zeros(self.nu)

        # weights handling
        if Qx is not None:
            if __is_matrix__(Qx) and Qx.shape[0] == self.nx and Qx.shape[1] == self.nx:
                self.Qx = Qx
            else:
                raise ValueError("Qx should be a matrix of shape (nx, nx)!")
        else:
            self.Qx = np.zeros((self.nx, self.nx)) # sparse

        if QxN is not None:
            if __is_matrix__(QxN) and QxN.shape[0] == self.nx and Qx.shape[1] == self.nx:
                self.QxN = QxN
            else:
                raise ValueError("QxN should be a square matrix of shape (nx, nx)!")
        else:
            self.QxN = self.Qx # sparse

        if Qu is not None:
            if __is_matrix__(Qu) and Qu.shape[0] == self.nu and Qu.shape[1] == self.nu:
                self.Qu = Qu
            else:
                raise ValueError("Qu should be a square matrix of shape (nu, nu)!")
        else:
            self.Qu = np.zeros((self.nu, self.nu))

        # constraints handling
        if umin is not None:
            if __is_vector__(umin) and umin.size == self.nu:
                self.umin = umin
            else:
                raise ValueError("umin should be a vector of shape (nu,)!")
        else:
            self.umin = -np.ones(self.nu)*np.inf

        if umax is not None:
            if __is_vector__(umax) and umax.size == self.nu:
                self.umax = umax
            else:
                raise ValueError("umax should be a vector of shape (nu,)!")
        else:
            self.umax = np.ones(self.nu)*np.inf


        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        self.u_failure = self.uref  # value provided when the MPC solver fails.


        # QP problem instance
        self.prob = osqp.OSQP()

        # Variables initialized by the setup() method
        self.res = None
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None
        self.x0_rh = None


    def setup(self, solve = False):
        self.x0_rh = np.copy(self.x0)
        self._compute_QP_matrices_()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, verbose=False, eps_abs=self.eps_rel, eps_rel=self.eps_abs)

        if solve:
            self.solve()


    def output(self):
        Np = self.Np
        nx = self.nx
        nu = self.nu

        # Extract first control input to the plant
        if self.res.info.status == 'solved':
            uMPC = self.res.x[(Np+1)*nx:(Np+1)*nx + nu]
        else:
            uMPC = self.u_failure
        
        return uMPC

        

    def update(self,x,solve=True):
        self.x0_rh = x
        self._update_QP_matrices_()
        if solve:
            self.solve()

    def solve(self):
        """ Solve the QP problem. """

        self.res = self.prob.solve()
        # Check solver status
        if self.res.info.status != 'solved':
            warnings.warn('OSQP did not solve the problem!')
            if self.raise_error:
                raise ValueError('OSQP did not solve the problem!')



    def _update_QP_matrices_(self):
        x0_rh = self.x0_rh
        nx = self.nx

        self.l[:nx] = -x0_rh
        self.u[:nx] = -x0_rh

        self.prob.update(l=self.l, u=self.u)


    def _compute_QP_matrices_(self):
        Np = self.Np
        Nc = self.Nc
        nx = self.nx
        nu = self.nu
        Qx = self.Qx
        QxN = self.QxN
        Qu = self.Qu
        xref = self.xref
        uref = self.uref
        Ad = self.Ad
        Bd = self.Bd
        x0 = self.x0
        umin = self.umin
        umax = self.umax


        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective

        P_X = sparse.csc_matrix(((Np+1)*nx, (Np+1)*nx))
        q_X = np.zeros((Np+1)*nx) 
        P_X += sparse.block_diag([sparse.kron(sparse.eye(Np), Qx),   # x0...x_N-1
                                    QxN])   
    
        q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),       # x0... x_N-1
                            -QxN.dot(xref)])                             # x_N

        # Filling P and q for J_U
        P_U = sparse.csc_matrix((Nc*nu, Nc*nu))
        q_U = np.zeros(Nc*nu) 
        P_U += sparse.kron(sparse.eye(Nc), Qu)
        q_U += np.kron(np.ones(Nc), -Qu.dot(uref))

        # Linear constraints

        # - linear dynamics x_k+1 = Ax_k + Bu_k and initial state
        Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), Ad)
        iBu = sparse.vstack([sparse.csc_matrix((1, Nc)),
                             sparse.eye(Nc)])
        Bu = sparse.kron(iBu, Bd)
        Aeq_dyn = sparse.hstack([Ax, Bu])

        leq_dyn = np.hstack([-x0, np.zeros(Np * nx)])
        ueq_dyn = leq_dyn # for equality constraints -> upper bound  = lower bound!


        # - bounds on u
        Aineq_u = sparse.hstack([sparse.csc_matrix((Nc*nu, (Np+1)*nx)), sparse.eye(Nc * nu)])
        lineq_u = np.kron(np.ones(Nc), umin)     # lower bound of inequalities
        uineq_u = np.kron(np.ones(Nc), umax)     # upper bound of inequalities

        A = sparse.vstack([Aeq_dyn, Aineq_u]).tocsc()
        l = np.hstack([leq_dyn, lineq_u])
        u = np.hstack([ueq_dyn, uineq_u])

        self.P = sparse.block_diag([P_X, P_U], format='csc')
        self.q = np.hstack([q_X, q_U])

        self.A = A
        self.l = l
        self.u = u

        
        





