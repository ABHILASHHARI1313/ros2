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
    """ This class implements a linear constrained MPC controller

    Attributes
    ----------
    Ad : 2D array_like. Size: (nx, nx)
         Discrete-time system matrix Ad.
    Bd : 2D array-like. Size: (nx, nu)
         Discrete-time system matrix Bd.
    Np : int
        Prediction horizon. Default value: 20.
    Nc : int
        Control horizon. It must be lower or equal to Np. If None, it is set equal to Np.
    x0 : 1D array_like. Size: (nx,)
         System state at time instant 0. If None, it is set to np.zeros(nx)
    xref : 1D array-like. Size: (nx,) or (Np, nx)
           System state reference (aka target, set-point). If size is (Np, nx), reference is time-dependent.
    uref : 1D array-like. Size: (nu, )
           System input reference. If None, it is set to np.zeros(nx)
    uminus1 : 1D array_like
             Input value assumed at time instant -1. If None, it is set to uref.
    Qx : 2D array_like
         State weight matrix. If None, it is set to eye(nx).
    QxN : 2D array_like
         State weight matrix for the last state. If None, it is set to eye(nx).
    Qu : 2D array_like
         Input weight matrix. If None, it is set to zeros((nu,nu)).
    QDu : 2D array_like
         Input delta weight matrix. If None, it is set to zeros((nu,nu)).
    xmin : 1D array_like
           State minimum value. If None, it is set to -np.inf*ones(nx).
    xmax : 1D array_like
           State maximum value. If None, it is set to np.inf*ones(nx).
    umin : 1D array_like
           Input minimum value. If None, it is set to -np.inf*ones(nx).
    umax : 1D array_like
           Input maximum value. If None, it is set to np.inf*ones(nx).
    Dumin : 1D array_like
           Input variation minimum value. If None, it is set to np.inf*ones(nx).
    Dumax : 1D array_like
           Input variation maximum value. If None, it is set to np.inf*ones(nx).
    eps_feas : float
               Scale factor for the matrix Q_eps. Q_eps = eps_feas*eye(nx).
    eps_rel : float
              Relative tolerance of the QP solver. Default value: 1e-3.
    eps_abs : float
              Absolute tolerance of the QP solver. Default value: 1e-3.
    """

    def __init__(self, Ad, Bd, Np=20, Nc=None,
                 x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None, umax=None, Dumin=None, Dumax=None,
                 eps_feas=1e6, eps_rel=1e-3, eps_abs=1e-3):

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

        if uminus1 is not None:
            if __is_vector__(uminus1) and uminus1.size == self.nu:
                self.uminus1 = uminus1
            else:
                raise ValueError("uminus1 should be a vector of shape (nu,)!")
        else:
            self.uminus1 = self.uref

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

        if QDu is not None:
            if __is_matrix__(QDu) and QDu.shape[0] == self.nu and QDu.shape[1] == self.nu:
                self.QDu = QDu
            else:
                raise ValueError("QDu should be a square matrix of shape (nu, nu)!")
        else:
            self.QDu = np.zeros((self.nu, self.nu))

        # constraints handling
        if xmin is not None:
            if __is_vector__(xmin) and xmin.size == self.nx:
                self.xmin = xmin.ravel()
            else:
                raise ValueError("xmin should be a vector of shape (nx,)!")
        else:
            self.xmin = -np.ones(self.nx)*np.inf

        if xmax is not None:
            if __is_vector__(xmax) and xmax.size == self.nx:
                self.xmax = xmax
            else:
                raise ValueError("xmax should be a vector of shape (nx,)!")
        else:
            self.xmax = np.ones(self.nx)*np.inf

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

        if Dumin is not None:
            if __is_vector__(Dumin) and Dumin.size == self.nu:
                self.Dumin = Dumin
            else:
                raise ValueError("Dumin should be a vector of shape (nu,)!")
        else:
            self.Dumin = -np.ones(self.nu)*np.inf

        if Dumax is not None:
            if __is_vector__(Dumax) and Dumax.size == self.nu:
                self.Dumax = Dumax
            else:
                raise ValueError("Dumax should be a vector of shape (nu,)!")
        else:
            self.Dumax = np.ones(self.nu)*np.inf

        self.eps_feas = eps_feas
        self.Qeps = eps_feas * sparse.eye(self.nx)

        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        self.u_failure = self.uref  # value provided when the MPC solver fails.

        # Hidden settings (for debug purpose)
        self.raise_error = False  # Raise an error when MPC optimization fails
        self.JX_ON = True  # Cost function terms in X active
        self.JU_ON = True  # Cost function terms in U active
        self.JDU_ON = True  # Cost function terms in Delta U active
        self.SOFT_ON = True  # Soft constraints active
        self.COMPUTE_J_CNST = False  # Compute the constant term of the MPC QP problem

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
        self.uminus1_rh = None
        self.J_CNST = None # Constant term of the cost function

    def setup(self, solve = True):
        """ Set-up the QP problem.

        Parameters
        ----------
        solve : bool
               If True, also solve the QP problem.

        """
        self.x0_rh = np.copy(self.x0)
        self.uminus1_rh = np.copy(self.uminus1)
        self._compute_QP_matrices_()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, verbose=False, eps_abs=self.eps_rel, eps_rel=self.eps_abs)

        if solve:
            self.solve()


    def output(self):
        """ Return the MPC controller output uMPC, i.e., the first element of the optimal input sequence and assign is to self.uminus1_rh.
            -------
            array_like (nu,)
                The first element of the optimal input sequence uMPC to be applied to the system.
            dict
                A dictionary with additional infos. It is returned only if one of the input flags return_* is set to True
        """
        Nc = self.Nc
        Np = self.Np
        nx = self.nx
        nu = self.nu

        # Extract first control input to the plant
        if self.res.info.status == 'solved':
            uMPC = self.res.x[(Np+1)*nx:(Np+1)*nx + nu]
        else:
            uMPC = self.u_failure
        
        self.uminus1_rh = uMPC

        return uMPC

        

    def update(self,x,solve=True):
        """ Update the QP problem.

        Parameters
        ----------
        x : array_like. Size: (nx,)
            The new value of x0.

        u : array_like. Size: (nu,)
            The new value of uminus1. If none, it is set to the previously computed u.

        xref : array_like. Size: (nx,)
            The new value of xref. If none, it is not changed

        solve : bool
               If True, also solve the QP problem.

        """
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


        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective

        P_X = sparse.csc_matrix(((Np+1)*nx, (Np+1)*nx))
        q_X = np.zeros((Np+1)*nx) 
        self.J_CNST = 0.0
        if self.JX_ON:
            P_X += sparse.block_diag([sparse.kron(sparse.eye(Np), Qx),   # x0...x_N-1
                                        QxN])   
        
            q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),       # x0... x_N-1
                               -QxN.dot(xref)])                             # x_N
        else:
            pass
        # Filling P and q for J_U
        P_U = sparse.csc_matrix((Nc*nu, Nc*nu))
        q_U = np.zeros(Nc*nu) 
        if self.JU_ON:
            self.J_CNST += 1/2*Np*(uref.dot(Qu.dot(uref)))
            P_U += sparse.kron(sparse.eye(Nc), Qu)
            q_U += np.kron(np.ones(Nc), -Qu.dot(uref))

        # Linear constraints

        # - linear dynamics x_k+1 = Ax_k + Bu_k
        Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), Ad)
        iBu = sparse.vstack([sparse.csc_matrix((1, Nc)),
                             sparse.eye(Nc)])
        Bu = sparse.kron(iBu, Bd)
        Aeq_dyn = sparse.hstack([Ax, Bu])

        leq_dyn = np.hstack([-x0, np.zeros(Np * nx)])
        ueq_dyn = leq_dyn # for equality constraints -> upper bound  = lower bound!


        A = sparse.vstack([Aeq_dyn]).tocsc()
        l = np.hstack([leq_dyn])
        u = np.hstack([ueq_dyn])

        self.P = sparse.block_diag([P_X, P_U], format='csc')
        self.q = np.hstack([q_X, q_U])

        self.A = A
        self.l = l
        self.u = u

        self.P_X = P_X
        
        





