import numpy as np
import control
import lib_control
import copy
from dynamics.dynamics import Dynamics
import warnings


class TwiprModel:
    def __init__(self, m_b, m_w, l, d_w, I_w, I_w2, I_y, I_x, I_z, c_alpha, r_w, tau_theta, tau_x):
        self.m_b = m_b
        self.m_w = m_w
        self.l = l
        self.d_w = d_w
        self.I_w = I_w
        self.I_w2 = I_w2
        self.I_y = I_y
        self.I_x = I_x
        self.I_z = I_z
        self.c_alpha = c_alpha
        self.r_w = r_w
        self.tau_theta = tau_theta
        self.tau_x = tau_x


ModelMichael = TwiprModel(m_b=2.5, m_w=0.636, l=0.026, d_w=0.28, I_w=5.1762e-4, I_w2=6.1348e-04, I_y=0.01648, I_x=0.02,
                          I_z=0.03, c_alpha=4.6302e-4, r_w=0.055, tau_theta=0, tau_x=0)


class ModelMichael_additionalMass(TwiprModel):
    def __init__(self, ma):
        super().__init__(m_b=2.5 + ma, m_w=0.636, l=(0.026*2.5+0.2*ma)/(ma+2.5), d_w=0.28, I_w=5.1762e-4,
                         I_w2=6.1348e-04, I_y=0.01648+0.2**2*ma, I_x=0.02, I_z=0.03, c_alpha=4.6302e-4, r_w=0.055, tau_theta=0,
                         tau_x=0)

class Twipr2D(Dynamics):
    model: TwiprModel

    def __init__(self, model, Ts, x0=None):
        super().__init__()
        self.Ts = Ts
        self.model = model

        self.n = 4
        self.p = 1
        self.q = 1

        if not (x0 is None):
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.n)

        self.state = self.x0
        self.state_names = ['x', 'x_dot', 'theta', 'theta_dot']
        self.K_cont = np.zeros((1, self.n))  # continous-time state controller
        self.K_disc = np.zeros((1, self.n))  # discrete-time state controller

        # get the linear continious model matrixes
        self.A, self.B, self.C, self.D = self.linear_model()
        # generate a linear continious model
        self.sys_cont = control.StateSpace(self.A, self.B, self.C, self.D, remove_useless=False)
        # convert the continious-time model to discrete-time
        self.sys_disc = control.c2d(self.sys_cont, self.Ts)
        self.A_d = np.asarray(self.sys_disc.A)
        self.B_d = np.asarray(self.sys_disc.B)
        self.C_d = np.asarray(self.sys_disc.C)
        self.D_d = np.asarray(self.sys_disc.D)

        self.A_hat = self.A
        self.A_hat_d = self.A_d

        self.ref_ctrl_is_set = False
        self.ref_ctrl = lib_control.PidController(0, 0, 0, self.Ts)
        self.ref_ctrl_state_num = 0

    def linear_model(self):
        g = 9.81
        C_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * self.model.m_b * self.model.l
        V_1 = (self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * (
                self.model.I_y + self.model.m_b * self.model.l ** 2) - self.model.m_b ** 2 * self.model.l ** 2
        D_22 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha + self.model.m_b * self.model.l * 2 * self.model.c_alpha / self.model.r_w
        D_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha / self.model.r_w + self.model.m_b * self.model.l * 2 * self.model.c_alpha / self.model.r_w ** 2
        C_11 = self.model.m_b ** 2 * self.model.l ** 2
        D_12 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w - self.model.m_b * self.model.l * 2 * self.model.c_alpha
        D_11 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w ** 2 - self.model.m_b * self.model.l * 2 * self.model.c_alpha / self.model.r_w

        A = [[0.000, 1, 0, 0],
             [0.000, -D_11 / V_1, -C_11 * g / V_1, D_12 / V_1],
             [0.000, 0, 0, 1],
             [0.000, D_21 / V_1, C_21 * g / V_1, -D_22 / V_1]]

        B_1 = (self.model.I_y + self.model.m_b * self.model.l ** 2) / self.model.r_w + self.model.m_b * self.model.l
        B_2 = self.model.m_b * self.model.l / self.model.r_w + self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2

        B = [[0],
             [B_1 / V_1],
             [0],
             [-B_2 / V_1]]

        C = [[0, 0, 1, 0]]

        D = 0

        return np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D)

    def set_poles(self, poles):
        poles = np.asarray(poles)
        self.K_cont = np.asarray(control.place(self.A, self.B, poles))
        self.K_disc = np.asarray(control.place(self.A_d, self.B_d, np.exp(poles * self.Ts)))
        self.A_hat = self.A - self.B @ self.K_cont
        self.sys_cont = control.StateSpace(self.A_hat, self.B, self.C, self.D, remove_useless=False)
        self.A_hat_d = self.A_d - self.B_d @ self.K_disc
        self.sys_disc = control.StateSpace(self.A_hat_d, self.B_d, self.C_d, self.D_d, self.Ts, remove_useless=False)
        return self.K_disc

    def set_state_controller(self, K):
        self.K_disc = K
        self.A_hat_d = self.A_d - self.B_d @ self.K_disc
        self.sys_disc = control.StateSpace(self.A_hat_d, self.B_d, self.C_d, self.D_d, self.Ts, remove_useless=False)

    def nonlinear_model(self, u):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        g = 9.81

        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]

        C_12 = (self.model.I_y + self.model.m_b * self.model.l ** 2) * self.model.m_b * self.model.l
        C_22 = self.model.m_b ** 2 * self.model.l ** 2 * np.cos(theta)
        C_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * self.model.m_b * self.model.l
        V_1 = (self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * (
                self.model.I_y + self.model.m_b * self.model.l ** 2) - self.model.m_b ** 2 * self.model.l ** 2 * np.cos(
            theta) ** 2
        D_22 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha + self.model.m_b * self.model.l * np.cos(
            theta) * 2 * self.model.c_alpha / self.model.r_w
        D_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha / self.model.r_w + self.model.m_b * self.model.l * np.cos(
            theta) * 2 * self.model.c_alpha / self.model.r_w ** 2
        C_11 = self.model.m_b ** 2 * self.model.l ** 2 * np.cos(theta)
        D_12 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w - self.model.m_b * self.model.l * np.cos(
            theta) * 2 * self.model.c_alpha
        D_11 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w ** 2 - 2 * self.model.m_b * self.model.l * np.cos(
            theta) * self.model.c_alpha / self.model.r_w
        B_2 = self.model.m_b * self.model.l / self.model.r_w * np.cos(
            theta) + self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2
        B_1 = (
                      self.model.I_y + self.model.m_b * self.model.l ** 2) / self.model.r_w + self.model.m_b * self.model.l * np.cos(
            theta)

        state_dot = np.zeros(4)
        state_dot[0] = x_dot
        state_dot[1] = np.sin(theta) / V_1 * (
                -C_11 * g + C_12 * theta_dot ** 2) - D_11 / V_1 * x_dot + D_12 / V_1 * theta_dot + B_1 / V_1 * u - self.model.tau_x * \
                       state_dot[0]
        state_dot[2] = theta_dot
        state_dot[3] = np.sin(theta) / V_1 * (
                C_21 * g - C_22 * theta_dot ** 2) + D_21 / V_1 * x_dot - D_22 / V_1 * theta_dot - B_2 / V_1 * u - self.model.tau_theta * \
                       state_dot[2]

        return state_dot

    def set_reference_controller(self, P, I, D, state_num):
        self.ref_ctrl_is_set = True
        self.ref_ctrl_state_num = state_num
        self.ref_ctrl = lib_control.PidController(P, I, D, self.Ts)

    def __reference_controller(self, w):

        # calculate the error
        e = w - self.state[self.ref_ctrl_state_num]
        u = self.ref_ctrl.calc(e)
        return u

    def __controller(self, u, mode):

        if self.ref_ctrl_is_set:
            u = self.__reference_controller(u)

        if mode == 'linear':
            u = u - self.K_disc @ self.state
        elif mode == 'nonlinear':
            u = u - self.K_cont @ self.state

        return u

    def __state_controller_linear(self, u):
        u = u - self.K_disc @ self.state
        return u

    def __state_controller_nonlinear(self, u):
        u = u - self.K_cont @ self.state
        return u

    def set_state(self, state):
        state = np.asarray(state, np.float)
        assert state.shape == (self.n,)
        self.state = state

    # simulate_step
    # @brief:
    def step(self, u, mode):
        u = np.atleast_1d(np.asarray(u))  # convert the input to an ndarray
        assert u.shape == (self.p,) or u.shape == (self.p, 1)
        y_out = []
        x_out = []
        if mode == 'linear':
            [y_out, x_out] = self.__simulate_step_linear(u)
        elif mode == 'nonlinear':
            [y_out, x_out] = self.__simulate_step_nonlinear(u)

        return y_out, x_out

    def simulate(self, u, mode, x0=None):
        # check the input and convert it to an array of appropriate size
        u = np.asarray(u)
        if len(u.shape) == 1:
            u = u[None, :]
        N = max(u.shape)
        if N < self.p:
            N = min(u.shape)
        u = u.reshape(self.p, N)

        # if an initial state is given, set the object state to the given state. If not, set the saved initial state
        if not (x0 is None):
            self.set_state(x0)
        else:
            self.set_state(self.x0)

        # reset the controllers and the saved input
        self.ref_ctrl.reset()

        if mode == 'linear':
            [y_out, x_out] = self.__simulate_linear(u, N)
        elif mode == 'nonlinear':
            [y_out, x_out] = self.__simulate_nonlinear(u, N)
        else:
            raise Exception("Wrong type of simulation mode!")
        return y_out, x_out

    # reset all the states and the controller
    def reset(self):
        self.ref_ctrl.reset()
        self.state = self.x0

    def __simulate_linear(self, u, N):
        y_out = np.zeros((self.q, N))
        x_out = np.zeros((self.n, N))
        x_out[:, 0] = self.state
        y_out[:, 0] - self.C @ self.state
        for k in range(1, N):
            [y_out[:, k], x_out[:, k]] = self.__simulate_step_linear(u[:, k - 1])
        return y_out, x_out

    def __simulate_nonlinear(self, u, N):
        y_out = np.zeros((self.q, N))
        x_out = np.zeros((self.n, N))
        x_out[:, 0] = self.state
        y_out[:, 0] - self.C @ self.state
        for k in range(1, N):
            [y_out[:, k], x_out[:, k]] = self.__simulate_step_nonlinear(u[:, k - 1])
        return y_out, x_out

    def __simulate_step_linear(self, w):
        u = self.__controller(w, 'linear')
        self.state = self.A_d @ self.state + self.B_d @ u
        x_out = self.state
        y_out = self.C @ self.state

        return y_out, x_out

    def __simulate_step_nonlinear(self, w):
        u = self.__controller(w, 'nonlinear')
        x_dot = self.nonlinear_model(u)
        self.state = self.state + x_dot * self.Ts
        x_out = self.state
        y_out = self.C @ self.state
        return y_out, x_out


def create_uncertain_copy(nominal_dynamics: Twipr2D, std_dev_mass):
    disturbed_dynamics = copy.deepcopy(nominal_dynamics)
    m_a = abs(std_dev_mass * np.random.randn())
    disturbed_dynamics.model.l = (disturbed_dynamics.model.l*disturbed_dynamics.model.m_b + 0.2 * m_a) / (m_a + disturbed_dynamics.model.m_b)
    disturbed_dynamics.model.I_y = disturbed_dynamics.model.I_y + 0.2**2*m_a
    disturbed_dynamics.model.m_b = disturbed_dynamics.model.m_b + m_a

    disturbed_dynamics.A, disturbed_dynamics.B, disturbed_dynamics.C, disturbed_dynamics.D = disturbed_dynamics.linear_model()
    # generate a linear continious model
    disturbed_dynamics.sys_cont = control.StateSpace(disturbed_dynamics.A, disturbed_dynamics.B, disturbed_dynamics.C, disturbed_dynamics.D, remove_useless=False)
    # convert the continious-time model to discrete-time
    disturbed_dynamics.sys_disc = control.c2d(disturbed_dynamics.sys_cont, disturbed_dynamics.Ts)
    disturbed_dynamics.A_d = np.asarray(disturbed_dynamics.sys_disc.A)
    disturbed_dynamics.B_d = np.asarray(disturbed_dynamics.sys_disc.B)
    disturbed_dynamics.C_d = np.asarray(disturbed_dynamics.sys_disc.C)
    disturbed_dynamics.D_d = np.asarray(disturbed_dynamics.sys_disc.D)

    disturbed_dynamics.A_hat_d = disturbed_dynamics.A_d - disturbed_dynamics.B_d @ disturbed_dynamics.K_disc
    disturbed_dynamics.sys_disc = control.StateSpace(disturbed_dynamics.A_hat_d, disturbed_dynamics.B_d, disturbed_dynamics.C_d, disturbed_dynamics.D_d, disturbed_dynamics.Ts, remove_useless=False)
    return disturbed_dynamics


class Twipr3D:
    model: TwiprModel

    def __init__(self, model, Ts, x0=None):
        self.Ts = Ts
        self.model = model

        self.n = 6
        self.p = 2
        self.q = 1

        if not (x0 is None):
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.n)

        self.state = self.x0
        self.state_names = ['x', 'x_dot', 'theta', 'theta_dot', 'psi', 'psi_dot']
        self.K_cont = np.zeros((1, self.n))  # continous-time state controller
        self.K_disc = np.zeros((1, self.n))  # discrete-time state controller

        # get the linear continious model matrixes
        self.A, self.B, self.C, self.D = self.linear_model()
        # generate a linear continious model
        self.sys_cont = control.StateSpace(self.A, self.B, self.C, self.D, remove_useless=False)
        # convert the continious-time model to discrete-time
        self.sys_disc = control.c2d(self.sys_cont, self.Ts)
        self.A_d = np.asarray(self.sys_disc.A)
        self.B_d = np.asarray(self.sys_disc.B)
        self.C_d = np.asarray(self.sys_disc.C)
        self.D_d = np.asarray(self.sys_disc.D)

        self.A_hat = self.A
        self.A_hat_d = self.A_d

        self.ref_ctrl_is_set = False
        self.ref_ctrl_1 = lib_control.PidController(0, 0, 0, self.Ts)
        self.ref_ctrl_2 = lib_control.PidController(0, 0, 0, self.Ts)
        self.ref_ctrl_1_state_num = 0
        self.ref_ctrl_2_state_num = 0

    def linear_model(self):
        g = 9.81
        C_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * self.model.m_b * self.model.l
        V_1 = (self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * (
                self.model.I_y + self.model.m_b * self.model.l ** 2) - self.model.m_b ** 2 * self.model.l ** 2
        D_22 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha + self.model.m_b * self.model.l * 2 * self.model.c_alpha / self.model.r_w
        D_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha / self.model.r_w + self.model.m_b * self.model.l * 2 * self.model.c_alpha / self.model.r_w ** 2
        C_11 = self.model.m_b ** 2 * self.model.l ** 2
        D_12 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w - self.model.m_b * self.model.l * 2 * self.model.c_alpha
        D_11 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w ** 2 - self.model.m_b * self.model.l * 2 * self.model.c_alpha / self.model.r_w
        D_33 = self.model.d_w / (2 * self.model.r_w ** 2) * self.model.c_alpha
        V_2 = self.model.I_z + 2 * self.model.I_w2 + (
                self.model.m_w + self.model.I_w / self.model.r_w ** 2) * self.model.d_w ** 2 / 2

        A = [[0, 1, 0, 0, 0, 0],
             [0, -D_11 / V_1, -C_11 * g / V_1, D_12 / V_1, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, D_21 / V_1, C_21 * g / V_1, -D_22 / V_1, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, -D_33 / V_2]]

        B_1 = (self.model.I_y + self.model.m_b * self.model.l ** 2) / self.model.r_w + self.model.m_b * self.model.l
        B_2 = self.model.m_b * self.model.l / self.model.r_w + self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2
        B_3 = self.model.d_w / (2 * self.model.r_w)

        B = [[0, 0],
             [B_1 / V_1, B_1 / V_1],
             [0, 0],
             [-B_2 / V_1, -B_2 / V_1],
             [0, 0],
             [-B_3 / V_2, B_3 / V_2]]

        C = [[0, 0, 1, 0, 0, 0]]

        D = [0, 0]

        return np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D)

    def set_eigenstructure(self, poles, eigenvectors):
        poles = np.asarray(poles)
        self.K_cont = lib_control.eigenstructure_assignment(self.A, self.B, poles, eigenvectors)
        self.K_disc = lib_control.eigenstructure_assignment(self.A_d, self.B_d, np.exp(poles * self.Ts), eigenvectors)

    def nonlinear_model(self, u):
        g = 9.81

        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]
        psi = self.state[4]
        psi_dot = self.state[5]

        C_12 = (self.model.I_y + self.model.m_b * self.model.l ** 2) * self.model.m_b * self.model.l
        C_22 = self.model.m_b ** 2 * self.model.l ** 2 * np.cos(theta)
        C_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * self.model.m_b * self.model.l
        V_1 = (self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * (
                self.model.I_y + self.model.m_b * self.model.l ** 2) - self.model.m_b ** 2 * self.model.l ** 2 * np.cos(
            theta) ** 2
        D_22 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha + self.model.m_b * self.model.l * np.cos(
            theta) * 2 * self.model.c_alpha / self.model.r_w
        D_21 = (
                       self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * 2 * self.model.c_alpha / self.model.r_w + self.model.m_b * self.model.l * np.cos(
            theta) * 2 * self.model.c_alpha / self.model.r_w ** 2
        C_11 = self.model.m_b ** 2 * self.model.l ** 2 * np.cos(theta)
        D_12 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w - self.model.m_b * self.model.l * np.cos(
            theta) * 2 * self.model.c_alpha
        D_11 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * 2 * self.model.c_alpha / self.model.r_w ** 2 - 2 * self.model.m_b * self.model.l * np.cos(
            theta) * self.model.c_alpha / self.model.r_w
        B_2 = self.model.m_b * self.model.l / self.model.r_w * np.cos(
            theta) + self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2
        B_1 = (
                      self.model.I_y + self.model.m_b * self.model.l ** 2) / self.model.r_w + self.model.m_b * self.model.l * np.cos(
            theta)
        C_31 = 2 * (self.model.I_z - self.model.I_x - self.model.m_b * self.model.l ** 2) * np.cos(theta)
        C_32 = self.model.m_b * self.model.l
        D_33 = self.model.d_w ** 2 / (2 * self.model.r_w ** 2) * self.model.c_alpha
        V_2 = self.model.I_z + 2 * self.model.I_w2 + (
                self.model.m_w + self.model.I_w / self.model.r_w ** 2) * self.model.d_w ** 2 / 2 - (
                      self.model.I_z - self.model.I_x - self.model.m_b * self.model.l ** 2) * np.sin(theta) ** 2
        B_3 = self.model.d_w / (2 * self.model.r_w)
        C_13 = (
                       self.model.I_y + self.model.m_b * self.model.l ** 2) * self.model.m_b * self.model.l + self.model.m_b * self.model.l * (
                       self.model.I_z - self.model.I_x - self.model.m_b * self.model.l ** 2) * np.cos(theta) ** 2
        C_23 = (self.model.m_b ** 2 * self.model.l ** 2 + (
                self.model.m_b + 2 * self.model.m_w + 2 * self.model.I_w / self.model.r_w ** 2) * (
                        self.model.I_z - self.model.I_x - self.model.m_b * self.model.l ** 2)) * np.cos(theta)

        state_dot = np.zeros(self.n)

        state_dot[0] = x_dot
        state_dot[1] = np.sin(theta) / V_1 * (
                -C_11 * g + C_12 * theta_dot ** 2 + C_13 * psi_dot ** 2) - D_11 / V_1 * x_dot + D_12 / V_1 * theta_dot + B_1 / V_1 * (
                               u[0] + u[1]) - self.model.tau_x * state_dot[0]
        state_dot[2] = theta_dot
        state_dot[3] = np.sin(theta) / V_1 * (
                C_21 * g - C_22 * theta ** 2 - C_23 * psi_dot ** 2) + D_21 / V_1 * x_dot - D_22 / V_1 * theta_dot - B_2 / V_1 * (
                               u[0] + u[1]) - self.model.tau_theta * state_dot[2]
        state_dot[4] = psi_dot
        state_dot[5] = np.sin(theta) / V_2 * (
                C_31 * theta_dot * psi_dot - C_32 * psi_dot * x_dot) - D_33 / V_2 * psi_dot - B_3 / V_2 * (
                               u[0] - u[1])

        return state_dot

    def set_reference_controller_pitch(self, P, I, D, state_num):
        self.ref_ctrl_is_set = True
        self.ref_ctrl_1_state_num = state_num
        self.ref_ctrl_1 = lib_control.PidController(P, I, D, self.Ts)

    def set_reference_controller_yaw(self, P, I, D, state_num):
        self.ref_ctrl_is_set = True
        self.ref_ctrl_2_state_num = state_num
        self.ref_ctrl_2 = lib_control.PidController(P, I, D, self.Ts)

    def __reference_controller(self, u):
        e_1 = u[0] - self.state[self.ref_ctrl_1_state_num]
        e_2 = u[1] - self.state[self.ref_ctrl_2_state_num]

        u_1 = self.ref_ctrl_1.calc(e_1)
        u_2 = self.ref_ctrl_2.calc(e_2)

        u = np.array([u_1 + u_2, u_1 - u_2])

        return u

    def __controller(self, u, mode):
        if self.ref_ctrl_is_set:
            u = self.__reference_controller(u)

        if mode == 'linear':
            u = u - self.K_disc @ self.state
        elif mode == 'nonlinear':
            u = u - self.K_cont @ self.state
        return u

    def __state_controller_linear(self, u):
        u = u - self.K_disc @ self.state
        return u

    def __state_controller_nonlinear(self, u):
        u = u - self.K_cont @ self.state
        return u

    def set_state(self, state):
        state = np.asarray(state, np.float)
        assert state.shape == (self.n,)
        self.state = state

    def simulate(self, u, mode, x0=None):
        # check the input and convert it to an array of appropriate size
        u = np.asarray(u)
        if len(u.shape) == 1:
            u = u[None, :]
        N = max(u.shape)
        if N < self.p:
            N = min(u.shape)
        u = u.reshape(self.p, N)

        # if an initial state is given, set the object state to the given state. If not, set the saved initial state
        if not (x0 is None):
            self.set_state(x0)
        else:
            self.set_state(self.x0)

        # reset the controllers and the saved input
        self.ref_ctrl_1.reset()
        self.ref_ctrl_2.reset()

        if mode == 'linear':
            [y_out, x_out] = self.__simulate_linear(u, N)
        elif mode == 'nonlinear':
            [y_out, x_out] = self.__simulate_nonlinear(u, N)
        else:
            raise Exception("Wrong type of simulation mode!")
        return y_out, x_out

    def simulate_step(self, u, mode):
        u = np.atleast_1d(np.asarray(u))  # convert the input to an ndarray
        assert u.shape == (self.p,) or u.shape == (self.p, 1)

        y_out = []
        x_out = []
        if mode == 'linear':
            [y_out, x_out] = self.__simulate_step_linear(u)
        elif mode == 'nonlinear':
            [y_out, x_out] = self.__simulate_step_nonlinear(u)
        return y_out, x_out

    def reset(self):
        self.state = self.x0
        self.ref_ctrl_1.reset()
        self.ref_ctrl_2.reset()

    def __simulate_linear(self, u, N):
        y_out = np.zeros((self.q, N))
        x_out = np.zeros((self.n, N))

        x_out[:, 0] = self.state
        y_out[:, 0] = self.C @ self.state

        for k in range(1, N):
            [y_out[:, k], x_out[:, k]] = self.__simulate_step_linear(u[:, k - 1])
        return y_out, x_out

    def __simulate_nonlinear(self, u, N):
        y_out = np.zeros((self.q, N))
        x_out = np.zeros((self.n, N))

        x_out[:, 0] = self.state
        y_out[:, 0] = self.C @ self.state

        for k in range(1, N):
            [y_out[:, k], x_out[:, k]] = self.__simulate_step_nonlinear(u[:, k - 1])
        return y_out, x_out

    def __simulate_step_linear(self, w):
        u = self.__controller(w, 'linear')
        self.state = self.A_d @ self.state + self.B_d @ u
        x_out = self.state
        y_out = self.C @ self.state

        return y_out, x_out

    def __simulate_step_nonlinear(self, w):
        u = self.__controller(w, 'nonlinear')
        x_dot = self.nonlinear_model(u)
        self.state = self.state + x_dot * self.Ts
        x_out = self.state
        y_out = self.C @ self.state
        return y_out, x_out
