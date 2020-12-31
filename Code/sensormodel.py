import numpy as np
import quaternions as qt


class Accelerometer:
    """
    A class to simulate an accelerometer
    ...
    Attributes
    -----------
    rx, ry, rz : double
        position of the accelerometer RF with respect to the robot RF [m]
    bias : double
        constant bias error of the accelerometer [m/s^2]
    wn_sigma : double
        standard deviation of the white noise error affecting the measurement
    bias_instability : double
        the part of the bias that is changing over time
    bias_instability_var : double
        standard deviation of the bias instability
    R : matrix
        Rotation matrix between the robot RF and the accelerometer RF

    Methods
    -------
    get_measurement
        gets a simulated measurement according to the current state of the robot
    """

    def __init__(self, rx=0, ry=0, rz=0, axis='', rotation=0, bias=0, wn_sigma=0, bias_instability_var=0):
        """
        Parameters
        ----------
        rx, ry, rz : double
            position of the accelerometer RF with respect to the robot RF [m]
        axis : str
            axis about  which the rotation between the accelerometer RF and the robot RF is given
        rotation : double
            rotation between the reference frames about the axis given in "axis" [degrees]
        bias : double
            constant bias error of the accelerometer
        wn_sigma : double
            standard deviation of the white noise error affecting the measurement
        bias_instability_var : double
            standard deviation of the bias instability
        """

        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.wn_sigma = np.radians(wn_sigma)
        self.bias = [np.random.normal(0, bias),
                     np.random.normal(0, bias),
                     np.random.normal(0, bias)]
        self.bias_instability_var = bias_instability_var
        self.bias_instability = [0, 0, 0]
        rotation = np.radians(rotation)
        if axis == 'x':
            self.R = [[1, 0, 0],
                      [0, np.cos(rotation), -np.sin(rotation)],
                      [0, np.sin(rotation), np.cos(rotation)]]
        elif axis == 'y':
            self.R = [[np.cos(rotation), 0, np.sin(rotation)],
                      [0, 1, 0],
                      [-np.sin(rotation), 0, np.cos(rotation)]]
        elif axis == 'z':
            self.R = [[np.cos(rotation), -np.sin(rotation), 0],
                      [np.sin(rotation), np.cos(rotation), 0],
                      [0, 0, 1]]
        else:
            self.R = np.identity(3)

    def get_measurement(self, robot, qkm1, vkm1, omegakm1):
        """
        returns the simulated measurement of the accelerometer

        Parameters
        ----------
        robot : class object
            robot in which the accelerometer is placed in
        qkm1 : vector
            previous orientation quaternion of the robot
        vkm1 : double
            previous state velocity of the robot
        omegakm1 : vector
            previous angular velocity of the robot in global reference frame

        Returns
        -------
        acceleration : list
            a list containing the 3 components of the measured acceleration [m/s^2]
        qk : vector
            a list containing the orientation quaternion of the robot
        angular_velocity : vector
            a list containing the 3 components of the angular velocity of the robot in global frame
        """

        g = 9.81

        yaw = robot.state[4]
        q_yaw = qt.quaternion(yaw, [0, 0, 1])

        # pitch quaternion
        pitch = robot.state[2]
        q_pitch = qt.quaternion(pitch, [0, 1, 0])

        # Current orientation quaternion
        qk = qt.multiply(q_yaw, q_pitch)

        dq = qt.relative_quaternion(qkm1, qk)
        angle = 2 * np.arccos(qt.clip(dq[0], -1, 1))
        if angle > np.spacing(1):
            axis = dq[1:] / np.linalg.norm(dq[1:])
        else:
            axis = np.array([0, 0, 0])
        angular_velocity = angle * axis / robot.Ts

        state = robot.state
        acc = (state[1] - vkm1) / robot.Ts
        roll_dot = angular_velocity[0]
        roll_2dot = (angular_velocity[0] - omegakm1[0]) / robot.Ts
        theta = state[2]
        theta_dot = angular_velocity[1]
        theta_2dot = (angular_velocity[1] - omegakm1[1]) / robot.Ts
        psi = state[4]
        psi_dot = angular_velocity[2]
        psi_2dot = (angular_velocity[2] - omegakm1[2]) / robot.Ts
        ang_acc = np.array([roll_2dot, theta_2dot, psi_2dot])

        # Angular velocity and acceleration in the body frame
        r = np.array([self.rx, self.ry, self.rz])

        acc_linear = np.array([acc * np.cos(theta), 0, acc * np.sin(theta)])

        acc_gravity = np.array([(-1) * g * np.sin(theta), 0, g * np.cos(theta)])

        acc_tang = np.cross(ang_acc, r)
        acc_norm = np.cross(angular_velocity, np.cross(angular_velocity, r))
        acc_rot = acc_tang + acc_norm

        accelerations = acc_linear + acc_gravity + acc_rot

        wn = [np.random.normal(0, self.wn_sigma),
              np.random.normal(0, self.wn_sigma),
              np.random.normal(0, self.wn_sigma)]

        self.bias_instability[0] = random_walk(self.bias_instability[0], self.bias_instability_var, 0.01)
        self.bias_instability[1] = random_walk(self.bias_instability[1], self.bias_instability_var, 0.01)
        self.bias_instability[2] = random_walk(self.bias_instability[2], self.bias_instability_var, 0.01)

        acceleration = (self.R @ accelerations) + self.bias + wn + self.bias_instability

        return acceleration, qk, angular_velocity


class Gyroscope:
    """
    A class to simulate a gyroscope
    ...
    Attributes
    -----------
    rx, ry, rz : double
        position of the gyroscope RF with respect to the robot RF [m]
    bias : double
        constant bias error of the gyroscope [degrees]
    wn_sigma : double
        standard deviation of the white noise error affecting the measurement
    bias_instability : double
        the part of the bias that is changing over time
    bias_instability_var : double
        standard deviation of the bias instability
    R : matrix
        Rotation matrix between the robot RF and the accelerometer RF

    Methods
    -------
    get_measurement
        gets a simulated measurement according to the current state of the robot
    """

    def __init__(self, rx=0, ry=0, rz=0, axis='', rotation=0, bias=0, wn_sigma=0, bias_instability_var=0):
        """
        Parameters
        ----------
        rx, ry, rz : double
            position of the gyroscope RF with respect to the robot RF [m]
        axis : str
            axis about  which the rotation between the gyroscope RF and the robot RF is given
        rotation : double
            rotation between the reference frames about the axis given in "axis" [degrees]
        bias : double
            constant bias error of the gyroscope
        wn_sigma : double
            standard deviation of the white noise error affecting the measurement
        bias_instability_var : double
            standard deviation of the bias instability
        """

        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.wn_sigma = np.radians(wn_sigma)
        self.bias = [np.random.normal(0, np.radians(bias)),
                     np.random.normal(0, np.radians(bias)),
                     np.random.normal(0, np.radians(bias))]
        self.bias_instability_var = np.radians(bias_instability_var)
        self.bias_instability = [0, 0, 0]
        rotation = np.radians(rotation)

        if axis == 'x':
            self.R = [[1, 0, 0],
                      [0, np.cos(rotation), -np.sin(rotation)],
                      [0, np.sin(rotation), np.cos(rotation)]]
        elif axis == 'y':
            self.R = [[np.cos(rotation), 0, np.sin(rotation)],
                      [0, 1, 0],
                      [-np.sin(rotation), 0, np.cos(rotation)]]
        elif axis == 'z':
            self.R = [[np.cos(rotation), -np.sin(rotation), 0],
                      [np.sin(rotation), np.cos(rotation), 0],
                      [0, 0, 1]]
        else:
            self.R = np.identity(3)

    def get_measurement(self, robot, qkm1):
        """
        returns the simulated measurement of the accelerometer

        Parameters
        ----------
        robot : class object
            robot in which the gyroscope is placed in
        qkm1 : vector
            previous orientation quaternion of the robot

        Returns
        -------
        gyro : list
            a list containing the 3 components of the measured angular velocities [degrees/s]
        qk : list
            a list containing the current orientation quaternion of the robot
        """
        # yaw quaternion
        yaw = robot.state[4]
        q_yaw = qt.quaternion(yaw, [0, 0, 1])

        # pitch quaternion
        pitch = robot.state[2]
        q_pitch = qt.quaternion(pitch, [0, 1, 0])

        # Current orientation quaternion
        qk = qt.multiply(q_yaw, q_pitch)
        dq = qt.relative_quaternion(qkm1, qk)
        angle = 2 * np.arccos(qt.clip(dq[0], -1, 1))
        if angle > np.spacing(1):
            axis = dq[1:] / np.linalg.norm(dq[1:])
        else:
            axis = np.array([0, 0, 0])
        gyro = angle * axis / robot.Ts

        wn = [np.random.normal(0, self.wn_sigma),
                np.random.normal(0, self.wn_sigma),
                np.random.normal(0, self.wn_sigma)]

        self.bias_instability[0] = random_walk(self.bias_instability[0], self.bias_instability_var, np.radians(0.01))
        self.bias_instability[1] = random_walk(self.bias_instability[1], self.bias_instability_var, np.radians(0.01))
        self.bias_instability[2] = random_walk(self.bias_instability[2], self.bias_instability_var, np.radians(0.01))

        gyro = gyro + self.bias + wn + self.bias_instability

        return gyro, qk


class Encoder:
    """
    A class to simulate an encoder
    ...
    Attributes
    -----------
    r : double
        radius of the wheel to which the encoder is attached [m]
    d : double
        distance between the two wheels [m]
    ticks_per_revolution : double
        number of ticks per revolution of the incremental encoder
    gear_ratio : double
        gear ratio between motor and wheel shaft

    Methods
    -------
    get_measurement
        gets a simulated measurement according to the current state of the robot
    """

    def __init__(self, wheel_r, d=0, ticks_per_revolution=0, gear_ratio=0):
        """
        Parameters
            ----------
        r : double
            radius of the wheel to which the encoder is attached [m]
        d : double
            distance between the two wheels [m]
        ticks_per_revolution : double
            number of ticks per revolution of the incremental encoder
        gear_ratio : double
            gear ratio between motor and wheel shaft
        """

        self.r = wheel_r
        self.d = d
        self.ticks_p_rev = ticks_per_revolution
        self.gear_ratio = gear_ratio
        self.left_angle = 0
        self.right_angle = 0
        self.left_vel = 0
        self.right_vel = 0

    def get_measurement(self, robot):
        """
        returns the simulated measurement of the encoders

        Parameters
        ----------
        robot : class object
            robot in which the encoders are placed in

        Returns
        -------
        left_wheel_angle : double
            a list containing the 3 components of the measured acceleration [m/s^2]
        right_wheel_angle : double
            the
        """

        state = robot.state
        x_dot = state[1]
        psi_dot = state[5]
        left_wheel_vel = (x_dot / self.r) - (self.d * psi_dot) / (2 * self.r)
        right_wheel_vel = (x_dot / self.r) + (self.d * psi_dot) / (2 * self.r)

        left_wheel_angle = self.left_angle + self.left_vel * robot.Ts
        right_wheel_angle = self.right_angle + self.right_vel * robot.Ts
        self.left_angle = left_wheel_angle
        self.right_angle = right_wheel_angle
        self.left_vel = left_wheel_vel
        self.right_vel = right_wheel_vel

        return left_wheel_angle, right_wheel_angle, left_wheel_vel, right_wheel_vel

    def get_measurement2(self, robot):
        state = robot.state
        x_dot = state[1]
        psi_dot = state[5]
        left_wheel_vel = (x_dot / self.r) - (self.d * psi_dot) / (2 * self.r)
        right_wheel_vel = (x_dot / self.r) + (self.d * psi_dot) / (2 * self.r)

        left_wheel_angle = self.left_angle + self.left_vel * robot.Ts
        right_wheel_angle = self.right_angle + self.right_vel * robot.Ts

        left_count = round(left_wheel_angle * self.ticks_p_rev * self.gear_ratio * 4 / 2 / np.pi)
        right_count = round(right_wheel_angle * self.ticks_p_rev * self.gear_ratio * 4 / 2 / np.pi)

        self.left_angle = left_wheel_angle
        self.right_angle = right_wheel_angle
        self.left_vel = left_wheel_vel
        self.right_vel = right_wheel_vel

        return [left_count, right_count]


def random_walk(bias, variance, limit):
    new_bias = bias + np.random.normal(0, variance)
    if new_bias > limit:
        return limit
    elif new_bias < -limit:
        return -limit
    else:
        return new_bias


def quat_mult(a, b):
    a1, b1, c1, d1 = a
    a2, b2, c2, d2 = b
    q0 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    q1 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    q2 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    q3 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return np.array([q0, q1, q2, q3])