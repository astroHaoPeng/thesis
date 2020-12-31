import numpy as np
from numpy import nan as nan
from matplotlib import pyplot as plt


def quaternion(angles, axis, rad=True):
    if not rad:
        angles = np.deg2rad(angles)
    angle = np.asarray([angles]) if np.isscalar(angles) else np.asarray(angles)
    if np.linalg.norm(axis) < np.spacing(1):
        n = len(angle)
        out = np.array([1, 0, 0, 0])
        out = np.tile(out, (n, 1))
        return out

    out = np.zeros((len(angle), 4))
    axis = axis / np.linalg.norm(axis)
    for i in range(len(angle)):
        out[i, :] = [np.cos(angle[i] / 2),
                    np.sin(angle[i] / 2) * axis[0],
                    np.sin(angle[i] / 2) * axis[1],
                    np.sin(angle[i] / 2) * axis[2]]

    if np.isscalar(angles):
        return np.squeeze(out)
    else:
        return out


def multiply(q1, q2):
    q1_aux = q1
    q1 = np.asarray([q1]) if isinstance(q1[0], float) else q1
    q2 = np.asarray([q2]) if isinstance(q2[0], float) else q2
    shapeq1 = np.shape(q1)[0]
    shapeq2 = np.shape(q2)[0]

    if shapeq1 == shapeq2:
        shape = shapeq1
    elif shapeq1 == 1:
        shape = shapeq2
    elif shapeq2 == 1:
        shape = shapeq1
    else:
        exit('Wrong quaternions dimensions')
    q3 = np.zeros((shape, 4))
    q3[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q3[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q3[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q3[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    if isinstance(q1_aux[0], float):
        return np.squeeze(q3)
    else:
        return q3


def conjugate(q):
    q_aux = q
    q = np.asarray([q]) if isinstance(q[0], float) else q
    if np.shape(q)[1] != 4:
        exit('input must be Nx4')

    q_conj = np.array([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])

    if isinstance(q_aux[0], float):
        return np.squeeze(q_conj)
    else:
        return np.transpose(q_conj)


def angle_from_q(q):
    q_aux = q
    q = np.asarray([q]) if isinstance(q[0], float) else q
    n = np.shape(q)[0]
    for i in range(n):
        q[i, :] = q[i, :] / np.linalg.norm(q[i, :])

    angle = 2 * np.arccos(q[:, 0])

    if isinstance(q_aux[0], float):
        return np.squeeze(angle)
    else:
        return angle


def axis_from_q(quat):
    q_aux = quat
    quat = np.asarray([quat]) if isinstance(quat[0], float) else quat
    n = np.shape(quat)[0]
    alpha = angle_from_q(quat)

    axis = np.zeros((3, n))
    for i in range(n):
        if alpha[i] == 0:
            axis[:, i] = [0, 0, 0]
        else:
            axis[:, i] = [quat[i, 1] / np.sin(alpha[i] / 2),
                          quat[i, 2] / np.sin(alpha[i] / 2),
                          quat[i, 3] / np.sin(alpha[i] / 2)]

    if isinstance(q_aux[0], float):
        return np.squeeze(axis)
    else:
        return np.transpose(axis)


def get_heading(quat):
    q_aux = quat
    quat = np.asarray([quat]) if isinstance(quat[0], float) else quat
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    w = quat[:, 0]

    axis = np.array([0, 0, 1])
    heading = 2 * np.arctan2(np.transpose(np.array([x, y, z])).dot(axis), w)

    q_rest = multiply(conjugate(quaternion(heading, [0, 0, 1])), quat)

    inclination = 2 * np.arccos(q_rest[:, 0])

    if isinstance(q_aux[0], float):
        return np.squeeze(heading), np.squeeze(inclination)
    else:
        return heading, inclination


def relative_quaternion(q1, q2):
    q1_aux = q1
    q1 = np.asarray([q1]) if isinstance(q1[0], float) else q1
    q2 = np.asarray([q2]) if isinstance(q2[0], float) else q2
    shape1 = np.shape(q1)[0]
    shape2 = np.shape(q2)[0]
    out = np.zeros((max(shape1, shape2), 4))

    out[:, 0] = q1[:, 0] * q2[:, 0] + q1[:, 1] * q2[:, 1] + q1[:, 2] * q2[:, 2] + q1[:, 3] * q2[:, 3]
    out[:, 1] = q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0] - q1[:, 2] * q2[:, 3] + q1[:, 3] * q2[:, 2]
    out[:, 2] = q1[:, 0] * q2[:, 2] + q1[:, 1] * q2[:, 3] - q1[:, 2] * q2[:, 0] - q1[:, 3] * q2[:, 1]
    out[:, 3] = q1[:, 0] * q2[:, 3] - q1[:, 1] * q2[:, 2] + q1[:, 2] * q2[:, 1] - q1[:, 3] * q2[:, 0]

    if isinstance(q1_aux[0], float):
        return np.squeeze(out)
    else:
        return out


def gyr_from_quat(quat, rate):
    N = np.shape(quat)[0]
    gyr = np.zeros((N, 3))
    dq = relative_quaternion(quat[0:-1, :], quat[1:, :])
    dq[dq[:, 0] < 0] = -dq[dq[:, 0] < 0]
    angle = []
    for i in range(N - 1):
        angle.append(2 * np.arccos(clip(dq[i, 0], -1, 1)))
    axis = np.zeros((N - 1, 3))

    for i in range(N - 1):
        if np.linalg.norm(dq[i, 1:]) < np.spacing(1):
            axis[i, :] = [0, 0, 0]
        else:
            axis[i, :] = dq[i, 1:] / np.linalg.norm(dq[i, 1:])
        gyr[i + 1, :] = angle[i] * axis[i, :] * rate

    return gyr


def gyr_from_quat2(quat, rate):
    N = np.shape(quat)[0]
    gyr = np.zeros((N, 3))
    dq = relative_quaternion(quat[0:-1, :], quat[1:, :])
    dq[dq[:, 0] < 0] = -dq[dq[:, 0] < 0]
    angle = []
    for i in range(N - 1):
        angle.append(2 * np.arccos(clip(dq[i, 0], -1, 1)))
    axis = np.zeros((N - 1, 3))

    nonzero = angle > np.spacing(1)
    axis[nonzero, :] = dq[nonzero, 1:] / np.linalg.norm(dq[nonzero, 1:])
    for i in range(N - 1):
        gyr[i + 1, :] = angle[i] * axis[i, :] * rate

    if N == 2:
        return gyr[-1, :]
    else:
        return gyr


def quaternion_rotate(quat, vec):
    q_aux = quat
    vec_aux = vec
    quat = np.asarray([quat]) if isinstance(quat[0], float) else quat
    vec = np.asarray([vec]) if isinstance(vec[0], float) else vec
    shapeq = np.shape(quat)[0]
    shapevec = np.shape(vec)[0]

    if shapeq == shapevec:
        shape = shapeq
    elif shapeq == 1:
        shape = shapevec
    elif shapevec == 1:
        shape = shapeq
    else:
        exit('Wrong quaternions dimensions')

    q_inv = conjugate(quat)
    new_vec = np.zeros((shape, 4))
    new_vec[:, 0] = np.zeros(shape)
    new_vec[:, 1] = vec[:, 0]
    new_vec[:, 2] = vec[:, 1]
    new_vec[:, 3] = vec[:, 2]
    qv = multiply(multiply(quat, new_vec), q_inv)
    v = qv[:, 1:]

    if isinstance(q_aux[0], float) and isinstance(vec_aux[0], float):
        return np.squeeze(v)
    else:
        return v


def euler_from_q(q, convention, intrinsic):
    q_aux = q
    q = np.asarray([q]) if isinstance(q[0], float) else q
    if intrinsic:
        convention = convention[::-1]

    a = conventionidentifier(convention[0])
    b = conventionidentifier(convention[1])
    c = conventionidentifier(convention[2])
    d = nan
    if a == c:
        if a != 1 and b != 1:
            d = 1
        elif a != 2 and b != 2:
            d = 2
        else:
            d = 3

    # Sign factor depending on axis order
    if b == (a % 3) + 1:
        s = 1  # Cyclic order
    else:
        s = -1  # anti-cyclic order

    angles = np.zeros((np.shape(q)[0], 3))
    if a == c:  # Proper Euler angles
        angles[:, 0] = np.arctan2(q[:, a] * q[:, b] - s * q[:, d] * q[:, 0],
                               q[:, b] * q[:, 0] + s * q[:, a] * q[:, d])
        angles[:, 1] = np.arccos(clip(q[:, 0] ** 2 + q[:, a] ** 2 - q[:, b] ** 2 - q[:, d] ** 2, -1, 1))
        angles[:, 2] = np.arctan2(q[:, a] * q[:, b] + s * q[:, c] * q[:, 0],
                               q[:, b] * q[:, 0] - s * q[:, a] * q[:, d])
    else:  # Tait Bryan
        angles[:, 0] = np.arctan2(2 * (q[:, a] * q[:, 0] + s * q[:, b] * q[:, c]),
                               q[:, 0] ** 2 - q[:, a] ** 2 - q[:, b] ** 2 + q[:, c] ** 2)
        angles[:, 1] = np.arcsin(clip(2 * (q[:, b] * q[:, 0] - s * q[:, a] * q[:, c]), -1, 1))
        angles[:, 2] = np.arctan2(2 * (s * q[:, a] * q[:, b] + q[:, c] * q[:, 0]),
                               q[:, 0] ** 2 + q[:, a] ** 2 - q[:, b] ** 2 - q[:, c] ** 2)

    if intrinsic:
        angles = angles[:, ::-1]

    if isinstance(q_aux[0], float):
        return np.squeeze(angles)
    else:
        return angles


def q_from_euler(angles, convention, intrinsic):
    axes = axes_from_convention(convention)

    q1 = quaternion(angles[0], axes[0, :])
    q2 = quaternion(angles[1], axes[1, :])
    q3 = quaternion(angles[2], axes[2, :])

    if intrinsic:
        q = multiply(multiply(q1, q2), q3)
    else:
        q = multiply(multiply(q3, q2), q1)

    return q


def axes_from_convention(convention):
    axes = np.zeros((3, 3))
    indx = 0
    for ax in convention:
        if ax == 'x':
            axes[:, indx] = [1, 0, 0]
        elif ax == 'y':
            axes[:, indx] = [0, 1, 0]
        elif ax == 'z':
            axes[:, indx] = [0, 0, 1]
        else:
            print('Not valid convention')
        indx += 1

    return np.transpose(axes)


def conventionidentifier(c):
    if c == 'x':
        n = 1
    elif c == 'y':
        n = 2
    elif c == 'z':
        n = 3

    return n


def clip(values, lower, upper):
    value = np.asarray([values]) if np.isscalar(values) else np.asarray(values)
    out = np.zeros(len(value))
    for i in range(len(value)):
        out[i] = max(lower, min(value[i], upper))

    if np.isscalar(values):
        return np.squeeze(out)
    else:
        return out


# pitch = np.linspace(0, np.pi / 4, 100)
# yaw = np.linspace(0, np.pi / 2, 100)
# roll = np.zeros(100)
# q_pitch = quaternion(pitch, [0, 1, 0])
# q_yaw = quaternion(yaw, [0, 0, 1])
# q = multiply(q_yaw, q_pitch)
# gyr = gyr_from_quat(q, 100)
# gyr_r = quaternion_rotate(q_pitch, gyr)
#
# figure, ax = plt.subplots(3)
# ax[0].plot(gyr[:, 0], 'r')
# ax[0].plot(gyr_r[:, 0], '--')
# ax[1].plot(gyr[:, 1], 'r')
# ax[1].plot(gyr_r[:, 1], '--')
# ax[2].plot(gyr[:, 2], 'r')
# ax[2].plot(gyr_r[:, 2], '--')
#
# plt.grid()
# plt.show()

