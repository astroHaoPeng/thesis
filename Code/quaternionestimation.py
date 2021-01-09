import numpy as np
import quaternions as qt


def quat_mult(a, b):
    a1, b1, c1, d1 = a
    a2, b2, c2, d2 = b
    q0 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    q1 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    q2 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    q3 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return np.array([q0, q1, q2, q3])


def system_update(xkm1, uk, ts):
    norm_uk = np.linalg.norm(uk)
    angle = norm_uk * ts / 2
    if norm_uk == 0:
        norm_uk = 1
    uk = uk / norm_uk
    d = np.array([np.cos(angle), np.sin(angle) * uk[0], np.sin(angle) * uk[1], np.sin(angle) * uk[2]])
    xk = qt.multiply(xkm1, d)

    return xk


def measurement_update(xk, g):
    yk = qt.quaternion_rotate(qt.conjugate(xk), np.array([0, 0, g]))

    yk2 = 2 * np.arctan2(xk[3], xk[0])

    return np.array([yk[0], yk[1], yk[2]])#, yk2])


def F_func(uk, ts):
    norm_uk = np.linalg.norm(uk)
    if norm_uk == 0:
        norm_uk = 1
    u0 = np.cos(0.5 * norm_uk * ts)
    u1 = np.sin(0.5 * norm_uk * ts) * uk[0] / norm_uk
    u2 = np.sin(0.5 * norm_uk * ts) * uk[1] / norm_uk
    u3 = np.sin(0.5 * norm_uk * ts) * uk[2] / norm_uk
    F_k = np.array([[u0, -u1, -u2, -u3],
                    [u1, u0, u3, -u2],
                    [u2, -u3, u0, u1],
                    [u3, u2, -u1, u0]])

    return F_k


def H_func(xk, g):
    wk, xk, yk, zk = xk / np.linalg.norm(xk)
    H_k = 2 * g * np.array([[-yk, zk, -wk, xk],
                               [xk, wk, zk, yk],
                               [wk, -xk, -yk, zk]])#,
                               #[-2 * zk / ((wk ** 2) + (zk ** 2)), 0, 0, 2 * wk / ((wk ** 2) + (zk ** 2))]])

    return H_k


def ekf(xkm1, uk, yk1, yk2, V, W, P, g, ts):
    F_k = F_func(uk, ts)

    # Time update of the state estimate
    x_prior = system_update(xkm1, uk, ts)
    P_prior = F_k @ P @ np.transpose(F_k) + V
    x_prior = x_prior / np.linalg.norm(x_prior)
    H_k = H_func(x_prior, g)
    K = P_prior @ np.transpose(H_k) @ np.linalg.inv((H_k @ P_prior @ np.transpose(H_k) + W))

    # Measurement update of the state estimate
    x_pos = x_prior + K.dot((np.array([yk1[0], yk1[1], yk1[2]]) - measurement_update(x_prior, g)))
    # x_pos = x_prior + K.dot((np.array([yk1[0], yk1[1], yk1[2], yk2]) - measurement_update(x_prior, g)))
    x_pos = x_pos / np.linalg.norm(x_pos)
    A = np.identity(4) - K @ H_k
    P_pos = A @ P_prior @ np.transpose(A) + K @ W @ np.transpose(K)

    return x_pos, P_pos


def quat2ypr(q):
    q0, q1, q2, q3 = q

    if q0 * q2 - q3 * q1 > 0.499:
        roll = 0
        pitch = np.pi / 2
        yaw = -2 * np.arctan2(q1, q0)
    elif q0 * q2 - q3 * q1 < -0.499:
        roll = 0
        pitch = -np.pi / 2
        yaw = 2 * np.arctan2(q1, q0)
    else:
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 * q2 ** 2))
        pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * ((q2 ** 2) * (q3 ** 2)))

    return yaw, pitch, roll


def inc_head(q):
    q = q / np.linalg.norm(q)
    k = np.array([0, 0, 1])
    w, x, y, z = q
    heading = 2 * np.arctan2(np.dot([x, y, z], k), w)
    q_head = np.array([np.cos(heading / 2), 0, 0, -np.sin(heading / 2)])

    q2 = quat_mult(q_head, q)
    w2, x2, y2, z2 = q2
    inclination = 2 * np.arccos(w2)

    return inclination, heading
