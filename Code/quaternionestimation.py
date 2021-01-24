import numpy as np
import quaternions as qt


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

