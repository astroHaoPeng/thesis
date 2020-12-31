import numpy as np
from matplotlib import pyplot as plt


def cart_polynomials(xi, xf, s, alfa, beta):
    x_s = np.zeros(len(s))
    for i in range(len(s)):
        x_s[i] = (s[i] ** 3) * xf - ((s[i] - 1) ** 3) * xi + alfa * (s[i] ** 2) * (s[i] - 1) + beta * s[i] * ((s[i] - 1) ** 2)

    return x_s


def deriv_s(x, s):
    x_d = np.zeros(len(x))
    for i in range (1, len(x)):
        x_d[i] = (x[i] - x[i - 1]) / (s[i] - s[i - 1])

    return x_d


def deriv_sanalitic(xi, xf, s, alfa, beta):
    x_ds = np.zeros(len(s))
    for i in range(len(s)):
        x_ds[i] = 3 * s[i] ** 2 * xf - 3 * xi * (s[i] - 1) ** 2 + alfa * (3 * s[i] ** 2 - 2 * s[i]) + beta * ((s[i] - 1) ** 2 + 2 * s[i] * (s[i] - 1))

    return x_ds


def path_and_cmd(xi, yi, xf, yf, theta_i, theta_f, k, t, N):
    s = np.linspace(0, 1, N)
    alfa_x = k * np.cos(theta_f) - 3 * xf
    alfa_y = k * np.sin(theta_f) - 3 * yf
    beta_x = k * np.cos(theta_i) + 3 * xi
    beta_y = k * np.sin(theta_i) + 3 * yi
    x_path = cart_polynomials(xi, xf, s, alfa_x, beta_x)
    y_path = cart_polynomials(yi, yf, s, alfa_y, beta_y)
    x_ds = deriv_sanalitic(0, xf, s, alfa_x, beta_x)
    y_ds = deriv_sanalitic(0, yf, s, alfa_y, beta_y)
    x_d2s = deriv_s(x_ds, s)
    y_d2s = deriv_s(y_ds, s)
    s_dt = deriv_s(s, t)
    theta_d = np.arctan2(y_ds, x_ds)

    v_rara = np.zeros(len(s))
    w_rara = np.zeros(len(s))

    for i in range(len(s)):
        v_rara[i] = np.sqrt(x_ds[i] ** 2 + y_ds[i] ** 2)
        if x_ds[i] ** 2 + y_ds[i] ** 2 == 0:
            w_rara[i] = 0
        else:
            w_rara[i] = (y_d2s[i] * x_ds[i] - x_d2s[i] * y_ds[i]) / (x_ds[i] ** 2 + y_ds[i] ** 2)

    v_cmd = v_rara * s_dt
    w_cmd = w_rara * s_dt

    return x_path, y_path, theta_d, v_cmd, w_cmd


def vel_nonlinear_filter(vr, vr_d, U, vm, vn, vn_d, Ts):
    yn = vn - vr
    yn_d = vn_d - vr_d
    zn = (yn / Ts + yn_d / 2) / (U * Ts)
    zn_d = yn_d / (U * Ts)
    m = int((1 + np.sqrt(1 + 8 * abs(zn))) / 2)
    sigman = zn_d + zn / m + (m - 1) * np.sign(zn) / 2
    un = -U * sigman * (1 + np.sign(vn_d * np.sign(sigman) + vm - Ts * U)) / 2

    return un, vn, vn_d


# definitions for the axes
# left, width = 0.07, 0.65
# bottom, height = 0.1, .8
# bottom_h = left_h = left+width+0.02
#
# rect_cones = [left, bottom, width, height]
# rect_box = [left_h, bottom + height / 2, 0.17, 0.17]
#
# fig = plt.figure()
# plt.ion()
# cones = plt.axes(rect_cones)
# box = plt.axes(rect_box)
# for i in range(50):
#     cones.clear()
#     x = np.linspace(0, 2 * np.pi, i * 10)
#     y = np.sin(x)
#     y2 = np.cos(x)
#
#     cones.plot(x, y)
#     cones.plot(x, -y)
#
#     box.plot(y, x)
#
#     plt.draw()
#     plt.pause(1)