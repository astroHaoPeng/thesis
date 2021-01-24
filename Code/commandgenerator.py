import numpy as np


def quadratic_command(yi, yf, angle, N, loc='initial'):
    x = np.linspace(0, 1, N)
    y = np.zeros(N)
    if loc == 'initial':
        c = yi
        b = angle
        a = yf - yi - angle
    elif loc == 'final':
        c = yi
        b = 2 * yf - 2 * yi - angle
        a = yf - yi - b
    for i in range(0, N):
        y[i] = a * x[i] ** 2 + b * x[i] + c

    return y


def linear_command(yi, yf, N):
    y = np.linspace(yi, yf, N)

    return y


def constant_command(cmd, N):
    y = cmd * np.ones(N)

    return y
