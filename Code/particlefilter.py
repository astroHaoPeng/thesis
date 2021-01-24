import numpy as np
import math
import random


class ParticleFilter:
    """
    A class to generate a particle filter
    ...
    Attributes
    -----------
    nx : double
        size of the state vector
    ns : double
        number of particles
    w : list
        list containing the current weight of each particle
    particles : list
        list containing the current particles

    Methods
    -------
    particlefilter
        performs the propagation of the state and updates the weights
    """

    def __init__(self, state, ns):
        """
        Parameters
        ----------
        state : list
            initial state of the system
        ns : double
            number of particles
        """

        self.nx = len(state)
        self.ns = ns
        self.w = np.ones(ns) / ns
        # self.particles = np.zeros((self.nx, ns))
        self.particles = gen_x0(0, 0, 0.01, self.ns)

    def particlefilter(self, uk, yk, ts, resampling_method):
        """
        performs the propagation of the particles and updates its weights

        Parameters
        ----------
        yk : list
            observation vector at time k
        uk : ndarray
            input vector at time k
        ts : double
            sampling period
        resampling_method : str
            resampling technique to be used

        Returns
        --------
        xhk : list
            estimated state
        ParticleFilter : class object
            the same structure but updated at new iteration
        """

        # Initialize variables
        wkm1 = self.w
        xkm1 = self.particles
        xk = np.zeros((self.nx, self.ns))
        wk = np.zeros(self.ns)

        sigma = 0.01

        for i in range(0, self.ns):
            xk[:, i] = system_update2(xkm1[:, i], uk, ts, sigma=sigma)

            if yk.size == 0:
                wk[i] = 1 / self.ns
            else:
                wk[i] = wkm1[i] * p_yk_given_xk(yk, xk[:, i], uk)

        if yk.size >= 2 and max(wk) <= 0.1:
            xk[0:2, 0] = yk[0:2]
            wk[0] = wkm1[0] * p_yk_given_xk(yk, xk[:, i], uk)

        # Normalize weight vector
        wk += 1.e-100               # Avoid round-off to zero
        wk = wk / np.sum(wk)

        # Compute effective sample size
        neff = 1 / np.sum(wk ** 2)

        # Resampling
        resample_percentage = 0.5
        nt = resample_percentage * self.ns

        if neff < nt:
            print('Resampling...' + '\n')
            xk, wk = resample(xk, wk, resampling_method, self.ns)

        # Compute estimated state
        xhk = np.average(xk, weights=wk, axis=1)

        # Store new weights and particles
        self.w = wk
        self.particles = xk

        return xhk


def system_update(xkm1, uk, ts, sigma):
    # Random noise added to the states
    noise = gen_sys_noise(xkm1[2], ts, uk[1], sigma=sigma)
    x = np.zeros(3)
    x[0] = xkm1[0] + xkm1[2] * ts * np.cos(uk[1]) + noise[0]
    x[1] = xkm1[1] + xkm1[2] * ts * np.sin(uk[1]) + noise[1]
    x[2] = xkm1[2] + uk[0] * ts + noise[2]

    return x


def system_update2(xkm1, uk, ts, sigma):
    # Noise added to the velocity
    vk = np.random.normal(0, sigma)
    y = 0.1 * xkm1[2] * ts * (2 * np.random.rand() - 1)
    x = np.zeros(3)
    x[0] = xkm1[0] + xkm1[2] * ts * np.cos(uk[1]) #- y * np.sin(uk[1])
    x[1] = xkm1[1] + xkm1[2] * ts * np.sin(uk[1]) #+ y * np.cos(uk[1])
    x[2] = xkm1[2] + uk[0] * ts + vk

    return x


def system_update3(xkm1, uk, ts, sigma):
    # Noise added to the velocity and the heading
    vk = np.random.normal(0, sigma)
    uk[1] += np.random.normal(0, np.deg2rad(1))
    y = 0.1 * xkm1[2] * ts * (2 * np.random.rand() - 1)
    x = np.zeros(3)
    x[0] = xkm1[0] + xkm1[2] * ts * np.cos(uk[1])  # - y * np.sin(uk[1])
    x[1] = xkm1[1] + xkm1[2] * ts * np.sin(uk[1])  # + y * np.cos(uk[1])
    x[2] = xkm1[2] + uk[0] * ts + vk

    return x


def meas_update(xk, yk, uk=0):
    if yk.size == 1:
        x = xk[2]
    elif yk.size == 2:
        x = xk[0:1]
    else:
        x = xk

    return x


def p_yk_given_xk(yk, xk, uk):
    """
    Computes the conditional probability P(yk|xk)

    Parameters
    ----------
    yk : list
        observation vector at time k
    xk : list
        the particle whose probability will be computed
    uk : ndarray
        input vector at time k

    Returns
    --------
    prob : double
        conditional probability of the particle
    """

    if yk.size == 1:
        prob = normpdf(yk - meas_update(xk, yk, uk), 0, 0.02)
        # if abs(yk) <= 0.01:
        #     prob = normpdf(yk - meas_update(xk, yk, uk), 0, 0.001)
    elif yk.size == 2:
        distance = np.linalg.norm(xk[0:2] - yk[0:2], 2)
        prob = normpdf(distance, 0, 0.005)
    elif yk.size == 3:
        distance = np.linalg.norm(xk[0:2] - yk[0:2], 2)
        prob_dist = normpdf(distance, 0, 0.005)
        prob_vel = normpdf(yk[-1] - xk[-1], 0, 0.01)
        prob = (prob_dist + 0.1 * prob_vel) / 2

    return prob


def gen_sys_noise(v, dt, heading, sigma):
    wk = np.random.normal(0, sigma)
    vk = np.random.normal(0, sigma)
    # x = v * dt * (2 * np.random.rand() - 1)
    x = v * dt * np.random.uniform(-0.6, 0.6, 1)
    y = 0.3 * v * dt * (2 * np.random.rand() - 1)
    x2 = x * np.cos(heading) - y * np.sin(heading)
    y2 = x * np.sin(heading) + y * np.cos(heading)
    wk = np.array([x2, y2, vk])

    return wk


def p_sys_noise(x, mean, sd):
    pdf = normpdf(x, mean, sd)

    return pdf


def gen_obs_noise(sigma):
    vk = np.random.normal(0, sigma)

    return vk


def p_obs_noise(x, mean, sd, variance):
    if np.isscalar(x):
        pdf_f = normpdf(x, mean, sd)
    else:
        pdf = np.zeros(len(x))
        for i in range(len(x)):
            if variance[i] == 0:
                variance[i] = 1
            if i == len(x) - 1:
                pdf[i] = normpdf(x[i] / np.sqrt(variance[i]), mean, 1) #sd * 10)
                # pdf[i] = normpdf(x[i], mean, np.sqrt(variance[i]))
            else:
                pdf[i] = normpdf(x[i] / np.sqrt(variance[i]), mean, 1) #sd)
                # pdf[i] = normpdf(x[i], mean, np.sqrt(variance[i]))
        if pdf.size == 1:
            pdf_f = np.mean(pdf)
        else:
            pdf_f = np.mean(pdf[0:2])

    return pdf_f


def normpdf(x, mean, sd):
    u = float((x - mean) / abs(sd))
    y = math.exp(-u * u / 2) / (np.sqrt(2 * np.pi) * abs(sd))

    return y


def gen_x0(x, y, std, ns):
    particles = np.empty((3, ns))
    particles[0, :] = x + (np.random.randn(ns) * std)
    particles[1, :] = y + (np.random.randn(ns) * std)
    particles[2, :] = 0

    return particles


def resample(xk, wk, resampling_method, n):
    """
    Resamples the particle set according to the specified method

    Parameters
    ----------
    xk : list
        set of particles
    wk : list
        the weights of the particles
    resampling_method : str
        resampling technique to be used
    n : double
        number of particles

    Returns
    --------
    xk : list
        new set of resampled particles
    wk : list
        normalized weights of the new set of particles
    """

    resampling_methods = {'multinomial_resampling': multinomial_resampling,
                          'systematic_resampling': systematic_resampling,
                          'residual_resampling': residual_resampling,
                          'residual_resampling2': residual_resampling2,
                          'stratified_resampling': stratified_resampling}
    indx = resampling_methods[resampling_method](wk, n)
    indx = indx.astype(int)
    xk = xk[:, indx]
    # wk = np.ones(len(wk)) / len(wk)
    wk = np.ones(n) / n

    return xk, wk


def multinomial_resampling(wk, n):
    # n = len(wk)
    w = wk / np.sum(wk)
    q = w.cumsum(axis=0)
    indx = np.zeros(n)
    i = -1
    while i < n - 1:
        i += 1
        sampl = random.uniform(0, 1)
        j = 0
        while q[j] < sampl:
            j += 1
        indx[i] = j

    return indx


def multinomial_resampling2(weights, n):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors

    return np.searchsorted(cumulative_sum, random(len(weights)))


def systematic_resampling(wk, n):
    n = len(wk)
    w = wk / np.sum(wk)
    q = w.cumsum(axis=0)
    indx = np.zeros(n)
    t = np.linspace(0, 1 - 1 / n, n) + random.uniform(0, 1) / n
    i = 0
    j = 0
    while i < n and j < n:
        while q[j] < t[i]:
            j += 1
        indx[i] = j
        i += 1

    return indx


def systematic_resampling2(weights, n):
    N = len(weights)

    # make N subdivisions, choose positions with a consistent random offset
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def residual_resampling(wk, n):
    # n = len(wk)
    w = wk / np.sum(wk)
    indx = np.zeros(n)
    ns = [math.floor(i) for i in (wk * n)]
    if n > len(wk):
        ns = np.concatenate((ns, ns))
    r = np.sum(ns)
    i = 0
    j = 0  # -1
    while j < n:
        j += 1
        cnt = 1
        while cnt <= ns[j - 1]:
            indx[i] = j - 1
            i += 1
            cnt += 1
    n_rdn = n - r
    ws = (n * w - ns) / n_rdn
    q = ws.cumsum(axis=0)
    while i <= n:
        sampl = random.uniform(0, 1)
        j = 0
        while q[j] < sampl:
            j += 1
        if i < n:
            indx[i] = j - 1
        i += 1

    return indx


def residual_resampling2(weights, n):
    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight
    num_copies = (N * np.asarray(weights)).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1

    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= sum(residual)  # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.  # ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random((N - k)))

    return indexes


def stratified_resampling(wk, n):
    n = len(wk)
    w = wk / np.sum(wk)
    q = w.cumsum(axis=0)
    indx = np.zeros(n)
    t = np.linspace(0, 1 - 1 / n, n) + np.ones(n) * random.uniform(0, 1) / n
    i = 0
    j = 0
    while i < n and j < n:
        while q[j] < t[i]:
            j += 1
        indx[i] = j
        i += 1

    return indx


def stratified_resampling2(weights, n):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def weighted_variance(particles, weights, mean):
    var = 0
    for i in range(len(weights)):
        var += weights[i] * (particles[:, i] - mean) ** 2

    return var * len(weights) / (len(weights) - 1)


def one_wheel_slip_detection(r, d, encoder, gyro):
    slipping = 0
    yaw_vel_enc = r * (encoder[0] - encoder[1]) / d
    if abs(gyro[2] - yaw_vel_enc) >= 0.2:
        slipping = 1.2

    return slipping


def two_wheel_slip_detection(r, prev_vel, encoder, acc, ts):
    slipping = 0
    vel_enc = r * (encoder[0] + encoder[1]) / 2
    vel_update = prev_vel + acc * ts
    if abs((vel_update - vel_enc)) >= 0.1:
        slipping = 0.98
    acc_1 = (vel_enc - prev_vel) / ts
    # if abs(acc_1 - acc) >= 0.5:
    #     slipping = 0.95

    return slipping
