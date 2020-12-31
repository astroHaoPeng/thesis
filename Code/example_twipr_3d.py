import numpy as np
from numpy import nan as nan
import matplotlib.pyplot as plt
from dynamics.twipr_dynamics import Twipr3D, ModelMichael
from sensormodel import Accelerometer, Gyroscope, Encoder
from particlefilter import ParticleFilter, one_wheel_slip_detection, two_wheel_slip_detection, weighted_variance
import quaternionestimation
import quaternions as qt
import geometricpath as path
import timing

# generate a new roboter object with the model from Michael and a simulation time of 0.02s
robot = Twipr3D(ModelMichael, 0.01)
accelerometer = Accelerometer(rz=0.195, axis='', rotation=90, bias=0.02, wn_sigma=0.003, bias_instability_var=0.001)
encoder = Encoder(wheel_r=0.055, d=0.25, ticks_per_revolution=1024, gear_ratio=5175/247)
gyroscope = Gyroscope(rz=0.195, axis='', rotation=90, bias=1, wn_sigma=0.075, bias_instability_var=0.01)

# set the poles and eigenvectors in order to decouple the pitch and yaw motion as well as stabilize the pitch motion
poles = [0, -20, -3 + 1j, -3 - 1j, 0, -1.5]
eigenvectors = np.array([[1, nan, nan, nan, 0, nan],
                         [nan, 1, nan, nan, nan, nan],
                         [nan, nan, 1, 1, nan, 0],
                         [nan, nan, nan, nan, nan, nan],
                         [0, nan, nan, nan, 1, 1],
                         [nan, 0, 0, 0, nan, nan]])

robot.set_eigenstructure(poles, eigenvectors)

# add reference controllers for x_dot and psi_dot
robot.set_reference_controller_pitch(-1.534 / 2, -2.81 / 2, -0.07264 / 2, 1)
robot.set_reference_controller_yaw(-0.3516, -1.288, -0.0002751, 5)

# make a simulation with x_dot_cmd = 1 and psi_dot_cmd = -1
t = np.arange(start=0, step=robot.Ts, stop=5)
N = len(t)

xdot_cmd = 0.5
psidot_cmd = 0.5

w = np.vstack((xdot_cmd * np.ones(t.shape), psidot_cmd * np.ones(t.shape)))

# Simulation of another path
s = np.linspace(0, 1, N)
k = 5
theta_f = 0
theta_i = 0
xf = 2
yf = 2
x_path, y_path, theta_d, v, w2 = path.path_and_cmd(0, 0, xf, yf, theta_i, theta_f, k, t, N)

# simulate all time steps at once
mode = 'nonlinear'
# [y, x] = robot.simulate(w, mode, [0, 0, 0, 0, 0, 0])

# simulate one time step each. But first reset the state of the robot and reset the controllers
robot.reset()
robot.set_state([0, 0, 0, 0, 0, 0])

# VARIABLE INITIALIZATION
# True state of the robot
x2 = np.zeros((6, N))                   # [X, X_dot, theta, theta_dot, psi, psi_dot]
x2[:, 0] = robot.state
real_position = np.zeros((2, N))        # X and Y coordinates

# Trajectory Tracking
e_x = np.zeros(N)
e_y = np.zeros(N)
e_t = np.zeros(N)

# Accelerometer
acceleration = np.zeros((3, N))         # Measurements
acceleration[:, 0] = [0, 0, 9.81]
qk = np.zeros((4, N))                   # Orientation of the robot
qk[0, :] = 1
omegak = np.zeros((3, N))               # Angular velocity of the robot

# Gyroscope
gyro = np.zeros((3, N))                 # Measurements
qk2 = np.zeros((4, N))
qk2[0, :] = 1                           # Orientation of the robot

# Encoders
enc_meas = np.zeros((4, N))             # Measurements

# Orientation Estimation
gyro_s = np.zeros((3, N))               # Gyroscope measurements in navigation frame
acc_s = np.zeros((3, N))                # Accelerometer measurements in navigation frame
acc_s[:, 0] = [0, 0, 9.81]
head_enc = 0                            # Heading given by the encoders
xh = np.zeros((4, N))                   # Orientation state estimate
xh[0, :] = 1
Ph = np.identity(4) * 1
heading = np.zeros((2, N))              # Heading and inclination from estimated orientation
euler = np.zeros((3, N))                # Orientation estimation in Euler angles 'zyx'
time_orientation = np.zeros(N)          # Execution time for the orientation estimation algorithm

# Position Estimation
x_pf = np.zeros((3, N))                 # State Estimate
x_pf[:, 0] = [0, 0, 0]
pf = ParticleFilter(x_pf[:, 0], 200)    # Class particle filter
vel_enc = np.zeros(N)                   # Velocity given by encoder measurements
theta_dot = np.zeros(N)                 # Pitch angular velocity
theta_2dot = np.zeros(N)                # Pitch angular acceleration
time_position = np.zeros(N)             # Execution time for the particle filter

# Slipping Detection
slipping_one = np.zeros(N)
slipping_two = np.zeros(N)
real_slip = np.zeros(N)

# Calibration
x2_cal = np.zeros((6, 1000))
x2_cal[:, 0] = robot.state
acceleration_cal = np.zeros((3, 1000))
acceleration_cal[:, 0] = [0, 0, 9.81]
qk_cal = np.zeros((4, 1000))
qk_cal[0, :] = 1
omegak_cal = np.zeros((3, 1000))
gyro_cal = np.zeros((3, 1000))
qk2_cal = np.zeros((4, 1000))
qk2_cal[0, :] = 1

al_est = np.zeros(N)
al_enc = np.zeros(N)

for i in range(1, 1000):
    [y, x2_cal[:, i]] = robot.simulate_step(np.array([0, 0]), mode)
    acceleration_cal[:, i], qk_cal[:, i], omegak_cal[:, i] = accelerometer.get_measurement(robot, qk_cal[:, i - 1], x2_cal[1, i - 1],
                                                                               omegak_cal[:, i - 1])
    acceleration_cal[:, i] -= np.array([0, 0, 9.81])
    gyro_cal[:, i], qk2_cal[:, i] = gyroscope.get_measurement(robot, qk2_cal[:, i - 1])
acc_bias = np.array([np.mean(acceleration_cal[0, :]), np.mean(acceleration_cal[1, :]), np.mean(acceleration_cal[2, :])])
gyr_bias = np.array([np.mean(gyro_cal[0, :]), np.mean(gyro_cal[1, :]), np.mean(gyro_cal[2, :])])

# fig, [globalplot, partplot] = plt.subplots(1, 2)
fig = plt.figure()
plt.get_current_fig_manager().window.showMaximized()
left, width = -0.1, 0.8
bottom, height = 0.1, 0.8
left_h, width_h = left + 0.6, 0.5
bottom_h, height_h = left + height / 2, 0.5
rect_cones = [left, bottom, width, height]
rect_box = [left_h, bottom_h, width_h, height_h]
globalplot = plt.axes(rect_cones)
partplot = plt.axes(rect_box)
prev_distancia = 0

for i in range(1, N):
    psi = x2[2, i - 1]
    e_x[i] = np.cos(psi) * (x_path[i] - real_position[0, i - 1]) + np.sin(psi) * (y_path[i] - real_position[1, i - 1])
    e_y[i] = -np.sin(psi) * (x_path[i] - real_position[0, i - 1]) + np.cos(psi) * (y_path[i] - real_position[1, i - 1])
    e_t[i] = theta_d[i] - x2[4, i - 1]
    kx = 2.5
    ky = 6
    kt = 4
    v_feed = v[i] * np.cos(e_t[i]) + kx * e_x[i]
    w_feed = w2[i] + ky * v[i] * e_y[i] + kt * v[i] * np.sin(e_t[i])
    v_feed = qt.clip(v_feed, -3, 3)
    w_feed = qt.clip(w_feed, -5, 5)

    if 160 <= i <= 200:
        v_feed = 0
        w_feed = 0
    # [y, x2[:, i]] = robot.simulate_step(w[:, i], mode)
    [y, x2[:, i]] = robot.simulate_step(np.array([v_feed, w_feed]), mode)

    # Sensor simulation
    acceleration[:, i], qk[:, i], omegak[:, i] = accelerometer.get_measurement(robot, qk[:, i - 1], x2[1, i - 1], omegak[:, i - 1])

    gyro[:, i], qk2[:, i] = gyroscope.get_measurement(robot, qk2[:, i - 1])

    enc_meas[:, i] = encoder.get_measurement(robot)

    # Bias removal
    acceleration[:, i] -= acc_bias
    gyro[:, i] -= gyr_bias

    # Slip simulation
    if np.random.random_sample() > 0.95:
        # One wheel slip
        enc_meas[-1, i] += 10
        enc_meas[-2, i] += 0
        real_slip[i] = 1.25
    elif np.random.random_sample() > 0.95 or 250 <= i <= 260:
        # Two wheel slip
        enc_meas[-1, i] += 10
        enc_meas[-2, i] += 10
        real_slip[i] = 1

    real_position[0, i] = real_position[0, i - 1] + x2[1, i] * np.cos(x2[4, i]) * robot.Ts
    real_position[1, i] = real_position[1, i - 1] + x2[1, i] * np.sin(x2[4, i]) * robot.Ts

    # Slip detection
    q_pitch1 = qt.quaternion(euler[1, i - 1], [0, 1, 0])
    gyro_slipping = qt.quaternion_rotate(q_pitch1, gyro[:, i])
    slipping_one[i] = one_wheel_slip_detection(encoder.r, encoder.d, np.array([enc_meas[-1, i], enc_meas[-2, i]]), gyro_slipping)

    # Orientation Estimation
    t_ori = timing.time()
    theta = x2[2, i]
    q_pitch = qt.quaternion(theta, [0, 1, 0])
    psi = x2[4, i]
    q_yaw = qt.quaternion(psi, [0, 0, 1])
    q_s = qt.multiply(q_yaw, q_pitch)
    gyro_s[:, i] = qt.quaternion_rotate(q_pitch, gyro[:, i])
    acc_s[:, i] = qt.quaternion_rotate(q_pitch, acceleration[:, i])

    R_IMU_b = np.identity(3)  # [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    acc = R_IMU_b @ acceleration[:, i]
    gyr = R_IMU_b @ gyro[:, i]

    # head_enc += robot.Ts * 0.055 * (enc_meas[3, i] - enc_meas[2, i]) / 0.25
    head_enc = heading[0, i - 1] + robot.Ts * 0.055 * (enc_meas[3, i] - enc_meas[2, i]) / 0.25

    V = np.identity(4) * 0.1
    W = np.identity(4) * 1000000
    W[3, 3] = 0.00001
    if slipping_one[i] == 1:
        W[3, 3] = 1000000000

    xh[:, i], Ph = quaternionestimation.ekf(xh[:, i - 1], gyr, acc, head_enc, V, W, Ph, 9.81, robot.Ts)
    heading[:, i] = qt.get_heading(xh[:, i])
    euler[:, i] = qt.euler_from_q(xh[:, i], 'zyx', True)
    time_orientation[i] = timing.time() - t_ori

    # Position Estimation
    t_pos = timing.time()
    vel_enc[i] = encoder.r * (enc_meas[2, i] + enc_meas[3, i]) / 2
    theta_dot[i] = gyro[1, i]
    theta_2dot[i] = (theta_dot[i] - theta_dot[i - 1]) / robot.Ts

    R_b_n = np.array([[np.cos(euler[1, i]), 0, np.sin(euler[1, i])],
                      [0, 1, 0],
                      [-np.sin(euler[1, i]), 0, np.cos(euler[1, i])]])
    acc_n = R_b_n @ acceleration[:, i]
    a_R = np.array([theta_2dot[i] * accelerometer.rz * np.cos(euler[1, i]) - theta_dot[i] ** 2 * accelerometer.rz * np.sin(euler[1, i]),
                    0,
                    -theta_2dot[i] * accelerometer.rz * np.sin(euler[1, i]) - theta_dot[i] ** 2 * accelerometer.rz * np.cos(euler[1, i])])
    a_T = acc_n - a_R - np.array([0, 0, 9.81])
    al_est[i] = a_T[0]
    al_enc[i] = (vel_enc[i] - vel_enc[i - 1]) / robot.Ts

    if slipping_one[i] == 0:
        slipping_two[i] = two_wheel_slip_detection(encoder.r, x_pf[-1, i - 1], np.array([enc_meas[-1, i], enc_meas[-2, i]]), a_T[0], robot.Ts)

    uk = np.array([a_T[0], heading[0, i], x2[5, i]])

    # First Method
    # if slipping_two[i] + slipping_one[i] >= 1:
    #     yk = np.array([real_position[0, i], real_position[1, i]])
    # else:
    #     yk = np.array([real_position[0, i], real_position[1, i], vel_enc[i]])
    # if 200 <= i <= 400:
    #     if slipping_one[i] + slipping_two[i] == 0:
    #         yk = np.array(vel_enc[i])
    #     else:
    #         yk = np.array([])

    # Second Method
    yk = np.array([])
    if i % 1 == 0 and i <= 100 or i >= 450:
        yk = np.append(yk, [real_position[0, i], real_position[1, i]])
    if slipping_two[i] + slipping_one[i] == 0:
        yk = np.append(yk, vel_enc[i])
    if i % 1 == 0:
        j = round(i / 1)
        x_pf[:, j] = pf.particlefilter(uk, yk, robot.Ts * 1, 'residual_resampling2')

        time_position[i] = timing.time() - t_pos
        # conf_interval = np.zeros((3, 2))
        # conf_interval[:, 1] = x_pf[:, i] + 1.96 * np.sqrt(variance / pf.ns)
        # conf_interval[:, 0] = x_pf[:, i] - 1.96 * np.sqrt(variance / pf.ns)

        # Plotting
        plt.ion()
        globalplot.clear()
        partplot.clear()
        for k in range(0, pf.ns):
            partplot.plot(pf.particles[0, k], pf.particles[1, k], 'bo', alpha=pf.w[k] * 20, label='_nolegend_')
            # partplot.plot(pf.particles[0, k], pf.particles[1, k], 'bo', alpha=0.5, label='_nolegend_')
        partplot.plot(real_position[0, 0:i + 1], real_position[1, 0:i + 1], 'r', label='Real position')
        globalplot.plot(real_position[0, 0:i + 1], real_position[1, 0:i + 1], 'r', label='Real position')
        globalplot.plot(x_pf[0, 0:j + 1], x_pf[1, 0:j + 1], 'y--', label='Estimated position')
        partplot.plot(x_pf[0, 0:j + 1], x_pf[1, 0:j + 1], 'y', label='Estimated position')
        partplot.plot(x_pf[0, j], x_pf[1, j], 'yo', markersize=5, alpha=0.8, label='_nolegend_')
        globalplot.axis('square')
        partplot.axis('square')
        globalplot.set_xlim([-0.1, 2.2])
        globalplot.set_ylim([-0.1, 2.2])
        dist_x = max(abs(min(pf.particles[0, :]) - x_pf[0, j]), abs(max(pf.particles[0, :]) - x_pf[0, j]))
        dist_y = max(abs(min(pf.particles[1, :]) - x_pf[1, j]), abs(max(pf.particles[1, :]) - x_pf[1, j]))
        distancia2 = max(dist_x, dist_y, 0.05)
        distancia = max(dist_x, dist_y, 0.05, prev_distancia)
        # ax[1].set_xlim([real_position[0, i] - 0.05, real_position[0, i] + 0.05])
        # ax[1].set_ylim([real_position[1, i] - 0.05, real_position[1, i] + 0.05])
        # partplot.set_xlim([x_pf[0, j] - 0.05, x_pf[0, j] + 0.05])
        # partplot.set_ylim([x_pf[1, j] - 0.05, x_pf[1, j] + 0.05])
        partplot.set_xlim([x_pf[0, j] - distancia, x_pf[0, j] + distancia])
        partplot.set_ylim([x_pf[1, j] - distancia, x_pf[1, j] + distancia])
        if 100 <= i <= 300:
            fig.suptitle('Time step: ' + str(i) + ' (time = ' + str(round(i * robot.Ts, 2)) + ' sec)' + '\n' + ' (No OptiTrack measurements)')
        else:
            fig.suptitle('Time step: ' + str(i) + ' (time = ' + str(round(i * robot.Ts, 2)) + ' sec)')
        partplot.set_title("Particles = " + str(pf.ns))
        globalplot.legend(loc='lower right')
        globalplot.set_xlabel('x [m]'), partplot.set_xlabel('x [m]')
        globalplot.set_ylabel('y [m]'), partplot.set_ylabel('y [m]')
        prev_distancia = distancia2
        plt.draw()
        plt.pause(0.01)


head_rmse = np.sqrt(((x2[4, :] - heading[0, :]) ** 2).mean())
perc1_90 = np.percentile(np.sqrt((x2[4, :] - heading[0, :]) ** 2), 90)
inc_rmse = np.sqrt(((abs(x2[2, :]) - heading[1, :]) ** 2).mean())
perc2_90 = np.percentile(np.sqrt((abs(x2[2, :]) - heading[1, :]) ** 2), 90)
position_rmse = np.sqrt(((real_position[0, :] - x_pf[0, :]) ** 2 + (real_position[1, :] - x_pf[1, :]) ** 2).mean())
perc3_90 = np.percentile(np.sqrt((real_position[0, :] - x_pf[0, :]) ** 2 + (real_position[1, :] - x_pf[1, :]) ** 2), 90)
print('inclination_rmse = ', np.around(inc_rmse * 180 / np.pi, 3), '\n', '90 perc = ', np.around(perc2_90 * 180 / np.pi, 3))
print('heading_rmse = ', np.around(head_rmse * 180 / np.pi, 3), '\n', '90 perc = ', np.around(perc1_90 * 180 / np.pi, 3))
print('position rmse = ', np.around(position_rmse, 3), '\n', '90 perc = ', np.around(perc3_90, 3))

pitch = x2[2, :]
yaw = x2[4, :]
q_pitch2 = qt.quaternion(pitch, [0, 1, 0])
q_yaw2 = qt.quaternion(yaw, [0, 0, 1])
q = qt.multiply(q_yaw2, q_pitch2)
gyro2 = qt.gyr_from_quat(q, 1 / robot.Ts)
gyr2 = qt.quaternion_rotate(q_pitch2, gyro2)

plt.ioff()
# PLOTS
plot_gyroscope_measurements = False
plot_accelerometer_measurements = False
plot_slip_detection = True
plot_execution_time = True
plot_heading_and_inclination = True
plot_position = True

# Gyroscope
if plot_gyroscope_measurements:
    fig1, ax1 = plt.subplots(3)

    ax1[0].plot(t, gyro_s[0, :], 'r')
    ax1[0].plot(t, gyr2[:, 0], '--')
    ax1[0].set_ylabel('x')

    ax1[1].plot(t, gyro_s[1, :], 'r')
    ax1[1].plot(t, gyr2[:, 1], '--')
    ax1[1].plot(t, x2[3, :], 'y')
    ax1[1].set_ylabel('y')

    ax1[2].plot(t, gyro_s[2, :], 'r')
    ax1[2].plot(t, gyr2[:, 2], '--')
    ax1[2].plot(t, x2[5, :], 'y')
    ax1[2].set_ylabel('z')

    fig1.suptitle('Gyroscope')

# Accelerometer
if plot_accelerometer_measurements:
    fig2, ax2 = plt.subplots(3)

    ax2[0].plot(t, acceleration[0, :], 'r')
    ax2[0].plot(t, acc_s[0, :], '--')
    ax2[0].set_ylabel('x')

    ax2[1].plot(t, acceleration[1, :], 'r')
    ax2[1].plot(t, acc_s[1, :], '--')
    ax2[1].set_ylabel('y')

    ax2[2].plot(t, acceleration[2, :], 'r')
    ax2[2].plot(t, acc_s[2, :], '--')
    ax2[2].set_ylabel('z')

    plt.legend(['body frame', 'nav frame'])
    fig2.suptitle('Accelerometer')

# Slipping Detection
if plot_slip_detection:
    fig3, ax3 = plt.subplots(2)

    # ax5[0].plot(t, x2[1, :], 'r', label='True velocity')
    # ax5[0].plot(t, x_pf[2, :], '--', label='Estimated velocity')
    # ax5[0].plot(t, vel_enc, 'y', label='Encoder velocity')
    # ax5[0].legend(loc='lower right')
    ax3[0].plot(t, enc_meas[-1, :], label='right wheel')
    ax3[0].plot(t, enc_meas[-2, :], label='left wheel')
    ax3[0].legend(loc='lower right')
    ax3[0].title.set_text('Encoder Measurements')
    ax3[1].plot(t, real_slip, 'ob', markersize=5)
    ax3[1].plot(t, slipping_one, 'or', markersize=5)
    ax3[1].plot(t, slipping_two, 'oy', markersize=5)
    ax3[1].set_ylim([0.9, 1.3])
    ax3[1].legend(['Real slip', 'One wheel slip', 'Two wheel slip'])

# Execution Time
if plot_execution_time:
    fig4, ax4 = plt.subplots(2)

    ax4[0].plot(t, time_orientation)
    ax4[0].set_ylabel('sec')
    ax4[0].title.set_text('Orientation Estimation')
    ax4[0].legend(['Average = ' + str(round(np.mean(time_orientation) * 1000)) + ' ms'])
    ax4[1].plot(t, time_position)
    ax4[1].set_ylabel('sec')
    ax4[1].title.set_text('Position Estimation')
    ax4[1].legend(['Average = ' + str(round(np.mean(time_position) * 1000)) + ' ms'])

# Heading and inclination
if plot_heading_and_inclination:
    fig5, ax5 = plt.subplots(2)
    ax5[0].plot(t, x2[4, :] * 180 / np.pi, 'r')
    ax5[0].plot(t, heading[0, :] * 180 / np. pi, '--')
    ax5[0].set_ylabel('deg')
    ax5[0].title.set_text('Heading')

    ax5[1].plot(t, abs(x2[2, :]) * 180 / np.pi, 'r')
    ax5[1].plot(t, heading[1, :] * 180 / np.pi, '--')
    ax5[1].set_ylabel('deg')
    ax5[1].title.set_text('Inclination')

    # ax3[2].plot(t, x2[2, :] * 180 / np.pi, 'r')
    # ax3[2].plot(t, euler[1, :] * 180 / np.pi, '--')
    # ax3[2].set_ylabel('deg')
    # ax3[2].title.set_text('Pitch angle')

    plt.legend(['True', 'Estimated'])

# Position
if plot_position:
    fig6, ax6 = plt.subplots(1)
    ax6.plot(real_position[0, :], real_position[1, :], 'r')
    ax6.plot(x_pf[0, :], x_pf[1, :], '--')
    ax6.title.set_text('Position')
    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('y [m]')

    plt.legend(['travelled path', 'estimated path'])

# Additional Plots
fig7, ax7 = plt.subplots(2)
ax7[0].plot(t, al_enc, 'r')
ax7[0].plot(t, al_est, '--')
ax7[0].legend(['acc from enc derivation', 'acc from accelerometer'])
ax7[1].plot(t, x2[1, :])
ax7[1].plot(t, x_pf[2, :], '--')
ax7[1].plot(t, vel_enc, 'y')
ax7[1].legend(['Real velocity', 'Estimated velocity', 'Encoder velocity'])

plt.grid()
plt.show()
