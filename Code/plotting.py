import matplotlib.pyplot as plt
import numpy as np

def plotparticles(particles, weights, x_pf, previa_distancia):
    figure, [noweight, yesweight] = plt.subplots(1, 2)
    for k in range(0, len(weights)):
        yesweight.plot(particles[0, k], particles[1, k], 'bo', alpha=weights[k] * 30, label='_nolegend_')
        noweight.plot(particles[0, k], particles[1, k], 'bo', alpha=0.5, label='_nolegend_')
    yesweight.plot(x_pf[0, :], x_pf[1, :], 'r', label='Estimated position')
    yesweight.plot(x_pf[0, -1], x_pf[1, -1], 'ro', markersize=5, alpha=0.8, label='_nolegend_')
    noweight.plot(x_pf[0, :], x_pf[1, :], 'r', label='Estimated position')
    noweight.plot(x_pf[0, -1], x_pf[1, -1], 'ro', markersize=5, alpha=0.8, label='_nolegend_')
    yesweight.axis('square')
    noweight.axis('square')
    dist_x = max(abs(min(particles[0, :]) - x_pf[0, -1]), abs(max(particles[0, :]) - x_pf[0, -1]))
    dist_y = max(abs(min(particles[1, :]) - x_pf[1, -1]), abs(max(particles[1, :]) - x_pf[1, -1]))
    distancia2 = max(dist_x, dist_y, 0.01)
    distancia = max(dist_x, dist_y, 0.01, previa_distancia)
    noweight.set_xlim([x_pf[0, -1] - distancia, x_pf[0, -1] + distancia])
    noweight.set_ylim([x_pf[1, -1] - distancia, x_pf[1, -1] + distancia])
    yesweight.set_xlim([x_pf[0, -1] - distancia, x_pf[0, -1] + distancia])
    yesweight.set_ylim([x_pf[1, -1] - distancia, x_pf[1, -1] + distancia])
    yesweight.set_xlabel('x [m]'), noweight.set_xlabel('x [m]')
    yesweight.set_ylabel('y [m]'), noweight.set_ylabel('y [m]')

    return distancia2



# fig = plt.figure(figsize=(15, 7.5))
# # plt.get_current_fig_manager().window.showMaximized()
# plt.get_current_fig_manager().window.state('zoomed')
# left, width = 0.1, 0.40
# bottom, height = 0.1, 0.8
# left_h, width_h = left + width + 0.2, 0.20
# bottom_h, height_h = left + height / 2, 0.4
# left_h2 = left + width + 0.2
# bottom_h2 = left
# rect_cones = [left, bottom, width, height]
# rect_box = [left_h, bottom_h, width_h, height_h]
# rect_box2 = [left_h2, bottom_h2, width_h, height_h]
# globalplot = plt.axes(rect_cones)
# partplot = plt.axes(rect_box)
# partplot2 = plt.axes(rect_box2)
#
# plt.ion()
# for i in range (1, 1000):
#     partplot.clear()
#     partplot2.clear()
#     globalplot.plot(np.cos(i), np.sin(i))
#     partplot.plot(1, np.cos(i))
#     partplot2.plot(1, np.sin(i))
#     plt.draw()
#     plt.pause(1)
#
#     # fig, [globalplot, partplot] = plt.subplots(1, 2)
#     fig = plt.figure()
#     # plt.get_current_fig_manager().window.showMaximized()
#     plt.get_current_fig_manager().window.state('zoomed')
#     left, width = -0.1, 0.8
#     bottom, height = 0.1, 0.8
#     left_h, width_h = left + 0.6, 0.5
#     bottom_h, height_h = left + height / 2, 0.5
#     rect_cones = [left, bottom, width, height]
#     rect_box = [left_h, bottom_h, width_h, height_h]
#     globalplot = plt.axes(rect_cones)
#     partplot = plt.axes(rect_box)