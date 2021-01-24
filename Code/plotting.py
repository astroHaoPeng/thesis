import matplotlib.pyplot as plt


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
