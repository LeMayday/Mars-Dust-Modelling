# Author: Chris May
# 3/15/2025

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple

SSP_RK3_Coeff = np.array([[3/4, 1/4], [1/3, 2/3]])

def subplots() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    return fig, ax

def plot_func(z, Z, time):
    fig, ax = subplots()
    ax.plot(z, Z)
    ax.set_ylim([min(-3.0, np.min(Z)), max(3.0, np.max(Z))])
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ax.text(xlims[1], ylims[1], "{:.2f}".format(time), verticalalignment='bottom', horizontalalignment='right')
    return fig

def euler_step(f, L, dt):
    return f + dt * L @ f

def SSP_RK3(f, L, dt, s):
    # s is stage
    if s == 1:
        return euler_step(f, L, dt)
    return SSP_RK3_Coeff[s-1 - 1, 0] * f + SSP_RK3_Coeff[s-1 - 1, 1] * euler_step(SSP_RK3(f, L, dt, s-1), L, dt)

def time_evolve(t_stop, z_grid, Zn, Phi, savename):
    filenames = []
    t_current = 0
    n = 0
    while t_current <= t_stop:
        if t_current != 0:
            Zn = SSP_RK3(Zn, Phi, dt, 3)
        if n % 25 == 0:
            fig = plot_func(z_grid, Zn, t_current)
            output_file = 'frame_%s.png' %(n)
            fig.savefig(output_file)
            plt.close(fig)
            filenames.append(output_file)
        t_current += dt
        n += 1
    create_movie(filenames, savename)
    delete_files(filenames)

zi = 0
zf = 1
num_z = 100
dz = (zf - zi) / num_z

t_stop = 0.1
z = np.linspace(zi, zf, num_z)

D = 1
w = 2
rho_a = 1 + 0*z

m = num_z - 2
L = np.zeros((num_z, num_z))
for i in range(1, num_z - 1):
    L[i, i-1] = D / dz**2 * rho_a[i] / rho_a[i-1] - w / dz
    L[i, i] = -2 * D / dz**2 + w / dz
    L[i, i+1] = D / dz**2 * rho_a[i] / rho_a[i+1]
L = L[1:-1, 1:-1]

evals, _ = np.linalg.eig(L)
e = min(evals)
dt = -np.real(e) / (np.real(e)**2 + np.imag(e)**2)

Z0 = 1/2 * np.sin(np.pi * z) - 3/2 * np.sin(3 * np.pi * z) + np.sin(8 * np.pi * z)

time_evolve(t_stop, z[1:-1], Z0[1:-1], L, "result_SSP_RK3.mp4")
