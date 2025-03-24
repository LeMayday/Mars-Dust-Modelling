# Author: Chris May
# 3/15/2025

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie

# c_vec = np.array([1/4, 3/4])
b_vec = np.array([1/2, 1/2])
a_mat = np.array([[1/4, 0], [1/2, 1/4]])

zi = 0
zf = 1
num_z = 1000
dz = (zf - zi) / num_z
dt = 0.001

n = 100
z = np.linspace(zi, zf, num_z)

D = 1
w = 0
rho_a = 1 + 0*z

m = num_z - 2
F = np.zeros((num_z, num_z))
for i in range(1, num_z - 1):
    F[i, i-1] = D / dz**2 * rho_a[i] / rho_a[i-1] - w / dz
    F[i, i] = -2 * D / dz**2 + w / dz
    F[i, i+1] = D / dz**2 * rho_a[i] / rho_a[i+1]
F = F[1:-1, 1:-1]

I = np.eye(m)

Phi_RK2 = I + b_vec[0] * dt * inv(I - dt * a_mat[0,0] * F) @ F + b_vec[1] * dt * inv(I - dt * a_mat[1,1] * F) @ F @ (I + dt * a_mat[0,1] * inv(I - dt * a_mat[0,0] * F) @ F)

Phi_RK1 = inv(I - dt * F)

Phi_RK1_1 = inv(I + dt * inv(I - dt * F) @ F)

def diagonalize(M):
    evals, evecs = np.linalg.eig(M)
    S = evecs
    D = np.diag(evals)
    return S, D


def integrate(n, Z0, S, D):
    if n == 0:
        return Z0
    return S @ D**n @ inv(S) @ Z0

def plot_func(z, Z):
    fig, ax = plt.subplots()
    ax.plot(z, Z)
    ax.set_ylim([min(-3.0, np.min(Z)), max(3.0, np.max(Z))])
    return fig

def time_evolve(n, z_grid, Z0, Phi, savename):
    S, D = diagonalize(Phi)
    filenames = []
    for num in range(n):
        fig = plot_func(z_grid, integrate(num, Z0, S, D))
        output_file = 'frame_%s.png' %(num)
        fig.savefig(output_file)
        plt.close(fig)
        filenames.append(output_file)
    create_movie(filenames, savename)
    delete_files(filenames)

Z0 = 1/2 * np.sin(np.pi * z) - 3/2 * np.sin(3 * np.pi * z) + np.sin(8 * np.pi * z)

time_evolve(n, z[1:-1], Z0[1:-1], Phi_RK2, "result_RK2.mp4")

# time_evolve(n, z[1:-1], Z0[1:-1], Phi_RK1, "reuslt_RK1.mp4")

# time_evolve(n, z[1:-1], Z0[1:-1], Phi_RK1_1, "reuslt_RK1_1.mp4")
