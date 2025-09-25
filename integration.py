# Author: Chris May
# 3/15/2025

############################## References ##############################
'''
[1] Numerical Methods for Fluid Dynamics, Dale R. Durran, 2nd Ed
Also see: https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html
https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html#region-of-absolute-stability
https://www.cfm.brown.edu/people/sg/SSPlinear.pdf
https://www.math.umd.edu/~tadmor/pub/linear-stability/Gottlieb-Shu-Tadmor.SIREV-01.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple

# [1] pg 56
SSP_RK3_Coeff = np.array([[3/4, 1/4], [1/3, 2/3]])
# problem definition
D = 1
w = -2
zi = 0
zf = 1

# xi and xf are domain values where BCs take effect
# function generates n evenly spaced points between xi and xf exclusive
def generate_grid_w_boundaries(xi, xf, n):
    dx = (xf - xi) / (n + 1)
    # linspace is inclusive by default
    return np.linspace(xi, xf, n + 2)[1:-1], dx

num_z = 100
rho_a = np.ones(num_z + 2)
z, dz = generate_grid_w_boundaries(zi, zf, num_z)

# flag for using Neumann BCs vs Dirichlet BCs
neumann = True

def main():
    t_stop = 1
    F = 10

    L = generate_RHS_matrix(num_z, dz, rho_a)

    # current method of finding stability is based on most negative eigenvalue (Euler). RK3 should have larger region of stability
    evals, _ = np.linalg.eig(L)
    e = min(evals)
    dt = -np.real(e) / (np.real(e)**2 + np.imag(e)**2)

    # test function that I found analytical solution to in Math 454
    Z0 = 1/2 * np.sin(np.pi * z) - 3/2 * np.sin(3 * np.pi * z) + np.sin(8 * np.pi * z)

    time_evolve(t_stop, dt, Z0, L, F, "result_SSP_RK3.mp4")

# redefined function because I wanted to get method hints for axes
def subplots() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    return fig, ax

def plot_func(z, Z, time, Z_min = 0, Z_max = 0):
    fig, ax = subplots()
    ax.plot(z, Z)
    if Z_min != 0 and Z_max != 0:
        ax.set_ylim(min(Z_min, np.min(Z)), max(Z_max, np.max(Z)))
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ax.text(xlims[1], ylims[1], "{:.2f}".format(time), verticalalignment='bottom', horizontalalignment='right')
    return fig

# n x n matrix
def generate_RHS_matrix(n, dz, rho_a):
    # the approach I take makes a 2-size larger matrix, fills in diagonal, then cuts border. Consider also scipy.sparse.diags
    L = np.zeros((n + 2, n + 2))
    for i in range(1, n+1):
        # make note of how rho_a indices are used - might have to flip direction of rho_a vector
        L[i, i-1] = D / dz**2 * rho_a[i] / rho_a[i-1] - w / dz
        L[i, i] = -2 * D / dz**2 + w / dz
        L[i, i+1] = D / dz**2 * rho_a[i] / rho_a[i+1]
    L = L[1:-1, 1:-1]
    if neumann:
        # -2 in rho is the same as -1 in L
        L[-1, -1] = L[-1, -1] + D / dz**2 * rho_a[-2] / rho_a[-1]
    return L

def euler_step(f, L, dt, F):
    # check supporting notes
    if neumann:
        flux_vec = np.zeros_like(f)
        c = D / dz**2 * rho_a[-2] / rho_a[-1]
        flux_vec[-1] = c * F/D * dz
        return f + dt * (L @ f + flux_vec)
    return f + dt * L @ f

def SSP_RK3(f, L, dt, F, s=3):
    # s is stage
    if s == 1:
        return euler_step(f, L, dt, F)
    # python does not respect tail recursion
    return SSP_RK3_Coeff[s-1 - 1, 0] * f + SSP_RK3_Coeff[s-1 - 1, 1] * euler_step(SSP_RK3(f, L, dt, F, s-1), L, dt, F)

def time_evolve(t_stop, dt, Zn, L, F, savename):
    filenames = []
    t_current = 0
    n = 0
    while t_current <= t_stop:
        # skips the first time just so it can be plotted
        if t_current != 0:
            Zn = SSP_RK3(Zn, L, dt, F)
        if n % 100 == 0:
            fig = plot_func(z, np.flip(Zn), t_current, -3.0, 3.0)
            output_file = 'frame_%s.png' %(n)
            fig.savefig(output_file)
            plt.close(fig)
            filenames.append(output_file)
        t_current += dt
        n += 1

    fig, ax = subplots()
    ax.plot(z, np.flip(Zn), label='Numerical')
    ax.plot(z, F/w * np.exp(w/D) + -F/w * np.exp(w/D * z), '--k', label='Analytical')
    ax.legend()
    ax.set_ylim(min(-1.0, np.min(Zn)), max(3.0, np.max(Zn)))
    fig.savefig('last_frame.png')
    plt.close(fig)
    create_movie(filenames, savename)
    delete_files(filenames)

if __name__ == "__main__":
    main()
