# Author: Chris May
# 2/1/2024

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

g = 375                     # cm / s^2
rho_p_over_rho = 240000
rho_p = 2650 / 1000         # g / cm^3

def A_func(D, U):
    nu = 11.19              # cm^2 / s
    B = U * D / nu
    K = 0.006               # g cm^0.5 / s^2
    f_D = (1 + K / (rho_p * g * D**2.5))**0.5
    f_B1 = 0.2 / (1 + 2.5 * B)**0.5
    f_B2 = 0.129 / (1.928 * B**0.092 - 1)**0.5
    #f_B2 = 0.120 * (1 - 0.0858 * torch.exp(-0.0617 * (B - 10)))

    A_out = f_B1 * f_D
    A_out[B > 0.3] = (f_B2 * f_D)[B > 0.3]
    #A_out[B > 10] = (f_B2 * f_D)[B > 10]
    return A_out

num_pts = 50
u_star_t = torch.as_tensor(np.geomspace(1, 10, num_pts)) * 1E2
D_p = torch.as_tensor(np.geomspace(10, 1000, num_pts)) * 1E-4
U, D = torch.meshgrid(u_star_t, D_p, indexing="ij")

A = U * (1/rho_p_over_rho / g / D) ** 0.5

ax = plt.subplot()
ax.contour(D * 1E4, U / 1E2, A - A_func(D, U), levels=[0])
ax.set_xscale('log')
ax.set_yscale('log')
# https://stackoverflow.com/a/33213196
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
plt.show()
