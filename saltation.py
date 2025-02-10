# Author: Chris May
# 2/1/2024

############################## References ##############################
'''
[1] Wind as a Geological Process, Greeley & Iversen, 1985
[2] Pankine and Ingersoll, Interannual Variability of Martian Global Dust Storms, 2002
[3] Boundary Layer Theory, Schlichting, 1955
'''
########################### Global Variables ###########################

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator

g = 375                                 # cm / s^2
rho_p_over_rho = 240000
rho_p = 2650 / 1000                     # g / cm^3
nu = 11.19                              # cm^2 / s
source_area_density = 1000               # g / cm

########################### Helper Functions ###########################

# [1] pg 81
def A_func(D, U):
    # friction Reynolds number
    B = U * D / nu
    K = 0.006               # g cm^0.5 / s^2
    f_D = (1 + K / (rho_p * g * D**2.5))**0.5
    # the dependence on B is piecewise
    f_B1 = 0.2 / (1 + 2.5 * B)**0.5
    f_B2 = 0.129 / (1.928 * B**0.092 - 1)**0.5
    f_B3 = 0.120 * (1 - 0.0858 * np.exp(-0.0617 * (B - 10)))

    A_out = f_B1 * f_D
    A_out[B > 0.3] = (f_B2 * f_D)[B > 0.3]
    A_out[B > 10] = (f_B3 * f_D)[B > 10]
    return A_out

def A_minus_A_func(D, U):
    return U * (1/rho_p_over_rho / g / D) ** 0.5 - A_func(D, U)

# plots 0 contour to recreate [1] Fig. 3.17 pg. 92
def plot_contour(d, u):
    D, U = np.meshgrid(d, u, indexing="ij")
    fig, ax = plt.subplots()
    # D in um, U in m/s
    ax.contour(D * 1E4, U / 1E2, A_minus_A_func(D, U), levels=[0])
    ax.set_xscale('log')
    ax.set_yscale('log')
    # make ticks plain numbers instead of exponential notation https://stackoverflow.com/a/33213196
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xlabel("Particle Diameter (um)")
    ax.set_ylabel("Threshold Friction Speed (m/s)")
    #plt.show()
    fig.savefig("diameter_v_speed.png")

# takes vector input and performs root finding for each element in the vector
def find_threshold_speeds(diameters: np.ndarray) -> np.ndarray:
    speeds = np.zeros_like(diameters)
    for i, d in np.ndenumerate(diameters):
        func = lambda u : A_minus_A_func(d, u)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
        # initial guess is 500 cm/s which is close for the full range of diameters
        speeds[i] = fsolve(func, 500)[0]
    return speeds

# [2]
# flux tensor is lookup table of particle flux as a function of diameter and freestream wind speed
def generate_flux_tensor(particle_diameters, u_freestream):
    u_threshold = find_threshold_speeds(particle_diameters)
    U_threshold, U_fs = np.meshgrid(u_threshold, u_freestream, indexing="ij")
    # skin-friction coefficient from [3] pg 577 eq. 18.76
    # advised to set as 0.5
    Cf = 0.5
    U_friction = U_fs * Cf
    # when R = 1, G = 0. If U_friction < U_threshold (R > 1), then particles will not lift off
    R = np.minimum(U_threshold / U_friction, np.ones_like(U_friction))
    # equation given in text has s0 proportionality parameter, but this is assumed to be 1 here
    return U_friction**3 * (1 - R) * (1 + R**2) * rho_p / rho_p_over_rho / g

# takes an array of N particle diameter sizes and a total supply and returns an array of N-1 densities for N-1 diameter buckets
def surface_dust_supply(diameters: np.ndarray) -> np.ndarray:
    l = 2
    max_d = np.max(diameters)
    min_d = np.min(diameters)
    A = 8 * source_area_density * (4 - l) / (rho_p * (max_d**(4 - l) - min_d**(4 - l)))
    supply_per_D = rho_p * (diameters / 2)**3 * A * diameters**(-l)
    # need to multiply by size of bucket (equivalent to differential element dD)
    for i in range(len(supply_per_D) - 1):
        # left hand value taken as size for bucket
        supply_per_D[i] = supply_per_D[i] * (diameters[i+1] - diameters[i])
    return supply_per_D[:-1]

def plot_bucket_depletion(diameters, bucket_times):
    fig, ax = plt.subplots()
    ax.plot(diameters[:-1], bucket_times)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xlabel("Particle Diameter (um)")
    ax.set_ylabel("Time to Deplete (s)")
    return fig

################################ Script ################################

num_pts = 200
u_star_t = np.geomspace(1, 10, num_pts) * 1E2                       # cm/s
particle_diameters = np.geomspace(10, 1000, num_pts) * 1E-4                        # cm
u_freestream = np.geomspace(0.5, 30, num_pts) * 1E2                 # cm/s

plot_contour(particle_diameters, u_star_t)

flux_tensor = generate_flux_tensor(particle_diameters, u_freestream)
np.save("flux_tensor.npy", flux_tensor)

interp = RegularGridInterpolator((particle_diameters, u_freestream), flux_tensor, bounds_error=False)

u_test = np.array([3, 10, 20]) * 1E2
bucket_densities = surface_dust_supply(particle_diameters)

for u in u_test:
    sample_pts = np.array([particle_diameters[:-1], np.full_like(bucket_densities, u)]).T
    fluxes = interp(sample_pts)
    times = bucket_densities / fluxes
    fig = plot_bucket_depletion(particle_diameters, times)
    fname = "diameter_v_deplete_time_%s_cm_s.png" %(u)
    fig.savefig(fname)
