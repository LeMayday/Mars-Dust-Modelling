# Author: Chris May
# 2/1/2025

############################## References ##############################
'''
[1] Wind as a Geological Process, Greeley & Iversen, 1985
[2] Pankine and Ingersoll, Interannual Variability of Martian Global Dust Storms, 2002
[3] Boundary Layer Theory, Schlichting, 1955
[4] Soil Transport by Winds on Mars, White, 1979
'''
########################### Global Variables ###########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple

g = 375                                 # cm / s^2
rho_p_over_rho = 240000
rho_p = 2650 / 1000                     # g / cm^3
nu = 11.19                              # cm^2 / s
source_area_density = 1000               # g / cm
l = 3

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
    fig, ax = plot_loglog_contour(D * 1E4, U / 1E2, A_minus_A_func(D, U), [0])
    ax.set_xlabel("Particle Diameter (um)")
    ax.set_ylabel("Threshold Friction Speed (m/s)")
    fig.savefig("diameter_v_speed.png")

def plot_flux(d, u, fname, levels) -> Figure:
    D, U = np.meshgrid(d, u, indexing="ij")
    fluxes = np.load(fname)
    fig, ax = plot_loglog_contour(D * 1E4, U / 1E2, fluxes, levels)
    ax.set_xlabel("Particle Diameter (um)")
    ax.set_ylabel("Freestream Wind Speed (m/s)")
    #fig.savefig(sname)
    return fig

def plot_loglog_contour(X, Y, Z, levels):
    fig, ax = plt.subplots()
    c = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(c, c.levels)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # make ticks plain numbers instead of exponential notation https://stackoverflow.com/a/33213196
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    return fig, ax

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
    # [4] eq 15 pg 4649
    return rho_p / rho_p_over_rho * U_friction * (1 - R)
    #return U_friction**3 * (1 - R) * (1 + R**2) * rho_p / rho_p_over_rho / g

# takes an array of N particle diameter sizes and a total supply and returns an array of N-1 densities for N-1 diameter buckets
def surface_dust_supply(diameters: np.ndarray) -> np.ndarray:
    max_d = np.max(diameters)
    min_d = np.min(diameters)
    if l == 4:
        A = 8 * source_area_density / (rho_p * np.log(max_d / min_d))
    else:
        A = 8 * source_area_density * (4 - l) / (rho_p * (max_d**(4 - l) - min_d**(4 - l)))
    supply_per_D = rho_p * (diameters / 2)**3 * A * diameters**(-l)
    # need to multiply by size of bucket (equivalent to differential element dD)
    for i in range(len(supply_per_D) - 1):
        # left hand value taken as size for bucket
        supply_per_D[i] = supply_per_D[i] * (diameters[i+1] - diameters[i])
    return supply_per_D[:-1]

def plot_bucket_depletion(diameters, bucket_times):
    fig, ax = plot_loglog(diameters[:-1], bucket_times, "Particle Diameter (um)", "Time to Deplete (s)")
    # plot vertical lines for cutoff diameters where flux drops to 0
    min_diameter_index = np.argmax(np.isfinite(bucket_times))
    ylims = ax.get_ylim()
    if min_diameter_index != 0:
        min_diameter = diameters[min_diameter_index]
        ax.loglog([min_diameter, min_diameter], ylims, '--k')
        # https://mkaz.blog/working-with-python/string-formatting/
        ax.text(min_diameter, ylims[1], "{:.2f}".format(min_diameter), verticalalignment='bottom', horizontalalignment='center')
    # minus 1 to hit the last finite value
    max_diameter_index = np.argmax(np.isinf(bucket_times[min_diameter_index:])) + min_diameter_index - 1
    if max_diameter_index != min_diameter_index - 1:
        max_diameter = diameters[max_diameter_index]
        ax.loglog([max_diameter, max_diameter], ylims, '--k')
        ax.text(max_diameter, ylims[1], "{:.1f}".format(max_diameter), verticalalignment='bottom', horizontalalignment='center')
    return fig

def plot_density_distribution(diameters, densities):
    fig, ax = plot_loglog(diameters[:-1] * 1E4, densities, "Particle Diameter (um)", "Bucket Density (mass per unit space)")
    return fig

def plot_loglog(x, y, xlabel, ylabel) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.loglog(x, y)
    #ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

################################ Script ################################

num_pts = 200
u_star_t = np.geomspace(1, 10, num_pts) * 1E2                       # cm/s
particle_diameters = np.geomspace(10, 1000, num_pts) * 1E-4                        # cm
u_freestream = np.geomspace(0.5, 30, num_pts) * 1E2                 # cm/s

#plot_contour(particle_diameters, u_star_t)

flux_tensor = generate_flux_tensor(particle_diameters, u_freestream)
np.save("flux_tensor.npy", flux_tensor)

fig = plot_flux(particle_diameters, u_freestream, "flux_tensor_old.npy", [0, 1, 2, 4, 6, 10, 15, 20, 30, 40, 60])
fig.savefig("fluxes_per_L.png")

fig = plot_flux(particle_diameters, u_freestream, "flux_tensor.npy", 10)
fig.savefig("fluxes.png")

interp = RegularGridInterpolator((particle_diameters, u_freestream), flux_tensor, bounds_error=False)

u_test = np.array([5, 10, 20]) * 1E2
bucket_densities = surface_dust_supply(particle_diameters)

fig = plot_density_distribution(particle_diameters, bucket_densities)
fig.savefig("diameter_v_bucket_densities_l_%s.png" %(l))

for u in u_test:
    sample_pts = np.array([particle_diameters[:-1], np.full_like(bucket_densities, u)]).T
    fluxes = interp(sample_pts)
    times = bucket_densities / fluxes
    fig = plot_bucket_depletion(particle_diameters * 1E4, times)
    fig.savefig("diameter_v_deplete_time_%s_cm_s.png" %(u))
