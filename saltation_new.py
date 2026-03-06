# saltation-related functions to be used with snapy model

import torch
from mars import *
from typing import Callable
from saltation import surface_dust_supply, plot_density_distribution, plot_bucket_depletion

A_N = 0.0123
y = 3E-4                # kg / s^2
z0 = 0.01               # m
rho_p = 2650            # kg / m^3 particle mass on Martian surface
inv_z0 = 1 / z0
K = 0.4                 # von Karman constant
inv_grav = 1 / grav


def vert_flux(vx: float, vy: float, rho: float, dx: float, dy: float, dz: float, D: torch.Tensor) -> torch.Tensor:
    H = hori_flux_calculator(rho, dz, D)
    return H(vx) * dy + H(vy) * dx


def hori_flux_calculator(rho: float, dz: float, D: torch.Tensor) -> Callable[[float], torch.Tensor]:
    '''
    Uses a closure to store values that are the same for the x and y fluxes
    '''
    a = K / torch.log(dz * inv_z0)
    b = 0.25 * rho * inv_grav
    v_fric_thresh_sq = A_N * (rho_p / rho * grav * D + y / D / rho)

    def hori_flux(v: float) -> torch.Tensor:
        v_fric = a * v
        v_ratio_sq = v_fric_thresh_sq / v_fric**2
        flux = b * v_fric**3 * (1 - v_ratio_sq) * (7.0 + 50.0 * v_ratio_sq)
        return torch.clamp(flux, min=0)

    return hori_flux

def main():
    dx = 80E3 / 256
    dy = 80E3 / 256
    dz = 20E3 / 64
    R = R_gas / M_bar
    rho0 = p0 / (R * Ts)

    vx = 0
    vy = 10
    nbins = 10
    D = torch.logspace(-6, -5, nbins + 1)   # m
    Q = vert_flux(vx, vy, rho0, dx, dy, dz, D)

    source_area_density = 1E6       # kg / m^2
    l = 3
    rho_p = 2650                    # kg / m^3
    bucket_densities = surface_dust_supply(D.numpy(), rho_p, source_area_density, l)
    fig = plot_density_distribution(D.numpy() * 1E6, bucket_densities)
    fig.savefig(f"diameter_v_bucket_densities_l_{l}.png")

    times = bucket_densities / Q
    fig = plot_bucket_depletion(D.numpy() * 1E6, times)
    fig.savefig(f"diameter_v_deplete_time_{vy}_m_s.png")

if __name__ == "__main__":
    main()
