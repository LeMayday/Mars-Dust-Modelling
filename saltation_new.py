# saltation-related functions to be used with snapy model

import torch
from mars import *

A_N = 0.0123
y = 3E-4                # kg / s^2
z0 = 0.01               # m
rho_p = 2650            # kg / m^3 particle mass on Martian surface
inv_z0 = 1 / z0
K = 0.4                 # von Karman constant
inv_grav = 1 / grav


def vert_flux(vx: torch.Tensor, vy: torch.Tensor, rho: torch.Tensor, dx: float, dy: float, dz: float, D: float) -> torch.Tensor:
    H = hori_flux_calculator(rho, dz, D)
    return H(vx) * dy + H(vy) * dx


def hori_flux_calculator(rho: torch.Tensor, dz: float, D: float) -> torch.Tensor:
    '''
    Uses a closure to store values that are the same for the x and y fluxes
    '''
    a = K / torch.log(dz * inv_z0)
    b = 0.25 * rho * inv_grav
    v_fric_thresh_sq = A_N * (rho_p / rho * grav * D + y / D / rho)

    def hori_flux(v: torch.Tensor) -> torch.Tensor:
        v_fric = a * v
        v_ratio = v_fric_thresh_sq / v_fric**2
        flux = b * v_fric**3 * (1 - v_ratio) * (7.0 + 50.0 * v_ratio)
        return torch.clamp(flux, min=0)

    return hori_flux

