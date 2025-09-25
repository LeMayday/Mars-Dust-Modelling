
import torch

p0 = 101.5E3    # Pa
rho0 = 1.25     # kg/m^3
T0 = 300        # K

R = 287         # J/kg/K
Cv = 5/2 * R
Cp = Cv + R
g = 9.8         # m/s^2

z_min = 0
z_max = 10E3
z_cells = 100

z = torch.linspace(z_min, z_max, z_cells)

# isothermal atmosphere

p_isotherm = p0 * torch.exp(-g * z / R / T0)
rho_isotherm = rho0 * torch.exp(-g * z / R / T0)

# adiabatic atmosphere

T_adiabatic = -g / Cp * z + T0
p_adiabatic = p0 * (T_adiabatic / T0) ** (Cp / R)
rho_adiabatic = rho0 * (T_adiabatic / T0) ** (Cv / R)

# 2D, Note first index is air column, second is row

x_cells = 100
p_isotherm_2D = p_isotherm.repeat(x_cells, 1)
rho_isotherm_2D = rho_isotherm.repeat(x_cells, 1)

p_adiabatic_2D = p_adiabatic.repeat(x_cells, 1)
rho_adiabatic_2D = rho_adiabatic.repeat(x_cells, 1)
print(rho_isotherm)
print(rho_adiabatic)
