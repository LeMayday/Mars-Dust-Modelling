# Author: Chris May
# 2/1/2024

import torch
import numpy as np

num_pts = 50
u_star_t = torch.as_tensor(np.geomspace(1, 10, num_pts))
D_p = torch.as_tensor(np.geomspace(10, 1000, num_pts))

nu = 11.19      # cm^2/s
g = 375         # cm/s^2
rho_p_rho = 24000
rho_p = 2650    # kg/m^3

A = torch.outer(u_star_t, torch.pow(1/D_p, 0.5)) * (1/rho_p_rho / g)
B = torch.outer(u_star_t, D_p) / nu

