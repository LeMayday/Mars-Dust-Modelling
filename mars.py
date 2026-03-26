# https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/mars-atmosphere-equation-metric/
# https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm

# atmosphere

p0 = 600            # Pa average surface pressure on Mars
Ts = 200            # K average surface temperature
gamma = 1.294
M_bar = 43.45E-3    # kg / mol
R_gas = 8.31446     # J / mol / K
species = [{'name': 'dry', 'composition': {'C': 0.95, 'O': 1.90, 'N': 0.06, 'Ar': 0.02}, 'cv_R': 3.4}]


# forcing

grav = 3.73         # m / s^2 Mars gravity constant
s0 = 580;           # W / m^2
# Teq = (s0 / 5.67E-8 / 4) ** (1/4)
q_dot = s0 / 4      # heat flux
# q_dot = q_dot / 2
