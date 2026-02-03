# following https://github.com/elijah-mullens/paddle/blob/main/docs/content/notebooks/Tutorial-Straka.ipynb
from enum import Enum
import yaml

class Res(Enum):
    COURSE = 'course'
    FINE = 'fine'

def get_res_multiplier(res: Res):
    assert(res in Res)
    match res:
        case Res.COURSE:
            return 1
        case Res.FINE:
            return 2

def configure_yaml(x1, x2, x3, threeD: bool, res: Res, implicit: bool, tlim):
    # define geometry

    geometry_type = 'cartesian'
    x1min = 0
    x1max = x1
    x2min = 0
    x2max = x2
    x3min = 0
    x3max = x3

    res_multiplier = get_res_multiplier(res)
    nx1 = 64 * res_multiplier
    nx2 = 256 * res_multiplier
    nx3 = 256 * res_multiplier if threeD else 1

    geometry_dict = {'type': geometry_type,
                     'bounds': {'x1min': x1min, 'x1max': x1max,
                                'x2min': x2min, 'x2max': x2max,
                                'x3min': x3min, 'x3max': x3max},
                     'cells':  {'nx1': nx1, 'nx2': nx2, 'nx3': nx3, 'nghost': 3}}

    # define dynamics

    eos_type = 'ideal-gas'
    # see https://descanso.jpl.nasa.gov/propagation/mars/MarsPub_sec4.pdf
    # and https://www.meteor.iastate.edu/classes/mt452/Class_Discussion/Mars-physical_and_orbital_statistics.pdf
    gamma = 1.3
    weight = 43.4E-3

    equation_of_state_dict = {'type': eos_type, 'gammad': gammad, 'weight':weight}

    vertical_projection_dict = {'type': 'temperature', 'pressure-margin': 1E6}
    
    # only explicit course has shock false
    reconstruct_dict = {'vertical'  : {'type' : 'weno5',
                                       'scale': False,
                                       'shock': threeD or implicit or res == Res.FINE},
                        'horizontal': {'type' : 'weno5',
                                       'scale': False,
                                       'shock': threeD or implicit or res == Res.FINE}}

    # explicit fine also uses hllc
    riemann_solver_dict = {'type': 'hllc' if (threeD or (not implicit and res == Res.FINE)) else 'lmars'}

    dynamics_dict = {'equation-of-state': equation_of_state_dict,
                    'vertical-projection': vertical_projection_dict,
                    'reconstruct': reconstruct_dict,
                    'riemann-solver': riemann_solver_dict}

    # define boundary conditions

    boundary_condition_dict = {'internal': {'solid-density': 1E3,
                                            'solid-pressure':1E9,
                                            'max-iter': 5},            
                               'external': {'x1-inner': 'reflecting',
                                            'x1-outer': 'reflecting',
                                            'x2-inner': 'reflecting',
                                            'x2-outer': 'reflecting',
                                            'x3-inner': 'reflecting',
                                            'x3-outer': 'reflecting'}}

    # define integration scheme

    integration_dict = {'type': 'rk3',
                        'cfl': 0.45 if (threeD and not implicit) else 0.9,
                        'implicit-scheme': int(implicit),
                        'nlim': -1,
                        'tlim': tlim,
                        'ncycle_out':1000}

    # define forcing

    forcing_dict = {'const-gravity': {'grav1': -3.73}}

