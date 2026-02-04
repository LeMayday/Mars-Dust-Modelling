# following https://github.com/elijah-mullens/paddle/blob/main/docs/content/notebooks/Tutorial-Straka.ipynb
from enum import Enum
import yaml
from typing import NamedTuple   # immutable class (like const struct)
from mars import grav

class Res(Enum):
    COURSE = 'course'
    FINE = 'fine'

class Sim_Properties(NamedTuple):
    Dx1: int
    Dx2: int
    Dx3: int
    tlim: int
    reference_state_dict = {'Tref': 0, 'Pref': 1E5}
    species = [{'name': 'dry', 'composition': {'C': 0.95, 'O': 1.90, 'N': 0.06, 'Ar': 0.02}, 'cv_R': 3.4}]

def get_res_multiplier(res: Res):
    assert(res in Res)
    match res:
        case Res.COURSE:
            return 1
        case Res.FINE:
            return 2

def is_implicit(exp_name: str) -> bool:
    return exp_name[0] == 'I'

def get_exp_res(exp_name: str) -> Res:
    match exp_name[1]:
        case 'C':
            return Res.COURSE
        case 'F':
            return Res.FINE

def is_3D(exp_name: str) -> bool:
    return '_3D' in exp_name

def configure_yaml(sim_properties: Sim_Properties, implicit: bool, res: Res, threeD: bool):
    # define geometry

    geometry_type = 'cartesian'
    x1min = 0
    x1max = sim_properties.Dx1
    x2min = 0
    x2max = sim_properties.Dx2
    x3min = 0
    x3max = sim_properties.Dx3

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

    equation_of_state_dict = {'type': eos_type, 'gammad': gamma, 'weight': weight, 'limiter': False}

    vertical_projection_dict = {'type': 'temperature', 'pressure-margin': 1.E-6}
    
    # only 2D explicit course has shock false
    fine = res == Res.FINE
    reconstruct_dict = {'vertical'  : {'type' : 'weno5',
                                       'scale': False,
                                       'shock': threeD or implicit or fine},
                        'horizontal': {'type' : 'weno5',
                                       'scale': False,
                                       'shock': threeD or implicit or fine}}

    # 2D explicit fine also uses hllc
    riemann_solver_dict = {'type': 'hllc' if (threeD or (not implicit and fine)) else 'lmars'}

    dynamics_dict = {'equation-of-state': equation_of_state_dict,
                    'vertical-projection': vertical_projection_dict,
                    'reconstruct': reconstruct_dict,
                    'riemann-solver': riemann_solver_dict}

    # define boundary conditions

    boundary_condition_dict = {'internal': {'solid-density': 1.E3,
                                            'solid-pressure':1.E9,
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
                        'tlim': sim_properties.tlim,
                        'ncycle_out':1000}

    # define forcing

    forcing_dict = {'const-gravity': {'grav1': -grav}}

    # define outputs

    # generate restart file every hour, generate nc file every 10 mins
    outputs_dict = [{'type': 'restart', 'dt': 3600},
                    {'type': 'netcdf', 'variables': ['prim', 'uov'], 'dt': 600}]

    full_dictionary = {'reference-state': Sim_Properties.reference_state_dict,
                       'species': Sim_Properties.species,
                       'geometry': geometry_dict,
                       'dynamics': dynamics_dict,
                       'boundary-condition': boundary_condition_dict,
                       'integration': integration_dict,
                       'forcing': forcing_dict,
                       'outputs': outputs_dict}
    return full_dictionary

def generate_yaml(sim_properties: Sim_Properties, file_base: str, exp_name: str) -> str:
    # Note: output files are generated with a basename that is the same as the yaml file
    # snapy 1.2.6 meshblock_options.cpp line 19 and netcdf.cpp line 86
    # so yaml files and output nc files should be stored in the same directory for a given experiment
    file_path = f"{file_base}_{exp_name}.yaml"
    with open(file_path, "w") as file_handler:
        full_dictionary = configure_yaml(sim_properties, is_implicit(exp_name), get_exp_res(exp_name), is_3D(exp_name))
        yaml.dump(full_dictionary, file_handler, sort_keys=False)
    return file_path

