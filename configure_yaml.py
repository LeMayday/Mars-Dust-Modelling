# following https://github.com/elijah-mullens/paddle/blob/main/docs/content/notebooks/Tutorial-Straka.ipynb
# pip modules
import yaml

# local files
from mars import grav, gamma, M_bar, species
from experiment import Res, Experiment
from mars_topography import Region, get_cell_topography

# default
from typing import Optional, Tuple
import os
import argparse

reference_state_dict = {'Tref': 0, 'Pref': 1E5}


def domain_bounds(exp: Experiment) -> Tuple[int, int, int]:
    '''
    Returns (Dx1, Dx2, Dx3)
    '''
    region = exp.region
    if region is None:
        return 20E3, 80E3, 80E3
    # 2nd dim will be lat, 3rd dim will be long
    _, nx2, nx3 = exp.get_num_cells()
    _, Dx2, Dx3 = get_cell_topography(region, nx2, nx3)
    Dx1 = 20E3
    return Dx1, Dx2, Dx3


def configure_yaml(exp: Experiment, tlim: int):
    # define geometry
    Dx1, Dx2, Dx3 = domain_bounds(exp)
    geometry_type = 'cartesian'
    x1min = 0
    x1max = Dx1
    x2min = 0
    x2max = Dx2
    x3min = 0
    x3max = Dx3

    nx1, nx2, nx3 = exp.get_num_cells()
    geometry_dict = {'type': geometry_type,
                     'bounds': {'x1min': x1min, 'x1max': x1max,
                                'x2min': x2min, 'x2max': x2max,
                                'x3min': x3min, 'x3max': x3max},
                     'cells':  {'nx1': nx1, 'nx2': nx2, 'nx3': nx3, 'nghost': 3}}

    # define dynamics
    eos_type = 'ideal-gas'
    # see https://descanso.jpl.nasa.gov/propagation/mars/MarsPub_sec4.pdf
    # and https://www.meteor.iastate.edu/classes/mt452/Class_Discussion/Mars-physical_and_orbital_statistics.pdf
    equation_of_state_dict = {'type': eos_type, 'gammad': gamma, 'weight': M_bar, 'limiter': False}
    # equation_of_state_dict = {'type': eos_type, 'density-floor': 1.E-10, 'pressure-floor': 1.E-10, 'limiter': False}

    vertical_projection_dict = {'type': 'temperature', 'pressure-margin': 1.E-6}
    
    # only 2D explicit course has shock false
    fine = exp.res() == Res.FINE
    threeD = exp.is_3D()
    implicit = exp.is_implicit()
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
                        'tlim': tlim,
                        'ncycle_out':1000}

    # define forcing
    forcing_dict = {'const-gravity': {'grav1': -grav}}

    # define outputs
    # generate restart file every 1.5 hours, generate nc file every 20 mins
    outputs_dict = [{'type': 'restart', 'dt': 5400},
                    {'type': 'netcdf', 'variables': ['prim', 'uov'], 'dt': 1200}]

    full_dictionary = {'reference-state': reference_state_dict,
                       'species': species,
                       'geometry': geometry_dict,
                       'dynamics': dynamics_dict,
                       'boundary-condition': boundary_condition_dict,
                       'integration': integration_dict,
                       'forcing': forcing_dict,
                       'outputs': outputs_dict}

    if (region := exp.region) is not None:
        region_dict = {'min-lat': region.min_lat, 'max-lat': region.max_lat,
                       'min-long': region.min_long, 'max-long': region.max_long}
        full_dictionary['region'] = region_dict

    return full_dictionary


def generate_yaml_file(exp: Experiment, tlim: int, prefix: str, output_parent_dir: Optional[str] = None) -> str:
    if output_parent_dir is not None:
        output_dir = f"{output_parent_dir}/output_{exp.name}"
    else:
        output_dir = f"output_{exp.name}"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    file_base = f"{output_dir}/{prefix}"
    # Note: output files are generated with a basename that is the same as the yaml file
    # snapy 1.2.6 meshblock_options.cpp line 19 and netcdf.cpp line 86
    # so yaml files and output nc files should be stored in the same directory for a given experiment
    file_path = f"{file_base}_{exp.name}.yaml"
    full_dictionary = configure_yaml(exp, tlim)
    with open(file_path, "w") as file_handler:
        yaml.dump(full_dictionary, file_handler, sort_keys=False)
    return file_path


def main():     # create new yaml file if necessary
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", required=True, type=str, help="Name of the experiment. First two letters must be implicit/explicit and course/fine")
    parser.add_argument("-t", "--time-limit", required=True, type=int, help="Time limit for integration in seconds.")
    parser.add_argument("-p", "--prefix", required=True, type=str, help="Prefix for yaml filename (no underscores).")
    parser.add_argument("-l", "--lat-long-bounds", type=float, nargs=4, help="List of min lat, max lat, min long, max long")
    parser.add_argument("-o", "--output-parent-dir", type=str, default = ".", help="Directory for output files.")
    args = parser.parse_args()

    region = Region(*args.lat_long_bounds) if args.lat_long_bounds is not None else None
    experiment = Experiment(args.experiment_name, region)
    
    generate_yaml_file(experiment, args.time_limit, args.prefix, args.output_parent_dir)


if __name__ == "__main__":
    main()
