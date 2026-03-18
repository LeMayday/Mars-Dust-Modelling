import torch
import torch.nn.functional as F
from configure_yaml import Sim_Properties, generate_yaml, get_num_cells_exp, nghost
from mars import *
import kintera
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV1, kIV2, kIV3, kIPR
import os
import argparse
from mars_topography import get_cell_topography, format_lat_long_string
from typing import Tuple, Optional
from saltation import surface_dust_supply

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
debug = False

# following https://github.com/elijah-mullens/paddle/blob/main/docs/content/notebooks/Tutorial-Straka.ipynb
def call_user_output(bvars, Rd, cp):
    hydro_w = bvars["hydro_w"]
    out = {}
    temp = hydro_w[kIPR] / (Rd * hydro_w[kIDN])
    # user defined variables: temp and potential temp
    out["temp"] = temp
    out["theta"] = temp * (p0 / hydro_w[kIPR]).pow(Rd / cp)
    return out


def generate_yaml_input_file(sim_properties: Sim_Properties, experiment_name: str, output_parent_dir: Optional[str] = None) -> str:
    if output_parent_dir is not None:
        output_dir = f"{output_parent_dir}/output_{experiment_name}"
    else:
        output_dir = f"output_{experiment_name}"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    input_file = generate_yaml(sim_properties, f"{output_dir}/surf_dust", experiment_name)
    print(f"Generated yaml file: {input_file}")
    return input_file


def run_with(input_file: str, output_dir: Optional[str] = None, restart_file: Optional[str] = None):
    # set hydrodynamic options
    print(f"Reading input file: {input_file}")
    # this still will set gas variables (weights, etc) from species list in yaml (see snapy equation_of_state.cpp line 66)
    op = MeshBlockOptions.from_yaml(input_file)
    print(f"Setting output directory: {output_dir}")
    op.output_dir(output_dir)
    # initialize block
    block = MeshBlock(op)
    if torch.cuda.is_available(): # and op.layout().backend() == "nccl":
        print("Attempting to use GPU")
        device = torch.device("cuda:0")
        print("device = ", device)
    else:
        print("Using CPU")
        device = torch.device("cpu")
    block.to(device)
    interior = block.part((0, 0, 0))
    # the first slice in interior is for the variables
    interior_geom = interior[1:]

    # get handles to modules
    coord = block.module("coord")
    # thermo = block.module("hydro.eos.thermo")
    eos = block.module("hydro.eos")
    surf = block.module("surface")

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )   # x3v is x, x2v is y, x1v is z
    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    nvar = 5

    Rd = kintera.constants.Rgas / kintera.species_weights()[0]
    cv = kintera.species_cref_R()[0] * Rd
    cp = cv + Rd

    block_vars = {}

    # determine how to initialize variables
    if restart_file is not None:
        print(f"Using restart file: {restart_file}")
        # current_time is the simulation time of the restart file
        block_vars, current_time = block.initialize_from_restart(restart_file)
        for key, val in block_vars.items():
            block_vars[key] = val.to(device)
        if "final" in restart_file:     # if restarting from the final file, extend time limit
            block.options.intg().tlim(current_time + block.options.intg().tlim())
    else:
        print("Initializing block variables.")
        # data is stored [x, y, z] so z is adjacent in memory, sometimes x is 1 (if 2D)
        w = torch.zeros((nvar, nc3, nc2, nc1), device=device)                   # initialize primitive variables (density, vx, vy, vz, pressure)
        temp = torch.full_like(x1v, Ts)                                         # isothermal condition

        # need to adjust x1v by where geopotential surface is
        print(f"Reference P: {p0} Pa, Reference T: {Ts}")
        w[kIPR] = p0 * torch.exp(-grav * x1v / (Rd * Ts))     # isothermal pressure
        w[kIDN] = w[kIPR] / (Rd * temp)                                         # ideal gas law

        # random initial velocity
        w[interior][kIV2] = torch.full_like(w[interior][kIV2], 50)

        block_vars["hydro_w"] = w

        diameters = torch.tensor(surf.options.diameters())
        print(surf.nbins())
        print(diameters)
        source_area_density = 1E2       # kg / m^2
        dx3 = coord.buffer("dx3f")[0]
        dx2 = coord.buffer("dx2f")[0]
        source_area_density *= dx3 * dx2    # kg
        l = 3
        rho_p = 2650                    # kg / m^3
        bucket_densities = torch.tensor(surface_dust_supply(diameters.numpy(), rho_p, source_area_density, l))
        block_vars["surface_r"] = bucket_densities.view(surf.nbins(), 1, 1).expand(surf.nbins(), nc3, nc2)

        block_vars, current_time = block.initialize(block_vars)

    # configure output
    block.set_user_output_func(lambda bvars: call_user_output(bvars, Rd, cp))

    # integration
    print(f"Forcing: {q_dot} W/m^2")
    if debug: return

    dz_inv = 1 / coord.buffer("dx1f")[0]
    block.make_outputs(block_vars, current_time)
    times_to_deplete = torch.zeros_like(block_vars["surface_r"])
    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        u = block_vars["hydro_u"]
        surface_u = block_vars["surface_s"]
        times_to_deplete[surface_u > 0] += dt
        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            # indices are rho -> rho, vi -> rho*vi, pr -> e
            # last_weight = block.intg.stages[stage].wght2()
            # u[interior][kIPR] += last_weight * q_dot * dz_inv * dt * q_mask

        err = block.check_redo(block_vars)
        if err > 0:
            continue    # redo current step
        if err < 0:
            break       # terminate

        current_time += dt
        block.make_outputs(block_vars, current_time)

    block.finalize(block_vars, current_time)
    torch.save(times_to_deplete, "surf_dust_test_deplete_times.pt")


def main(args):
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("-t", "--time-limit", required=True, type=int, help="Time limit for integration in seconds.")
    parser.add_argument("--3D", action="store_true", help="Whether to perform a 3D experiment")
    parser.add_argument("-r", "--restart-file", type=str, help="Continue integrating from a file")
    parser.add_argument("-o", "--output-parent-dir", type=str, default = ".", help="Directory for output files.")
    parser.add_argument("-i", "--input-file", type=str, help="Input file")
    args = parser.parse_args(args)
    experiment_name = args.experiment_name
    threeD = vars(args)['3D']       # 3D is not a valid python identifier, but it can be used as a dict key
    if threeD:
        # modify experiment name if 3D
        experiment_name = experiment_name + "_3D"
    nx1, nx2, nx3 = get_num_cells_exp(experiment_name)
    Dx1 = 20E3
    Dx2 = 80E3
    Dx3 = 80E3
    print(f"Size: z: {Dx1:.2f}, y: {Dx2:.2f}, x: {Dx3:.2f} [m]")
    print(f"Res: z: {Dx1/nx1:.2f}, y: {Dx2/nx2:.2f}, x: {Dx3/nx3:.2f} [m/cell]")
    print(f"Experiment name: {experiment_name}")
    sim_properties = Sim_Properties(Dx1, Dx2, Dx3, args.time_limit)
    output_dir = f"{args.output_parent_dir}/output_{experiment_name}"
    # determine yaml input file
    if args.input_file is None:
        input_file = generate_yaml_input_file(sim_properties, experiment_name, args.output_parent_dir)
    else:
        input_file = f"{output_dir}/{args.input_file}"
    run_with(input_file, output_dir, args.restart_file)
    

if __name__ == "__main__":
    boot_parser = argparse.ArgumentParser(add_help=False)           # add_help=False avoids conflicts
    boot_parser.add_argument('-d', '--debug', action='store_true')
    boot_args, extras = boot_parser.parse_known_args()

    # if boot_args.debug:
    #     print("--- RUNNING IN DEBUG MODE ---")
    #     debug = True
    #     # copied from print statement for test arguments I've been using
    #     if '--3D' in extras:
    #         extras = ['-e', 'IC', '-t', '43200', '--3D', '-l', '-31', '-29', '74', '76']
    #     else:
    #         extras = ['-e', 'IC', '-t', '43200', '-l', '-35', '-25', '70', '80']
    main(extras)
