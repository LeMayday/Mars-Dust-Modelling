import torch
import math
import time
# IMPORTANT!! importing nc2pt after snapy causes a seg fault (for some reason)
import nc2pt
from configure_yaml import Sim_Properties, generate_yaml, get_num_cells_exp
from mars import *
import kintera
import snapy
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV1, kIV2, kIV3, kIPR
import os
import argparse
import re
from mars_topography import get_cell_topography

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


# assign some key values
q_dot = s0 / 4      # heat flux
q_dot = q_dot / 2
print(f"Forcing: {q_dot} W/m^2")

# following https://github.com/elijah-mullens/paddle/blob/main/docs/content/notebooks/Tutorial-Straka.ipynb
def call_user_output(bvars, Rd, cp):
    hydro_w = bvars["hydro_w"]
    out = {}
    temp = hydro_w[kIPR] / (Rd * hydro_w[kIDN])
    out["temp"] = temp
    out["theta"] = temp * (p0 / hydro_w[kIPR]).pow(Rd / cp)
    return out


def generate_yaml_input_file(sim_properties: Sim_Properties, experiment_name: str) -> str:
    output_dir = f"output_{experiment_name}"
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    input_file = generate_yaml(sim_properties, f"{output_dir}/topography", experiment_name)
    print(f"Generated yaml file: {input_file}")
    return input_file


def run_with(input_file: str, restart_file: str):
    # set hydrodynamic options
    print(f"Reading input file: {input_file}")
    # this still will set gas variables (weights, etc) from species list in yaml (see snapy equation_of_state.cpp line 66)
    op = MeshBlockOptions.from_yaml(input_file)
    # initialize block
    block = MeshBlock(op)
    if torch.cuda.is_available() and op.layout().backend() == "nccl":
        device = torch.device("cuda:0")
        print("device = ", device)
    else:
        device = torch.device("cpu")
    block.to(device)
    interior = block.part((0, 0, 0))

    # get handles to modules
    coord = block.module("coord")
    thermo = block.module("hydro.eos.thermo")
    eos = block.module("hydro.eos")

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
    if restart_file != '':
        module = torch.jit.load(restart_file)
        for name, data in module.named_buffers():
            block_vars[name] = data.to(device)
    else:
        w = torch.zeros((nvar, nc3, nc2, nc1), device=device)       # initialize primitive variables (density, vx, vy, vz, pressure)
        temp = torch.full_like(x1v, Ts)                             # isothermal condition

        # need to adjust x1v by where geopotential surface is
        w[kIPR] = p0 * torch.exp(-grav * x1v / Rd / Ts)             # isothermal pressure
        w[kIDN] = w[kIPR] / (Rd * temp)                             # ideal gas law

        # random initial velocity
        w[interior][index.ivy] = torch.randn_like(w[interior][index.ivy])

        block_vars["hydro_w"] = w
        block_vars, current_time = block.initialize(block_vars)

    block.set_user_output_func(lambda bvars: call_user_output(bvars, Rd, cp))

    # integration
    block.make_outputs(block_vars, current_time)
    while not block.intg.stop(block.inc.cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for state in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)

        err = block.check_redo(block_vars)
        if err > 0:
            continue    # redo current step
        if err < 0:
            break       # terminate

        current_time += dt
        block.make_outputs(block_vars, current_time)

    block.finalize(block_vars, current_time)


# raise Exception("Oh no!")




# dz for heat flux
bottom_row_height = torch.full((nc3, nc2), coord.buffer("dx1f")[interior[-1]][0])[interior[1:3]]
bottom_row_height = bottom_row_height.to(device)
top_row_height = torch.full((nc3, nc2), coord.buffer("dx1f")[interior[-1]][-1])[interior[1:3]]
top_row_height = top_row_height.to(device)  

    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)
        u = block_vars["hydro_u"]
        last_weight = block.intg.stages[stage].wght2()
        u[interior][index.ipr][:, :, 0] += last_weight * q_dot / bottom_row_height * dt     # heating from the bottom
        u[interior][index.ipr][:, :, -1] -= last_weight * q_dot / top_row_height * dt       # cooling from the top
    
    count += 1
    current_time += dt

print("elapsed time = ", time.time() - start_time)

def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("-t", "--time-limit", required=True, type=int, help="Time limit for integration in seconds.")
    parser.add_argument("--3D", action="store_true", help="Whether to perform a 3D experiment")
    parser.add_argument("-c", "--continue-from", type=str, help="Continue integrating from a file")
    parser.add_argument("-l", "--lat-long-bounds", type=float, nargs=4, help="List of min lat, max lat, min long, max long")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    # 3D is not a valid python identifier, but it can be used as a dict key
    # modify experiment name if 3D
    if vars(args)['3D']:
        experiment_name = experiment_name + "_3D"
    # determine restart file
    restart_file = args.continue_from if args.continue_from is not None else ''
    # determine topographical information
    if args.lat_long_bounds is not None:
        nx1, nx2, nx3 = get_num_cells_exp(experiment_name)
        min_lat, max_lat, min_long, max_long = *args.lat_long_bounds
        # 2nd dim will be lat, 3rd dim will be long
        mars_data, Dx2, Dx3 = get_cell_topography(min_lat, max_lat, min_long, max_long, nx2, nx3)
        Dx1 = Dx2 / nx2 * nx1
        assert Dx1 >= 20E3, "Domain height is not >~ 1.5 Mars scale heights"
    else:
        Dx1 = 20E3
        Dx2 = 80E3
        Dx3 = 80E3
    sim_properties = Sim_Properties(Dx1, Dx2, Dx3, args.time_limit)
    # determine yaml input file
    input_file = generate_yaml_input_file(sim_properties, experiment_name)
    

