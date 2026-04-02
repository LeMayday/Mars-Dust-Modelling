# pip modules
import torch
import torch.nn.functional as F
import kintera
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV1, kIV2, kIV3, kIPR
import yaml

# local files
from mars import q_dot
from mars_topography import get_mars_data_from_yaml_config
from experiment import handle_input
from convection import call_user_output, select_device, shift_terrain_data, assign_solid_tensor, heat_flux_mask, pad_tensor

# default
from typing import Optional

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


def run_with(input_file: str, output_dir: Optional[str] = None, restart_file: Optional[str] = None, mars_data: Optional[torch.Tensor] = None):
    print(f"Reading input file: {input_file}")
    # this still will set gas variables (weights, etc) from species list in yaml (see snapy equation_of_state.cpp line 66)
    op = MeshBlockOptions.from_yaml(input_file)
    print(f"Setting output directory: {output_dir}")
    op.output_dir(output_dir)
    block = MeshBlock(op)
    device = select_device(block, op)
    print("device = ", device)
    block.to(device)
    interior = block.part((0, 0, 0))
    # the first slice in interior is for the variables
    interior_geom = interior[1:]

    coord = block.module("coord")
    eos = block.module("hydro.eos")

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )   # x3v is x, x2v is y, x1v is z
    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    nvar = eos.nvar()

    Rd = kintera.constants.Rgas / kintera.species_weights()[0]
    cv = kintera.species_cref_R()[0] * Rd
    cp = cv + Rd

    block_vars = {}
    # define solid region, pad with ghost zones
    if mars_data is not None:
        mars_data, min_elevation = shift_terrain_data(mars_data)
        # since x1v is stacked from meshgrid, without doing the same, can simply subtract the min value
        x1f = x1v[interior_geom] - torch.min(x1v[interior_geom])
        solid_tensor = assign_solid_tensor(mars_data.to(device), x1f.to(device))
        solid_tensor = solid_tensor.to(device)
        # need to pad tensor here, tensor must be boolean
        # block_vars["solid"] = pad_tensor(solid_tensor.char()).bool()
    else:
        # no topography
        solid_tensor = torch.zeros_like(x1v[interior_geom]).to(device)
        min_elevation = 0

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
        with open(input, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        Ts = float(config["problem"]["Ts"]),
        p0 = float(config["problem"]["Ps"]),
        grav = -float(config["forcing"]["const-gravity"]["grav1"])

        w = torch.zeros((nvar, nc3, nc2, nc1), device=device)                   # initialize primitive variables (density, vx, vy, vz, pressure)
        temp = torch.full_like(x1v, Ts)                                         # isothermal condition
        # need to adjust x1v by where geopotential surface is
        print(f"Reference P: {p0} Pa, Reference T: {Ts}")
        w[kIPR] = p0 * torch.exp(-grav * (x1v + min_elevation) / (Rd * Ts))     # isothermal pressure
        w[kIDN] = w[kIPR] / (Rd * temp)                                         # ideal gas law

        # random initial velocity
        w[interior][kIV2] = torch.randn_like(w[interior][kIV2])

        block_vars["hydro_w"] = w
        block_vars, current_time = block.initialize(block_vars)

    # configure output
    block.set_user_output_func(lambda bvars: call_user_output(bvars, p0, Rd, cp))

    # integration
    print(f"Forcing: {q_dot} W/m^2")
    # solid_tensor is NOT padded
    assert solid_tensor.shape != (nc3, nc2, nc1), "solid_tensor includes ghost zones where it shouldn't"
    # solid_tensor can be either bool or int type depending on which branch of if/else, so convert to int8 (.char())
    q_mask = heat_flux_mask(solid_tensor.char()).to(device)

    dz_inv = 1 / coord.buffer("dx1f")[0]
    block.make_outputs(block_vars, current_time)
    padded_solid_tensor = pad_tensor(solid_tensor.char()).bool()
    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        u = block_vars["hydro_u"]
        w = block_vars["hydro_w"]
        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            # indices are rho -> rho, vi -> rho*vi, pr -> e
            last_weight = block.intg.stages[stage].wght2()
            u[interior][kIPR] += last_weight * q_dot * dz_inv * dt * q_mask
            u[kIV1] -= last_weight * w[kIV1] * 0.1 * w[kIDN] * padded_solid_tensor    # assume tau = 10*dt
            u[kIV2] -= last_weight * w[kIV2] * 0.1 * w[kIDN] * padded_solid_tensor
            u[kIV3] -= last_weight * w[kIV3] * 0.1 * w[kIDN] * padded_solid_tensor

        err = block.check_redo(block_vars)
        if err > 0:
            continue    # redo current step
        if err < 0:
            break       # terminate

        current_time += dt
        block.make_outputs(block_vars, current_time)

    block.finalize(block_vars, current_time)


def main():
    input_file, output_dir, restart_file = handle_input()
    mars_data = get_mars_data_from_yaml_config(input_file)
    if mars_data is None:
        run_with(input_file, output_dir, restart_file)
    else:
        run_with(input_file, output_dir, restart_file, torch.from_numpy(mars_data))


if __name__ == "__main__":
    main()
