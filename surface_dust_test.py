# pip modules
import torch
import kintera
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV1, kIV2, kIV3, kIPR
import yaml

# local files
from mars import q_dot
from saltation import surface_dust_supply
from experiment import handle_input
from convection import select_device, call_user_output

# default
from typing import Optional

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


def run_with(input_file: str, output_dir: Optional[str] = None, restart_file: Optional[str] = None):
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
    surf = block.module("surface")

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
        w[kIPR] = p0 * torch.exp(-grav * x1v / (Rd * Ts))     # isothermal pressure
        w[kIDN] = w[kIPR] / (Rd * temp)                                         # ideal gas law

        # random initial velocity
        w[interior][kIV2] = torch.full_like(w[interior][kIV2], 50)

        block_vars["hydro_w"] = w

        diameters = torch.tensor(surf.options.diameters())
        source_area_density = 1E2       # kg / m^2
        dx3: torch.Tensor = coord.buffer("dx3f")[0]
        dx2: torch.Tensor = coord.buffer("dx2f")[0]
        source_area_density *= dx3 * dx2    # kg (note fluxes are computed as kg/s)
        source_area_density = source_area_density.cpu().numpy() # convert tensor of scalar value to numpy
        l = 3
        rho_p = 2650                    # kg / m^3
        bucket_densities = torch.tensor(surface_dust_supply(diameters.numpy(), rho_p, source_area_density, l))  # essentially kg/cell not kg/m^2
        block_vars["surface_r"] = bucket_densities.view(surf.nbins(), 1, 1).expand(surf.nbins(), nc3, nc2).to(device)

        block_vars, current_time = block.initialize(block_vars)

    # configure output
    block.set_user_output_func(lambda bvars: call_user_output(bvars, p0, Rd, cp))

    # integration
    print(f"Forcing: {q_dot} W/m^2")

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


def main():
    run_with(*handle_input())
    

if __name__ == "__main__":
    main()
