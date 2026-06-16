# pip modules
import torch
import kintera
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIV1, kIV2, kIV3, kIPR
import yaml
from amrex import space3d as amr

# local files
from mars import q_dot
from experiment import handle_input
from convection import call_user_output, select_device, shift_terrain_data, assign_solid_tensor, heat_flux_mask, pad_tensor

# default
from typing import Optional

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
debug = False


def calculate_cell_properties_2D(a: int, b: int, coord):
    f = lambda x2: -a * torch.abs(1/b * (torch.remainder(x2, 2*b) - b)) + a
    nc3 = coord.ku() - coord.kl() + 1
    nc2 = coord.ju() - coord.jl() + 1
    nc1 = coord.iu() - coord.il() + 1
    x1f = torch.arange(nc1 + 1)
    x2f = torch.arange(nc2 + 1)
    x3f = torch.arange(nc3 + 1)
    X3f, X2f, X1f = torch.meshgrid(x3f, x2f, x1f, indexing="ij")    # nc3 + 1 x nc2 + 1 x nc1 + 1
    cfa3_frac = torch.ones(nc3 + 1, nc2, nc1)   # nc3 + 1, nc2, nc1
    cfa2_frac = torch.minimum(X1f[:-1, :, 1:] - torch.maximum(f(X2f) - X1f)[:-1, :, :-1], 0)    # nc3 x nc2 + 1 x nc1

    cfa1_x = (X1f - f(X2f)[:, :-1, :]) / (f(X2f)[:, 1:, :] - f(X2f)[:, :-1, :]) + X2f[:, :-1, :]

    x1v = torch.arange(nc1)
    x2v = torch.arange(nc2)
    X2v, X1v = torch.meshgrid(x2v, x1v, indexing="ij")
    intercepts2 = f(x2f)
    areas2 = torch.remainder(intercepts2, 1)
    below_ground = intercepts2
    
    
    
    # return cfa2_frac, cfa1, cvol


def func(coord):
    amr.initialize()

    # Define interior domain
    nx, ny, nz = 64, 64, 16

    rb = amr.RealBox(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0]))
    coord = 0  
    domain = amr.Box(amr.IntVect(0,0,0), amr.IntVect(nx-1, ny-1, nz-1))
    geom = amr.Geometry(domain, rb, 0, False)

    dx, dy, dz = geom.CellSize()
    dV = dx * dy * dz

    # Define geometry and build
    sphere_implicit = amr.EB2.SphereIF(0.3, [0.5, 0.5, 0.5], True)
    gshop = amr.EB2.makeShop(sphere_implicit)
    amr.EB2_Build(gshop, geom, 0, 0)

    # Build distribution maps (forcing a single global patch for extraction)
    ba = amr.BoxArray(domain)
    ba.maxSize(max(nx, ny, nz)) 
    dm = amr.DistributionMapping(ba)

    # Request zero ghost cells (ngrow = 0)
    ngrow = amr.Vector_int([0, 0, 0])
    eb_factory = amr.makeEBFabFactory(geom, ba, dm, ngrow, amr.EBSupport.full)

    vol_frac_mf = eb_factory.getVolFrac()
    area_frac_mf = eb_factory.getAreaFrac() # Index 0=X, 1=Y, 2=Z

    # Loop over the singular main domain tile
    for mfi in vol_frac_mf:
        # mfi.tilebox() restricts the extracted array strictly to the 64x64x16 interior
        interior_box = mfi.tilebox()
        
        # 1. Extract Volumes -> Shape: (64, 64, 16)
        vol_frac_array = vol_frac_mf[mfi].to_array(interior_box)
        interior_volumes = vol_frac_array * dV
    
    


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
        block_vars["solid"] = pad_tensor(solid_tensor.char(), coord.options.nghost()).bool()
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
        with open(input_file, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        Ts = float(config["problem"]["Ts"])
        p0 = float(config["problem"]["Ps"])
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
    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        u = block_vars["hydro_u"]
        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            # indices are rho -> rho, vi -> rho*vi, pr -> e
            last_weight = block.intg.stages[stage].wght2()
            u[interior][kIPR] += last_weight * q_dot * dz_inv * dt * q_mask

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
    run_with(input_file, output_dir, restart_file)


if __name__ == "__main__":
    main()
