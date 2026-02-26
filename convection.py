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
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    input_file = generate_yaml(sim_properties, f"{output_dir}/topography", experiment_name)
    print(f"Generated yaml file: {input_file}")
    return input_file


def heat_flux_mask(solid_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Expect input tensor is int8.
    '''
    assert solid_tensor.dtype == torch.int8, "Expected input tensor type as int8."
    # need to use slice (:1) to preserve tensor shape
    # recall 0 index is bottom
    padded = torch.cat((torch.ones_like(solid_tensor[:, :, :1]), solid_tensor), dim=2)
    # padded[ignore top z] - padded[ignore bottom z]
    q_mask = padded[:, :, :-1] - padded[:, :, 1:]
    # make whole top z row = -1
    q_mask[:, :, -1] = -1
    return q_mask


def pad_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Requires input tensor not be boolean.
    '''
    assert input_tensor.dtype != torch.bool, "Padding does not work for boolean type tensors."
    assert input_tensor.ndim == 3, "Function is not defined for non-3D tensors"
    if input_tensor.shape[0] != 1:      # 3D
        # pad in 3D only works for 4D tensor-- add/remove dummy dimension
        # pads last 3 dims of 4D tensor
        temp = input_tensor.unsqueeze(0)
        temp = F.pad(temp, (nghost, nghost, nghost, nghost, nghost, nghost), mode='replicate')
        temp = temp.squeeze(0)
    else:                               # 2D
        # pads last 2 dims of 3D tensor
        temp = F.pad(input_tensor, (nghost, nghost, nghost, nghost), mode='replicate')
    return temp


def shift_terrain_data(cell_data: torch.Tensor) -> Tuple[torch.Tensor, float]:
    '''
    Shift terrain data so lowest point is 0 and return min value
    '''
    min_elevation = torch.min(cell_data)
    # shift data up
    return cell_data - min_elevation, min_elevation


def assign_solid_tensor(cell_data: torch.Tensor, x1f: torch.Tensor) -> torch.Tensor:
    '''
    Cast celled topography data to tensor of 1s and 0s compatible with model geometry.
    Assume x1f is just interior - want f for cell edge so lowest point can be at bottom row
    '''
    nc_lat = cell_data.shape[0]
    nc_long = cell_data.shape[1]
    nc_height = x1f.shape[2]
    # cell_data is [lat, long] and I want [long, lat, height]
    cell_data = torch.transpose(cell_data, 0, 1).reshape(nc_long, nc_lat, 1)
    # repeat in the height dimension by height of x1f
    cell_data = cell_data.repeat(1, 1, nc_height)
    # boolean mask with tensor of vertical heights
    solid_data = cell_data > x1f
    return solid_data


def debug_plot(x1v: torch.Tensor, x2v: torch.Tensor, x3v: torch.Tensor,
               solid_tensor: torch.Tensor, q_mask: torch.Tensor, mars_data: torch.Tensor):
    '''
    Assumes x1v, x2v, and x3v are interior
    '''
    import matplotlib.pyplot as plt
    import pickle
    x1f = x1v - torch.min(x1v)
    # surface is computed using x1f, redefine x1v = x1f since I wrote plotting code with x1v first
    x1v = x1f
    x1v = x1v.cpu().numpy()
    x2v = x2v.cpu().numpy()
    x3v = x3v.cpu().numpy()
    solid_tensor = solid_tensor.cpu().numpy()
    q_mask = q_mask.cpu().numpy()
    mars_data = mars_data.cpu().numpy()

    if solid_tensor.shape[0] != 1:      # 3D
        skip = 3
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', computed_zorder=False)
        # mars_data is converted to [long, lat] in assign_solid_tensor, but not in here
        # mars_data is [lat, long], all the other ones are [long, lat]
        ax.plot_surface(x2v[:, :, 0], x3v[:, :, 0], mars_data, color='k', alpha=0.5, zorder=2)
        ax.scatter(x3v[solid_tensor == 1][::skip], x2v[solid_tensor == 1][::skip], x1v[solid_tensor == 1][::skip], s=0.5, alpha=0.2, c='blue', zorder=1)
        ax.scatter(x3v[q_mask == 1][::skip], x2v[q_mask == 1][::skip], x1v[q_mask == 1][::skip], s=0.5, alpha=0.2, c='orange', zorder=3)
        # ax.scatter(x3v[q_mask == -1], x2v[q_mask == -1], x1v[q_mask == -1], s=1, alpha=0.8, c='lightblue')
        ax.set_xlabel("Longitude [m] -> W")
        ax.invert_xaxis()   # to match what a 2d plot looks like
        ax.set_ylabel("N <- Latitude [m]")
        ax.invert_yaxis()
        ax.view_init(elev=10, azim=65)
        filename = "debug_terrain_plot_3D.png"
        with open('FigureObject.fig.pickle', 'wb') as f:
            pickle.dump(fig, f)
    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x2v[0, :, 0], mars_data[:, 0], color='k')
        ax.scatter(x2v[0, :, :][solid_tensor[0, :, :] == 1], x1v[0, :, :][solid_tensor[0, :, :] == 1], s=0.5, c='blue')
        ax.scatter(x2v[0, :, :][q_mask[0, :, :] == 1], x1v[0, :, :][q_mask[0, :, :] == 1], s=0.5, c='orange')
        ax.set_xlabel("Latitude [m] -> N")
        filename = "debug_terrain_plot_2D.png"
    fig.savefig("output/" + filename, dpi=300, bbox_inches='tight')


def run_with(input_file: str, output_dir: Optional[str] = None, restart_file: Optional[str] = None, mars_data: Optional[torch.Tensor] = None):
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
    # define solid region, pad with ghost zones
    if mars_data is not None:
        mars_data, min_elevation = shift_terrain_data(mars_data)
        # since x1v is stacked from meshgrid, without doing the same, can simply subtract the min value
        x1f = x1v[interior_geom] - torch.min(x1v[interior_geom])
        solid_tensor = assign_solid_tensor(mars_data.to(device), x1f.to(device))
        solid_tensor = solid_tensor.to(device)
        # need to pad tensor here, tensor must be boolean
        block_vars["solid"] = pad_tensor(solid_tensor.char()).bool()
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
    block.set_user_output_func(lambda bvars: call_user_output(bvars, Rd, cp))

    # integration
    print(f"Forcing: {q_dot} W/m^2")
    # solid_tensor is NOT padded
    assert solid_tensor.shape != (nc3, nc2, nc1), "solid_tensor includes ghost zones where it shouldn't"
    # solid_tensor can be either bool or int type depending on which branch of if/else, so convert to int8 (.char())
    q_mask = heat_flux_mask(solid_tensor.char()).to(device)
    if debug and mars_data is not None: debug_plot(x1v[interior_geom], x2v[interior_geom], x3v[interior_geom], solid_tensor, q_mask, mars_data)
    if debug: return

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


def main(args):
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("-t", "--time-limit", required=True, type=int, help="Time limit for integration in seconds.")
    parser.add_argument("--3D", action="store_true", help="Whether to perform a 3D experiment")
    parser.add_argument("-r", "--restart-file", type=str, help="Continue integrating from a file")
    parser.add_argument("-l", "--lat-long-bounds", type=float, nargs=4, help="List of min lat, max lat, min long, max long")
    parser.add_argument("-o", "--output-parent-dir", type=str, default = ".", help="Directory for output files.")
    args = parser.parse_args(args)
    experiment_name = args.experiment_name
    threeD = vars(args)['3D']       # 3D is not a valid python identifier, but it can be used as a dict key
    if threeD:
        # modify experiment name if 3D
        experiment_name = experiment_name + "_3D"
    # determine topographical information
    topography = args.lat_long_bounds is not None
    nx1, nx2, nx3 = get_num_cells_exp(experiment_name)
    if topography:
        # must append lat/long at the end since other functions look at first two chars for I/E and C/F
        experiment_name = experiment_name + "_" + format_lat_long_string(*args.lat_long_bounds)
        # 2nd dim will be lat, 3rd dim will be long
        # mars_data is a numpy array 
        mars_data, Dx2, Dx3 = get_cell_topography(*args.lat_long_bounds, nx2, nx3)
        # Dx1 = Dx2 / nx2 * nx1
        Dx1 = 20E3
        assert Dx1 >= 16E3, "Domain height is not >~ 1.5 Mars scale heights"
        # TODO: need a way to appropriately choose atmospheric height if vertical domain is very large
    else:
        Dx1 = 20E3
        Dx2 = 80E3
        Dx3 = 80E3
    print(f"Size: z: {Dx1:.2f}, y: {Dx2:.2f}, x: {Dx3:.2f} [m]")
    print(f"Res: z: {Dx1/nx1:.2f}, y: {Dx2/nx2:.2f}, x: {Dx3/nx3:.2f} [m/cell]")
    print(f"Experiment name: {experiment_name}")
    sim_properties = Sim_Properties(Dx1, Dx2, Dx3, args.time_limit)
    # determine yaml input file
    input_file = generate_yaml_input_file(sim_properties, experiment_name, args.output_parent_dir)
    output_dir = f"{args.output_parent_dir}/output_{experiment_name}"
    if topography:
        run_with(input_file, output_dir, args.restart_file, torch.from_numpy(mars_data))
    else:
        run_with(input_file, output_dir, args.restart_file)
    

if __name__ == "__main__":
    boot_parser = argparse.ArgumentParser(add_help=False)           # add_help=False avoids conflicts
    boot_parser.add_argument('-d', '--debug', action='store_true')
    boot_args, extras = boot_parser.parse_known_args()

    if boot_args.debug:
        print("--- RUNNING IN DEBUG MODE ---")
        debug = True
        # copied from print statement for test arguments I've been using
        if '--3D' in extras:
            extras = ['-e', 'IC', '-t', '43200', '--3D', '-l', '-31', '-29', '74', '76']
        else:
            extras = ['-e', 'IC', '-t', '43200', '-l', '-35', '-25', '70', '80']
    main(extras)

