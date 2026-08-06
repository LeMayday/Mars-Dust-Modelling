# pip modules
import torch
import kintera
from snapy import MeshBlockOptions, MeshBlock, Cartesian
from snapy import kIDN, kIV1, kIV2, kIV3, kIPR
import yaml
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# local files
from mars import q_dot
from experiment import handle_input
from convection import call_user_output, select_device, shift_terrain_data, assign_solid_tensor, heat_flux_mask, pad_tensor

# default
from typing import Optional

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
debug = False


def add_edge_crossings_across_z(xpts: torch.Tensor, f_at_xpts: torch.Tensor, x1f: torch.Tensor):
    X1f, F_at_xpts = torch.meshgrid(x1f, f_at_xpts, indexing="ij")
    _, Xpts = torch.meshgrid(x1f, xpts, indexing="ij")
    F_dist_from_X1f = F_at_xpts - X1f
    F0 = F_dist_from_X1f[:, :-1]
    F1 = F_dist_from_X1f[:, 1:]
    where_crossings = F1 * F0 < 0
    X_intercepts = Xpts[:, :-1] + (Xpts[:, 1:] - Xpts[:, :-1]) / (F1 - F0) * (X1f[:, :-1] - )
    indices_right = []
    vals_to_insert = []
    for idx, f in enumerate(f_at_xpts[:-1]):
        if 


# assume F = f(X,Y)

def add_edge_crossings_across_dim(xgridpts: torch.Tensor, xpts: torch.Tensor, F: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim in [2, 3]
    sl_none = [None for _ in range(5)]
    sl_all = [slice(None) for _ in range(5)]

    slgrid = list(sl_none); slgrid[3 - dim] = slice(None)
    slpts = list(sl_none);  slpts[6 - dim] = slice(None)
    Xif = xgridpts[tuple(slgrid)]                                               # (nx+1, 1, 1, 1, 1) or (1, nx+1, 1, 1, 1)
    Xipts = xpts[tuple(slpts)]                                                  # (1, 1, 1, npts, 1) or (1, 1, 1, 1, npts)

    Xipts_dist_from_Xif = Xipts - Xif                                           # uses broadcasting to create mesh of pts - gridlines
    sl_x0 = list(sl_all); sl_x0[6 - dim] = slice(None, -1)
    sl_x1 = list(sl_all); sl_x1[6 - dim] = slice(1, None)
    Xi_dist0 = Xipts_dist_from_Xif[tuple(sl_x0)]
    Xi_dist1 = Xipts_dist_from_Xif[tuple(sl_x1)]

    where_crossings = torch.nonzero(Xi_dist0 * Xi_dist1 < 0, as_tuple=True)     # a crossing only occurs if x_left and x_right are on opposite sides of a gridline
    grid_indices = where_crossings[3 - dim]                                     # the gridlines where a crossing occurs
    pt_indices = where_crossings[6 - dim]                                       # the xpts preceding where a crossing occurs

    xpts_0 = xpts[pt_indices]
    xpts_1 = xpts[pt_indices + 1]
    xgridpts_crossed = xgridpts[grid_indices]

    sl_f0 = list(sl_all); sl_f0[6 - dim] = pt_indices;      sl_f0 = sl_f0[3:]   
    sl_f1 = list(sl_all); sl_f1[6 - dim] = pt_indices + 1;  sl_f1 = sl_f1[3:]
    F0 = F[tuple(sl_f0)]
    F1 = F[tuple(sl_f1)]
    t = (xgridpts_crossed - xpts_0) / (xpts_1 - xpts_0)                         # linear interpolation factor
    t = t.unsqueeze(2 - dim)                                                    # -1 if dim=3, 0 if dim=2

    F_new = F0 + t * (F1 - F0)                                                  # interpolate values

    xpts_w_grid = torch.cat([xpts, xgridpts_crossed])
    F_at_xpts_w_grid = torch.cat([F, F_new], dim=(3-dim))

    sorted_indices = torch.argsort(xpts_w_grid)                                 # sort values such that x is increasing
    xpts_w_grid = xpts_w_grid[sorted_indices]
    sl = list(sl_all); sl[6 - dim] = sorted_indices; sl = sl[3:]
    F_at_xpts_w_grid = F_at_xpts_w_grid[tuple(sl)]
    return xpts_w_grid, F_at_xpts_w_grid




def find_edge_crossings(x3pts: torch.Tensor, x2pts: torch.Tensor, F: torch.Tensor, coord: Cartesian):
    nc3 = coord.ku() - coord.kl() + 1
    nc2 = coord.ju() - coord.jl() + 1
    nc1 = coord.iu() - coord.il() + 1
    res = 2
    x1f = torch.arange(nc1 + 1)
    x2f = torch.arange(nc2 + 1)
    x3f = torch.arange(nc3 + 1)

    X3f = x3f[:, None, None, None, None, None]
    X2f = x2f[None, :, None, None, None, None]
    X1f = x1f[None, None, :, None, None, None]

    x3pts = torch.linspace(0, 1, res + 1)
    x2pts = torch.linspace(0, 1, res + 1)
    x1pts = torch.linspace(0, 1, res + 1)

    cell_X3 = X3f + x3pts[None, None, None, :, None, None]
    cell_X2 = X2f + x2pts[None, None, None, None, :, None]
    cell_X1 = X1f + x1pts[None, None, None, None, None, :]
    F_in_cell = F(cell_X3, cell_X2, cell_X1)

    cell_volumes = torch.empty(nc3, nc2, nc1)
    cell_areas_x3x1 = torch.empty(nc3, nc2 + 1, nc1)
    cell_areas_x2x1 = torch.empty(nc3 + 1, nc2, nc1)
    cell_areas_x2x3 = torch.empty(nc3, nc2, nc1 + 1)
    

    F_bl = F_in_cell[:-1, :, :-1, :-1, :, :-1]
    F_br = F_in_cell[1:,  :, :-1,  1:, :, :-1]
    F_ur = F_in_cell[1:,  :,  1:,  1:, :,  1:]
    F_ul = F_in_cell[:-1, :,  1:, :-1, :,  1:]

    X3_bl = cell_X3[:-1, :, :-1, :-1, :, :-1]; X1_bl = cell_X1[:-1, :, :-1, :-1, :, :-1]
    X3_br = cell_X3[1:,  :, :-1,  1:, :, :-1]; X1_br = cell_X1[1:,  :, :-1,  1:, :, :-1]
    X3_ur = cell_X3[1:,  :,  1:,  1:, :,  1:]; X1_ur = cell_X1[1:,  :,  1:,  1:, :,  1:]
    X3_ul = cell_X3[:-1, :,  1:, :-1, :,  1:]; X1_ul = cell_X1[:-1, :,  1:, :-1, :,  1:]

    F_corners = torch.stack([F_bl, F_br, F_ur, F_ul], dim=0)
    X3_corners = torch.stack([X3_bl, X3_br, X3_ur, X3_ul], dim=0)
    X1_corners = torch.stack([X1_bl, X1_br, X1_ur, X1_ul], dim=0)

    # Shifting by +1 to find pairs for edge-crossings (i + 1)%4
    F_next = torch.roll(F_corners, shifts=-1, dims=0)
    X3_next = torch.roll(X3_corners, shifts=-1, dims=0)
    X1_next = torch.roll(X1_corners, shifts=-1, dims=0)

    # 3. Compute linear intercepts along the 4 edges
    denom = F_next - F_corners
    t = -F_corners / torch.where(denom == 0, 1e-9, denom)
    X3_intercepts = X3_corners + t * (X3_next - X3_corners)
    X1_intercepts = X1_corners + t * (X1_next - X1_corners)

    # 4. Gather all potential fluid vertices (4 corners + 4 edge intercepts) Interleaved!
    # To keep a valid loop order, we alternate: Corner 0, Edge 0->1, Corner 1, Edge 1->2...
    all_X3 = torch.stack([
        X3_corners[0], X3_intercepts[0],
        X3_corners[1], X3_intercepts[1],
        X3_corners[2], X3_intercepts[2],
        X3_corners[3], X3_intercepts[3]
    ], dim=0) # Shape: (8, nc3, nc2+1, nc1)

    all_X1 = torch.stack([
        X1_corners[0], X1_intercepts[0],
        X1_corners[1], X1_intercepts[1],
        X1_corners[2], X1_intercepts[2],
        X1_corners[3], X1_intercepts[3]
    ], dim=0)

    # 5. Build the Boolean Mask matching your pseudocode conditions
    # Condition A: Corner is fluid (F < 0 since <0 is fluid)
    mask_corners = (F_corners > 0) 
    # Condition B: Edge is crossed (t must lie securely within the segment bounds [0, 1])
    mask_edges = (F_corners * F_next < 0)

    # Interleave the masks to match the 8-point vertex structure
    valid_mask = torch.stack([
        mask_corners[0], mask_edges[0],
        mask_corners[1], mask_edges[1],
        mask_corners[2], mask_edges[2],
        mask_corners[3], mask_edges[3]
    ], dim=0) # Shape: (8, nc3, nc2+1, nc1)

    # 6. Corrected Vectorized Shoelace via Forward-Filling
    # We reshape to make sequential operations easier over the vertex dimension (dim=0)

    # Ensure the very first vertex is valid to anchor our forward-fill.
    # If nothing is valid (all solid), the mask will be all False, which we handle later.
    X3_valid = all_X3.clone()
    X1_valid = all_X1.clone()

    # Forward-fill: If a vertex is invalid, replace it with the coordinates of the previous one
    for v in range(1, 8):
        X3_valid[v] = torch.where(valid_mask[v], all_X3[v], X3_valid[v-1])
        X1_valid[v] = torch.where(valid_mask[v], all_X1[v], X1_valid[v-1])

    # Handle the cyclic boundary condition: the loop closes back to vertex 0.
    # If vertex 0 itself was invalid, we cyclic-fill it from the last valid element.
    for v in reversed(range(7)):
        X3_valid[v] = torch.where(valid_mask[v], X3_valid[v], X3_valid[v+1])
        X1_valid[v] = torch.where(valid_mask[v], X1_valid[v], X1_valid[v+1])

    # Roll the arrays to find the adjacent closed-loop pairs
    X3_next = torch.roll(X3_valid, shifts=-1, dims=0)
    X1_next = torch.roll(X1_valid, shifts=-1, dims=0)

    # Evaluate the cross product sum
    cross_product_sum = torch.sum(X3_valid * X1_next - X3_next * X1_valid, dim=0)

    # Micro-face areas matrix (shape: nc3, nc2+1, nc1, res, res)
    micro_areas = 0.5 * torch.abs(cross_product_sum)

    # --- 7. FINAL INTEGRATION STAGE ---
    # Sum across the trailing subgrid dimensions to get total open area per main cell face
    cell_areas_x3x1 = torch.sum(micro_areas, dim=(-2, -1)) # Final Shape: (nc3, nc2+1, nc1)


    cell_X3_0 = cell_X3[..., :-1, :, :]
    cell_X3_1 = cell_X3[..., 1:, :, :]

    t_X3 = -F_in_cell[..., :-1, :, :] / (F_in_cell[..., 1:, :, :] - F_in_cell[..., :-1, :, :] + 1e-9)              # (1, 1, nx3+1, x3pts-1, n2pts)
    X3_intercept = cell_X3_0 + t_X3 * (cell_X3_1 - cell_X3_0)                                       # (1, 1, nx3+1, num_intercepts, :)
    target_shape = X3_intercept.shape
    bound_min = cell_X3_0.expand(target_shape)
    bound_max = cell_X3_1.expand(target_shape)
    cell_X3_intercepts = torch.clamp(X3_intercept, min=bound_min, max=bound_max)


    cell_X1_0 = cell_X1[..., :, :, :-1]
    cell_X1_1 = cell_X1[..., :, :, 1:]
    t_X1 = -F_in_cell[..., :, :, :-1] / (F_in_cell[..., :, :, 1:] - F_in_cell[..., :, :, :-1] + 1e-9)
    X1_intercept = cell_X1_0 + t_X1 * (cell_X1_1 - cell_X1_0)
    target_shape = X1_intercept.shape
    bound_min = cell_X1_0.expand(target_shape)
    bound_max = cell_X1_1.expand(target_shape)
    cell_X1_intercepts = torch.clamp(X1_intercept, min=bound_min, max=bound_max)

    






    x3_i0 = X3_intercept[:, :, :-1, :, :, :]
    x3_i1 = X3_intercept[:, :, 1:, :, :, :]

    x3_min = torch.minimum(x3_i0, x3_i1)
    x3_max = torch.maximum(x3_i0, x3_i1)

    target_shape = x3_min.shape  # (1, 1, nc1, P3-1, P2)
    bound_min = cell_X3_0.expand(target_shape)
    bound_max = cell_X3_1.expand(target_shape)

    first_pt = torch.clamp(x3_min, min=bound_min, max=bound_max)
    second_pt = torch.clamp(x3_max, min=bound_min, max=bound_max)
    h0 = torch.clamp(F_in_cell[:, :, :-1, :-1, :, :], 0, 1)
    h1 = torch.clamp(F_in_cell[:, :, :-1, 1:, :, :], 0, 1)
    Area_under_curve_X3 = (first_pt - cell_X3_0) * h0 + 0.5 * (second_pt - first_pt) * (h0 + h1) + (cell_X3_1 - second_pt) * h1

    



    Area_under_curve_X3 = 0.5 * (cell_X3[..., 1:, :] - cell_X3[..., :-1, :]) * (F_in_cell[..., 1:, :] + F_in_cell[..., :-1, :])       # (1, 1, 1, x3pts-1, n2pts) careful! this is all X3, not gridpoints
    where_crossing = F_in_cell[..., :-1, :] * F_in_cell[..., 1:, :] < 0


def func(xpts: torch.Tensor, f_at_xpts: torch.Tensor, coord: Cartesian):
    '''
    Requires x_pos and f_at_xpos to be same size.
    Assumes xpos is strictly increasing.
    '''
    nc3 = coord.ku() - coord.kl() + 1
    nc2 = coord.ju() - coord.jl() + 1
    nc1 = coord.iu() - coord.il() + 1
    x1f = torch.arange(nc1 + 1)
    x2f = torch.arange(nc2 + 1)
    x3f = torch.arange(nc3 + 1)

    idx = torch.searchsorted(xpts, x2f, side='right')       # same length as x2f
    f_at_cell_walls = f_at_xpts[idx - 1] + (f_at_xpts[idx] - f_at_xpts[idx - 1]) / (xpts[idx] - xpts[idx - 1]) * (x2f - xpts[idx - 1])      # y-intercepts at cell borders

    # for the next part, I want to guarantee the x-grid pts appear in xpts

    # determine which values do not overlap
    where_no_repeat = (x2f - xpts[idx - 1]) != 0
    idx_no_repeat = idx[where_no_repeat]

    # allocate new arrays that will be the union of the sets
    full_len = len(xpts) + len(idx_no_repeat)
    xpts_w_j = torch.empty(full_len)
    f_at_xpts_w_j = torch.empty(full_len)

    # locations where to insert -- takes care of repeated indices
    idx_to_insert = torch.arange(len(idx_no_repeat)) + idx_no_repeat
    insert_mask = torch.zeros(full_len, dtype=torch.bool)
    insert_mask[idx_to_insert] = True

    xpts_w_j[insert_mask] = x2f[where_no_repeat]
    xpts_w_j[~insert_mask] = xpts                                       # now contains union of xpts and x2f
    f_at_xpts_w_j[insert_mask] = f_at_cell_walls[where_no_repeat]
    f_at_xpts_w_j[~insert_mask] = f_at_xpts                             # now contains union of f_at_xpts and f_at_cell_walls




def calculate_cell_properties_2D(a: int, b: int, coord: Cartesian):
    f = lambda x2: -a * torch.abs(1/b * (torch.remainder(x2, 2*b) - b)) + a
    nc3 = coord.ku() - coord.kl() + 1
    nc2 = coord.ju() - coord.jl() + 1
    nc1 = coord.iu() - coord.il() + 1
    x1f = torch.arange(nc1 + 1)
    x2f = torch.arange(nc2 + 1)
    x3f = torch.arange(nc3 + 1)
    X3f, X2f, X1f = torch.meshgrid(x3f, x2f, x1f, indexing="ij")    # nc3 + 1 x nc2 + 1 x nc1 + 1
    cfa3_frac = torch.ones(nc3 + 1, nc2, nc1)   # nc3 + 1, nc2, nc1
    f_X2f = f(X2f)
    ilj = X2f[:-1, :-1, :]
    irj = X2f[:-1, 1:, :]
    ij = X2f[:-1, :, :-1]
    jl = X1f[:-1, :, :-1]
    jr = X1f[:-1, :, 1:]
    cfa2_frac = torch.minimum(jr - torch.maximum(f_X2f[:-1, :, :-1], jl), 0)    # nc3 x nc2 + 1 x nc1

    cfa1_frac = torch.ones(nc3, nc2, nc1 + 1)
    cells_where_intersect_left = 0 < cfa2_frac[:, :-1, :] < 1
    cells_where_intersect_right = 0 < cfa2_frac[:, 1:, :] < 1
    cfa1_frac[:, :, :-1][torch.logical_and(cells_where_intersect_left, cells_where_intersect_right)] = 0
    slope_sign = f_X2f[:-1, 1:, :-1] - f_X2f[:-1, :-1, :-1] # cell f(i+1) - f(i)
    

    g = lambda x1: b/a * x1
    
    cfa1_xintercept = X1f[:-1, :-1, ] - f(X2f[:-1, :-1, :]) / ()
    cfa1_frac_first_left = torch.minimum(torch.maximum(g(X1f)- X2f)[:-1, :-1, :] - X2f[:-1, 1:, :], 0)
    cvol_frac = torch.ones(nc3, nc2, nc1)
    for i in range(nc2):
        place_slice = (slice(None), slice(i, min(i+b, nc2-i)), slice(None))
        ref_slice = (slice(None), slice(0, min(b, nc2-i)), slice(None))
        if i % 2*b == 0:
            F = lambda x2: 0.5 * a/b * x2**2 
            cfa1_frac[place_slice] = cfa1_frac_first_left[ref_slice]
            top_fa = cfa1_frac[:, 0:min(b, nc2-i), 1:]
            bot_fa = cfa1_frac[:, 0:min(b, nc2-i), :-1]
            cvol_frac[place_slice] = top_fa + X1f[:, 0:min(b, nc2-i), :-1] * (top_fa - bot_fa) - F(X2f[ref_slice] + top_fa) + F(X2f[ref_slice] + bot_fa)
        elif i % 2*b == b:
            F = lambda x2: -0.5 * a/b * x2**2 
            cfa1_frac[place_slice] = cfa1_frac_first_left[ref_slice]
            top_fa = cfa1_frac[:, 0:min(b, nc2-i), 1:]
            bot_fa = cfa1_frac[:, 0:min(b, nc2-i), :-1]
            cvol_frac[place_slice] = top_fa + X1f[:, 0:min(b, nc2-i), :-1] * (top_fa - bot_fa) - F(X2f[ref_slice] + top_fa) + F(X2f[ref_slice] + bot_fa)

    cfa1_x = (X1f - f_X2f[:-1, :-1, :]) / (f_X2f[:-1, 1:, :] - f_X2f[:-1, :-1, :]) + X2f[:-1, :-1, :]    # nc3 x nc2 x nc1 + 1 --- x-intercepts
    cfa1_frac_left = cfa1_x - X2f[:-1, :-1, :]      # x-intercepts minus left coordinates (only gets area where fluid to the left of boundary)
    cfa1_frac_right = X2f[:-1, 1:, :] - cfa1_x      # fluid is on the right of boundary
    cfa1_frac = cfa1_frac_left
    cfa1_frac[f_X2f[:-1, :-1, :] < f(cfa1_x)] = cfa1_frac_right[f_X2f[:-1, :-1, :] < f(cfa1_x)]    # if f(left coordinate) < f(x-intercept), fluid is on the right

    x1v = torch.arange(nc1)
    x2v = torch.arange(nc2)
    X2v, X1v = torch.meshgrid(x2v, x1v, indexing="ij")
    intercepts2 = f(x2f)
    areas2 = torch.remainder(intercepts2, 1)
    below_ground = intercepts2
    
    
    
    # return cfa2_frac, cfa1, cvol  


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
