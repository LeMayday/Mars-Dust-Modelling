import torch
import math
import time
import kintera
import snapy
from snapy import (
        index,
        MeshBlockOptions,
        MeshBlock,
        OutputOptions,
        NetcdfOutput
        )
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from movie_from_pngs import delete_files, create_movie
from integration import plot_func

torch.set_default_dtype(torch.float64)

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/mars-atmosphere-equation-metric/
# https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/gas_constant.htm
# put these values here and / or in yaml file -- no, these values have priority?
p0 = 669            # average surface pressure on Mars
Ts = -31 + 273      # average surface temperature
grav = 3.73         # Mars gravity constant
Rd = 189.0          # gas constant
gamma = 1.29
o = 5.67E-8         # W / m^2 / K^4
s0 = 580;           # W / m^2
Teq = (s0 / o / 4) ** (1/4)
q_dot = o * Teq **4     # heat flux

# device
device = torch.device("cuda:0")

# set hydrodynamic options
op = MeshBlockOptions.from_yaml("convection.yaml")

# initialize block
block = MeshBlock(op)
block.to(device)

# get handles to modules
coord = block.hydro.module("coord")
thermo = block.hydro.module("eos.thermo")
eos = block.hydro.module("eos")

# thermodynamics
Rd = kintera.constants.Rgas / kintera.species_weights()[0]
cv = kintera.species_cref_R()[0] * Rd
cp = cv + Rd

# set initial condition

x3v, x2v, x1v = torch.meshgrid(
    coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
)   # x3v is x, x2v is y, x1v is z

# dimensions
nc3 = coord.buffer("x3v").shape[0]
nc2 = coord.buffer("x2v").shape[0]
nc1 = coord.buffer("x1v").shape[0]
nvar = 5

w = torch.zeros((nvar, nc3, nc2, nc1), device=device)       # initialize primitive variables (density, vx, vy, vz, pressure)

# temp = Ts - grav * x1v / cp       # adiabatic condition
temp = torch.full_like(x1v, Ts)     # isothermal condition

# w[index.ipr] = p0 * torch.pow(temp / Ts, cp / Rd)
w[index.ipr] = p0 * torch.exp(-grav * x1v / Rd / Ts)        # isothermal pressure
w[index.idn] = w[index.ipr] / (Rd * temp)                   # ideal gas law

block_vars = {}
block_vars["hydro_w"] = w
block_vars = block.initialize(block_vars)
block_vars["scalar_x"] = torch.tensor(0.)
block_vars["scalar_v"] = torch.tensor(0.)

# make output
out2 = NetcdfOutput(OutputOptions().file_basename("output/convection").fid(2).variable("prim"))
out3 = NetcdfOutput(OutputOptions().file_basename("output/convection").fid(3).variable("uov"))

block.set_uov("temp", temp)
block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

activities = [ProfilerActivity.CPU]

# integration
count = 0
start_time = time.time()
interior = block.part((0, 0, 0))
current_time = 0.0

bottom_row_height = torch.full((nc3, nc2), coord.buffer("dx1f")[interior[-1]][0])[interior[1:3]]
bottom_row_height = bottom_row_height.to(device)
top_row_height = torch.full((nc3, nc2), coord.buffer("dx1f")[interior[-1]][-1])[interior[1:3]]
top_row_height = top_row_height.to(device)  

w[interior][index.ivx] = torch.randn_like(w[interior][index.ivx])
filenames = []
# with profile(activities=activities, record_shapes=True) as prof:
while not block.intg.stop(count, current_time):
    dt = block.max_time_step(block_vars)

    if count % 1000 == 0:
        print(f"count = {count}, dt = {dt}, time = {current_time}")
        u = block_vars["hydro_u"]
        print("mass = ", u[interior][index.idn].sum())

        ivol = thermo.compute("DY->V", (w[index.idn], w[index.icy:]))
        temp = thermo.compute("PV->T", (w[index.ipr], ivol))

        block.set_uov("temp", temp)
        block.set_uov("theta", temp * (p0 / w[index.ipr]).pow(Rd / cp))

        for out in [out2, out3]:
            out.increment_file_number()
            out.write_output_file(block, block_vars, current_time)
            out.combine_blocks()

        fig = plot_func(coord.buffer("x1v")[interior[-1]].cpu(), torch.mean(temp[interior[1:]], dim=[0, 1]).cpu(), current_time)
        output_file = f"frame_{count}.png"
        fig.savefig(output_file)
        plt.close(fig)
        filenames.append(output_file)

    for stage in range(len(block.intg.stages)):
        block.forward(dt, stage, block_vars)
        u = block_vars["hydro_u"]
        last_weight = block.intg.stages[stage].wght2()
        u[interior][index.ipr][:, :, 0] += last_weight * q_dot / bottom_row_height * dt     # heating from the bottom
        u[interior][index.ipr][:, :, -1] -= last_weight * q_dot / top_row_height * dt       # cooling from the top
    
    count += 1
    current_time += dt

print("elapsed time = ", time.time() - start_time)
create_movie(filenames, "vertical_temp.mp4")
delete_files(filenames)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
