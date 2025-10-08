# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie
from plotting import *
import os
#import kintera
#from snapy import MeshBlockOptions

def get_nc_files(directory):
    out2_files = []
    out3_files = []
    for filename in sorted(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, filename)):
            root, ext = os.path.splitext(filename)
            if ext.lower() == ".nc":
                if "out2" in root:
                    out2_files.append(os.path.join(directory, filename))
                elif "out3" in root:
                    out3_files.append(os.path.join(directory, filename))
    return out2_files, out3_files 

# for some reason, importing kintera and snapy makes the code not work due to some h5py issue
g = 3.75
#MeshBlockOptions.from_yaml("convection.yaml")
#Rd = kintera.constants.Rgas / kintera.species_weights()[0]
#cv = kintera.species_cref_R()[0] * Rd
#cp = cv + Rd
cp = 842

experiment_name = input("Experiment Name:\n")
working_dir = f"output_{experiment_name}"
nc2_files, nc3_files = get_nc_files(working_dir)
ref_data = xr.open_dataset(nc2_files[0]).isel(time=0)
X, Y, Z = np.meshgrid(ref_data['x2'], ref_data['x3'], ref_data['x1'])

plot_dict = {}

plot_dict["vertical_temp_theta"] = {"flag": 1, "files": []}
plot_dict["horizontal_temp_theta"] = {"flag": 1, "files": []}
plot_dict["vertical_vel_top_bottom"] = {"flag": 1, "files": []}
plot_dict["horizontal_vel"] = {"flag": 1, "files": []}
plot_dict["vel_vector"] = {"flag": 0, "files": []}

# nc2_files and nc3_files should have the same size
for n in range(len(nc2_files)):
    print(f"Reading file {n + 1}")
    data2 = xr.open_dataset(nc2_files[n]).isel(time=0)
    data3 = xr.open_dataset(nc3_files[n]).isel(time=0)
    current_time = data2['time']

    for key, value in plot_dict.items():
        if not value["flag"]:
            continue
        if key == "vertical_temp_theta":
            fig = plt.figure()
            fig.set_size_inches(12, 8)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            temp_data = data3['temp']
            plot_func(temp_data.mean(dim=['x2', 'x3']), temp_data['x1'], current_time, ax = ax1)
            plot_func(-g / cp * temp_data['x1'] + 260, temp_data['x1'], current_time, ax = ax1)
            ax1.set_title("Temp")

            theta_data = data3['theta']
            plot_func(theta_data.mean(dim=['x2', 'x3']), theta_data['x1'], current_time, ax = ax2)
            ax2.set_title("Theta")
            fig.tight_layout()

        elif key == "horizontal_temp_theta":
            fig = plt.figure()
            fig.set_size_inches(12, 8)
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

            temp_data = data3['temp']
            plot_func(temp_data['x2'], temp_data.isel(x1 = 0).mean(dim=['x3']), current_time, ax = ax1)
            plot_func(temp_data['x2'], temp_data.isel(x1 = -1).mean(dim=['x3']), current_time, ax = ax1)
            ax1.set_title("Temp")

            theta_data = data3['theta']
            plot_func(theta_data['x2'], theta_data.isel(x1 = 0).mean(dim=['x3']), current_time, ax = ax2)
            plot_func(theta_data['x2'], theta_data.isel(x1 = -1).mean(dim=['x3']), current_time, ax = ax2)
            ax2.set_title("Theta")
            fig.tight_layout()

        elif key == "vertical_vel_top_bottom":
            fig = plt.figure()
            fig.set_size_inches(12, 8)
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

            vel_data_bottom = data2['vel1'].isel(x1=0).transpose()
            vel_data_top = data2['vel1'].isel(x1=-1).transpose()
            vmin = min(float(vel_data_bottom.min()), float(vel_data_top.min()))
            vmax = min(float(vel_data_bottom.max()), float(vel_data_top.max()))

            im = ax1.imshow(vel_data_bottom, cmap='cividis', aspect='auto', vmin=vmin, vmax=vmax)
            ax1.text(1, 1, f"{float((vel_data_bottom > 0).sum())/vel_data_bottom.size*100:.1f}% +, Time: {current_time:.2f}", transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='right')
            # cbar = fig.colorbar(im, ax=ax1, orientation='vertical')
            ax1.set_title("Vz Bottom")

            im = ax2.imshow(vel_data_top, cmap='cividis', aspect='auto', vmin=vmin, vmax=vmax)
            ax2.text(1, 1, f"{float((vel_data_top > 0).sum())/vel_data_top.size*100:.1f}% +, Time: {current_time:.2f}", transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='right')
            # cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
            ax2.set_title("Vz Top")
            fig.tight_layout()
            fig.colorbar(im, ax=[ax1, ax2], orientation='vertical')

        elif key == "horizontal_vel":
            fig = plt.figure()
            fig.set_size_inches(12, 8)
            ax1 = fig.add_subplot(1, 1, 1)
            
            hor_vel_data = data2['vel2']
            plot_func(hor_vel_data.mean(dim=['x2', 'x3']), hor_vel_data['x1'], current_time, ax = ax1)
            ax1.set_title('Mean Horizontal Vel')
            fig.tight_layout()

        elif key == "vel_vector":
            fig = plt.figure()
            fig.set_size_inches(12, 8)
            ax1 = fig.add_subplot(1, 1, 1)

            vert_vel_data = data2['vel1']
            hor_vel_data = data2['vel2']

            plot_2D_vectors(X, Z, hor_vel_data.isel(x3 = 0).transpose(), vert_vel_data.isel(x3 = 0).transpose(), current_time, ax=ax1)
            fig.tight_layout()

        output_file = f"{key}_frame_{n}.png"
        fig.savefig(output_file)
        plt.close(fig)
        value["files"].append(output_file)

    data2.close()
    data3.close()

for key, value in plot_dict.items():
    filenames = value["files"]
    if len(filenames) != 0:
        create_movie(filenames, f"{key}_exp{experiment_name}.mp4")
        delete_files(filenames)

