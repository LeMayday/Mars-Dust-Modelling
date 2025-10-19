# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie
from plotting import *
import os
import argparse
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
                    out2_file = os.path.join(directory, filename)
                elif "out3" in root:
                    out3_file = os.path.join(directory, filename)
    return out2_file, out3_file 

# for some reason, importing kintera and snapy makes the code not work due to some h5py issue
g = 3.73
#MeshBlockOptions.from_yaml("convection.yaml")
#Rd = kintera.constants.Rgas / kintera.species_weights()[0]
#cv = kintera.species_cref_R()[0] * Rd
#cp = cv + Rd
cp = 842

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment-names", required=True, type=str, help="Name of the experiments to analyze as one string (e.g. ACD)")
args = parser.parse_args()
experiment_names = sorted(list(args.experiment_names.upper()))

ref_file, _ = get_nc_files(f"output_{experiment_names[0]}")
ref_data = xr.open_dataset(ref_file).isel(time=0)
X, Y, Z = np.meshgrid(ref_data['x2'], ref_data['x3'], ref_data['x1'])

nc2_files = []
nc3_files = []
for exp in experiment_names:
    nc2_file, nc3_file = get_nc_files(f"output_{exp}")
    nc2_files.append(nc2_file)
    nc3_files.append(nc3_file)

plot_dict = {}
plot_dict["vert_temp_theta"] = {"flag": 1, "files": []}
plot_dict["hori_temp_theta"] = {"flag": 0, "files": []}
plot_dict["vert_vel_top_bot"] = {"flag": 1, "files": []}
plot_dict["hori_vel"] = {"flag": 1, "files": []}

# configure plot settings
for key, value in plot_dict.items():
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    value["fig"] = fig
    if key == "vert_temp_theta":
        value["ax1"] = fig.add_subplot(1, 2, 1)
        value["ax2"] = fig.add_subplot(1, 2, 2)
    if key == "hori_temp_theta":
        value["ax1"] = fig.add_subplot(2, 1, 1)
        value["ax2"] = fig.add_subplot(2, 1, 2)
    if key == "vert_vel_top_bot":
        value["ax1"] = fig.add_subplot(2, 1, 1)
        value["ax2"] = fig.add_subplot(2, 1, 2)
    if key == "hori_vel":
        value["ax1"] = fig.add_subplot(1, 1, 1)

legend_labels = ["Experiment " + exp for exp in experiment_names]

for key, value in plot_dict.items():
    if not value["flag"]:
        continue
    fig = value["fig"]
    if key == "vert_temp_theta":
        ax1 = value["ax1"]
        ax1.set_title("Temp")
        ax2 = value["ax2"]
        ax2.set_title("Theta")

        for i, exp in enumerate(experiment_names):
            data3 = xr.open_dataset(nc3_files[i]).isel(time=0)

            temp_data = data3['temp']
            ax1.plot(temp_data.mean(dim=['x2', 'x3']), temp_data['x1'])

            theta_data = data3['theta']
            ax2.plot(theta_data.mean(dim=['x2', 'x3']), theta_data['x1'])
            data3.close()

        ax1.plot(-g / cp * temp_data['x1'] + 260, temp_data['x1'], 'k--')
        ax1.legend(legend_labels)
        ax2.legend(legend_labels)
        fig.tight_layout()

    elif key == "hori_vel":
        ax1 = value["ax1"]
        ax1.set_title('Mean Horizontal Vel')

        for i, exp in enumerate(experiment_names):
            data2 = xr.open_dataset(nc2_files[i]).isel(time=0)

            hor_vel_data = data2['vel2']
            ax1.plot(hor_vel_data.mean(dim=['x2', 'x3']), hor_vel_data['x1'])
            data2.close()

        ax1.legend(legend_labels)
        fig.tight_layout()

    elif key == "vert_vel_top_bot":
        ax1 = value["ax1"]
        ax1.set_title("Vz Bottom")
        ax2 = value["ax2"]
        ax2.set_title("Vz Top")

        num_bins = 50
        for i, exp in enumerate(experiment_names):
            data2 = xr.open_dataset(nc2_files[i]).isel(time=0)
            
            vel_data_bottom = data2['vel1'].isel(x1=0)
            vel_data_top = data2['vel1'].isel(x1=-1)

            ax1.hist(vel_data_top, bins=num_bins, histtype='step', density=True)
            ax2.hist(vel_data_bottom, bins=num_bins, histtype='step', density=True)
            data2.close()

        ax1.legend(legend_labels)
        ax2.legend(legend_labels)
        fig.tight_layout()

    output_file = f"{key}_steady_state.png"
    fig.savefig(output_file)
    plt.close(fig)

