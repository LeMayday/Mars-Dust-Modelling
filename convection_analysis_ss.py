# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie
from plotting import *
import os
import argparse

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
g = 3.73
cp = 842

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment-names", required=True, type=str, help="Name of the experiments to analyze as one string (e.g. ACD)")
group = parser.add_mutually_exclusive_group()
group.add_argument("--3D", action="store_true", help="Whether to use 3D data")
group.add_argument("--compare-with", type=str, default="", help="List of 3D data to compare with")
parser.add_argument("-n", "--num-file", type=int, default=1, help="Number of files to average over from the end")
args = parser.parse_args()
experiment_names = sorted(list(args.experiment_names.upper()))
compare_names = sorted(list(args.compare_with.upper()))
if vars(args)['3D']:
    experiment_names = [exp_name + "_3D" for exp_name in experiment_names]
elif len(compare_names) > 0:
    experiment_names = experiment_names + [exp_name + "_3D" for exp_name in compare_names]
num_files = args.num_file

nc2_data_by_exp = []
nc3_data_by_exp = []
# average over time dimension for last n files and store in an array according to experiment
for exp in experiment_names:
    nc2_files, nc3_files = get_nc_files(f"output_{exp}")
    nc2_data = []
    for nc2 in nc2_files[-num_files:]:
        with xr.open_dataset(nc2) as ds:
            nc2_data.append(ds)
    nc2_data_concat = xr.concat(nc2_data, dim='time')
    nc2_data_by_exp.append(nc2_data_concat.mean('time'))

    nc3_data = []
    for nc3 in nc3_files[-num_files:]:
        with xr.open_dataset(nc3) as ds:
            nc3_data.append(ds)
    nc3_data_concat = xr.concat(nc3_data, dim='time')
    nc3_data_by_exp.append(nc3_data_concat.mean('time'))

plot_dict = {}
plot_dict["vert_temp_theta"] = {"flag": 0, "files": []}
plot_dict["hori_theta"] = {"flag": 1, "files": []}
plot_dict["vert_vel_top_bot"] = {"flag": 0, "files": []}
plot_dict["hori_vel"] = {"flag": 0, "files": []}

# configure plot settings
for key, value in plot_dict.items():
    fig = plt.figure()
    value["fig"] = fig
    if key == "vert_temp_theta":
        fig.set_size_inches(12, 8)
        value["ax1"] = fig.add_subplot(1, 2, 1)
        value["ax2"] = fig.add_subplot(1, 2, 2)
    if key == "hori_theta":
        fig.set_size_inches(12, 12)
        value["ax1"] = fig.add_subplot(3, 1, 1)
        value["ax2"] = fig.add_subplot(3, 1, 2)
        value["ax3"] = fig.add_subplot(3, 1, 3)
    if key == "vert_vel_top_bot":
        fig.set_size_inches(12, 8)
        value["ax1"] = fig.add_subplot(2, 1, 1)
        value["ax2"] = fig.add_subplot(2, 1, 2)
    if key == "hori_vel":
        fig.set_size_inches(12, 8)
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
            data3 = nc3_data_by_exp[i]

            temp_data = data3['temp']
            ax1.plot(temp_data.mean(dim=['x2', 'x3']), temp_data['x1'])

            theta_data = data3['theta']
            ax2.plot(theta_data.mean(dim=['x2', 'x3']), theta_data['x1'])

        ax1.plot(-g / cp * temp_data['x1'] + 260, temp_data['x1'], 'k--')
        ax1.legend(legend_labels)
        ax2.legend(legend_labels)
        fig.tight_layout()

    elif key == "hori_vel":
        ax1 = value["ax1"]
        ax1.set_title('Mean Horizontal Vel')

        for i, exp in enumerate(experiment_names):
            data2 = nc2_data_by_exp[i]

            hor_vel_data = data2['vel2']
            vels = hor_vel_data.mean(dim=['x2', 'x3'])
            if vels.isel(x1=0) < 0:     # have all velocity profiles oriented the same way
                vels = -vels
            ax1.plot(vels, hor_vel_data['x1'])

        ax1.legend(legend_labels)
        fig.tight_layout()

    elif key == "vert_vel_top_bot":
        ax1 = value["ax1"]
        ax1.set_title("Vz Top")
        ax2 = value["ax2"]
        ax2.set_title("Vz Bottom")

        bin_width = 0.5
        # bins = 100
        bins = np.arange(-20, 20 + bin_width, bin_width)
        sample_pts = np.linspace(-20, 20, 100)
        for i, exp in enumerate(experiment_names):
            data2 = nc2_data_by_exp[i]
            
            vel_data_bottom = data2['vel1'].isel(x1=0).stack(x3x2=('x3','x2'))
            vel_data_top = data2['vel1'].isel(x1=-1).stack(x3x2=('x3','x2'))

            ax1.hist(vel_data_top, bins=bins, histtype='step', density=True, linewidth=2, alpha=0.6)
            ax2.hist(vel_data_bottom, bins=bins, histtype='step', density=True, linewidth=2, alpha=0.6)
            ax2.sharex(ax1)
            ax1.set_xlim([-20, 20])

        ax1.legend(legend_labels)
        ax2.legend(legend_labels)
        fig.tight_layout()

    elif key == "hori_theta":
        ax1 = value["ax1"]
        ax1.set_title("Theta Top 1/4")
        ax2 = value["ax2"]
        ax2.set_title("Theta Middle")
        ax3 = value["ax3"]
        ax3.set_title("Theta Bottom 1/4")

        for i, exp in enumerate(experiment_names):
            theta_data_exp = [[], [], []]
            time = []
            _, nc3_files = get_nc_files(f"output_{exp}")
            for j, nc3 in enumerate(nc3_files):
                with xr.open_dataset(nc3).isel(time=0) as data3:
                    time.append(float(data3.time))
                    theta_data = data3['theta']
                    mean_theta = theta_data.mean(dim=['x2', 'x3'])
                    nx1 = data3.x1.size
                    theta_data_exp[0].append(mean_theta.isel(x1=int(nx1*3/4)))
                    theta_data_exp[1].append(mean_theta.isel(x1=int(nx1/2)))
                    theta_data_exp[2].append(mean_theta.isel(x1=int(nx1/4)))

            ax1.plot(time, theta_data_exp[0])
            ax2.plot(time, theta_data_exp[1])
            ax3.plot(time, theta_data_exp[2])
            ax2.sharex(ax1)
            ax3.sharex(ax1)

        ax1.legend(legend_labels)
        ax2.legend(legend_labels)
        ax3.legend(legend_labels)
        fig.tight_layout()

    if vars(args)['3D']:
        output_file = f"{key}_steady_state_3D.png"
    elif len(compare_names) > 0:
        output_file = f"{key}_steady_state_2D_3D.png"
    else:
        output_file = f"{key}_steady_state.png"
    fig.savefig("analysis_output/" + output_file)
    plt.close(fig)

