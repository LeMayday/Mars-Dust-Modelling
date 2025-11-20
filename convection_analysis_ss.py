# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
import argparse
from typing import List, Tuple

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

def averaged_exp_data(exp_name: str, num_files: int) -> Tuple[xr.Dataset, xr.Dataset]:
    nc2_files, nc3_files = get_nc_files(f"output_{exp_name}")
    nc2_data = []
    for nc2 in nc2_files[-num_files:]:
        with xr.open_dataset(nc2) as ds:
            nc2_data.append(ds)
    nc2_data_concat: xr.Dataset = xr.concat(nc2_data, dim='time')

    nc3_data = []
    for nc3 in nc3_files[-num_files:]:
        with xr.open_dataset(nc3) as ds:
            nc3_data.append(ds)
    nc3_data_concat: xr.Dataset = xr.concat(nc3_data, dim='time')
    return nc2_data_concat.mean('time'), nc3_data_concat.mean('time')

# for some reason, importing kintera and snapy makes the code not work due to some h5py issue
g = 3.73
cp = 842
s0 = 580;           # W / m^2
q_dot = s0 / 4      # heat flux

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment-names", required=True, type=str, nargs='+', help="Name of the experiments to analyze as one string (e.g. ACD)")
group = parser.add_mutually_exclusive_group()
group.add_argument("--3D", action="store_true", help="Whether to use 3D data")
group.add_argument("--compare-with", type=str, nargs='+', default="", help="List of 3D data to compare with")
parser.add_argument("-n", "--num-file", type=int, default=1, help="Number of files to average over from the end")
args = parser.parse_args()
experiment_names = args.experiment_names
compare_names = args.compare_with
if vars(args)['3D']:
    experiment_names = [exp_name + "_3D" for exp_name in experiment_names]
elif len(compare_names) > 0:
    experiment_names = experiment_names + [exp_name + "_3D" for exp_name in compare_names]
num_files = args.num_file

# make plot output directory if it doesn't already exist
save_directory = f"analysis_output"
try:
    os.mkdir(save_directory)
except FileExistsError:
    pass

plot_dict = {}
plot_dict["vert_temp_theta"] = {"flag": 0, "subplots": [1, 2]}
plot_dict["hori_theta"] = {"flag": 0, "subplots": [3, 1]}
plot_dict["vert_vel_dist"] = {"flag": 0, "subplots": [5, 1]}
plot_dict["hori_vel"] = {"flag": 0, "subplots": [1, 1]}
plot_dict["gravity_wave"] = {"flag": 0, "subplots": [1, 1]}
plot_dict["KE_flux"] = {"flag": 0, "subplots": [1, 1]}
plot_dict["KE_power"] = {"flag": 1, "subplots": [3, 1]}

skip_main_loop = True
for key, value in plot_dict.items():
    if key != "hori_theta" and value["flag"]:
        skip_main_loop = False

# configure plot size and axes
print("Preparing plots...")
for key, value in plot_dict.items():
    fig = plt.figure()
    value["fig"] = fig
    subplot_array_dims = value["subplots"]
    fig.set_size_inches(12, max(8, 4*subplot_array_dims[0]))
    for i in range(1, np.prod(subplot_array_dims) + 1):
        value[f"ax{i}"] = fig.add_subplot(subplot_array_dims[0], subplot_array_dims[1], i)

# configure plot coloring
linestyles = {"color": [], "style": [], "marker": [], "comb": []}
for exp in experiment_names:
    if "E" in exp:
        color = 'b'
    elif "I" in exp:
        color = 'r'
    linestyles["color"].append(color)
    if "C" in exp:
        style = '--'
        marker = 'v'
    elif "F" in exp:
        style = "-"
        marker = 'o'
    linestyles["style"].append(style)
    linestyles["marker"].append(marker)
    linestyles["comb"].append(f"{style}{color}")

legend_labels = ["Experiment " + exp for exp in experiment_names]

last = len(experiment_names) - 1
for i, exp in enumerate(experiment_names):
    if skip_main_loop:
        continue
    # outer loop is experiment so only one set of data is loaded at a time
    print(f"Loading data from experiment {exp} ...")
    nc2_data, nc3_data = averaged_exp_data(exp, num_files)
    for key, value in plot_dict.items():
        # inner loop is which experiments need info to be analyzed
        if not value["flag"]:
            continue
        fig: Figure = value["fig"]
        match key:
            case "vert_temp_theta":
                ax1: Axes = value["ax1"]
                ax1.set_title("Temp")
                ax2: Axes = value["ax2"]
                ax2.set_title("Theta")

                temp_data = nc3_data['temp']
                ax1.plot(temp_data.mean(dim=['x2', 'x3']), temp_data['x1'], linestyles["comb"][i])

                theta_data = nc3_data['theta']
                ax2.plot(theta_data.mean(dim=['x2', 'x3']), theta_data['x1'], linestyles["comb"][i])

                if i == last:
                    # plot comparison line at very end
                    ax1.plot(-g / cp * temp_data['x1'] + 260, temp_data['x1'], 'k:')
                ax1.legend(legend_labels)
                ax2.legend(legend_labels)
                fig.tight_layout()

            case "hori_vel":
                ax1: Axes = value["ax1"]
                ax1.set_title('Mean Horizontal Vel')

                hor_vel_data = nc2_data['vel2']
                vels = hor_vel_data.mean(dim=['x2', 'x3'])
                if vels.isel(x1=0) < 0:     # have all velocity profiles oriented the same way
                    vels = -vels
                ax1.plot(vels, hor_vel_data['x1'], linestyles["comb"][i])

                ax1.legend(legend_labels)
                fig.tight_layout()

            case "vert_vel_dist":
                titles = ["Vz Top", "Vz Top 1/4", "Vz Middle", "Vz Bottom 1/4", "Vz Bottom"]
                axes: List[Axes] = [value[f"ax{i}"] for i in range(1, 6)]

                bin_width = 0.5
                # bins = 100
                # bins = np.arange(-100, 100 + bin_width, bin_width)
                # sample_pts = np.linspace(-20, 20, 100)
                nx1 = nc2_data.x1.size
                idxs = [-1, int(nx1*3/4), int(nx1/2), int(nx1/4), 0]
                data_min = 0
                data_max = 0
                for j, ax in enumerate(axes):
                    vel_data = nc2_data['vel1'].isel(x1=idxs[j]).stack(x3x2=('x3','x2'))
                    bin_min = np.floor(vel_data.min() / bin_width) * bin_width
                    bin_max = np.floor(vel_data.max() / bin_width) * bin_width
                    bins = np.arange(bin_min, bin_max + bin_width, bin_width)
                    if bin_min < data_min:
                        data_min = bin_min
                    if bin_max > data_max:
                        data_max = bin_max
                    ax.hist(nc2_data['vel1'].isel(x1=idxs[j]).stack(x3x2=('x3','x2')), bins=bins, histtype='step', density=True, linewidth=2, alpha=0.6, linestyle=linestyles["style"][i], color=linestyles["color"][i])
                    # if i > 0:
                    #     ax.sharex(axes[0])
                    ax.set_title(titles[j])
                    ax.legend(legend_labels)
                axes[0].set_xlim([data_min, data_max])
                fig.tight_layout()

            case "gravity_wave":
                ax1: Axes = value["ax1"]
                ax1.set_title(r'$(\dot{q}/\rho)^{1/3} N^{-1}$')

                rho_data = nc2_data['rho'].mean(dim=['x2', 'x3'])
                theta_data = nc3_data['theta'].mean(dim=['x2', 'x3'])
                dtheta_dz = theta_data.differentiate('x1')
                # drho_dz = rho_data.differentiate('x1')
                # N_sq = -g / rho_data * drho_dz
                N_sq = g / theta_data * dtheta_dz
                val = (q_dot / rho_data)**(1/3) * N_sq**(-1/2)

                ax1.plot(val, rho_data['x1'], linestyles["comb"][i])
                ax1.set_xlim([0, 50E3])

                ax1.legend(legend_labels)
                fig.tight_layout()

            case "KE_flux":
                ax1: Axes = value["ax1"]

                rho = nc2_data['rho']
                u = nc2_data['vel3']
                v = nc2_data['vel2']
                w = nc2_data['vel1']

                KE_flux = w * 0.5 * (u**2 + v**2 + w**2) * rho
                mean_KE_flux = KE_flux.mean(dim=['x2', 'x3'])

                ax1.plot(mean_KE_flux, mean_KE_flux['x1'], linestyles["comb"][i])
                if not "3D" in exp and i == last:
                    ax1.plot(q_dot + 0 * mean_KE_flux['x1'], mean_KE_flux['x1'], 'k:')
                    legend_labels.append('Forcing')

                ax1.legend(legend_labels)
                ax1.set_xlabel('Energy Flux (W/m^2)')
                ax1.set_ylabel('Height (m)')
                fig.tight_layout()

            case "KE_power":
                # num samples
                nx1 = nc2_data.x1.size
                nx2 = nc2_data.x2.size
                nx3 = nc2_data.x3.size

                titles = ["KE Power Top 1/4", "KE Power Middle", "KE Power Bottom 1/4"]
                idxs = [int(nx1*3/4), int(nx1/2), int(nx1/4)]
                axes: List[Axes] = [value[f"ax{i}"] for i in range(1, 4)]

                # sampling frequency
                L = 80E3        # size of domain
                Fs2 = nx2 / L   # sampling frequency in x2 direction
                Fs3 = nx3 / L   # sampling frequency in x3 direction

                for j, ax in enumerate(axes):
                    u = nc2_data['vel3'].isel(x1=idxs[j])
                    v = nc2_data['vel2'].isel(x1=idxs[j])
                    w = nc2_data['vel1'].isel(x1=idxs[j])
                    if "3D" in exp:
                        kx2 = np.fft.fftfreq(nx2, d=1/Fs2)
                        kx2 = np.fft.fftshift(kx2)
                        kx3 = np.fft.fftfreq(nx3, d=1/Fs3)
                        kx3 = np.fft.fftshift(kx3)
                        Kx2, Kx3 = np.meshgrid(kx2, kx3)
                        k = np.sqrt(Kx2**2 + Kx3**2)

                        u_hat = np.fft.fft2(u)
                        u_hat = np.fft.fftshift(u_hat)
                        v_hat = np.fft.fft2(v)
                        v_hat = np.fft.fftshift(v_hat)
                        w_hat = np.fft.fft2(w)
                        w_hat = np.fft.fftshift(w_hat)

                        KE_hat = u_hat**2 + v_hat**2 + w_hat**2
                        KE_ps_2D = abs(KE_hat)**2

                        # ASSUMES nx2 = nx3!!!
                        freqs = kx2[kx2 >= 0]
                        hist_counts, _ = np.histogram(k.ravel(), bins=freqs)
                        hist_power, _ = np.histogram(k.ravel(), bins=freqs, weights=KE_ps_2D.ravel())

                        KE_ps = np.divide(hist_power, hist_counts, where=hist_counts != 0)
                        freqs = np.convolve(freqs, [0.5, 0.5])[1:-1]

                    else:
                        freqs = np.fft.rfftfreq(nx2, d=1/Fs2)
                        # (nx, 1) arrays do not undergo 1D ffts correctly, need (nx,)
                        v = v.isel(x3=0)
                        w = w.isel(x3=0)

                        v_hat = np.fft.rfft(v)
                        w_hat = np.fft.rfft(w)

                        KE_hat = v_hat**2 + w_hat**2
                        KE_ps = abs(KE_hat)**2

                    ax.plot(freqs, KE_ps / np.max(KE_ps), marker=linestyles["marker"][i], color=linestyles["color"][i], linestyle='None')
                    if i == last:
                        ax.plot(freqs, np.pow(freqs, -5/3) / np.max(np.pow(freqs, -5/3)), 'k:')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_title(titles[j])
                    ax.set_ylabel('Wave Power (Log Magnitude), Normalized')
                    ax.legend([*legend_labels, "k^(-5/3)"])

                axes[-1].set_xlabel('Wavenumber (1/m)')
                fig.tight_layout()

# handle time series data separately
if (value := plot_dict["hori_theta"])["flag"]:
    print("Loading time series...")
    titles = ["Theta Top 1/4", "Theta Middle", "Theta Bottom 1/4"]
    axes: List[Axes] = [value[f"ax{i}"] for i in range(1, 4)]
    fig: Figure = value["fig"]

    for i, exp in enumerate(experiment_names):
        theta_data_exp = [[], [], []]
        _, nc3_files = get_nc_files(f"output_{exp}")
        time = np.zeros(len(nc3_files))
        for j, nc3 in enumerate(nc3_files):
            with xr.open_dataset(nc3).isel(time=0) as data3:
                time[j] = float(data3.time) / 60
                theta_data = data3['theta']
                mean_theta = theta_data.mean(dim=['x2', 'x3'])
                nx1 = data3.x1.size
                theta_data_exp[0].append(mean_theta.isel(x1=int(nx1*3/4)))
                theta_data_exp[1].append(mean_theta.isel(x1=int(nx1/2)))
                theta_data_exp[2].append(mean_theta.isel(x1=int(nx1/4)))

        for j, ax in enumerate(axes):
            ax.plot(time.tolist(), theta_data_exp[j], linestyles["comb"][i])
            if j > 0:
                ax.sharex(axes[0])

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_ylabel('Theta (K)')
        ax.legend(legend_labels)

    axes[-1].set_xlabel('Time (min)')
    fig.tight_layout()

# finally, save all plots
print("Saving plots...")
for key, value in plot_dict.items():
    # inner loop is which experiments need info to be analyzed
    if not value["flag"]:
        continue
    fig: Figure = value["fig"]
    if vars(args)['3D']:
        output_file = f"{key}_steady_state_3D.png"
    elif len(compare_names) > 0:
        output_file = f"{key}_steady_state_2D_3D.png"
    else:
        output_file = f"{key}_steady_state.png"
    fig.savefig(f"{save_directory}/{output_file}", dpi=300)
    plt.close(fig)
