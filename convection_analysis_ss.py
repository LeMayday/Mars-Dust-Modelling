# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
import argparse
from typing import List, Dict, TypedDict, Callable
from configure_yaml import is_implicit, is_3D, get_exp_res, Res
from mars_topography import format_lat_long_string
from mars import grav, gamma, M_bar, R_gas, q_dot 
# for some reason, importing kintera and snapy makes the code not work due to some h5py issue
cp = gamma * R_gas / M_bar / (gamma - 1)


class Analysis_Config(TypedDict):
    flag: bool
    subplots: List[int]


def add_slope_triangle_loglog(ax: Axes, x_start_frac: float, y_start_frac: float, slope_str: str, width_frac: float = 0.1):
    # thanks Gemini
    assert slope_str in ["-5/3", "-3"]      # make sure not evaluating other expression
    slope = eval(slope_str)
    trans = ax.transAxes + ax.transData.inverted()

    p1_data = trans.transform((x_start_frac, y_start_frac))     # define points in axis coordinated
    p2_axes = (x_start_frac + width_frac, y_start_frac)
    p2_data = trans.transform(p2_axes)

    dx_ratio = p2_data[0] / p1_data[0]                          # but compute slope based on actual data
    dy_ratio = dx_ratio**slope
    p3_data = (p2_data[0], p1_data[1] * dy_ratio)

    pts = np.array([p1_data, p2_data, p3_data])
    ax.add_patch(Polygon(pts, facecolor='none', edgecolor='k', transform=ax.transData))
    ax.text(p3_data[0], p3_data[1], slope_str, transform=ax.transData, va='center')


def get_nc_files(directory: str) -> List[str]:
    out_files = []
    for filename in sorted(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, filename)):
            root, ext = os.path.splitext(filename)
            if ext.lower() == ".nc":
                out_files.append(os.path.join(directory, filename))
    return out_files


def averaged_exp_data(file_path: str, num_files: int) -> xr.Dataset:
    nc_files = get_nc_files(file_path)
    nc_data = []
    for nc in nc_files[-num_files:]:
        with xr.open_dataset(nc) as ds:
            nc_data.append(ds)
    nc_data_concat: xr.Dataset = xr.concat(nc_data, dim='time')
    return nc_data_concat.mean('time')


def plot_vert_temp_theta(analysis_dict: Analysis_Config, temp: xr.Dataset, theta: xr. Dataset, linestyle: str, legend_labels: List[str], last: bool):
    ax1: Axes = analysis_dict["ax1"]
    ax1.set_title("Temp")
    ax2: Axes = analysis_dict["ax2"]
    ax2.set_title("Theta")

    ax1.plot(temp.mean(dim=['x2', 'x3']), temp['x1'], linestyle)
    ax2.plot(theta.mean(dim=['x2', 'x3']), theta['x1'], linestyle)

    if last:        # plot comparison line at very end
        ax1.plot(-grav / cp * temp['x1'] + 260, temp['x1'], 'k:')

    ax1.legend([*legend_labels, "Adiabatic profile"])
    ax2.legend(legend_labels)
    ax2.sharey(ax1)
    ax1.set_ylabel("Height (m)")


def plot_vert_vel_dist(analysis_dict: Analysis_Config, vel1: xr.Dataset, color: str, style: str, legend_labels: List[str]):
    titles = ["Vz Top", "Vz Top 1/4", "Vz Middle", "Vz Bottom 1/4", "Vz Bottom"]
    axes: List[Axes] = [analysis_dict[f"ax{i}"] for i in range(1, 6)]

    bin_width = 0.5
    # bins = 100
    # bins = np.arange(-100, 100 + bin_width, bin_width)
    # sample_pts = np.linspace(-20, 20, 100)
    nx1 = vel1.x1.size
    idxs = [-1, int(nx1*3/4), int(nx1/2), int(nx1/4), 0]
    data_min = 0
    data_max = 0
    for j, ax in enumerate(axes):
        vel_data = vel1.isel(x1=idxs[j]).stack(x3x2=('x3','x2'))
        bin_min = np.floor(vel_data.min() / bin_width) * bin_width
        bin_max = np.floor(vel_data.max() / bin_width) * bin_width
        bins = np.arange(bin_min, bin_max + bin_width, bin_width)
        if bin_min < data_min:
            data_min = bin_min
        if bin_max > data_max:
            data_max = bin_max
        ax.hist(vel1.isel(x1=idxs[j]).stack(x3x2=('x3','x2')), bins=bins, histtype='step', density=True, linewidth=2, alpha=0.6, linestyle=style, color=color)
        if j > 0:
            ax.sharex(axes[0])
        ax.set_title(titles[j])
        ax.legend(legend_labels)
    axes[0].set_xlim([data_min, data_max])


def plot_gravity_wave(analysis_dict: Analysis_Config, rho: xr.Dataset, theta: xr.Dataset, linestyle: str, legend_labels: List[str]):
    ax1: Axes = analysis_dict["ax1"]
    ax1.set_title(r'$(\dot{q}/\rho)^{1/3} N^{-1}$')

    rho_mean = rho.mean(dim=['x2', 'x3'])
    theta_mean = theta.mean(dim=['x2', 'x3'])
    dtheta_dz = theta_mean.differentiate('x1')
    # drho_dz = rho_data.differentiate('x1')
    # N_sq = -g / rho_data * drho_dz
    N_sq = grav / theta_mean * dtheta_dz
    val = (q_dot / rho_mean)**(1/3) * N_sq**(-1/2)

    ax1.plot(val, rho_mean['x1'], linestyle)
    ax1.set_xlim([0, 50E3])

    ax1.legend(legend_labels)


def plot_KE_flux(analysis_dict: Analysis_Config, rho: xr.Dataset, vel1: xr.Dataset, vel2: xr.Dataset, vel3: xr.Dataset,
                 linestyle: str, legend_labels: List[str], last: bool):
    ax1: Axes = analysis_dict["ax1"]

    # u**2 + v**2 + w**2
    KE_flux = vel1 * 0.5 * (vel3**2 + vel2**2 + vel1**2) * rho
    mean_KE_flux = KE_flux.mean(dim=['x2', 'x3'])

    ax1.plot(mean_KE_flux, mean_KE_flux['x1'], linestyle)
    if last:
        ax1.plot(q_dot + 0 * mean_KE_flux['x1'], mean_KE_flux['x1'], 'k:')
        legend_labels.append('Forcing')

    ax1.legend(legend_labels)
    ax1.set_xlabel('Energy Flux (W/m^2)')
    ax1.set_ylabel('Height (m)')


def plot_KE_power(analysis_dict: Analysis_Config, vel1: xr.Dataset, vel2: xr.Dataset, vel3: xr.Dataset,
                  marker: str, color: str, legend_labels: List[str], last: bool, threeD: bool):
    # num samples
    nx1 = vel1.x1.size
    nx2 = vel1.x2.size
    nx3 = vel1.x3.size

    titles = ["KE Power Top 1/4", "KE Power Middle", "KE Power Bottom 1/4"]
    idxs = [int(nx1*3/4), int(nx1/2), int(nx1/4)]
    axes: List[Axes] = [analysis_dict[f"ax{i}"] for i in range(1, 4)]

    # sampling frequency
    L = 80E3        # size of domain
    Fs2 = nx2 / L   # sampling frequency in x2 direction
    Fs3 = nx3 / L   # sampling frequency in x3 direction

    for j, ax in enumerate(axes):
        u = vel3.isel(x1=idxs[j])
        v = vel2.isel(x1=idxs[j])
        w = vel1.isel(x1=idxs[j])
        if threeD:
            kx2 = np.fft.fftfreq(nx2, d=1/Fs2)
            kx2 = np.fft.fftshift(kx2)
            kx3 = np.fft.fftfreq(nx3, d=1/Fs3)
            kx3 = np.fft.fftshift(kx3)
            Kx2, Kx3 = np.meshgrid(kx2, kx3)
            k = np.sqrt(Kx2**2 + Kx3**2)

            KE = u**2 + v**2 + w**2
            KE_hat = np.fft.fft2(KE)
            KE_hat = np.fft.fftshift(KE_hat)
            KE_ps_2D = abs(KE_hat)**2

            # ASSUMES nx2 = nx3!!!
            freqs = kx2[kx2 >= 0]
            hist_counts, _ = np.histogram(k.ravel(), bins=freqs)
            hist_power, _ = np.histogram(k.ravel(), bins=freqs, weights=KE_ps_2D.ravel())

            KE_ps = np.where(hist_counts != 0, hist_power / hist_counts, 0)
            # KE_ps = np.divide(hist_power, hist_counts, where=hist_counts != 0)
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

            freqs = freqs[1:]
            KE_ps = KE_ps[1:]

        ax.plot(freqs, KE_ps, marker=marker, color=color, linestyle='None', markerfacecolor='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1E-4, np.max(freqs)])
        if last:
            slopes = ["-5/3", "-3"]
            for s in slopes:
                add_slope_triangle_loglog(ax, 0.85, 0.85, s, 0.15)
        ax.set_title(titles[j])
        ax.set_ylabel('Wave Power (Log Magnitude), Normalized')
        ax.legend(legend_labels)

    axes[-1].set_xlabel('Wavenumber (1/m)')


def plot_theta_time_series(analysis_dict: Analysis_Config, experiment_names: List[str], filepath_constructor, linestyles: Dict[str, List[str]], legend_labels: List[str]):
    titles = ["Theta Top 1/4", "Theta Middle", "Theta Bottom 1/4"]
    axes: List[Axes] = [analysis_dict[f"ax{i}"] for i in range(1, 4)]

    for i, exp in enumerate(experiment_names):
        theta_data_exp = [[], [], []]
        nc_files = get_nc_files(filepath_constructor(exp))
        time = np.zeros(len(nc_files))
        for j, nc3 in enumerate(nc_files):
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


def make_plots(plot_dict: Dict[str, Analysis_Config], experiment_names: List[str], num_files_to_avg: int,
               filepath_constructor: Callable[[str], str]):
 
    skip_main_loop = True
    for key, analysis_dict in plot_dict.items():
        if key != "hori_theta" and analysis_dict["flag"]:
            skip_main_loop = False

    # configure plot size and axes
    print("Preparing plots...")
    for key, analysis_dict in plot_dict.items():
        fig = plt.figure()
        analysis_dict["fig"] = fig
        subplot_array_dims = analysis_dict["subplots"]
        fig.set_size_inches(max(12, 6*subplot_array_dims[1]), max(8, 4*subplot_array_dims[0]))
        for i in range(1, np.prod(subplot_array_dims) + 1):
            analysis_dict[f"ax{i}"] = fig.add_subplot(subplot_array_dims[0], subplot_array_dims[1], i)

    # configure plot coloring
    linestyles = {"color": [], "style": [], "marker": [], "comb": []}
    for exp in experiment_names:
        if is_implicit(exp):
            color = 'r'
        else:
            color = 'b'
        if get_exp_res(exp) == Res.COURSE:
            style = '--'
            marker = 'v'
        elif get_exp_res(exp) == Res.FINE:
            style = "-"
            marker = 'o'
        linestyles["color"].append(color)
        linestyles["style"].append(style)
        linestyles["marker"].append(marker)
        linestyles["comb"].append(f"{style}{color}")

    # configure legend labels
    legend_labels = ["Experiment " + exp for exp in experiment_names]
    for i, exp in enumerate(experiment_names):          # outer loop is experiment so only one set of data is loaded at a time
        if skip_main_loop:
            continue
        last = i == len(experiment_names) - 1
        print(f"Loading data from experiment {exp} ...")
        nc_data = averaged_exp_data(filepath_constructor(exp), num_files_to_avg)
        for key, analysis_dict in plot_dict.items():    # inner loop is which experiments need info to be analyzed
            if not analysis_dict["flag"]:
                continue
            fig: Figure = analysis_dict["fig"]
            match key:
                case "vert_temp_theta":
                    plot_vert_temp_theta(analysis_dict, nc_data['temp'], nc_data['theta'], linestyles["comb"][i], legend_labels, last)
                    fig.tight_layout()

                case "vert_vel_dist":
                    plot_vert_vel_dist(analysis_dict, nc_data['vel1'], linestyles["color"][i], linestyles["style"][i], legend_labels)
                    fig.tight_layout()

                case "gravity_wave":
                    plot_gravity_wave(analysis_dict, nc_data['rho'], nc_data['theta'], linestyles["comb"][i], legend_labels)
                    fig.tight_layout()

                case "KE_flux":
                    plot_KE_flux(analysis_dict, nc_data['rho'], nc_data['vel1'], nc_data['vel2'], nc_data['vel3'], linestyles["comb"][i], legend_labels, (not is_3D(exp) and last))
                    fig.tight_layout()

                case "KE_power":
                    plot_KE_power(analysis_dict, nc_data['vel1'], nc_data['vel2'], nc_data['vel3'], linestyles["marker"][i], linestyles["color"][i], legend_labels, last, is_3D(exp))
                    fig.tight_layout()
    
    if (analysis_dict := plot_dict["hori_theta"])["flag"]:
        print("Loading time series...")
        fig: Figure = analysis_dict["fig"]
        plot_theta_time_series(analysis_dict, experiment_names, filepath_constructor, linestyles, legend_labels)
        fig.tight_layout()


def save_plots(plot_dict: Dict[str, Analysis_Config], save_dir: str, file_index: str):
    # finally, save all plots
    print("Saving plots...")
    for key, value in plot_dict.items():
        # inner loop is which experiments need info to be analyzed
        if not value["flag"]:
            continue
        fig: Figure = value["fig"]
        output_file = f"{key}_ss{file_index}.png"
        fig.savefig(f"{save_dir}/{output_file}", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-names", required=True, type=str, nargs='+', help="Name of the experiments to analyze as a list")
    parser.add_argument("-l", "--lat-long-bounds", type=float, nargs=4, help="List of min lat, max lat, min long, max long")
    parser.add_argument("-n", "--num-file", type=int, default=1, help="Number of files to average over from the end")
    parser.add_argument("-i", "--index", type=str, default="", help="Index for file naming (in case of duplicate names)")
    parser.add_argument("-o", "--output-parent-dir", type=str, default = ".", help="Directory for output files.")
    args = parser.parse_args()
    file_index: str = args.index
    if file_index != "": file_index = "_" + file_index
    experiment_names: List[str] = args.experiment_names
    lat_long_str = format_lat_long_string(*args.lat_long_bounds) if args.lat_long_bounds is not None else ""
    num_files: int = args.num_file
    filepath_constructor: Callable[[str], str] = lambda exp_name: f"{args.output_parent_dir}/output_{exp_name}_{lat_long_str}"

    # make plot output directory if it doesn't already exist
    save_directory = f"output/analysis_output"
    try:
        os.mkdir(save_directory)
    except FileExistsError:
        pass

    plot_dict: Dict[str, Analysis_Config] = {}
    plot_dict["vert_temp_theta"]    = {"flag": 1, "subplots": [1, 2]}
    plot_dict["hori_theta"]         = {"flag": 1, "subplots": [3, 1]}
    plot_dict["vert_vel_dist"]      = {"flag": 1, "subplots": [5, 1]}
    plot_dict["gravity_wave"]       = {"flag": 1, "subplots": [1, 1]}
    plot_dict["KE_flux"]            = {"flag": 1, "subplots": [1, 1]}
    plot_dict["KE_power"]           = {"flag": 1, "subplots": [1, 3]}

    make_plots(plot_dict, experiment_names, num_files, filepath_constructor)
    save_plots(plot_dict, save_directory, file_index)


if __name__ == "__main__":
    main()
