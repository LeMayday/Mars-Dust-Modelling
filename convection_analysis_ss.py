# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
import argparse
from typing import List, Dict, TypedDict, Callable
from configure_yaml import is_implicit, is_3D, get_exp_res, Res, get_num_cells_exp
from mars_topography import format_lat_long_string, get_cell_topography, configure_plot_axis_lat_long_labels
from mars import grav, gamma, M_bar, R_gas, q_dot 
from convection import assign_solid_tensor, heat_flux_mask, shift_terrain_data
# for some reason, importing kintera and snapy makes the code not work due to some h5py issue
cp = gamma * R_gas / M_bar / (gamma - 1)


class Analysis_Config(TypedDict):
    flag: bool
    subplots: List[int]
    all_experiments: bool


def axes_list(analysis_dict: Analysis_Config) -> List[Axes]:
    return [analysis_dict[f"ax{i}"] for i in range(1, np.prod(analysis_dict["subplots"]) + 1)]


def configure_fig_and_axes(analysis_dict: Analysis_Config):
    fig = plt.figure()
    analysis_dict["fig"] = fig
    subplot_array_dims = analysis_dict["subplots"]
    fig.set_size_inches(max(12, 6*subplot_array_dims[1]), max(8, 4*subplot_array_dims[0]))
    for i in range(1, np.prod(subplot_array_dims) + 1):
        analysis_dict[f"ax{i}"] = fig.add_subplot(subplot_array_dims[0], subplot_array_dims[1], i)


def add_legend_labels(plot_dict: Dict[str, Analysis_Config], experiment_names: List[str]):
    print("Adding legends...")
    for key, analysis_dict in plot_dict.items():
        if analysis_dict["flag"] and analysis_dict["all_experiments"]:
            legend_labels = ["Experiment " + exp for exp in experiment_names]
            axes = axes_list(analysis_dict)
            if key == "vert_temp_theta":
                legend_labels.append("Adiabatic profile")
            elif key == "KE_flux":
                legend_labels.append("Forcing")

            for ax in axes:
                ax.legend(legend_labels)


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


def plot_vert_temp_theta(axes: List[Axes], temp: xr.Dataset, theta: xr. Dataset, linestyle: str, last: bool):
    ax1 = axes[0]
    ax1.set_title("Temp")
    ax2 = axes[1]
    ax2.set_title("Theta")

    ax1.plot(temp.mean(dim=['x2', 'x3']), temp['x1'], linestyle)
    ax2.plot(theta.mean(dim=['x2', 'x3']), theta['x1'], linestyle)

    if last:        # plot comparison line at very end
        ax1.plot(-grav / cp * temp['x1'] + 260, temp['x1'], 'k:')

    ax2.sharey(ax1)
    ax1.set_ylabel("Height (m)")


def plot_vert_vel_dist(axes: List[Axes], vel1: xr.Dataset, color: str, style: str):
    titles = ["Vz Top", "Vz Top 1/4", "Vz Middle", "Vz Bottom 1/4", "Vz Bottom"]

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
    axes[0].set_xlim([data_min, data_max])


def plot_gravity_wave(axes: List[Axes], rho: xr.Dataset, theta: xr.Dataset, linestyle: str):
    ax1 = axes[0]
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


def plot_KE_flux(axes: List[Axes], rho: xr.Dataset, vel1: xr.Dataset, vel2: xr.Dataset, vel3: xr.Dataset,
                 linestyle: str, last: bool):
    ax1 = axes[0]

    # u**2 + v**2 + w**2
    KE_flux = vel1 * 0.5 * (vel3**2 + vel2**2 + vel1**2) * rho
    mean_KE_flux = KE_flux.mean(dim=['x2', 'x3'])

    ax1.plot(mean_KE_flux, mean_KE_flux['x1'], linestyle)
    if last:
        ax1.plot(q_dot + 0 * mean_KE_flux['x1'], mean_KE_flux['x1'], 'k:')

    ax1.set_xlabel('Energy Flux (W/m^2)')
    ax1.set_ylabel('Height (m)')


def plot_KE_power(axes: List[Axes], vel1: xr.Dataset, vel2: xr.Dataset, vel3: xr.Dataset,
                  marker: str, color: str, threeD: bool, last: bool):
    # num samples
    nx1 = vel1.x1.size
    nx2 = vel1.x2.size
    nx3 = vel1.x3.size

    titles = ["KE Power Top 1/4", "KE Power Middle", "KE Power Bottom 1/4"]
    idxs = [int(nx1*3/4), int(nx1/2), int(nx1/4)]

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
                add_slope_triangle_loglog(ax, 0.75, 0.85, s, 0.15)
        ax.set_title(titles[j])
        ax.set_ylabel('Wave Power (Log Magnitude), Normalized')

    axes[-1].set_xlabel('Wavenumber (1/m)')


def plot_theta_time_series(axes: List[Axes], experiment_names: List[str], filepath_constructor, linestyles: Dict[str, List[str]]):
    titles = ["Theta Top 1/4", "Theta Middle", "Theta Bottom 1/4"]

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

    axes[-1].set_xlabel('Time (min)')


def plot_slope_winds(fig: Figure, axes: List[Axes], velx: NDArray, vely: NDArray, velz: NDArray):
    '''
    Requires that velx, vely, and velz are collapsed datasets
    '''
    ax1 = axes[0]
    ax1.set_title("Horizontal Wind")
    ax2 = axes[1]
    ax2.set_title("Vertical Wind")

    skip = 8
    velx = velx
    vely = vely
    magnitude = np.sqrt(velx**2 + vely**2)
    # add a tiny epsilon to avoid division by zero
    u_norm = velx / (magnitude + 1e-10)
    v_norm = vely / (magnitude + 1e-10)

    y_idx, x_idx = np.indices(u_norm.shape)
    q = ax1.quiver(x_idx[::skip, ::skip], y_idx[::skip, ::skip], u_norm[::skip, ::skip], v_norm[::skip, ::skip],
                   magnitude[::skip, ::skip], cmap='cividis', pivot='middle')
    ax1.set_aspect('equal')
    fig.colorbar(q, ax=ax1, label='Wind Speed (m/s)', location='bottom')

    im = ax2.imshow(velz, cmap='cividis')
    fig.colorbar(im, ax=ax2, label='Wind Velocity (m/s)', location='bottom')
    ax2.invert_yaxis()


def make_plots(plot_dict: Dict[str, Analysis_Config], experiment_names: List[str], num_files_to_avg: int,
               filepath_constructor: Callable[[str], str]):
    skip_main_loop = True
    for key, analysis_dict in plot_dict.items():
        if key != "hori_theta" and analysis_dict["flag"] and analysis_dict["all_experiments"]:
            skip_main_loop = False

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
            axes = axes_list(analysis_dict)
            match key:
                case "vert_temp_theta":
                    plot_vert_temp_theta(axes, nc_data['temp'], nc_data['theta'], linestyles["comb"][i], last)

                case "vert_vel_dist":
                    plot_vert_vel_dist(axes, nc_data['vel1'], linestyles["color"][i], linestyles["style"][i])

                case "gravity_wave":
                    plot_gravity_wave(axes, nc_data['rho'], nc_data['theta'], linestyles["comb"][i])

                case "KE_flux":
                    plot_KE_flux(axes, nc_data['rho'], nc_data['vel1'], nc_data['vel2'], nc_data['vel3'], linestyles["comb"][i], (not is_3D(exp) and last))

                case "KE_power":
                    plot_KE_power(axes, nc_data['vel1'], nc_data['vel2'], nc_data['vel3'], linestyles["marker"][i], linestyles["color"][i], is_3D(exp), last)

            fig.tight_layout()

    if (analysis_dict := plot_dict["hori_theta"])["flag"]:
        print("Loading time series...")
        axes = axes_list(analysis_dict)
        plot_theta_time_series(axes, experiment_names, filepath_constructor, linestyles)
        nc_data = averaged_exp_data(filepath_constructor(exp), num_files_to_avg)

    add_legend_labels(plot_dict, experiment_names)


def make_BL_plots(plot_dict: Dict[str, Analysis_Config], experiment_names: List[str], num_files_to_avg: int,
                  filepath_constructor: Callable[[str], str], lat_long_bounds: List[str], save_dir: str, file_index: int):
    for i, exp in enumerate(experiment_names):          # outer loop is experiment so only one set of data is loaded at a time
        print("Retrieving Topography Data")
        _, nx2, nx3 = get_num_cells_exp(exp)
        mars_data, _, _ = get_cell_topography(*lat_long_bounds, nx2, nx3)
        mars_data, _ = shift_terrain_data(torch.from_numpy(mars_data))
        print(f"Loading data from experiment {exp} ...")
        nc_data = averaged_exp_data(filepath_constructor(exp), num_files_to_avg)

        _, _, x1f = np.meshgrid(nc_data['x3f'].values[:-1], nc_data['x2f'].values[:-1], nc_data['x1f'].values[:-1], indexing="ij")
        x1f = torch.from_numpy(x1f)
        solid_tensor = assign_solid_tensor(mars_data, x1f)
        mask_torch = heat_flux_mask(solid_tensor.char()) == 1
        # this is stored as (x3, x2, x1), but nc_data is stored as (x1, x3, x2)
        mask_torch = mask_torch.permute(2, 0, 1)

        # nc_data also stores x1f, x2f, x3f as dims, but those aren't in any of the variables
        true_dims = nc_data["rho"].dims
        boundary_mask = xr.DataArray(mask_torch.numpy(), coords={k: nc_data.coords[k] for k in true_dims}, dims=true_dims)
        # keeps values where mask is True, others become NaN
        boundary_data_3d = nc_data.where(boundary_mask)
        # Since only 1 value exists per (x,y) column, 'max' or 'sum' extracts that single value
        boundary_data = boundary_data_3d.sum(dim='x1', skipna=True)
        # images are oriented with lat along y and long along x
        # by default, data is (x3, x2), which is long along y and lat long x
        velx = boundary_data['vel3'].transpose('x2', 'x3')
        vely = boundary_data['vel2'].transpose('x2', 'x3')
        velz = boundary_data['vel1'].transpose('x2', 'x3')

        for key, analysis_dict in plot_dict.items():    # inner loop is which experiments need info to be analyzed
            if not analysis_dict["flag"] or analysis_dict["all_experiments"]:
                continue
            configure_fig_and_axes(analysis_dict)
            fig: Figure = analysis_dict["fig"]
            axes = axes_list(analysis_dict)
            match key:
                case "slope_winds":
                    plot_slope_winds(fig, axes, velx.values, vely.values, velz.values)
                    for ax in axes:
                        ax.set_ylabel("Latitude -> N")
                        ax.set_xlabel("Longitude -> E")
                        configure_plot_axis_lat_long_labels(ax, *lat_long_bounds, nx2, nx3)
                        ax.contour(mars_data, 10, cmap='gray')

            fig.tight_layout()
            output_file = f"slope_winds_ss_{exp}{format_lat_long_string(*lat_long_bounds)}{file_index}.png"
            fig.savefig(f"{save_dir}/{output_file}", dpi=300)
            fig.clear()


def save_plots(plot_dict: Dict[str, Analysis_Config], save_dir: str, lat_long_str: str, file_index: str):
    # finally, save all plots
    print("Saving plots...")
    for key, value in plot_dict.items():
        # inner loop is which experiments need info to be analyzed
        if value["flag"] and value["all_experiments"]:
            fig: Figure = value["fig"]
            output_file = f"{key}_ss{lat_long_str}{file_index}.png"
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
    if (topography := args.lat_long_bounds is not None):
        lat_long_str = f"_{format_lat_long_string(*args.lat_long_bounds)}"
    else:
        lat_long_str = ""
    num_files: int = args.num_file
    filepath_constructor: Callable[[str], str] = lambda exp_name: f"{args.output_parent_dir}/output_{exp_name}{lat_long_str}"

    # make plot output directory if it doesn't already exist
    save_directory = f"output/analysis_output"
    try:
        os.mkdir(save_directory)
    except FileExistsError:
        pass

    plot_dict: Dict[str, Analysis_Config] = {}
    plot_dict["vert_temp_theta"]    = {"flag": 0, "subplots": [1, 2], "all_experiments": 1}
    plot_dict["hori_theta"]         = {"flag": 0, "subplots": [3, 1], "all_experiments": 1}
    plot_dict["vert_vel_dist"]      = {"flag": 0, "subplots": [5, 1], "all_experiments": 1}
    plot_dict["gravity_wave"]       = {"flag": 0, "subplots": [1, 1], "all_experiments": 1}
    plot_dict["KE_flux"]            = {"flag": 0, "subplots": [1, 1], "all_experiments": 1}
    plot_dict["KE_power"]           = {"flag": 0, "subplots": [1, 3], "all_experiments": 1}
    plot_dict["slope_winds"]        = {"flag": 1, "subplots": [1, 2], "all_experiments": 0}

    # configure plot size and axes
    for key, analysis_dict in plot_dict.items():
        configure_fig_and_axes(analysis_dict)

    make_plots(plot_dict, experiment_names, num_files, filepath_constructor)
    if topography:
        make_BL_plots(plot_dict, experiment_names, num_files, filepath_constructor, args.lat_long_bounds, save_directory, file_index)
    save_plots(plot_dict, save_directory, lat_long_str, file_index)


if __name__ == "__main__":
    main()
