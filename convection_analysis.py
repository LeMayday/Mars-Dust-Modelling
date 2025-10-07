# script to analyze netcdf output of convection.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from movie_from_pngs import delete_files, create_movie
from plotting import *
import os

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

experiment_name = input("Experiment Name:\n")
working_dir = f"output_{experiment_name}"
nc2_files, nc3_files = get_nc_files(working_dir)
ref_data = xr.open_dataset(nc2_files[0]).isel(time=0)
X, Y, Z = np.meshgrid(ref_data['x2'], ref_data['x3'], ref_data['x1'])

plot_temp_flag = 0
filenames1 = []

plot_vert_vel_flag = 1
filenames2 = []

plot_hor_vel_flag = 1
filenames3 = []

plot_vel_vector_flag = 0
filenames4 = []

# nc2_files and nc3_files should have the same size
for n in range(len(nc2_files)):
    print(f"Reading file {n + 1}")
    data2 = xr.open_dataset(nc2_files[n]).isel(time=0)
    data3 = xr.open_dataset(nc3_files[n]).isel(time=0)
    current_time = data2['time']

    if plot_temp_flag:
        fig1 = plt.figure()
        fig1.set_size_inches(12, 8)
        ax1 = fig1.add_subplot(1, 2, 1)
        ax2 = fig1.add_subplot(1, 2, 2)

        temp_data = data3['temp']
        plot_func(temp_data.mean(dim=['x2', 'x3']), temp_data['x1'], current_time, ax = ax1)
        ax1.set_title("Temp")

        theta_data = data3['theta']
        plot_func(theta_data.mean(dim=['x2', 'x3']), theta_data['x1'], current_time, ax = ax2)
        ax2.set_title("Theta")
        fig1.tight_layout()

        output_file = f"fig1_frame_{n}.png"
        fig1.savefig(output_file)
        plt.close(fig1)
        filenames1.append(output_file)

    if plot_vert_vel_flag:
        fig2 = plt.figure()
        fig2.set_size_inches(12, 8)
        ax1 = fig2.add_subplot(2, 1, 1)
        ax2 = fig2.add_subplot(2, 1, 2)

        vel_data_bottom = data2['vel1'].isel(x1=0).transpose()
        vel_data_top = data2['vel1'].isel(x1=-1).transpose()
        vmin = min(float(vel_data_bottom.min()), float(vel_data_top.min()))
        vmax = min(float(vel_data_bottom.max()), float(vel_data_top.max()))

        im = ax1.imshow(vel_data_bottom, cmap='cividis', aspect='auto', vmin=vmin, vmax=vmax)
        ax1.text(1, 1, f"{float((vel_data_bottom > 0).sum())/vel_data_bottom.size*100:.1f}% +, Time: {current_time:.2f}", transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='right')
        # cbar = fig2.colorbar(im, ax=ax1, orientation='vertical')
        ax1.set_title("Vz Bottom")

        im = ax2.imshow(vel_data_top, cmap='cividis', aspect='auto', vmin=vmin, vmax=vmax)
        ax2.text(1, 1, f"{float((vel_data_top > 0).sum())/vel_data_top.size*100:.1f}% +, Time: {current_time:.2f}", transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='right')
        # cbar = fig2.colorbar(im, ax=ax2, orientation='vertical')
        ax2.set_title("Vz Top")
        fig2.tight_layout()
        fig2.colorbar(im, ax=[ax1, ax2], orientation='vertical')

        output_file = f"fig2_frame_{n}.png"
        fig2.savefig(output_file)
        plt.close(fig2)
        filenames2.append(output_file)

    if plot_hor_vel_flag:
        fig3 = plt.figure()
        fig3.set_size_inches(12, 8)
        ax1 = fig3.add_subplot(1, 1, 1)
        
        hor_vel_data = data2['vel2']
        plot_func(hor_vel_data.mean(dim=['x2', 'x3']), hor_vel_data['x1'], current_time, ax = ax1)
        ax1.set_title('Mean Horizontal Vel')
        fig3.tight_layout()

        output_file = f"fig3_frame_{n}.png"
        fig3.savefig(output_file)
        plt.close(fig3)
        filenames3.append(output_file)

    if plot_vel_vector_flag:
        fig4 = plt.figure()
        fig4.set_size_inches(12, 8)
        ax1 = fig4.add_subplot(1, 1, 1)

        vert_vel_data = data2['vel1']
        hor_vel_data = data2['vel2']

        plot_2D_vectors(X, Z, hor_vel_data.isel(x3 = 0).transpose(), vert_vel_data.isel(x3 = 0).transpose(), current_time, ax=ax1)
        fig4.tight_layout()

        output_file = f"fig4_frame_{n}.png"
        fig4.savefig(output_file)
        plt.close(fig4)
        filenames4.append(output_file)

    data2.close()
    data3.close()

if plot_temp_flag:
    create_movie(filenames1, f"vertical_temp_theta_exp{experiment_name}.mp4")
    delete_files(filenames1)
if plot_vert_vel_flag:
    create_movie(filenames2, f"vertical_vel_top_bottom_exp{experiment_name}.mp4")
    delete_files(filenames2)
if plot_hor_vel_flag:
    create_movie(filenames3, f"horizontal_vel_exp{experiment_name}.mp4")
    delete_files(filenames3)
if plot_vel_vector_flag:
    create_movie(filenames4, f"vel_vector_exp{experiment_name}.mp4")
    delete_files(filenames4)

