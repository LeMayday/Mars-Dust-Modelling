# https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window
from scipy.interpolate import interpn
from typing import Tuple

file_path = "image_files/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
height_px = 23040
width_px = 46080
# m_per_px = 463.093541550370901
m_per_px = 463


def load_MOLA_DEM_data(window: Window) -> NDArray:
    # rasterio supposedly handles determining metadata and all that
    # note if using np.memmap that metadata is split across header and trailer, and that even though label says little endian, the data is big endian
    with rasterio.open(file_path) as src:
        assert src.shape[0] == height_px
        assert src.shape[1] == width_px
        mars_data = src.read(1, window=window)
    return mars_data


def lat_to_pix(input_lat):
    return (-input_lat + 90) * height_px // 180


def long_to_pix(input_long):
    return (input_long + 180) * width_px // 360


def get_region_topography(min_lat, max_lat, min_long, max_long) -> NDArray:
    '''
    lat should be [-90, 90]
    long should be [-180, 180]
    '''
    if min_lat >= max_lat or min_long >= max_long:
        raise ValueError("Min value cannot be greater than max value.")
    if min_lat < -90 or max_lat > 90:
        raise ValueError("Lat bounds must be in [-90, 90].")
    if min_long < -180 or max_long > 180:
        raise ValueError("Long bounds must be in [-180, 180].")

    # (max_lat, min_lat) is right for viewing, but not for data being stored in order of increasing latitude
    # need to window (max_lat, min_lat) since image has (0,0) in top left
    w = Window.from_slices((lat_to_pix(max_lat), lat_to_pix(min_lat)), (long_to_pix(min_long), long_to_pix(max_long)))
    mars_data = load_MOLA_DEM_data(w)
    # flip about lat axis so min lat is at pos 0
    mars_data = np.flip(mars_data, axis=0)
    return mars_data


def get_cell_topography(min_lat, max_lat, min_long, max_long, num_cells_lat, num_cells_long) -> Tuple[NDArray, int, int]:
    '''
    returns interpolated values plus (lat, long) dimensions in m
    '''
    mars_data = get_region_topography(min_lat, max_lat, min_long, max_long)
    num_pixels_lat = mars_data.shape[0]
    num_pixels_long = mars_data.shape[1]

    lat_idx = np.arange(num_pixels_lat)     # get indices of data pts in lat (y)
    long_idx = np.arange(num_pixels_long)    # get indices of data pts in long (x)

    lat_sample = (np.arange(num_cells_lat) + 0.5) / num_cells_lat * num_pixels_lat      # sample for num_cells_lat boxes in the center
    long_sample = (np.arange(num_cells_long) + 0.5) / num_cells_long * num_pixels_long   # sample for num_cells_long boxes in the center
    Lats, Longs = np.meshgrid(lat_sample, long_sample, indexing='ij')
    sample_pts = np.vstack([Lats.ravel(), Longs.ravel()]).T
    cell_data = interpn((lat_idx, long_idx), mars_data, sample_pts, method='splinef2d')
    cell_data = cell_data.reshape((num_cells_lat, num_cells_long))

    return cell_data, m_per_px * num_pixels_lat, m_per_px * num_pixels_long 


def main():
    import matplotlib.pyplot as plt
    min_lat = -35
    max_lat = -25
    min_long = 70
    max_long = 80

    # reference region against https://oderest.rsl.wustl.edu/GDSWeb/GDSMOLAPEDR.html
    mars_data_true = get_region_topography(min_lat, max_lat, min_long, max_long)
    mars_data_false, _, _ = get_cell_topography(min_lat, max_lat, min_long, max_long, 256, 256)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    im1 = ax1.imshow(mars_data_true, cmap='gray')
    fig.colorbar(im1, ax=ax1)
    ax1.invert_yaxis()      # image format has (0,0) at the top left, but want top to be max lat
    im2 = ax2.imshow(mars_data_false, cmap='gray')
    fig.colorbar(im2, ax=ax2)
    ax2.invert_yaxis()
    fig.savefig("data_import_test.png", dpi=300)


if __name__ == "__main__":
    main()
