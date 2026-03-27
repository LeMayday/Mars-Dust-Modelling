# https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window
from scipy.interpolate import interpn
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import yaml
from dataclasses import dataclass

file_path = "image_files/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
height_px = 23040
width_px = 46080
# m_per_px = 463.093541550370901
m_per_px = 463


@dataclass(frozen=True)
class Region():
    min_lat: int
    max_lat: int
    min_long: int
    max_long: int

    def __post_init__(self):
        '''
        lat should be [-90, 90]
        long should be [-180, 180]
        '''
        if self.min_lat >= self.max_lat or self.min_long >= self.max_long:
            raise ValueError("Min value cannot be greater than max value.")
        if self.min_lat < -90 or self.max_lat > 90:
            raise ValueError("Lat bounds must be in [-90, 90].")
        if self.min_long < -180 or self.max_long > 180:
            raise ValueError("Long bounds must be in [-180, 180].")

    def to_string(self) -> str:
        conv_lat = lambda lat: f"{abs(lat)}" + ("N" if lat >= 0 else "S")
        conv_long = lambda lon: f"{abs(lon)}" + ("W" if lon >= 0 else "E")
        return (f"{conv_lat(self.min_lat)}{conv_long(self.min_long)}"
                f"{conv_lat(self.max_lat)}{conv_long(self.max_long)}")


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


def get_region_topography(region: Region) -> NDArray:
    # (max_lat, min_lat) is right for viewing, but not for data being stored in order of increasing latitude
    # need to window (max_lat, min_lat) since image has (0,0) in top left
    w = Window.from_slices((lat_to_pix(region.max_lat), lat_to_pix(region.min_lat) + 1),
                           (long_to_pix(region.min_long), long_to_pix(region.max_long) + 1))
    mars_data = load_MOLA_DEM_data(w)
    # flip about lat axis so min lat is at pos 0
    mars_data = np.flip(mars_data, axis=0)
    return mars_data


def get_cell_topography(region: Region, num_cells_lat, num_cells_long) -> Tuple[NDArray, int, int]:
    '''
    returns interpolated values plus (lat, long) dimensions in m
    '''
    mars_data = get_region_topography(region)
    num_pixels_lat = mars_data.shape[0]
    num_pixels_long = mars_data.shape[1]

    lat_idx = np.arange(num_pixels_lat)         # get indices of data pts in lat (y)
    long_idx = np.arange(num_pixels_long)       # get indices of data pts in long (x)

    lat_sample = (np.arange(num_cells_lat) + 0.5) / num_cells_lat * (num_pixels_lat - 1)        # sample for num_cells_lat boxes in the center
    long_sample = (np.arange(num_cells_long) + 0.5) / num_cells_long * (num_pixels_long - 1)    # sample for num_cells_long boxes in the center
    Lats, Longs = np.meshgrid(lat_sample, long_sample, indexing='ij')
    sample_pts = np.vstack([Lats.ravel(), Longs.ravel()]).T
    cell_data = interpn((lat_idx, long_idx), mars_data, sample_pts, method='splinef2d')
    cell_data = cell_data.reshape((num_cells_lat, num_cells_long))

    return cell_data, m_per_px * num_pixels_lat, m_per_px * num_pixels_long 


def get_mars_data_from_yaml_config(input_file: str) -> NDArray | None:
    with open(input_file, "r", encoding="utf-8") as stream:
        config: dict = yaml.safe_load(stream)
    region_info = config.get("region")
    if region_info is None:
        return None
    min_lat = float(region_info["min-lat"])
    max_lat = float(region_info["max-lat"])
    min_long = float(region_info["min-long"])
    max_long = float(region_info["max-long"])
    nx2 = config["geometry"]["cells"]["nx2"]
    nx3 = config["geometry"]["cells"]["nx3"]
    mars_data, Dx2, Dx3 = get_cell_topography(Region(min_lat, max_lat, min_long, max_long), nx2, nx3)
    Dx2_yaml = config["geometry"]["bounds"]["x2max"] - config["geometry"]["bounds"]["x2min"]
    Dx3_yaml = config["geometry"]["bounds"]["x3max"] - config["geometry"]["bounds"]["x3min"]
    assert Dx2 == Dx2_yaml and Dx3 == Dx3_yaml, "Domain size does not match region size."
    return mars_data


def configure_plot_axis_lat_long_labels(ax: Axes, region: Region, num_cells_lat: int, num_cells_long):
    xtick_locs = np.linspace(0, num_cells_long - 1, 5)
    ytick_locs = np.linspace(0, num_cells_lat - 1, 5)
    ax.set_xticks(xtick_locs); ax.set_yticks(ytick_locs)
    xtick_labels = xtick_locs / num_cells_long * (region.max_long - region.min_long) + region.min_long
    ytick_labels = ytick_locs / num_cells_lat * (region.max_lat - region.min_lat) + region.min_lat
    ax.set_xticklabels([f'{x:.1f}' for x in xtick_labels])
    ax.set_yticklabels([f'{y:.1f}' for y in ytick_labels])


def main():
    min_lat = -31 #-35
    max_lat = -29 #-25
    min_long = 74 #70
    max_long = 76 #80

    # reference region against https://oderest.rsl.wustl.edu/GDSWeb/GDSMOLAPEDR.html
    region = Region(min_lat, max_lat, min_long, max_long)
    mars_data_true = get_region_topography(region)
    mars_data_false, _, _ = get_cell_topography(region, 256, 256)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    im1 = ax1.imshow(mars_data_true, cmap='gray')
    fig.colorbar(im1, ax=ax1)
    ax1.invert_yaxis()      # image format has (0,0) at the top left, but want top to be max lat
    im2 = ax2.imshow(mars_data_false, cmap='gray')
    fig.colorbar(im2, ax=ax2)
    ax2.invert_yaxis()
    fig.savefig("output/data_import_test.png", dpi=300)


if __name__ == "__main__":
    main()
