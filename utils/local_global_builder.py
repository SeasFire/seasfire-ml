import logging
import os
import numpy as np
import torch
import xarray as xr
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class LocalGlobalBuilder:
    def __init__(
        self,
        cube,
        cube_resolution,
        input_vars,
        oci_input_vars,
        target_var,
        output_folder,
        radius,
        timeseries_weeks,
        target_count,
        global_scale_factor,
    ):
        self._input_vars = input_vars
        self._oci_input_vars = oci_input_vars
        self._target_var = target_var

        self._cube = cube
        if cube_resolution == "25km":
            self._sp_res = 0.25
            self._lat_min = -89.875
            self._lat_max = 89.875
            self._lon_min = -179.875
            self._lon_max = 179.875
        elif cube_resolution == "100km":
            self._sp_res = 1
            self._lat_min = -89.5
            self._lat_max = 89.5
            self._lon_min = -179.5
            self._lon_max = 179.5
        else:
            raise ValueError("Invalid cube resolution")

        # create output cache folder
        self._cache_folder = os.path.join(output_folder, "cache")
        for folder in [self._cache_folder]:
            logger.info("Creating cache folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        self._radius = radius
        self._global_scale_factor = global_scale_factor
        self._timeseries_weeks = timeseries_weeks
        self._target_count = target_count

    @property
    def cube(self):
        return self._cube

    @property
    def target_var(self):
        return self._target_var

    def _create_local_data(self, lat, lon, time, radius):
        logger.debug(
            "Creating local data for lat={}, lon={}, time={}".format(lat, lon, time)
        )
        # find time in time coords
        time_idx = np.where(self._cube["time"] == time)[0][0]
        time_slice = slice(time_idx - self._timeseries_weeks + 1, time_idx + 1)

        lat_slice = slice(lat + radius * self._sp_res, lat - radius * self._sp_res)
        lon_slice = slice(lon - radius * self._sp_res, lon + radius * self._sp_res)

        data = (
            self._cube[self._input_vars + self._oci_input_vars]
            .sel(latitude=lat_slice, longitude=lon_slice)
            .isel(time=time_slice)
            .load()
        )
        data = data.transpose("latitude", "longitude", "time")

        # lat_values = data["latitude"].values
        # logger.info("lat values={}".format(lat_values))
        # lon_values = data["longitude"].values
        # logger.info("lon values={}".format(lon_values))
        # values = data.transpose('latitude', 'longitude', 'time').to_array().values
        # logger.info("values={}".format(values))

        return data

    def _create_global_data(self, time):
        result = self._read_from_cache(key="global_{}".format(time))
        if result is not None:
            return result

        global_region = self._cube
        lat_target = len(global_region.coords["latitude"]) // self._global_scale_factor
        lon_target = len(global_region.coords["longitude"]) // self._global_scale_factor
        logger.debug("Global view dimensions = ({},{})".format(lat_target, lon_target))
        global_agg = global_region.coarsen(
            latitude=lat_target, longitude=lon_target
        ).mean(skipna=True)

        # find center_time in time coords
        time_idx = np.where(global_region["time"] == time)[0][0]
        time_slice = slice(time_idx - self._timeseries_weeks + 1, time_idx + 1)

        data = (
            global_agg[self._input_vars + self._oci_input_vars]
            .isel(time=time_slice)
            .load()
        )

        timeseries_len = len(data.coords["time"])
        if timeseries_len != self._timeseries_weeks:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._timeseries_weeks
                )
            )
            raise ValueError("Invalid time series length")

        data = data.transpose("latitude", "longitude", "time")

        self._write_to_cache(key="global_{}".format(time), data=data)
        return data

    def _compute_area(self, lat, lon):
        area = self._cube["area"].sel(latitude=lat, longitude=lon)
        area_in_hectares = area / 10000.0
        # area_in_hectares = area.values / 10000.0
        return area_in_hectares

    def _compute_ground_truth(self, lat, lon, time):
        # find time in time coords
        time_idx = np.where(self._cube["time"] == time)[0][0]
        time_slice = slice(time_idx + 1, time_idx + 1 + self._target_count)
        logger.debug(
            "Computing ground truth for lat={}, lon={}, time={}, time_slice={}".format(
                lat, lon, time, time_slice
            )
        )

        target = (
            self._cube[self._target_var]
            .sel(
                latitude=lat,
                longitude=lon,
            )
            .isel(time=time_slice)
        )

        timeseries_len = len(target.coords["time"])
        if timeseries_len != self._target_count:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._target_count
                )
            )
            raise ValueError("Invalid time series length")

        logger.debug("Ground truth target values={}".format(target.values))
        return target

    def create(self, lat, lon, time):
        logger.debug("Creating data for lat={}, lon={}, time={}".format(lat, lon, time))

        local_dataset = self._create_local_data(lat, lon, time, self._radius)
        global_dataset = self._create_global_data(time)
        ground_truth_dataset = self._compute_ground_truth(lat, lon, time)
        area_dataset = self._compute_area(lat, lon)

        return local_dataset, global_dataset, ground_truth_dataset, area_dataset

    def _read_from_cache(self, key):
        try:
            return torch.load(
                os.path.join(self._cache_folder, "cache_item_{}.pt".format(key))
            )
        except FileNotFoundError:
            return None

    def _write_to_cache(self, key, data):
        output_path = os.path.join(self._cache_folder, "cache_item_{}.pt".format(key))
        torch.save(data, output_path)
