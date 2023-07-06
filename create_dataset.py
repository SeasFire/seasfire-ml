#!/usr/bin/env python3

import argparse
import logging
import os
import xarray as xr
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(
        self,
        cube_path,
        cube_resolution,
        global_cube_path,
        load_cube_in_memory,
        output_folder,
        split,
        radius,
        positive_samples_threshold,
        positive_samples_size,
        generate_all_samples,
        seed,
        timeseries_weeks,
        target_count,
        include_oci_variables,
        global_scale_factor,
    ):
        self._log_preprocess_input_vars = ["tp", "pop_dens"]
        self._input_vars = [
            "mslp",
            "tp",
            "vpd",
            "sst",
            "t2m_mean",
            "ssrd",
            "swvl1",
            "lst_day",
            "ndvi",
            "pop_dens",
        ]
        self._oci_input_vars = (
            [
                "oci_wp",
                "oci_pna",
                "oci_nao",
                "oci_soi",
                "oci_gmsst",
                "oci_pdo",
                "oci_ea",
                "oci_epo",
                "oci_nina34_anom",
                "oci_censo",
            ]
            if include_oci_variables
            else []
        )

        # one of gwis_ba, BurntArea, frpfire, co2fire, FCCI_BA, co2fire
        self._target_var = "gwis_ba"
        logger.info("Using target variable: {}".format(self._target_var))

        self._ndvi_var = "ndvi"
        logger.info("Using ndvi variable: {}".format(self._ndvi_var))

        logger.info("Using seed: {}".format(seed))
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        # validate split
        if split not in ["train", "test", "val"]:
            raise ValueError("Wrong split type")
        self._split = split
        logger.info("Using split: {}".format(self._split))

        # create output split folder
        self._output_folder = os.path.join(output_folder, self._split)
        for folder in [self._output_folder]:
            logger.info("Creating output folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        logger.info("Cube path={}".format(cube_path))
        self._cube_path = cube_path
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
        logger.info("Using cube resolution: {}".format(cube_resolution))

        self._global_sp_res = 1
        self._global_lat_min = -89.5
        self._global_lat_max = 89.5
        self._global_lon_min = -179.5
        self._global_lon_max = 179.5

        # open global cube if present
        self._global_cube = None
        if global_cube_path is None:
            if cube_resolution == "25km":
                raise ValueError("Global cube not found")
        else:
            logger.info("Opening global cube zarr file: {}".format(global_cube_path))
            self._global_cube = xr.open_zarr(global_cube_path, consolidated=False)
            if load_cube_in_memory:
                logger.info("Loading global cube into memory")
                self._global_cube.load()

            for var_name in self._log_preprocess_input_vars:
                logger.info("Log-transforming input var: {}".format(var_name))
                self._global_cube[var_name] = xr.DataArray(
                    np.log(1.0 + self._global_cube[var_name].values),
                    coords=self._global_cube[var_name].coords,
                    dims=self._global_cube[var_name].dims,
                    attrs=self._global_cube[var_name].attrs,
                )

        # open local cube and display basic info
        logger.info("Opening local cube zarr file: {}".format(self._cube_path))
        self._cube = xr.open_zarr(self._cube_path, consolidated=False)
        if load_cube_in_memory:
            logger.info("Loading the whole cube in memory.")
            self._cube.load()

        if self._global_cube is None:
            logger.info("Using local cube as global")
            self._global_cube = self._cube

        logger.info("Cube: {}".format(self._cube))
        logger.info("Vars: {}".format(self._cube.data_vars))

        for var_name in self._log_preprocess_input_vars:
            logger.info("Log-transforming input var: {}".format(var_name))
            self._cube[var_name] = xr.DataArray(
                np.log(1.0 + self._cube[var_name].values),
                coords=self._cube[var_name].coords,
                dims=self._cube[var_name].dims,
                attrs=self._cube[var_name].attrs,
            )

        for input_var in self._input_vars:
            logger.debug(
                "Var name {}, description: {}".format(
                    input_var, self._cube[input_var].description
                )
            )

        for oci_var in self._oci_input_vars:
            logger.debug(
                "Oci name {}, description: {}".format(
                    oci_var, self._cube[oci_var].description
                )
            )

        # Maximum radius supported for grid graph
        self._radius = radius
        logger.info("Will add max radius padding={} around area".format(self._radius))

        self._global_scale_factor = global_scale_factor
        logger.info("Using global scale factor={}".format(self._global_scale_factor))

        # Threshold for fires
        self._positive_samples_threshold = positive_samples_threshold
        # How many samples to take above the threshold
        self._positive_samples_size = positive_samples_size
        # Whether to generate all samples
        self._generate_all_samples = generate_all_samples

        self._number_of_train_years = 16
        self._days_per_week = 8
        self._timeseries_weeks = timeseries_weeks  # 12 months before = 48 weeks
        self._year_in_weeks = 48

        self._max_week_with_data = 918
        logger.info(
            "Maximum week with valid data = {}".format(self._max_week_with_data)
        )

        # how many targets periods to generate in the future, each period is one week
        self._target_count = target_count

        logger.info(
            "Will generate {} target periods (weeks).".format(self._target_count)
        )

        # split time periods
        self._time_train = (
            self._timeseries_weeks,
            self._year_in_weeks * self._number_of_train_years - self._target_count,
        )
        logger.info("Train time in weeks: {}".format(self._time_train))

        self._time_val = (
            self._year_in_weeks * self._number_of_train_years + self._timeseries_weeks,
            self._year_in_weeks * self._number_of_train_years
            + 2 * self._timeseries_weeks,
        )
        logger.info("Val time in weeks: {}".format(self._time_val))

        self._time_test = (
            self._year_in_weeks * self._number_of_train_years
            + 2 * self._timeseries_weeks,
            self._max_week_with_data - self._target_count,
        )
        logger.info("Test time in weeks: {}".format(self._time_test))

        if self._split == "train":
            self._start_time, self._end_time = self._time_train
        elif self._split == "val":
            self._start_time, self._end_time = self._time_val
        elif self._split == "test":
            self._start_time, self._end_time = self._time_test
        else:
            raise ValueError("Invalid split type")

    def _try_sample_wrt_threshold(
        self,
        sample_region,
        sample_region_target_var_per_area,
        sample_region_ndvi,
        strategy,
        threshold,
        sample_size,
        increase_threshold = False,
    ):
        cur_sample_size = sample_size
        cur_threshold = threshold
        idx = 0

        while True:
            if idx > 50 or cur_threshold <= 0.0:
                logger.warning("Failed to find proper threshold") 
                return sample
            
            sample = self._sample_wrt_threshold(
                sample_region,
                sample_region_target_var_per_area,
                sample_region_ndvi,
                strategy,
                cur_threshold,
                cur_sample_size,
            )
            if len(sample) >= sample_size: 
                return sample

            if increase_threshold: 
                cur_threshold *= 2
            else:
                cur_threshold /= 2
            idx += 1 


    def _sample_wrt_threshold(
        self,
        sample_region,
        sample_region_target_var_per_area,
        sample_region_ndvi,
        strategy,
        threshold,
        sample_size,
    ):
        logger.info(
            "Sampling for strategy={}, sample size={} and threshold={}".format(strategy, sample_size, threshold)
        )
        if strategy == "above-threshold":
            sample_region_target_var_per_area_wrt_threshold = (
                sample_region_target_var_per_area > threshold
            ) & (~np.isnan(sample_region_ndvi))
        elif strategy == "positive-below-threshold":
            sample_region_target_var_per_area_wrt_threshold = (
                (sample_region_target_var_per_area <= threshold)
                & (sample_region_target_var_per_area > 0.0)
                & (~np.isnan(sample_region_ndvi))
            )
        elif strategy == "zero":
            sample_region_target_var_per_area_wrt_threshold = (
                sample_region_target_var_per_area <= 0.0
            ) & (~np.isnan(sample_region_ndvi))
        else:
            raise ValueError("Invalid strategy")

        sample_len_wrt_threshold = (
            sample_region_target_var_per_area_wrt_threshold.sum().values
        )
        logger.info(
            "Samples for strategy={} are {}".format(strategy, sample_len_wrt_threshold)
        )

        result = []
        all_wrt_threshold_samples_index = np.argwhere(
            sample_region_target_var_per_area_wrt_threshold.values
        )
        if sample_size > len(all_wrt_threshold_samples_index):
            logger.warning(
                "Cannot sample {} from {} elements. Will include all.".format(
                    sample_size, len(all_wrt_threshold_samples_index)
                )
            )

        if sample_size > len(all_wrt_threshold_samples_index):
            wrt_threshold_samples_index = all_wrt_threshold_samples_index
        else:
            wrt_threshold_samples_index = self._rng.choice(
                all_wrt_threshold_samples_index,
                size=sample_size,
                replace=False,
            )

        for index in wrt_threshold_samples_index:
            result.append(
                (
                    sample_region.latitude.values[index[1]],
                    sample_region.longitude.values[index[2]],
                    sample_region.time.values[index[0]],
                ),
            )
        return result

    def _all_wrt_threshold(
        self,
        sample_region,
        sample_region_target_var_per_area,
        sample_region_ndvi,
        strategy,
    ):
        if strategy == "above-threshold":
            sample_region_target_var_per_area_wrt_threshold = (
                sample_region_target_var_per_area > self._positive_samples_threshold
            ) & (~np.isnan(sample_region_ndvi))
        elif strategy == "positive-below-threshold":
            sample_region_target_var_per_area_wrt_threshold = (
                (sample_region_target_var_per_area <= self._positive_samples_threshold)
                & (sample_region_target_var_per_area > 0.0)
                & (~np.isnan(sample_region_ndvi))
            )
        elif strategy == "zero":
            sample_region_target_var_per_area_wrt_threshold = (
                sample_region_target_var_per_area <= 0.0
            ) & (~np.isnan(sample_region_ndvi))
        else:
            raise ValueError("Invalid strategy")

        sample_len_wrt_threshold = (
            sample_region_target_var_per_area_wrt_threshold.sum().values
        )
        logger.info(
            "Samples for strategy={} are {}".format(strategy, sample_len_wrt_threshold)
        )

        result = []
        all_wrt_threshold_samples_index = np.argwhere(
            sample_region_target_var_per_area_wrt_threshold.values
        )

        for index in all_wrt_threshold_samples_index:
            result.append(
                (
                    sample_region.latitude.values[index[1]],
                    sample_region.longitude.values[index[2]],
                    sample_region.time.values[index[0]],
                ),
            )
        return result

    def _generate_all_samples_lists(
        self, min_lon, min_lat, max_lon, max_lat, target_week
    ):
        # define sample region
        sample_region = self._cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        ).isel(time=slice(self._start_time, self._end_time))

        # find target variable in region
        sample_region_target_var = sample_region[self._target_var].fillna(0)
        sample_region_ndvi = sample_region[self._ndvi_var]

        # compute area in sample region and add time dimension
        sample_region_area = (
            self._cube["area"]
            .sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            .expand_dims(dim={"time": sample_region.time}, axis=0)
        )

        # Convert area in hectars
        sample_region_area = sample_region_area / 10000.0

        # Shift target var by amount of target in the future
        sample_region_target_var_shifted = sample_region_target_var.shift(
            time=-target_week, fill_value=0
        )

        # compute target variable per area
        sample_region_target_var_per_area = (
            sample_region_target_var_shifted / sample_region_area
        )

        above_threshold_samples_list = self._all_wrt_threshold(
            sample_region,
            sample_region_target_var_per_area,
            sample_region_ndvi,
            strategy="above-threshold",
        )

        below_threshold_samples_list = self._all_wrt_threshold(
            sample_region,
            sample_region_target_var_per_area,
            sample_region_ndvi,
            strategy="positive-below-threshold",
        )

        zero_threshold_samples_list = self._all_wrt_threshold(
            sample_region,
            sample_region_target_var_per_area,
            sample_region_ndvi,
            strategy="zero",
        )

        return (
            above_threshold_samples_list,
            below_threshold_samples_list,
            zero_threshold_samples_list,
        )

    def _generate_threshold_samples_lists(
        self, min_lon, min_lat, max_lon, max_lat, target_week
    ):
        # define sample region
        sample_region = self._cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        ).isel(time=slice(self._start_time, self._end_time))

        # find target variable in region
        sample_region_target_var = sample_region[self._target_var].fillna(0)
        sample_region_ndvi = sample_region[self._ndvi_var]

        # compute area in sample region and add time dimension
        sample_region_area = (
            self._cube["area"]
            .sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            .expand_dims(dim={"time": sample_region.time}, axis=0)
        )

        # Convert area in hectars
        sample_region_area = sample_region_area / 10000.0

        # Shift target var by amount of target in the future
        sample_region_target_var_shifted = sample_region_target_var.shift(
            time=-target_week, fill_value=0
        )

        # compute target variable per area
        sample_region_target_var_per_area = (
            sample_region_target_var_shifted / sample_region_area
        )

        above_threshold_samples_list = self._try_sample_wrt_threshold(
            sample_region,
            sample_region_target_var_per_area,
            sample_region_ndvi,
            strategy="above-threshold",
            threshold=self._positive_samples_threshold,
            sample_size=self._positive_samples_size,
        )

        below_threshold_samples_list = self._try_sample_wrt_threshold(
            sample_region,
            sample_region_target_var_per_area,
            sample_region_ndvi,
            strategy="positive-below-threshold",
            threshold=self._positive_samples_threshold,
            sample_size=self._positive_samples_size,
            increase_threshold=True
        )

        zero_threshold_samples_list = self._sample_wrt_threshold(
            sample_region,
            sample_region_target_var_per_area,
            sample_region_ndvi,
            strategy="zero",
            threshold=self._positive_samples_threshold,
            sample_size=2 * self._positive_samples_size,
        )

        return (
            above_threshold_samples_list,
            below_threshold_samples_list,
            zero_threshold_samples_list,
        )

    def _generate_samples_lists(self, min_lon, min_lat, max_lon, max_lat, target_week):
        logger.info("Generating sample list for split={}".format(self._split))
        logger.info(
            "Generating from week={} to week={} using target week={}".format(
                self._start_time, self._end_time, target_week
            )
        )

        if self._generate_all_samples:
            logger.info("Will generate all samples")
            return self._generate_all_samples_lists(
                min_lon, min_lat, max_lon, max_lat, target_week
            )
        else:
            logger.info("Will samples based on threshold")
            return self._generate_threshold_samples_lists(
                min_lon, min_lat, max_lon, max_lat, target_week
            )

    def _create_local_data(self, min_lat, min_lon, max_lat, max_lon):
        # add radius padding around location of interest
        lat_slice = slice(
            max_lat + self._radius * self._sp_res, min_lat - self._radius * self._sp_res
        )
        lon_slice = slice(
            min_lon - self._radius * self._sp_res, max_lon + self._radius * self._sp_res
        )
        data = self._cube[self._input_vars + self._oci_input_vars].sel(
            latitude=lat_slice, longitude=lon_slice
        )
        return data

    def _create_global_data(self):
        global_region = self._global_cube
        lat_target = len(global_region.coords["latitude"]) // self._global_scale_factor
        lon_target = len(global_region.coords["longitude"]) // self._global_scale_factor
        logger.debug("Global view dimensions = ({},{})".format(lat_target, lon_target))
        global_agg = global_region.coarsen(
            latitude=lat_target, longitude=lon_target
        ).mean(skipna=True)

        data = global_agg[self._input_vars + self._oci_input_vars]
        data = data.transpose("latitude", "longitude", "time")
        return data

    def _create_stats(self, data, name):
        stats_filename = "{}/mean_std_stats_{}.pk".format(self._output_folder, name)
        if not os.path.exists(stats_filename):
            features = self._input_vars + self._oci_input_vars
            mean_std = np.zeros((len(features), 2))
            for idx, var_name in enumerate(features):
                logger.info("Computing mean-std for variable={}".format(var_name))
                mean_std[idx] = [
                    data[var_name].mean().values,
                    data[var_name].std().values,
                ]
                logger.info(
                    "mean-std for variable={} are={}".format(var_name, mean_std[idx])
                )
            logger.debug("mean-std={}".format(mean_std))
            torch.save(
                mean_std, "{}/mean_std_stats_{}.pk".format(self._output_folder, name)
            )
        else:
            logger.info(
                "Skipping {} features stats computation. Found file: {}".format(
                    name, stats_filename
                )
            )

    def _create_area_data(self):
        area = self._cube["area"]
        area_in_hectares = area / 10000.0
        return area_in_hectares

    def _create_target_var_data(self, min_lat, min_lon, max_lat, max_lon):
        # add radius padding around area of interest
        lat_slice = slice(
            max_lat + self._radius * self._sp_res, min_lat - self._radius * self._sp_res
        )
        lon_slice = slice(
            min_lon - self._radius * self._sp_res, max_lon + self._radius * self._sp_res
        )
        data = self._cube[self._target_var].sel(latitude=lat_slice, longitude=lon_slice)
        return data

    def run(self, area):
        min_lat, min_lon, max_lat, max_lon = area

        logger.info(
            "Running for min_lat={}, min_lon={}, max_lat={}, max_lon={}".format(
                min_lat, min_lon, max_lat, max_lon
            )
        )

        logger.info("Creating global dataset")
        global_dataset = self._create_global_data()
        global_latlon_shape = (
            global_dataset["latitude"].shape[0],
            global_dataset["longitude"].shape[0],
        )
        self._write_dataset_to_disk(global_dataset, "global")
        self._create_stats(global_dataset, "global")

        logger.info("Creating local dataset")
        local_dataset = self._create_local_data(min_lat, min_lon, max_lat, max_lon)
        self._write_dataset_to_disk(local_dataset, "local")
        self._create_stats(local_dataset, "local")

        logger.info("Creating area data")
        area_dataset = self._create_area_data()
        self._write_dataset_to_disk(area_dataset, "area")

        logger.info("Creating ground truth data")
        ground_truth_dataset = self._create_target_var_data(
            min_lat, min_lon, max_lat, max_lon
        )
        self._write_dataset_to_disk(ground_truth_dataset, "ground_truth")

        logger.info("Creating samples index")
        for target_week in range(1, self._target_count + 1):
            logger.info("Creating from target week={}".format(target_week))
            # create list of samples to generate
            (
                gt_threshold_samples,
                le_threshold_samples,
                zero_threshold_samples,
            ) = self._generate_samples_lists(
                min_lon=min_lon,
                min_lat=min_lat,
                max_lon=max_lon,
                max_lat=max_lat,
                target_week=target_week,
            )
            logger.info(
                "About to create {} samples above the threshold".format(
                    len(gt_threshold_samples)
                )
            )
            self._write_data_to_disk(
                gt_threshold_samples, "t{}_gt_threshold_samples".format(target_week)
            )

            logger.info(
                "About to create {} samples below the threshold".format(
                    len(le_threshold_samples)
                )
            )
            self._write_data_to_disk(
                le_threshold_samples, "t{}_le_threshold_samples".format(target_week)
            )

            logger.info(
                "About to create {} zero samples".format(len(zero_threshold_samples))
            )
            self._write_data_to_disk(
                zero_threshold_samples, "t{}_zero_threshold_samples".format(target_week)
            )

        metadata = {
            "input_vars": self._input_vars,
            "oci_input_vars": self._oci_input_vars,
            "target_var": self._target_var,
            "sp_res": self._sp_res,
            "global_sp_res": self._global_sp_res,
            "global_latlon_shape": global_latlon_shape,
            "max_radius": self._radius,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "timeseries_weeks": self._timeseries_weeks,
            "target_count": self._target_count,
            "positive_samples_threshold": self._positive_samples_threshold,
            "gt_threshold_samples": len(gt_threshold_samples),
            "le_threshold_samples": len(le_threshold_samples),
            "zero_threshold_samples": len(zero_threshold_samples),
        }
        self._write_metadata_to_disk(metadata)

    def _write_metadata_to_disk(self, data):
        output_path = os.path.join(self._output_folder, "metadata.pt")
        torch.save(data, output_path)

    def _write_data_to_disk(self, data, name, index=None):
        if index is not None:
            output_path = os.path.join(
                self._output_folder, "{}_{}.pt".format(name, index)
            )
        else:
            output_path = os.path.join(self._output_folder, "{}.pt".format(name))
        torch.save(data, output_path)

    def _write_dataset_to_disk(self, dataset, name, index=None):
        if index is not None:
            output_path = os.path.join(
                self._output_folder, "{}_{}.h5".format(name, index)
            )
        else:
            output_path = os.path.join(self._output_folder, "{}.h5".format(name))
        dataset.to_netcdf(output_path)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    builder = DatasetBuilder(
        args.cube_path,
        args.cube_resolution,
        args.global_cube_path,
        args.load_cube_in_memory,
        args.output_folder,
        args.split,
        args.radius,
        args.positive_samples_threshold,
        args.positive_samples_size,
        args.generate_all_samples,
        args.seed,
        args.timeseries_weeks,
        args.target_count,
        args.include_oci_variables,
        args.global_scale_factor,
    )

    builder.run(args.area)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "--cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="../seasfire_1deg.zarr",
        help="Cube path",
    )
    parser.add_argument(
        "--cube-resolution",
        metavar="RESOLUTION",
        type=str,
        action="store",
        dest="cube_resolution",
        default="100km",
        help="Cube resolution. Can be 100km or 25km.",
    )
    parser.add_argument(
        "--global-cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="global_cube_path",
        default=None,
        help="Global cube path",
    )
    parser.add_argument(
        "--output-folder",
        metavar="KEY",
        type=str,
        action="store",
        dest="output_folder",
        default="data",
        help="Output folder",
    )
    parser.add_argument(
        "--split",
        metavar="KEY",
        type=str,
        action="store",
        dest="split",
        default="train",
        help="Split type. Can be train, test, val.",
    )
    parser.add_argument(
        "--radius",
        metavar="KEY",
        type=int,
        action="store",
        dest="radius",
        default=7,
        help="Radius of grid graph",
    )
    parser.add_argument(
        "--global-scale-factor",
        metavar="KEY",
        type=int,
        action="store",
        dest="global_scale_factor",
        default=18,
        help="Global scaling factor (coarsen)",
    )
    parser.add_argument(
        "--positive-samples-threshold",
        metavar="KEY",
        type=float,
        action="store",
        dest="positive_samples_threshold",
        default=0.01,
        help="Positive sample threshold",
    )
    parser.add_argument(
        "--positive-samples-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="positive_samples_size",
        default=50000,
        help="Positive samples size.",
    )
    parser.add_argument(
        "--area",
        metavar="KEY",
        type=str,
        action="store",
        dest="area",
        default="-35,-18,30,51",  # Africa
        # default="36,-25,72,50", # Europe
        # default="-82.5,-172.5,82.5,172.5", # Global
        help="Area as min_lat,min_lon,max_lat,max_lon",
    )
    parser.add_argument(
        "--generate-all-samples", dest="generate_all_samples", action="store_true"
    )
    parser.add_argument(
        "--no-generate-all-samples", dest="generate_all_samples", action="store_false"
    )
    parser.set_defaults(generate_all_samples=False)
    parser.add_argument(
        "--load-cube-in-memory", dest="load_cube_in_memory", action="store_true"
    )
    parser.add_argument(
        "--no-load-cube-in-memory", dest="load_cube_in_memory", action="store_false"
    )
    parser.set_defaults(load_cube_in_memory=False)
    parser.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        action="store",
        dest="seed",
        default=17,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "--timeseries-weeks",
        metavar="KEY",
        type=int,
        action="store",
        dest="timeseries_weeks",
        default=24,
        help="How many weeks will each timeseries contain.",
    )
    parser.add_argument(
        "--target-count",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_count",
        default=24,
        help="Target count. How many targets in the future to generate.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.add_argument(
        "--include-oci-variables", dest="include_oci_variables", action="store_true"
    )
    parser.add_argument(
        "--no-include-oci-variables", dest="include_oci_variables", action="store_false"
    )
    parser.set_defaults(include_oci_variables=True)

    args = parser.parse_args()

    args.area = args.area.split(",")
    if len(args.area) != 4:
        raise ValueError(
            "Area should be in the form of min_lat, min_lon, max_lat, max_lon"
        )
    args.area = (
        float(args.area[0]),
        float(args.area[1]),
        float(args.area[2]),
        float(args.area[3]),
    )

    main(args)
