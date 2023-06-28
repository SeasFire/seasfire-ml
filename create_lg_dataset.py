#!/usr/bin/env python3

import argparse
import logging
import os
from tqdm import tqdm
import xarray as xr
import numpy as np
import torch
from utils import LocalGlobalBuilder, LocalGlobalDataset

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(
        self,
        cube_path,
        cube_resolution,
        load_cube_in_memory,
        output_folder,
        split,
        radius,
        positive_samples_threshold,
        positive_samples_size,
        generate_all_samples,
        first_sample_index,
        seed,
        timeseries_weeks,
        target_count,
        include_oci_variables,
        global_scale_factor,
    ):
        self._cube_path = cube_path
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
        self._oci_input_vars = [
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
        self._include_oci_variables = include_oci_variables

        # one of gwis_ba, BurntArea, frpfire, co2fire, FCCI_BA, co2fire
        self._target_var = "gwis_ba"
        logger.info("Using target variable: {}".format(self._target_var))

        logger.info("Using seed: {}".format(seed))
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        if cube_resolution not in ["100km", "25km"]:
            raise ValueError("Wrong cube resolution")
        logger.info("Using cube resolution: {}".format(cube_resolution))

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

        # open zarr and display basic info
        logger.info("Opening zarr file: {}".format(self._cube_path))
        self._cube = xr.open_zarr(self._cube_path, consolidated=False)
        if load_cube_in_memory:
            logger.info("Loading the whole cube in memory.")
            self._cube.load()
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

        # Radius for grid graph
        self._radius = radius
        logger.info("Using radius={} for grid graph".format(self._radius))

        self._global_scale_factor = global_scale_factor
        logger.info("Using global scale factor={}".format(self._global_scale_factor))

        # Threshold for fires
        self._positive_samples_threshold = positive_samples_threshold
        # How many samples to take above the threshold
        self._positive_samples_size = positive_samples_size
        # Whether to generate all samples
        self._generate_all_samples = generate_all_samples

        # sample index to start generation from
        self._first_sample_index = first_sample_index

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

        # create graph builder
        self._builder = LocalGlobalBuilder(
            cube=self._cube,
            cube_resolution=cube_resolution,
            input_vars=self._input_vars,
            oci_input_vars=self._oci_input_vars,
            target_var=self._target_var,
            output_folder=output_folder,
            radius=self._radius,
            timeseries_weeks=self._timeseries_weeks,
            target_count=self._target_count,
            global_scale_factor=self._global_scale_factor,
        )

    def _sample_wrt_threshold(
        self, sample_region, sample_region_gwsi_ba_per_area, strategy
    ):
        if strategy == "above-threshold":
            sample_region_gwsi_ba_per_area_wrt_threshold = (
                sample_region_gwsi_ba_per_area > self._positive_samples_threshold
            )
            size = self._positive_samples_size
        elif strategy == "positive-below-threshold":
            sample_region_gwsi_ba_per_area_wrt_threshold = (
                sample_region_gwsi_ba_per_area <= self._positive_samples_threshold
            ) & (sample_region_gwsi_ba_per_area > 0.0)
            size = self._positive_samples_size
        elif strategy == "zero":
            sample_region_gwsi_ba_per_area_wrt_threshold = (
                sample_region_gwsi_ba_per_area <= 0.0
            )
            size = 2 * self._positive_samples_size
        else:
            raise ValueError("Invalid strategy")

        sample_len_wrt_threshold = np.sum(sample_region_gwsi_ba_per_area_wrt_threshold)
        logger.info(
            "Samples for strategy={} are {}".format(strategy, sample_len_wrt_threshold)
        )

        result = []
        all_wrt_threshold_samples_index = np.argwhere(
            sample_region_gwsi_ba_per_area_wrt_threshold
        )
        if size > len(all_wrt_threshold_samples_index):
            raise ValueError("Not enough samples to sample from.")

        wrt_threshold_samples_index = self._rng.choice(
            all_wrt_threshold_samples_index,
            size=size,
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

    def _generate_all_samples_lists(self, min_lon, min_lat, max_lon, max_lat):
        sample_region = self._cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        ).isel(time=slice(self._start_time, self._end_time))

        sample_region_gwsi_ba_values = sample_region.gwis_ba.values
        sample_region_gwsi_non_nan = sample_region_gwsi_ba_values >= 0.0

        sample_len_non_nan = np.sum(sample_region_gwsi_non_nan)
        logger.debug("Samples={}".format(sample_len_non_nan))

        result = []
        all_samples_index = np.argwhere(sample_region_gwsi_non_nan)

        for index in all_samples_index:
            result.append(
                (
                    sample_region.latitude.values[index[1]],
                    sample_region.longitude.values[index[2]],
                    sample_region.time.values[index[0]],
                ),
            )

        return result

    def _generate_threshold_samples_lists(self, min_lon, min_lat, max_lon, max_lat):
        # define sample region
        sample_region = self._cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        ).isel(time=slice(self._start_time, self._end_time))

        # find target variable in region
        sample_region_gwsi_ba_values = sample_region.gwis_ba.values

        # compute area in sample region and add time dimension
        sample_region_area = (
            self._cube["area"]
            .sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            .expand_dims(dim={"time": sample_region.time}, axis=0)
        )

        # Convert area in hectars
        sample_region_area_values = sample_region_area.values / 10000.0
        logger.info(
            "Unique area values: {}".format(np.unique(sample_region_area_values))
        )

        # compute target variable per area
        sample_region_gwsi_ba_per_area = (
            sample_region_gwsi_ba_values / sample_region_area_values
        )

        above_threshold_samples_list = self._sample_wrt_threshold(
            sample_region, sample_region_gwsi_ba_per_area, strategy="above-threshold"
        )

        below_threshold_samples_list = self._sample_wrt_threshold(
            sample_region,
            sample_region_gwsi_ba_per_area,
            strategy="positive-below-threshold",
        )

        zero_threshold_samples_list = self._sample_wrt_threshold(
            sample_region, sample_region_gwsi_ba_per_area, strategy="zero"
        )

        return (
            above_threshold_samples_list
            + below_threshold_samples_list
            + zero_threshold_samples_list
        )

    def generate_samples_lists(self, min_lon, min_lat, max_lon, max_lat):
        logger.info("Generating sample list for split={}".format(self._split))
        logger.info(
            "Generating from week={} to week={}".format(
                self._start_time, self._end_time
            )
        )

        if self._generate_all_samples:
            logger.info("Will generate all samples")
            return self._generate_all_samples_lists(min_lon, min_lat, max_lon, max_lat)
        else:
            logger.info("Will samples based on threshold")
            return self._generate_threshold_samples_lists(
                min_lon, min_lat, max_lon, max_lat
            )

    def run(self):
        ##Africa
        # min_lon = -18
        # min_lat = -35
        # max_lon = 51
        # max_lat = 30

        # Europe
        min_lon = -25
        min_lat = 36
        max_lon = 50
        max_lat = 72

        # create list of samples to generate
        samples = self.generate_samples_lists(
            min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
        )

        local_latlon_shape = None

        logger.info("Creating global dataset")
        global_dataset = self._builder.create_global_data()
        global_latlon_shape = (global_dataset["latitude"].shape[0], global_dataset["longitude"].shape[0])
        self._write_dataset_to_disk(global_dataset, "global")

        logger.info("Creating area data")
        area_dataset = self._builder.create_area_data()
        self._write_dataset_to_disk(area_dataset, "area")

        logger.info("Creating ground truth data")
        ground_truth_dataset = self._builder.create_target_var_data(min_lat, min_lon, max_lat, max_lon)
        self._write_dataset_to_disk(ground_truth_dataset, "ground_truth")

        logger.info("Creating samples index")
        self._write_data_to_disk(samples, "samples")
        logger.info("About to create {} samples".format(len(samples)))

        # now generate them and write to disk
        for idx in tqdm(range(0, len(samples))):
            if idx < self._first_sample_index:
                continue
            if self._is_dataset_present(idx):
                # logger.info("Skipping sample {} generation.".format(idx))
                continue

            center_lat, center_lon, center_time = samples[idx]

            local_dataset = self._builder.create(
                lat=center_lat, lon=center_lon, time=center_time
            )

            if local_latlon_shape is None: 
                local_latlon_shape = (local_dataset["latitude"].shape[0], local_dataset["longitude"].shape[0])

            self._write_dataset_to_disk(local_dataset, "local", idx)

        metadata = {
            "input_vars": self._input_vars,
            "oci_input_vars": self._oci_input_vars,
            "target_var": self._target_var,
            "local_latlon_shape": local_latlon_shape,
            "global_latlon_shape": global_latlon_shape,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "timeseries_weeks": self._timeseries_weeks,
            "target_count": self._target_count,
        }
        self._write_metadata_to_disk(metadata)

        # Second pass to compute statistics
        # Only if we generated at least one sample on this run
        if local_latlon_shape is not None:
            self._compute_statistics()

    def _compute_statistics(self): 
        dataset = LocalGlobalDataset(
            root_dir=self._output_folder,
        )

        local_stats_filename = "{}/mean_std_stats_local.pk".format(self._output_folder)
        if not os.path.exists(local_stats_filename): 
            local_features_count = len(self._input_vars) + len(self._oci_input_vars)
            logger.info("Local features={}".format(local_features_count))

            local_mean_std = np.zeros((local_features_count, 2))
            logger.info("Computing local features mean/std")
            for feature_idx in range(local_features_count):
                logger.info("Computing mean, std for feature idx={}".format(feature_idx))

                temp = np.concatenate([graph.x[:, feature_idx, :].flatten() for graph in dataset])
                mean_std = [np.nanmean(temp), np.nanstd(temp)]
                local_mean_std[feature_idx] = mean_std
                logger.info("Mean, std for idx={} are {}".format(feature_idx, mean_std))
            torch.save(local_mean_std, "{}/mean_std_stats_local.pk".format(self._output_folder))
        else:
            logger.info("Skipping local features stats computation. Found file: {}".format(local_stats_filename))

        global_stats_filename = "{}/mean_std_stats_global.pk".format(self._output_folder)
        if not os.path.exists(global_stats_filename):         
            global_features_count = local_features_count
            logger.info("Global features={}".format(global_features_count))

            global_mean_std = np.zeros((global_features_count, 2))
            logger.info("Computing global features mean/std")
            for feature_idx in range(global_features_count):
                logger.info("Computing mean, std for feature idx={}".format(feature_idx))

                temp = np.concatenate([graph.global_x[:, feature_idx, :].flatten() for graph in dataset])
                mean_std = [np.nanmean(temp), np.nanstd(temp)]
                global_mean_std[feature_idx] = mean_std
                logger.info("Mean, std for idx={} are {}".format(feature_idx, mean_std))

            torch.save(global_mean_std, "{}/mean_std_stats_global.pk".format(self._output_folder))
        else:
            logger.info("Skipping global features stats computation. Found file: {}".format(global_stats_filename))

    def _write_metadata_to_disk(self, data):
        output_path = os.path.join(self._output_folder, "metadata.pt")
        torch.save(data, output_path)

    def _write_data_to_disk(self, data, name, index=None):
        if index is not None:
            output_path = os.path.join(self._output_folder, "{}_{}.pt".format(name, index))
        else: 
            output_path = os.path.join(self._output_folder, "{}.pt".format(name))
        torch.save(data, output_path)

    def _write_dataset_to_disk(self, dataset, name, index=None):
        if index is not None:
            output_path = os.path.join(self._output_folder, "{}_{}.h5".format(name, index))
        else:
            output_path = os.path.join(self._output_folder, "{}.h5".format(name))
        dataset.to_netcdf(output_path)

    def _is_dataset_present(self, index):
        for key in ["local", "ground_truth", "area"]: 
            output_path = os.path.join(
                self._output_folder, "{}_{}.h5".format(key, index)
            )
            if not os.path.exists(output_path): 
                return False
        return True

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
        args.load_cube_in_memory,
        args.output_folder,
        args.split,
        args.radius,
        args.positive_samples_threshold,
        args.positive_samples_size,
        args.generate_all_samples,
        args.first_sample_index,
        args.seed,
        args.timeseries_weeks,
        args.target_count,
        args.include_oci_variables,
        args.global_scale_factor,
    )
    builder.run()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "--cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="../1d_SeasFireCube.zarr",
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
        default=0.002,
        help="Positive sample threshold",
    )
    parser.add_argument(
        "--positive-samples-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="positive_samples_size",
        default=100,
        help="Positive samples size.",
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
        "--first-sample-index",
        metavar="INT",
        type=int,
        action="store",
        dest="first_sample_index",
        default=0,
        help="Generate samples starting from a specific sample index. Allows to resume dataset creation.",
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
    main(args)
