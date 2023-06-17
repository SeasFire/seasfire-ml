#!/usr/bin/env python3

import argparse
import logging
import os
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import torch
from torch_geometric.data import Data

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
        target_length,
        include_oci_variables,
        global_scale_factor,
    ):
        self._cube_path = cube_path
        self._input_vars = [
            "lst_day",
            "mslp",
            "ndvi",
            "pop_dens",
            "rel_hum",
            "ssrd",
            "sst",
            "t2m_mean",
            "tp",
            "vpd",
        ]

        # one of gwis_ba, BurntArea, frpfire, co2fire, FCCI_BA, co2fire
        self._target_var = "gwis_ba"
        logger.info("Using target variable: {}".format(self._target_var))

        logger.info("Using seed: {}".format(seed))
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        if cube_resolution not in ["100km", "25km"]:
            raise ValueError("Wrong cube resolution")
        self._cube_resolution = cube_resolution
        logger.info("Using cube resolution: {}".format(self._cube_resolution))

        if cube_resolution == "25km":
            self._sp_res = 0.25
            self._lat_min = -89.875
            self._lat_max = 89.875
            self._lon_min = -179.875
            self._lon_max = 179.875
        else:
            self._sp_res = 1
            self._lat_min = -89.5
            self._lat_max = 89.5
            self._lon_min = -179.5
            self._lon_max = 179.5

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

        # create output split folder
        self._cache_folder = os.path.join(output_folder, "cache")
        for folder in [self._cache_folder]:
            logger.info("Creating cache folder {}".format(folder))
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

        self._oci_input_vars = [
            # "oci_censo",
            # "oci_ea",
            # "oci_epo",
            # "oci_gmsst",
            "oci_nao",
            # "oci_nina34_anom",
            # "oci_pdo",
            # "oci_pna",
            # "oci_soi",
            # "oci_wp",
        ]
        self._include_oci_variables = include_oci_variables

        # for oci_var in self._oci_input_vars:
        #     logger.debug(
        #         "Oci name {}, description: {}".format(
        #             oci_var, self._cube[oci_var].description
        #         )
        #     )
        # print(self._cube.longitude.values)
        # for x in self._cube.longitude.values:
        #     print(x)
        # print(self._cube.latitude.values)
        # for x in self._cube.longitude.values:
        #     print(x)

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

        # how many targets periods to generate in the future
        # e.g. 6 means the next six months (if target length is 4 weeks)
        # e.g. 24 means the next six months (if target length is 1)
        self._target_count = target_count
        # length of each target period in weeks, e.g. 4
        # length of the target period is now 1 week (8 days)
        self._target_length = target_length

        logger.info("Will generate {} target periods.".format(self._target_count))
        for p in range(self._target_count):
            logger.info(
                "Target period {} is weeks in the future: [{},{}]".format(
                    p,
                    p * self._target_length,
                    (p + 1) * self._target_length,
                )
            )

        # split time periods
        self._time_train = (
            self._timeseries_weeks,
            self._year_in_weeks * self._number_of_train_years
            - (self._target_count * self._target_length),
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
            self._max_week_with_data - (self._target_count * self._target_length),
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

    def _create_local_vertices(self, center_lat, center_lon, center_time, radius):
        # Create a grid graph around the center vertex.
        grid = list(
            map(
                self._normalize_lat_lon,
                self._create_neighbors(
                    (center_lat, center_lon), include_self=True, radius=radius
                ),
            )
        )

        vertices = []
        vertices_idx = {}
        for cur in grid:
            cur_vertex = (cur[0], cur[1])
            vertices_idx[cur_vertex] = len(vertices)
            vertices.append(cur_vertex)

        # find center_time in time coords
        center_time_idx = np.where(self._cube["time"] == center_time)[0][0]
        time_slice = slice(
            center_time_idx - self._timeseries_weeks + 1, center_time_idx + 1
        )
        lat_slice = slice(
            center_lat + (radius + 1) * self._sp_res,
            center_lat - (radius + 1) * self._sp_res,
        )
        lon_slice = slice(
            center_lon - (radius + 1) * self._sp_res,
            center_lon + (radius + 1) * self._sp_res,
        )

        input_vars = self._input_vars
        if self._include_oci_variables:
            input_vars += self._oci_input_vars
        points_input_vars = (
            self._cube[input_vars]
            .sel(
                latitude=lat_slice,
                longitude=lon_slice,
            )
            .isel(time=time_slice)
            .load()
        )
        timeseries_len = len(points_input_vars.coords["time"])
        if timeseries_len != self._timeseries_weeks:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._timeseries_weeks
                )
            )
            raise ValueError("Invalid time series length")

        # Create vertex feature tensors
        vertices_input_vars = points_input_vars.stack(vertex=("latitude", "longitude"))
        vertex_features = []
        vertex_positions = []
        for vertex in vertices:
            # get all input vars and append lat-lon
            v_features = (
                vertices_input_vars.sel(vertex=vertex)
                .to_array(dim="variable", name=None)
                .values
            )
            v_position = [
                np.cos(vertex[0]),
                np.sin(vertex[0]),
                np.cos(vertex[1]),
                np.sin(vertex[1]),
            ]
            vertex_features.append(v_features)
            vertex_positions.append(v_position)

        return vertices, vertices_idx, vertex_features, vertex_positions

    def _create_local_edges(self, center_lat, center_lon, radius, vertices_idx):
        edges = []
        for lat_inc in range(-radius, radius + 1):
            for lon_inc in range(-radius, radius + 1):
                # vertex that we care about
                cur = (
                    center_lat + lat_inc * self._sp_res,
                    center_lon + lon_inc * self._sp_res,
                )
                cur_idx = vertices_idx[(cur[0], cur[1])]
                # logger.info("cur = {}, cur_idx={}".format(cur, cur_idx))

                # 1-hop neighbors
                cur_neighbors = self._create_neighbors(
                    cur, radius=1, include_self=False
                )
                # logger.info("cur 1-neighbors = {}".format(cur_neighbors))

                # 1-hop neighbors inside our bounding box from the center vertex
                cur_neighbors_bb = [
                    neighbor
                    for neighbor in cur_neighbors
                    if self._in_bounding_box(
                        neighbor,
                        center_lat_lon=(center_lat, center_lon),
                        radius=radius,
                    )
                ]
                cur_neighbors_bb = list(map(self._normalize_lat_lon, cur_neighbors_bb))
                cur_neighbors_bb_idx = [
                    vertices_idx[(x[0], x[1])] for x in cur_neighbors_bb
                ]
                # logger.info("cur 1-neighbors in bb = {}".format(cur_neighbors_bb))
                # logger.info("cur_idx 1-neighbors in bb = {}".format(cur_neighbors_bb_idx))

                for neighbor_idx in cur_neighbors_bb_idx:
                    # add only one direction, the other will be added by the other vertex
                    edges.append((cur_idx, neighbor_idx))
        return edges

    def _create_global_vertices(self, center_time):
        result = self._read_from_cache(key="global_{}".format(center_time))
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
        center_time_idx = np.where(global_region["time"] == center_time)[0][0]
        time_slice = slice(
            center_time_idx - self._timeseries_weeks + 1, center_time_idx + 1
        )

        input_vars = self._input_vars
        if self._include_oci_variables:
            input_vars += self._oci_input_vars
        points_input_vars = global_agg[input_vars].isel(time=time_slice).load()

        timeseries_len = len(points_input_vars.coords["time"])
        if timeseries_len != self._timeseries_weeks:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._timeseries_weeks
                )
            )
            raise ValueError("Invalid time series length")

        # Create list of vertices
        vertices = []
        vertices_idx = {}
        for lat in global_agg.coords["latitude"].values:
            for lon in global_agg.coords["longitude"].values:
                cur_vertex = (lat, lon)
                vertices_idx[cur_vertex] = len(vertices)
                vertices.append(cur_vertex)

        # Create vertex feature tensors
        vertices_input_vars = points_input_vars.stack(vertex=("latitude", "longitude"))
        vertex_features = []
        vertex_positions = []
        for vertex in vertices:
            # get all input vars and append lat-lon
            v_features = (
                vertices_input_vars.sel(vertex=vertex)
                .to_array(dim="variable", name=None)
                .values
            )
            v_position = [
                np.cos(vertex[0]),
                np.sin(vertex[0]),
                np.cos(vertex[1]),
                np.sin(vertex[1]),
            ]
            vertex_features.append(v_features)
            vertex_positions.append(v_position)

        result = (vertices, vertices_idx, vertex_features, vertex_positions)
        self._write_to_cache(key="global_{}".format(center_time), data=result)

        return result

    def _create_sample_data(
        self,
        center_lat,
        center_lon,
        center_time,
        center_area,
        ground_truth,
        radius,
    ):
        logger.info(
            "Creating sample for center_lat={}, center_lon={}, center_time={}".format(
                center_lat, center_lon, center_time
            )
        )

        # compute local vertices
        (
            local_vertices,
            local_vertices_idx,
            local_vertices_features,
            local_vertices_positions,
        ) = self._create_local_vertices(
            center_lat=center_lat,
            center_lon=center_lon,
            center_time=center_time,
            radius=radius,
        )

        local_edges = self._create_local_edges(
            center_lat=center_lat,
            center_lon=center_lon,
            radius=radius,
            vertices_idx=local_vertices_idx,
        )

        # compute global vertices
        (
            global_vertices,
            global_vertices_idx,
            global_vertices_features,
            global_vertices_positions,
        ) = self._create_global_vertices(center_time=center_time)

        # TODO: combine local and global vertices
        # TODO: add edges

        # logger.info("Local vertices features={}".format(local_vertices_features))
        # logger.info("Global vertices features={}".format(global_vertices_features))

        vertices_features = local_vertices_features + global_vertices_features
        vertices_features = np.array(vertices_features)
        vertices_features = torch.from_numpy(vertices_features).type(torch.float32)
        vertices_positions = local_vertices_positions + global_vertices_positions
        vertices_positions = np.array(vertices_positions)
        vertices_positions = torch.from_numpy(vertices_positions).type(torch.float32)

        graph_level_ground_truth = torch.from_numpy(np.array(ground_truth)).type(
            torch.float32
        )
        assert len(graph_level_ground_truth) == self._target_count

        area = torch.from_numpy(np.array(center_area)).type(torch.float32)

        # Create edge index tensor
        edges = local_edges
        sources, targets = zip(*edges)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        logger.debug("Computed edge tensor= {}".format(edge_index))

        data = Data(
            x=vertices_features,
            y=graph_level_ground_truth,
            edge_index=edge_index,
            pos=vertices_positions,
            area=area,
            center_lat=center_lat,
            center_lon=center_lon,
            center_time=center_time,
        )

        logger.info("Computed sample={}".format(data))
        return data

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

    def _compute_area(self, lat, lon):
        area = self._cube["area"].sel(latitude=lat, longitude=lon)
        area_in_hectares = area.values / 10000.0
        return area_in_hectares

    def compute_ground_truth(self, center_lat, center_lon, center_time):

        # find center_time in time coords
        center_time_idx = np.where(self._cube["time"] == center_time)[0][0]
        time_slice = slice(
            center_time_idx + 1,
            center_time_idx + 1 + self._target_length * self._target_count,
            self._target_length,
        )
        logger.debug(
            "Computing ground truth for lat={}, lon={}, time={}, time_slice={}".format(
                center_lat, center_lon, center_time, time_slice
            )
        )

        gwis_ba = (
            self._cube["gwis_ba"]
            .sel(
                latitude=center_lat,
                longitude=center_lon,
            )
            .isel(time=time_slice)
        )

        timeseries_len = len(gwis_ba.coords["time"])
        if timeseries_len != self._target_count:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._target_count
                )
            )
            raise ValueError("Invalid time series length")

        logger.debug("Ground truth target values={}".format(gwis_ba.values))
        return gwis_ba.values

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

        logger.info("About to create {} samples".format(len(samples)))
        # now generate them and write to disk
        for idx in tqdm(range(0, len(samples))):
            if idx < self._first_sample_index:
                continue
            if self._is_sample_present(idx):
                # logger.info("Skipping sample {} generation.".format(idx))
                continue
            center_lat, center_lon, center_time = samples[idx]
            ground_truth = self.compute_ground_truth(
                center_lat, center_lon, center_time
            )
            center_area = self._compute_area(center_lat, center_lon)

            graph = self._create_sample_data(
                center_lat=center_lat,
                center_lon=center_lon,
                center_time=center_time,
                center_area=center_area,
                ground_truth=ground_truth,
                radius=self._radius,
            )

            self._write_sample_to_disk(graph, idx)

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

    def _write_sample_to_disk(self, data, index):
        output_path = os.path.join(self._output_folder, "graph_{}.pt".format(index))
        torch.save(data, output_path)

    def _is_sample_present(self, index):
        output_path = os.path.join(self._output_folder, "graph_{}.pt".format(index))
        return os.path.exists(output_path)

    def _in_bounding_box(self, lat_lon, center_lat_lon, radius):
        lat, lon = lat_lon
        center_lat, center_lon = center_lat_lon
        return (
            lat <= center_lat + radius * self._sp_res
            and lat >= center_lat - radius * self._sp_res
            and lon >= center_lon - radius * self._sp_res
            and lon <= center_lon + radius * self._sp_res
        )

    def _create_neighbors(self, lat_lon, radius=1, include_self=False, normalize=False):
        """Create list of all neighbors inside a radius. Radius is measured in multiples of
        the spatial resolution.
        """
        lat, lon = lat_lon
        neighbors = []
        for lat_inc in range(-radius, radius + 1):
            for lon_inc in range(-radius, radius + 1):
                if not include_self and lat_inc == 0 and lon_inc == 0:
                    continue
                neighbors.append(
                    (lat + lat_inc * self._sp_res, lon + lon_inc * self._sp_res)
                )
        if normalize:
            neighbors = list(map(self._normalize_lat_lon, neighbors))
        return neighbors

    def _create_periphery_neighbors(
        self, lat_lon, radius=4, include_self=False, normalize=False
    ):
        """Create a list of neighbors in the periphery of a square with a specific radius around
        a vertex.
        """
        lat, lon = lat_lon
        neighbors = []
        for lat_inc in [-radius, 0, radius]:
            for lon_inc in [-radius, 0, radius]:
                if not include_self and lat_inc == 0 and lon_inc == 0:
                    continue
                neighbors.append(
                    (lat + lat_inc * self._sp_res, lon + lon_inc * self._sp_res)
                )
        if normalize:
            neighbors = list(map(self._normalize_lat_lon, neighbors))
        return neighbors

    def _normalize_lat_lon(self, lat_lon):
        lat, lon = lat_lon
        while lat > self._lat_max:
            lat -= 180.0
        while lat < self._lat_min:
            lat += 180.0
        while lon < self._lon_min:
            lon += 360.0
        while lon > self._lon_max:
            lon -= 360.0
        return lat, lon

    def _datetime64_to_ts(self, dt64):
        return (dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")


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
        args.target_length,
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
        default=0.01,
        help="Positive sample threshold",
    )
    parser.add_argument(
        "--positive-samples-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="positive_samples_size",
        default=2500,
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
        default=48,
        help="How many weeks will each timeseries contain.",
    )
    parser.add_argument(
        "--target-count",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_count",
        default=6,
        help="Target count. How many targets in the future to generate.",
    )
    parser.add_argument(
        "--target-length",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_length",
        default=4,
        help="Target length. How long does the target period last. Measured in weeks.",
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
