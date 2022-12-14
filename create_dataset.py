#!/usr/bin/env python3

import argparse
import logging
import os
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

OCI_BOUNDING_BOXES = {
    "oci_nao": [
        {
            "min_lat": 20.0,
            "max_lat": 55.0,
            "min_lon": -90,
            "max_lon": 60.0,
        },
        {
            "min_lat": 55.0,
            "max_lat": 90.0,
            "min_lon": -90.0,
            "max_lon": 60.0,
        },
    ],
    "oci_nina34_anom": [
        {
            "min_lat": -170.0,
            "max_lat": 120.0,
            "min_lon": -5.0,
            "max_lon": 5.0,
        }
    ],
    "oci_pdo": [
        {
            "min_lat": 20.0,
            "max_lat": 70.0,
            "min_lon": 115.0,
            "max_lon": 250.0,
        }
    ],
    "oci_pna": [
        {
            "min_lat": 20.0,
            "max_lat": 80.0,
            "min_lon": -60.0,
            "max_lon": 120.0,
        }
    ],
    "oci_soi": [
        {
            "min_lat": -18.625,
            "max_lat": -16.375,
            "min_lon": -150.875,
            "max_lon": -148.625,
        },
        {
            "min_lat": -13.625,
            "max_lat": -11.375,
            "min_lon": -132.125,
            "max_lon": -129.875,
        },
    ],
    "oci_wp": [
        {
            "min_lat": 50.0,
            "max_lat": 70.0,
            "min_lon": -150.0,
            "max_lon": 140.0,
        },
        {
            "min_lat": 25.0,
            "max_lat": 40.0,
            "min_lon": -150.0,
            "max_lon": 140.0,
        },
    ],
}


class DatasetBuilder:
    def __init__(
        self,
        cube_path,
        cube_resolution,
        output_folder,
        split,
        positive_samples_threshold,
        positive_samples_size,
        first_sample_index,
        seed,
        target_shift,
        target_length,
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

        # open zarr and display basic info
        logger.info("Opening zarr file: {}".format(self._cube_path))
        self._cube = xr.open_zarr(self._cube_path, consolidated=False)
        logger.info("Cube: {}".format(self._cube))
        logger.info("Vars: {}".format(self._cube.data_vars))

        self._oci_input_vars = [
            "oci_censo",
            "oci_ea",
            "oci_epo",
            "oci_gmsst",
            "oci_nao",
            "oci_nina34_anom",
            "oci_pdo",
            "oci_pna",
            "oci_soi",
            "oci_wp",
        ]
        self._oci_locations = self._vertices_per_oci()

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

        # Threshold for fires
        self._positive_samples_threshold = positive_samples_threshold
        # How many samples to take above the threshold
        self._positive_samples_size = positive_samples_size

        # sample index to start generation
        self._first_sample_index = first_sample_index

        self._number_of_train_years = 16
        self._days_per_week = 8
        self._timeseries_weeks = 48
        self._aggregation_in_weeks = 4  # aggregate per month
        self._target_shift = target_shift  # in weeks, e.g. 0
        self._target_length = target_length  # in weeks, e.g. 4
        self._year_in_weeks = 48

        logger.info(
            "Target period weeks in the future: [{},{}]".format(
                self._target_shift, self._target_shift + self._target_length
            )
        )

        # split time periods
        self._time_train = (
            self._timeseries_weeks,
            self._year_in_weeks * self._number_of_train_years
            - (self._target_shift + self._target_length),
        )  # 46 - 778 (782-0-4) week -> 17 years
        logger.info("Train time in weeks: {}".format(self._time_train))

        self._time_val = (
            self._year_in_weeks * self._number_of_train_years + self._timeseries_weeks,
            self._year_in_weeks * self._number_of_train_years
            + 2 * self._timeseries_weeks,
        )  # 782+46 - 782+2*46 week -> 2 years
        logger.info("Val time in weeks: {}".format(self._time_val))

        self._time_test = (
            self._year_in_weeks * self._number_of_train_years
            + 2 * self._timeseries_weeks,
            918 - (self._target_shift + self._target_length),
        )  # 828 (782+46) - 914 (918-0-4) week -> 2 years
        logger.info("Test time in weeks: {}".format(self._time_test))

        if self._split == "train":
            self._start_time, self._end_time = self._time_train
        elif self._split == "val":
            self._start_time, self._end_time = self._time_val
        elif self._split == "test":
            self._start_time, self._end_time = self._time_test
        else:
            raise ValueError("Invalid split type")

    def _get_oci_features(
        self,
        center_time,
    ):
        """Get the OCI features for a certain timerange"""
        first_week = center_time - np.timedelta64(
            self._timeseries_weeks * self._days_per_week, "D"
        )
        aggregation_in_days = "{}D".format(
            self._aggregation_in_weeks * self._days_per_week
        )
        points_oci_vars = self._cube[self._oci_input_vars].sel(
            time=slice(first_week, center_time)
        )
        points_oci_vars = points_oci_vars.resample(
            time=aggregation_in_days, closed="left"
        ).mean(skipna=True)
        return points_oci_vars

    def create_sample(
        self,
        center_lat,
        center_lon,
        center_time,
        ground_truth,
        small_radius=2,
        medium_radius=4,
    ):
        logger.info(
            "Creating sample for center_lat={}, center_lon={}, center_time={}".format(
                center_lat, center_lon, center_time
            )
        )

        # Initialize basic graph representation
        vertices = []
        vertices_idx = {}
        edges = []

        # Create a small grid graph around the center vertex (small radius).
        grid = list(
            map(
                self._normalize_lat_lon,
                self._create_neighbors(
                    (center_lat, center_lon), include_self=True, radius=small_radius
                ),
            )
        )
        for cur in grid:
            cur_vertex = (cur[0], cur[1])
            vertices_idx[cur_vertex] = len(vertices)
            vertices.append(cur_vertex)

        # Create small grid edges
        for lat_inc in range(-small_radius, small_radius + 1):
            for lon_inc in range(-small_radius, small_radius + 1):
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
                        radius=small_radius,
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

        # Create medium radius periphery
        periphery_vertices = self._create_periphery_neighbors(
            (center_lat, center_lon), radius=medium_radius, normalize=True
        )
        center_idx = vertices_idx[(center_lat, center_lon)]
        for cur_p_vertex in periphery_vertices:
            vertices_idx[cur_p_vertex] = len(vertices)
            cur_p_vertex_idx = len(vertices)
            vertices.append(cur_p_vertex)

            edges.append((center_idx, cur_p_vertex_idx))
            edges.append((cur_p_vertex_idx, center_idx))

            cur_p_neighbors = self._create_neighbors(
                cur_p_vertex, radius=1, include_self=False, normalize=True
            )
            for cur_p_neighbor in cur_p_neighbors:
                vertices_idx[cur_p_neighbor] = len(vertices)
                cur_p_neighbor_idx = len(vertices)
                vertices.append(cur_p_neighbor)

                edges.append((cur_p_vertex_idx, cur_p_neighbor_idx))
                edges.append((cur_p_neighbor_idx, cur_p_vertex_idx))

        vertex_features, vertex_positions = self._compute_local_vertices_features(
            vertices, center_lat, center_lon, center_time, small_radius, medium_radius
        )

        # Create long range vertices to OCI
        for cur_oci_vertex in self._oci_locations:
            logger.debug("Oci = {}".format(cur_oci_vertex))
            vertices_idx[cur_oci_vertex] = len(vertices)
            cur_oci_vertex_idx = len(vertices)
            vertices.append(cur_oci_vertex_idx)

            oci_features, oci_positions = self._compute_vertex_features(
                cur_oci_vertex[0], cur_oci_vertex[1], center_time
            )
            vertex_features.extend(oci_features)
            vertex_positions.extend(oci_positions)

            edges.append((center_idx, cur_oci_vertex_idx))
            edges.append((cur_oci_vertex_idx, center_idx))

            cur_oci_vertex_neighbors = self._create_neighbors(
                cur_oci_vertex, radius=1, include_self=False, normalize=True
            )

            for cur_oci_vertex_neighbor in cur_oci_vertex_neighbors:
                vertices_idx[cur_oci_vertex_neighbor] = len(vertices)
                cur_oci_vertex_neighbor_idx = len(vertices)
                vertices.append(cur_oci_vertex_neighbor)

                oci_features, oci_positions = self._compute_vertex_features(
                    cur_oci_vertex_neighbor[0], cur_oci_vertex_neighbor[1], center_time
                )
                vertex_features.extend(oci_features)
                vertex_positions.extend(oci_positions)

                edges.append((cur_oci_vertex_idx, cur_oci_vertex_neighbor_idx))
                edges.append((cur_oci_vertex_neighbor_idx, cur_oci_vertex_idx))

        vertex_features = np.array(vertex_features)
        vertex_features = torch.from_numpy(vertex_features).type(torch.float32)
        vertex_positions = np.array(vertex_positions)
        vertex_positions = torch.from_numpy(vertex_positions).type(torch.float32)

        graph_level_ground_truth = torch.from_numpy(np.array([ground_truth])).type(
            torch.float32
        )

        # Create edge index tensor
        sources, targets = zip(*edges)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        logger.debug("Computed edge tensor= {}".format(edge_index))

        return Data(
            x=vertex_features,
            y=graph_level_ground_truth,
            edge_index=edge_index,
            pos=vertex_positions,
        )

    def _compute_local_vertices_features(
        self,
        vertices,
        center_lat,
        center_lon,
        center_time,
        small_radius=2,
        medium_radius=4,
    ):
        first_week = center_time - np.timedelta64(
            self._timeseries_weeks * self._days_per_week, "D"
        )

        aggregation_in_days = "{}D".format(
            self._aggregation_in_weeks * self._days_per_week
        )
        logger.debug("Using aggregation period in days: {}".format(aggregation_in_days))

        max_radius = max(medium_radius + 1, small_radius + 1)
        lat_slice = slice(
            center_lat + max_radius * self._sp_res,
            center_lat - max_radius * self._sp_res,
        )
        lon_slice = slice(
            center_lon - max_radius * self._sp_res,
            center_lon + max_radius * self._sp_res,
        )

        points_input_vars = self._cube[self._input_vars].sel(
            latitude=lat_slice, longitude=lon_slice, time=slice(first_week, center_time)
        )

        points_input_vars = points_input_vars.resample(
            time=aggregation_in_days, closed="left"
        ).mean(skipna=True)

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

        return vertex_features, vertex_positions

    def _compute_vertex_features(
        self,
        vertex_lat,
        vertex_lon,
        vertex_time,
    ):
        logger.debug(
            "Computing oci features for ({},{},{})".format(
                vertex_lat, vertex_lon, vertex_time
            )
        )
        first_week = vertex_time - np.timedelta64(
            self._timeseries_weeks * self._days_per_week, "D"
        )
        aggregation_in_days = "{}D".format(
            self._aggregation_in_weeks * self._days_per_week
        )
        logger.debug("Using aggregation period in days: {}".format(aggregation_in_days))

        points_input_vars = self._cube[self._input_vars].sel(
            latitude=vertex_lat,
            longitude=vertex_lon,
            time=slice(first_week, vertex_time),
        )

        points_input_vars = points_input_vars.resample(
            time=aggregation_in_days, closed="left"
        ).mean(skipna=True)

        # Create vertex feature tensors
        vertex_features = []
        vertex_positions = [
            [
                np.cos(vertex_lat),
                np.sin(vertex_lat),
                np.cos(vertex_lon),
                np.sin(vertex_lon),
            ]
        ]
        v_features = points_input_vars.to_array(dim="variable", name=None).values
        vertex_features.append(v_features)

        return vertex_features, vertex_positions

    def _sample_wrt_threshold(
        self, sample_region, sample_region_gwsi_ba_per_area, strategy
    ):
        size = 0

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
            size = 2* self._positive_samples_size
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
        if self._positive_samples_size > len(all_wrt_threshold_samples_index):
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

    def generate_samples_lists(self, min_lon, min_lat, max_lon, max_lat):
        logger.info("Generating sample list for split={}".format(self._split))
        logger.info(
            "Generating from week={} to week={}".format(
                self._start_time, self._end_time
            )
        )

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

    def compute_ground_truth(self, lat, lon, time):
        start_time = time + np.timedelta64(
            self._target_shift * self._days_per_week, "D"
        )
        end_time = time + np.timedelta64(
            (self._target_shift + self._target_length) * self._days_per_week, "D"
        )
        logger.debug(
            "Computing ground truth for lat={}, lon={}, time=[{},{}]".format(
                lat, lon, start_time, end_time
            )
        )

        values = (
            self._cube["gwis_ba"]
            .sel(
                latitude=lat,
                longitude=lon,
                time=slice(
                    start_time,
                    end_time,
                ),
            )
            .values
        )
        ground_truth = sum(values)

        return ground_truth

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
            graph = self.create_sample(
                center_lat=center_lat,
                center_lon=center_lon,
                center_time=center_time,
                ground_truth=ground_truth,
            )

            self._write_sample_to_disk(graph, idx)

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

    def _vertices_per_oci(self):
        all_long_range = []
        for oci in self._oci_input_vars:
            if oci not in OCI_BOUNDING_BOXES:
                continue
            for candidate_bb in OCI_BOUNDING_BOXES[oci]:

                lat_values = self._cube.latitude
                lat_values = lat_values.where(
                    (lat_values >= candidate_bb["min_lat"])
                    & (lat_values <= candidate_bb["max_lat"])
                )
                lat_values = lat_values[~np.isnan(lat_values)]
                lat_values = lat_values.values
                candidate_lat = np.take(lat_values, lat_values.size // 2)

                lon_values = self._cube.longitude
                lon_values = lon_values.where(
                    (lon_values >= candidate_bb["min_lon"])
                    & (lon_values <= candidate_bb["max_lon"])
                )
                lon_values = lon_values[~np.isnan(lon_values)]
                lon_values = lon_values.values
                candidate_lon = np.take(lon_values, lon_values.size // 2)

                logger.info(
                    "Using ({},{}) as location for oci {}".format(
                        candidate_lat, candidate_lon, oci
                    )
                )
                all_long_range.append((candidate_lat, candidate_lon))

        return all_long_range


def main(args):
    logging.basicConfig(level=logging.INFO)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    builder = DatasetBuilder(
        args.cube_path,
        args.cube_resolution,
        args.output_folder,
        args.split,
        args.positive_samples_threshold,
        args.positive_samples_size,
        args.first_sample_index,
        args.seed,
        args.target_shift,
        args.target_length,
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
        "--positive-samples-threshold",
        metavar="KEY",
        type=float,
        action="store",
        dest="positive_samples_threshold",
        default=0.001,
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
        "--target-shift",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_shift",
        default=0,
        help="Target shift. How far in the future does the target period start. Measured in weeks.",
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
    args = parser.parse_args()
    main(args)
