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


class DatasetBuilder:
    def __init__(
        self, cube_path, output_folder, split, positive_samples_threshold, target_shift
    ):
        self.cube_path = cube_path
        self.input_vars = [
            "ndvi",
            "t2m_min",
            "tp",
            "drought_code_max",
            "sst",
            "rel_hum",
            "lst_day",
        ]
        # one of gwis_ba, BurntArea, frpfire, co2fire, FCCI_BA, co2fire
        self.target_var = "gwis_ba"
        logger.info("Using target variable: {}".format(self.target_var))

        # validate split
        if split not in ["train", "test", "val"]:
            raise ValueError("Wrong split type")
        self.split = split
        logger.info("Using split: {}".format(self.split))

        # create output split folder
        self.output_folder = os.path.join(output_folder, self.split)
        for folder in [self.output_folder]:
            logger.info("Creating output folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        # open zarr and display basic info
        logger.info("Opening zarr file: {}".format(self.cube_path))
        self.cube = xr.open_zarr(self.cube_path, consolidated=False)
        logger.info("Cube: {}".format(self.cube))
        logger.info("Vars: {}".format(self.cube.data_vars))

        # positive example threshold
        self.positive_samples_threshold = positive_samples_threshold
        self.number_of_positive_samples = 0

        self.number_of_train_years = 13
        self.days_per_week = 8
        self.weeks = 48
        self.aggregation_in_weeks = 12
        self.target_shift = target_shift  # in weeks, e.g. 4
        self.year_in_weeks = 46

        # Spatial resolution
        self._sp_res = 0.25
        self._lat_min = -89.875
        self._lat_max = 89.875
        self._lon_min = -179.875
        self._lon_max = 179.875

        # mean and std dictionary
        self.mean_std_dict = {}

        # split time periods
        self.time_train = (
            self.weeks,
            self.year_in_weeks * self.number_of_train_years - self.target_shift,
        )  # 58-594 week -> 13 years
        logger.info("Train time in weeks: {}".format(self.time_train))
        self.time_val = (
            self.year_in_weeks * self.number_of_train_years + self.weeks,
            self.year_in_weeks * self.number_of_train_years + 2 * self.weeks,
        )  # 598-715 week -> 2.5 years
        logger.info("Val time in weeks: {}".format(self.time_val))
        self.time_test = (
            self.year_in_weeks * self.number_of_train_years + 2 * self.weeks,
            916,
        )  # 714-916 week -> 4.5 years
        logger.info("Test time in weeks: {}".format(self.time_test))

        if self.split == "train":
            self._start_time, self._end_time = self.time_train
        elif self.split == "val":
            self._start_time, self._end_time = self.time_val
        elif self.split == "test":
            self._start_time, self._end_time = self.time_test
        else:
            raise ValueError("Invalid split type")

    # def compute_mean_std_dict(self):
    #     # TODO: compute mean_std_dict
    #     first_time, last_time = self.time_train

    #     for var in self.input_vars + [self.target_var]:
    #         self.mean_std_dict[var + '_mean'] = self.cube[var].isel(time=slice(first_time, last_time)).mean()
    #         self.mean_std_dict[var + '_std'] = self.cube[var].isel(time=slice(first_time, last_time)).std()

    #     # Store data
    #     with open('mean_std_dict.pickle', 'wb') as handle:
    #         pkl.dump(self.mean_std_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    #     # # # Load data
    #     # data = pkl.load(open('mean_std_dict.pickle', 'rb'))
    #     # print(data['ndvi_mean'].values)
    #     # print(data['ndvi_std'].values)

    def create_sample(
        self,
        center_lat,
        center_lon,
        center_time,
        ground_truth,
        radius=2,
    ):
        logger.info(
            "Creating sample for center_lat={}, center_lon={}, center_time={}, radius={}".format(
                center_lat, center_lon, center_time, radius
            )
        )
        first_week = center_time - np.timedelta64(self.weeks * self.days_per_week, "D")
        logger.debug("Sampling from time period: [{}, {}]".format(first_week, center_time))

        aggregation_in_days = "{}D".format(self.aggregation_in_weeks * self.days_per_week)
        logger.debug("Using aggregation period: {}".format(aggregation_in_days))

        lat_slice = slice(
            center_lat + radius * self._sp_res, center_lat - radius * self._sp_res
        )
        lon_slice = slice(
            center_lon - radius * self._sp_res, center_lon + radius * self._sp_res
        )

        points_input_vars = self.cube[self.input_vars].sel(
            latitude=lat_slice, longitude=lon_slice, time=slice(first_week, center_time)
        )
        points_input_vars = points_input_vars.resample(time=aggregation_in_days, closed='left').mean(
            skipna=True
        )

        time_dim = points_input_vars.time.shape[0]
        latitude_dim = points_input_vars.latitude.shape[0]
        longitude_dim = points_input_vars.longitude.shape[0]
        logger.info(
            "Sample dimensions: time={}, lat={}, lon={}".format(
                time_dim, latitude_dim, longitude_dim
            )
        )

        # Compute time coordinates from present to past
        time_coords = np.flip(points_input_vars.time.to_numpy())
        logger.info("Time coordinates in reverse: {}".format(time_coords))

        # Create list of vertices and mapping from vertices to integer indices
        vertices = []
        vertices_idx = {}
        grid = list(
            map(
                self._normalize_lat_lon,
                self._create_neighbors(
                    (center_lat, center_lon), include_self=True, radius=radius
                ),
            )
        )
        for _, time_val in enumerate(time_coords):
            for cur in grid:
                cur_vertex = (cur[0], cur[1], time_val)
                cur_vertex_index = len(vertices)
                vertices_idx[cur_vertex] = cur_vertex_index
                vertices.append(cur_vertex)
        logger.info("Final graph will have {} vertices".format(len(vertices)))

        # Create edges
        edges = []
        for time_idx, time_val in enumerate(time_coords):
            for lat_inc in range(-radius, radius + 1):
                for lon_inc in range(-radius, radius + 1):
                    # vertex that we care about
                    cur = (
                        center_lat + lat_inc * self._sp_res,
                        center_lon + lon_inc * self._sp_res,
                    )
                    cur_idx = vertices_idx[(cur[0], cur[1], time_val)]
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
                    cur_neighbors_bb = list(
                        map(self._normalize_lat_lon, cur_neighbors_bb)
                    )
                    cur_neighbors_bb_idx = [
                        vertices_idx[(x[0], x[1], time_val)] for x in cur_neighbors_bb
                    ]
                    # logger.info("cur 1-neighbors in bb = {}".format(cur_neighbors_bb))
                    # logger.info("cur_idx 1-neighbors in bb = {}".format(cur_neighbors_bb_idx))

                    for neighbor_idx in cur_neighbors_bb_idx:
                        # add only on direction, the other will be added by the other vertex
                        edges.append((cur_idx, neighbor_idx))

                    if time_idx < len(time_coords) - 1:
                        past_time_val = time_coords[time_idx + 1]
                        cur_past_neighbors_bb_idx = [
                            vertices_idx[(x[0], x[1], past_time_val)]
                            for x in cur_neighbors_bb
                        ]
                        # logger.info("past cur_idx 1-neighbors in bb = {}".format(cur_past_neighbors_bb_idx))

                        for neighbor_idx in cur_past_neighbors_bb_idx:
                            # add both directions to and from the past
                            edges.append((cur_idx, neighbor_idx))
                            edges.append((neighbor_idx, cur_idx))

        # Create edge index tensor
        logger.info("Total edges added in graph = {}".format(len(edges)))
        sources, targets = zip(*edges)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        logger.info("Computed edge tensor= {}".format(edge_index))

        # Create vertex feature tensors
        # Now that we have our graph, compute variables per vertex
        vertices_input_vars = points_input_vars.stack(
            vertex=("latitude", "longitude", "time")
        )
        vertex_features = []
        vertex_positions = []
        for vertex in vertices:
            # get all input vars and append lat-lon
            v_features = (
                vertices_input_vars.sel(vertex=vertex)
                .to_array(dim="variable", name=None)
                .values
            )
            v_position = [vertex[0], vertex[1], self._datetime64_to_ts(vertex[2])]
            vertex_features.append(v_features)
            vertex_positions.append(v_position)

        vertex_features = np.array(vertex_features)
        vertex_features = torch.from_numpy(vertex_features).type(torch.float32)
        vertex_positions = np.array(vertex_positions)
        vertex_positions = torch.from_numpy(vertex_positions).type(torch.float32)

        graph_level_ground_truth = torch.from_numpy(np.array([ground_truth])).type(
            torch.float32
        )

        return Data(
            x=vertex_features,
            y=graph_level_ground_truth,
            edge_index=edge_index,
            pos=vertex_positions,
        )

    def generate_samples(self, min_lon, min_lat, max_lon, max_lat):
        logger.info("Generating sample list for split={}".format(self.split))
        logger.info(
            "Generating from week={} to week={}".format(
                self._start_time, self._end_time
            )
        )

        # define sample region
        sample_region = self.cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        ).isel(time=slice(self._start_time, self._end_time))

        # find target variable in region
        sample_region_gwsi_ba_values = sample_region.gwis_ba.values

        # compute area in sample region and add time dimension
        sample_region_area = (
            self.cube["area"]
            .sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            .expand_dims(dim={"time": sample_region.time}, axis=0)
        )
        sample_region_area_values = sample_region_area.values

        # compute target variable per area and apply threshold
        sample_region_gwsi_ba_per_area = (
            sample_region_gwsi_ba_values / sample_region_area_values
        )
        sample_region_gwsi_ba_per_area_above_threshold = (
            sample_region_gwsi_ba_per_area > self.positive_samples_threshold
        )
        logger.info(
            "Samples above threshold={}".format(
                np.sum(sample_region_gwsi_ba_per_area_above_threshold)
            )
        )

        # Percentage of burnt area to all grid area without Wetlands, permanent snow_ice and water bodies
        # grid_area_land = (self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values
        #                   -(self.cube.lccs_class_4.sel(latitude=lat_inc, longitude=lon_inc).isel(time=58).values * self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values/100.0)
        #                   -(self.cube.lccs_class_7.sel(latitude=lat_inc, longitude=lon_inc).isel(time=58).values * self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values/100.0)
        #                   -(self.cube.lccs_class_8.sel(latitude=lat_inc, longitude=lon_inc).isel(time=58).values * self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values)/100.0)
        # burnt_grid_area_percentage = burnt_grid_area / grid_area_land
        # print(grid_area_land)

        # return (lat, lon, time) of samples
        samples_list = []
        samples_index = np.argwhere(sample_region_gwsi_ba_per_area_above_threshold)
        for index in samples_index:
            samples_list.append(
                (
                    sample_region.latitude.values[index[1]],
                    sample_region.longitude.values[index[2]],
                    sample_region.time.values[index[0]],
                ),
            )

        return samples_list

    def compute_ground_truth(self, lat, lon, time):
        logger.debug(
            "Computing ground truth for lat={}, lon={}, time={}".format(lat, lon, time)
        )
        time_slice = slice(time, time + np.timedelta64(self.target_shift, "W"))

        values = (
            self.cube["gwis_ba"]
            .sel(latitude=lat, longitude=lon, time=time_slice)
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

        # call generate samples
        samples = self.generate_samples(
            min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
        )

        logger.info("About to create {} samples".format(len(samples)))
        # self.number_of_positive_samples = 95
        # call create sample
        for idx in tqdm(range(0, len(samples))):
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

            self._write_sample_to_disk(graph, self.number_of_positive_samples)
            self.number_of_positive_samples += 1

    def _write_sample_to_disk(self, data, index):
        output_path = os.path.join(self.output_folder, "graph_{}.pt".format(index))
        torch.save(data, output_path)

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
    logging.basicConfig(level=logging.INFO)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    builder = DatasetBuilder(
        args.cube_path,
        args.output_folder,
        args.split,
        args.positive_samples_threshold,
        args.target_shift,
    )
    builder.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "--cubepath",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="../seasfire.zarr",
        help="Cube path",
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
        help="Split type",
    )
    parser.add_argument(
        "--positive-samples-threshold",
        metavar="KEY",
        type=float,
        action="store",
        dest="positive_samples_threshold",
        default=0.00001,
        help="Positive sample threshold",
    )
    parser.add_argument(
        "--target-shift",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_shift",
        default=4,
        help="Target shift",
    )
    args = parser.parse_args()
    main(args)
