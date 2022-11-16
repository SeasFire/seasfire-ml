#!/usr/bin/env python3

import argparse
import json
import time
import logging
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, cube_path, output_folder, split, positive_samples_threshold):
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

        # validate split
        if split not in ["train", "test", "val"]:
            raise ValueError("Wrong split type")
        self.split = split

        # positive example threshold
        self.positive_samples_threshold = positive_samples_threshold

        # split time periods
        self.time_train = (0, 874)  # 2001-2019
        self.time_test = (874, 920)  # 2020
        self.time_eval = (920, 966)  # 2021

        # Spatial resolution
        self._sp_res = 0.25
        self._lat_min = -89.875
        self._lat_max = 89.875
        self._lon_min = -179.875
        self._lon_max = 179.875

        # mean and std dictionary
        self.mean_std_dict = {}

        # create output split folder
        self.output_folder = os.path.join(output_folder, self.split)
        for folder in [self.output_folder]:
            logger.info("Creating output folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        # open zarr and display basic info
        logger.info("Opening zarr file {}".format(self.cube_path))
        self.cube = xr.open_zarr(self.cube_path, consolidated=False)
        logger.info("Cube: {}".format(self.cube))
        logger.info("Vars: {}".format(self.cube.data_vars))
        # print(self.cube.longitude.to_numpy())

    def compute_mean_std_dict():
        # TODO: compute mean_std_dict
        pass

    def create_samples(self, min_lon, min_lat, max_lon, max_lat):
        # TODO
        pass

    def create_sample(
        self,
        center_lat,
        center_lon,
        center_time,
        radius=2,
        weeks=58,
        time_aggregation="3M",
    ):
        logger.info(
            "Creating sample for center_lat={}, center_lon={}, center_time={}, radius={}, weeks={}, time_aggregation={}".format(
                center_lat, center_lon, center_time, radius, weeks, time_aggregation
            )
        )

        lat_slice = slice(
            center_lat + radius * self._sp_res, center_lat - radius * self._sp_res
        )
        lon_slice = slice(
            center_lon - radius * self._sp_res, center_lon + radius * self._sp_res
        )
        time_slice = slice(center_time - weeks, center_time)
        points_input_vars = (
            self.cube[self.input_vars]
            .sel(latitude=lat_slice, longitude=lon_slice)
            .isel(time=time_slice)
        )
        points_target_vars = (
            self.cube[self.input_vars]
            .sel(latitude=lat_slice, longitude=lon_slice)
            .isel(time=time_slice)
        )

        points_input_vars = points_input_vars.resample(time=time_aggregation).mean(
            skipna=True
        )
        points_target_vars = points_target_vars.resample(time=time_aggregation).mean(
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
                self._create_neighbors((center_lat, center_lon), include_self=True, radius=radius)
            )
        )
        for _, time_val in enumerate(time_coords):
            for cur in grid:
                cur_vertex = (cur[0], cur[1], time_val)
                cur_vertex_index = len(vertices)
                vertices_idx[cur_vertex] = cur_vertex_index
                vertices.append(cur_vertex)
        # logger.debug("All vertices: {}".format(vertices))
        logger.info("Final graph will have {} vertices".format(len(vertices)))

        #
        # DEBUG for debug print grid
        #
        # for _, time_val in enumerate(time_coords):        
        #     for lat_inc in range(-radius, radius + 1):
        #         for lon_inc in range(-radius, radius + 1):
        #             # vertex that we care about
        #             cur = (
        #                 center_lat + lat_inc * self._sp_res,
        #                 center_lon + lon_inc * self._sp_res,
        #             )
        #             #print(*cur, end=" ")
        #             #print((*cur, time_val), end=" ")
        #             print(vertices_idx[(*cur, time_val)], end=" ")
        #         print("")
        #     print("")    

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
                    #logger.info("cur = {}, cur_idx={}".format(cur, cur_idx))

                    # 1-hop neighbors
                    cur_neighbors = self._create_neighbors(
                        cur, radius=1, include_self=False
                    )
                    #logger.info("cur 1-neighbors = {}".format(cur_neighbors))

                    # 1-hop neighbors inside our bounding box from the center vertex
                    cur_neighbors_bb = [
                        neighbor
                        for neighbor in cur_neighbors
                        if self._in_bounding_box(
                            neighbor, center_lat_lon=(center_lat, center_lon), radius=radius
                        )
                    ]
                    cur_neighbors_bb  = list(map(self._normalize_lat_lon, cur_neighbors_bb))
                    cur_neighbors_bb_idx = [vertices_idx[(x[0], x[1], time_val)] for x in cur_neighbors_bb]
                    #logger.info("cur 1-neighbors in bb = {}".format(cur_neighbors_bb))
                    #logger.info("cur_idx 1-neighbors in bb = {}".format(cur_neighbors_bb_idx))

                    for neighbor_idx in cur_neighbors_bb_idx:
                        # add only on direction, the other will be added by the other vertex 
                        edges.append((cur_idx, neighbor_idx))

                    if time_idx < len(time_coords)-1: 
                        past_time_val = time_coords[time_idx+1]
                        cur_past_neighbors_bb_idx = [vertices_idx[(x[0], x[1], past_time_val)] for x in cur_neighbors_bb]
                        #logger.info("past cur_idx 1-neighbors in bb = {}".format(cur_past_neighbors_bb_idx))

                        for neighbor_idx in cur_past_neighbors_bb_idx: 
                            # add both directions to and from the past
                            edges.append((cur_idx, neighbor_idx))
                            edges.append((neighbor_idx, cur_idx))
                    


        #logger.info("Edges = {}".format(edges))
        logger.info("Total edges added in graph = {}".format(len(edges)))

        # Now that we have our graph, compute variables per vertex
        vertices_input_vars = points_input_vars.stack(
            vertex=("latitude", "longitude", "time")
        )
        # print(vertices_input_vars)
        vertices_target_vars = points_target_vars.stack(
            vertex=("latitude", "longitude", "time")
        )
        # TODO

        # print(vertices_input_vars.vertex)
        # print(vertices_input_vars)
        # tmp  = vertices_input_vars.sel(vertex=(-23.875, 19.875, pd.Timestamp('2010-08-31 00:00:00')))
        # print(tmp["ndvi"].to_numpy())

        # TODO: at the end wrap all these into a pytorch Data()
        # TODO: pass target variable or label as an input to this function

        pass

    def run(self):
        # Depending on split

        # compute mean and std

        # Sample nodes above threshold

        # call create sample

        # write to folder

        # self.create_samples(
        #    min_lon=20.375, min_lat=-24.375, max_lon=24.625, max_lat=-17.875
        # )
        self.create_sample(center_lat=-24.375, center_lon=20.375, center_time=500)
        pass

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


def main(args):
    logging.basicConfig(level=logging.DEBUG)

    builder = DatasetBuilder(
        args.cube_path, args.output_folder, args.split, args.positive_samples_threshold
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
        default="dataset",
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
        default=0.1,
        help="Positive sample threshold",
    )
    parser.add_argument(
        "--todo",
        metavar="FLAG",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="todo",
        default=True,
        help="todo",
    )
    args = parser.parse_args()
    main(args)
