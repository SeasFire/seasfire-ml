#!/usr/bin/env python3

import argparse
import json
import time
import logging
import os
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data


logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, cube_path, output_folder, split, positive_samples_threshold, target_shift):
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

        # positive example threshold
        self.positive_samples_threshold = positive_samples_threshold
        self.number_of_positive_samples = 0

        self.number_of_train_years = 13
        self.weeks = 58
        self.target_shift = target_shift #in weeks
        self.year_in_weeks = 46
        # self.valid_mask = np.count_nonzero((self.cube.gwis_ba_valid_mask.values)==1)

        # split time periods
        self.time_train = (0, self.year_in_weeks*self.number_of_train_years+target_shift)  # 0-591 week -> 13 yeaars
        self.time_val = (self.year_in_weeks*self.number_of_train_years, 
                         self.year_in_weeks*self.number_of_train_years+2*self.weeks+target_shift)  # 598-715 week -> 2.5 years
        self.time_test = (self.year_in_weeks*self.number_of_train_years+2*self.weeks, 916)  # 714-916 week -> 4.5 years

        # Spatial resolution
        self._sp_res = 0.25
        self._lat_min = -89.875
        self._lat_max = 89.875
        self._lon_min = -179.875
        self._lon_max = 179.875

        # mean and std dictionary
        self.mean_std_dict = {}
        

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

    #     pass


    def create_sample(
        self,
        center_lat,
        center_lon,
        center_time,
        ground_truth,
        radius=2,
        time_aggregation="3M",
    ):
        print("\n")
        logger.info(
            "Creating sample for center_lat={}, center_lon={}, center_time={}, radius={}, time_aggregation={}".format(
                center_lat, center_lon, center_time, radius, time_aggregation
            )
        )

        lat_slice = slice(
            center_lat + radius * self._sp_res, center_lat - radius * self._sp_res
        )
        lon_slice = slice(
            center_lon - radius * self._sp_res, center_lon + radius * self._sp_res
        )
        time_slice = slice(center_time - self.weeks, center_time)
        points_input_vars = (
            self.cube[self.input_vars]
            .sel(latitude=lat_slice, longitude=lon_slice)
            .isel(time=time_slice)
        )
        points_input_vars = points_input_vars.resample(time=time_aggregation).mean(
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
        # logger.info("Edges = {}".format(edges))
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

        return Data(x=vertex_features, y=graph_level_ground_truth, edge_index=edge_index, pos=vertex_positions)


    def generate_samples(self, min_lon, min_lat, max_lon, max_lat):
        # time depending on split
        start_time = 0
        end_time = 0

        if self.split == 'train':    
            start_time, end_time = self.time_train
            start_time += self.weeks
            end_time = end_time - self.target_shift
        if self.split == 'val':    
            start_time, end_time = self.time_val
            start_time += self.weeks
            end_time = end_time - self.target_shift
        if self.split == 'test':    
            start_time, end_time = self.time_test
            end_time = end_time - self.target_shift # last week does not have ground_truth
        
        total_time = end_time-start_time
        
        logger.info("Creating sample region")

        # define sample region
        sample_region = self.cube.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)).isel(time=slice(start_time, end_time))  
        samples_list = []

        # Sample nodes above threshold
        # for each node call create_sample() to build the graph
        for time_index in range(0, total_time):
            for lat_inc in sample_region.latitude.values:
                for lon_inc in sample_region.longitude.values:
                    burnt_grid_area = sample_region.gwis_ba.sel(latitude=lat_inc, longitude=lon_inc).isel(time=time_index).values
                    if burnt_grid_area > 0.0:

                        # Percentage of burnt area to all grid area 
                        grid_area = self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values
                        burnt_grid_area_percentage = burnt_grid_area / grid_area
                        print(grid_area)

                        # Percentage of burnt area to all grid area without Wetlands, permanent snow_ice and water bodies
                        # grid_area_land = (self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values 
                        #                   -(self.cube.lccs_class_4.sel(latitude=lat_inc, longitude=lon_inc).isel(time=58).values * self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values/100.0) 
                        #                   -(self.cube.lccs_class_7.sel(latitude=lat_inc, longitude=lon_inc).isel(time=58).values * self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values/100.0) 
                        #                   -(self.cube.lccs_class_8.sel(latitude=lat_inc, longitude=lon_inc).isel(time=58).values * self.cube.area.sel(latitude=lat_inc, longitude=lon_inc).values)/100.0)
                        # burnt_grid_area_percentage = burnt_grid_area / grid_area_land
                        # print(grid_area_land)

                        if burnt_grid_area_percentage >= self.positive_samples_threshold:
                            logger.info("Positive sample found.")

                            target_period = time_index + self.target_shift
                            ground_truth = sum(sample_region.gwis_ba.sel(latitude=lat_inc, longitude=lon_inc).isel(time=slice(time_index, target_period)).values)

                            center_lat = lat_inc
                            center_lon = lon_inc
                            center_time = start_time + time_index

                            samples_list.append((center_lat, center_lon, center_time, ground_truth))
                
        return samples_list      

    def run(self):
        # compute mean and std
#         self.compute_mean_std_dict()

        ##Africa
        min_lon = -18
        min_lat = -35
        max_lon = 51
        max_lat = 30

        # Europe
        # min_lon = -25
        # min_lat = 36
        # max_lon = 50
        # max_lat = 72

        # call generate samples
        samples = self.generate_samples(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)
                        
        # call create sample
        for idx in tqdm(range(0, len(samples[:]))):
            center_lat, center_lon, center_time, ground_truth = samples[idx]
            graph = self.create_sample(center_lat=center_lat, center_lon=center_lon, center_time=center_time, ground_truth=ground_truth)
        
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
    logging.basicConfig(level=logging.DEBUG)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    builder = DatasetBuilder(
        args.cube_path, args.output_folder, args.split, args.positive_samples_threshold, args.target_shift
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
    # parser.add_argument(
    #     "--todo",
    #     metavar="FLAG",
    #     type=bool,
    #     action=argparse.BooleanOptionalAction,
    #     dest="todo",
    #     default=True,
    #     help="todo",
    # )
    args = parser.parse_args()
    main(args)
