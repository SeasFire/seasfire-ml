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
    def __init__(
        self,
        cube_path,
        output_folder,
        split,
        positive_samples_threshold
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

        if split not in ["train", "test", "val"]: 
            raise ValueError("Wrong split type")
        self.split = split

        self.positive_samples_threshold = positive_samples_threshold

        # split time periods
        self.time_train = (0, 874)  # 2001-2019
        self.time_test = (874, 920)  # 2020
        self.time_eval = (920, 966)  # 2021

        self.sp_res = 0.25  # Spatial resolution
        self.mean_std_dict = {}

        self.output_folder = os.path.join(output_folder, self.split)
        for folder in [self.output_folder]:
            logger.info("Creating output folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        logger.info("Opening zarr file {}".format(self.cube_path))
        self.cube = xr.open_zarr(self.cube_path, consolidated=False)
        logger.info("Cube: {}".format(self.cube))
        logger.info("Vars: {}".format(self.cube.data_vars))

    def compute_mean_std_dict(): 
        # compute mean_std_dict
        pass

    def create_samples(self, min_lon, min_lat, max_lon, max_lat):
        logger.info("Creating samples")
        region = self.cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )  # .isel(time=35)

        print(region)

        pass

    def create_sample(self, lat, lon, time, radius=2, weeks=58, time_aggregation="3M"):
        logger.info(
            "Creating sample for lat={}, lon={}, time={}, radius={}, weeks={}, time_aggregation={}".format(
                lat, lon, time, radius, weeks, time_aggregation
            )
        )

        lat_slice = slice(lat + radius * self.sp_res, lat - radius * self.sp_res)
        lon_slice = slice(lon - radius * self.sp_res, lon + radius * self.sp_res)

        time_slice = slice(time - weeks, time)
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

        time_coords = np.flip(points_input_vars.time.to_numpy())
        logger.info("Coordinates in (reverse) time {}".format(time_coords))

        vertices_index = []
        for time_index in time_coords: 
            for lat_index in np.linspace(lat + radius * self.sp_res, lat - radius * self.sp_res, 2*radius+1):
                for lon_index in np.linspace(lon - radius * self.sp_res, lon + radius * self.sp_res, 2*radius+1): 
                    vertices_index.append((lat_index, lon_index,time_index))
        #print(vertices_index)

        #print("TODO")
        vertices_input_vars = points_input_vars.stack(vertex=("latitude", "longitude", "time"))
        #print(vertices_input_vars)
        vertices_target_vars = points_target_vars.stack(vertex=("latitude", "longitude", "time"))
        vertices_dim = vertices_input_vars.vertex.shape[0]
        #print(vertices_input_vars.vertex)
        logger.info("Graph will have {} vertices".format(vertices_dim))

        #print(vertices_input_vars.vertex)
        #print(vertices_input_vars)
        #tmp  = vertices_input_vars.sel(vertex=(-23.875, 19.875, pd.Timestamp('2010-08-31 00:00:00')))
        #print(tmp["ndvi"].to_numpy())
        
        #print(x)

        # TODO: figure out a way to define by std and mean

        # TODO: build a graph

        # data = self.cube.sel(
        #    latitude=lat, longitude=lon
        # )#.isel(time=time)
        # print(dataset)
        # print(data["drought_code_max"].to_numpy())

        # print(region["area"])

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
        self.create_sample(lat=-24.375, lon=20.375, time=500)
        pass


def main(args):
    logging.basicConfig(level=logging.DEBUG)

    builder = DatasetBuilder(
        args.cube_path,
        args.output_folder,
        args.split,
        args.positive_samples_threshold
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
