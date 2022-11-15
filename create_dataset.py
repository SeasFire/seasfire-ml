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
    ):
        self.cube_path = cube_path
        self.output_folder = output_folder
        self.input_vars = [
            "ndvi",
            "t2m_min",
            "tp",
            "drought_code_max",
            "sst",
            "lst_day",
        ]
        # one of gwis_ba, BurntArea, frpfire, co2fire, FCCI_BA, co2fire
        self.target_var = "gwis_ba"

        self.time_train = (0, 874)    # 2001-2019
        self.time_test  = (874,920)   # 2020
        self.time_eval  = (920, 966)  # 2021

        self.sp_res = 0.25  # Spatial resolution

        for folder in [self.output_folder]:
            logger.info("Creating output folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        logger.info("Opening zarr file {}".format(self.cube_path))
        self.cube = xr.open_zarr(self.cube_path, consolidated=False)
        logger.info("Cube: {}".format(self.cube))
        #print(self.cube.data_vars)


    def create_samples(self, min_lon, min_lat, max_lon, max_lat):
        logger.info("Creating samples")
        region = self.cube.sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )#.isel(time=35)

        print(region)

        pass

    def create_sample(self, lon, lat, time, radius=2, past_weeks=46, time_aggregation='3M'):
        logger.info("Creating samples")
        #region = self.cube.sel(
        #    latitude=lat, longitude=lon
        #).isel(time=35)

        #print (self.cube.coords["latitude"].to_numpy())
        #print (self.cube.coords["longitude"].to_numpy())

        lat_slice=slice(lat+radius*self.sp_res, lat-radius*self.sp_res)
        lon_slice=slice(lon-radius*self.sp_res, lon+radius*self.sp_res)

        cur_point_input_vars = self.cube[self.input_vars].sel(latitude=lat_slice, longitude=lon_slice).isel(time=time)
        cur_point_target_var = self.cube[self.target_var].sel(latitude=lat_slice, longitude=lon_slice).isel(time=time)

        time_slice=slice(time-past_weeks, time)
        past_points_input_vars = self.cube[self.input_vars].sel(latitude=lat_slice, longitude=lon_slice).isel(time=time_slice)
        past_points_target_vars = self.cube[self.input_vars].sel(latitude=lat_slice, longitude=lon_slice).isel(time=time_slice)

        past_points_input_vars = past_points_input_vars.resample(time=time_aggregation).mean(skipna=True)
        past_points_target_vars = past_points_target_vars.resample(time=time_aggregation).mean(skipna=True)
        past_points_count = past_points_input_vars.time.shape[0]

        # TODO: figure out a way to define by std and mean 

        # TODO: build a graph

        #data = self.cube.sel(
        #    latitude=lat, longitude=lon
        #)#.isel(time=time)
        #print(dataset)
        #print(data["drought_code_max"].to_numpy())
        
        #print(region["area"])

        pass

    def run(self):
        #self.create_samples(
        #    min_lon=20.375, min_lat=-24.375, max_lon=24.625, max_lat=-17.875
        #)
        self.create_sample(
            lon=20.375, lat=-24.375, time=500
        )
        pass


def main(args):
    logging.basicConfig(level=logging.DEBUG)

    builder = DatasetBuilder(
        args.cube_path,
        args.output_folder,
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
