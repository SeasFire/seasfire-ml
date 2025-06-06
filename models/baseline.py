import logging
import xarray as xr
import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaselineModel:
    def __init__(
        self,
        cube_path,
        load_cube_in_memory,
        method,
    ):
        self._target_var = "gwis_ba"
        logger.info("Using target variable: {}".format(self._target_var))

        self._method = method

        logger.info("Cube path={}".format(cube_path))
        self._cube_path = cube_path
        logger.info("Opening local cube zarr file: {}".format(self._cube_path))
        self._cube = xr.open_zarr(self._cube_path, consolidated=False)
        if load_cube_in_memory:
            logger.info("Loading the whole cube in memory.")
            self._cube.load()

        logger.info("Cube: {}".format(self._cube))
        logger.info("Vars: {}".format(self._cube.data_vars))

        self._year_in_weeks = 48

    def __call__(self, x): 
        lat_batch, lon_batch, time_batch = x
        batches = lat_batch.shape[0]

        result_list = []
        for b in range(batches): 
            lat = lat_batch[b]
            lon = lon_batch[b]
            time = time_batch[b]

            result = self._predict(lat, lon, time)
            result_list.append(result)

        result_batch = torch.tensor(result_list)

        logger.debug("results={}".format(result_batch))

        return result_batch

    def _predict(self, lat, lon, time):
        logger.debug(
            "Predicting for lat={}, lon={}, time={}".format(
                lat, lon, time
            )
        )

        time_idx = np.where(self._cube["time"] == time)[0][0]
        logger.debug("time idx={}".format(time_idx))

        target = (
            self._cube[self._target_var].sel(
                latitude=lat,
                longitude=lon,
            )
            .fillna(0)
        )

        previous = target[time_idx-self._year_in_weeks::-self._year_in_weeks]

        if self._method == "mean":
            mean = previous.mean()
            logger.debug("mean={}".format(mean.values))
            return torch.as_tensor(mean.values)
        elif self._method == "majority":
            non_zero_count = (previous != 0).sum().values
            majority = non_zero_count / previous.size
            logger.debug("majority={} since {} (non-zeros) in total {}".format(majority, non_zero_count, previous.size))
            return torch.as_tensor(majority)
        else: 
            raise ValueError("Invalid method")