import logging
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)


class BaselineModel:
    def __init__(
        self,
        cube_path,
        load_cube_in_memory,
    ):
        # one of gwis_ba, BurntArea, frpfire, co2fire, FCCI_BA, co2fire
        self._target_var = "gwis_ba"
        logger.info("Using target variable: {}".format(self._target_var))

        logger.info("Cube path={}".format(cube_path))
        self._cube_path = cube_path

        # open local cube and display basic info
        logger.info("Opening local cube zarr file: {}".format(self._cube_path))
        self._cube = xr.open_zarr(self._cube_path, consolidated=False)
        if load_cube_in_memory:
            logger.info("Loading the whole cube in memory.")
            self._cube.load()

        logger.info("Cube: {}".format(self._cube))
        logger.info("Vars: {}".format(self._cube.data_vars))

        self._number_of_train_years = 16
        self._days_per_week = 8
        self._year_in_weeks = 48

    def __call__(self, x): 
        lat_batch, lon_batch, time_batch = x
        batches = lat_batch.shape[0]

        for b in range(batches): 
            lat = lat_batch[b]
            lon = lon_batch[b]
            time = time_batch[b]

            self._predict(lat, lon, time)    

    def _predict(self, lat, lon, time):
        logger.info(
            "Predicting for lat={}, lon={}, time={}".format(
                lat, lon, time
            )
        )

        time_idx = np.where(self._cube["time"] == time)[0][0]
        logger.info("time idx={}".format(time_idx))

        # Compute ground truth - target
        target = (
            self._cube[self._target_var].sel(
                latitude=lat,
                longitude=lon,
            )
            .fillna(0)
        )
        logger.info("target={}".format(target))
        logger.info("target shape={}".format(target.values.shape))
        logger.info("target={}".format(target.values))


