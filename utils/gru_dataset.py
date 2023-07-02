import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Dataset, Data
import os
import xarray as xr
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUDataset(Dataset):
    def __init__(
        self,
        root_dir,
        include_oci_variables=True,
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        self._metadata = torch.load(os.path.join(self.root_dir, "metadata.pt"))
        logger.info("Metadata={}".format(self._metadata))

        self._timeseries_weeks = self._metadata["timeseries_weeks"]
        self._target_count = self._metadata["target_count"]
        self._target_var = self._metadata["target_var"]
        self._input_vars = self._metadata["input_vars"]

        logger.info("Found input vars={}".format(self._input_vars))
        if include_oci_variables:
            self._oci_input_vars = self._metadata["oci_input_vars"]
            logger.info("Found oci input vars={}".format(self._oci_input_vars))
        else:
            self._oci_input_vars = []

        self._sp_res = self._metadata["sp_res"]
        logger.info("spatial resolution (sp_res)={}".format(self._sp_res))

        gt_threshold_samples = torch.load(
            os.path.join(self.root_dir, "gt_threshold_samples.pt")
        )
        self._gt_threshold_samples_count = len(gt_threshold_samples)
        logger.info("Samples (>threshold)={}".format(self._gt_threshold_samples_count))

        le_threshold_samples = torch.load(
            os.path.join(self.root_dir, "le_threshold_samples.pt")
        )
        self._le_threshold_samples_count = len(le_threshold_samples)
        logger.info(
            "Samples (>0 and <=threshold)={}".format(
                self._le_threshold_samples_count
            )
        )

        zero_threshold_samples = torch.load(
            os.path.join(self.root_dir, "zero_threshold_samples.pt")
        )
        self._zero_threshold_samples_count = len(zero_threshold_samples)
        logger.info("Samples (=0)={}".format(self._zero_threshold_samples_count))

        self._samples = gt_threshold_samples + le_threshold_samples + zero_threshold_samples

        self._indices = list(
            range(
                self._gt_threshold_samples_count
                + self._le_threshold_samples_count
                + self._zero_threshold_samples_count
            )
        )

        logger.info("Loading ground truth dataset")
        with xr.open_dataset(os.path.join(self.root_dir, "ground_truth.h5")) as ds:
            self._ground_truth_ds = ds.load()
        logger.debug("ground_truth_ds={}".format(self._ground_truth_ds))

        logger.info("Loading local dataset")
        with xr.open_dataset(os.path.join(self.root_dir, "local.h5")) as ds:
            self._local_ds = ds.load()
        logger.debug("local_ds={}".format(self._local_ds))

        # Compute local graph features
        for oci_var in self._oci_input_vars:
            self._local_ds[oci_var] = self._local_ds[oci_var].expand_dims(
                dim={
                    "latitude": self._local_ds["latitude"],
                    "longitude": self._local_ds["longitude"],
                }
            )
        logger.debug("local_ds={}".format(self._local_ds))

    @property
    def len(self):
        return (
            self._gt_threshold_samples_count
            + self._le_threshold_samples_count
            + self._zero_threshold_samples_count
        )

    @property
    def local_features(self):
        return self._input_vars + self._oci_input_vars

    def get(self, idx: int) -> Data:
        lat, lon, time = self._samples[idx]
        logger.debug(
            "Generating sample for idx={},lat={}, lon={}, time={}".format(
                idx, lat, lon, time
            )
        )

        # Compute ground truth - target
        time_idx = np.where(self._ground_truth_ds["time"] == time)[0][0]
        time_slice = slice(time_idx + 1, time_idx + 1 + self._target_count)
        target = (
            self._ground_truth_ds.sel(
                latitude=lat,
                longitude=lon,
            )
            .isel(time=time_slice)
            .fillna(0)
        )

        timeseries_len = len(target.coords["time"])
        if timeseries_len != self._target_count:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._target_count
                )
            )
            raise ValueError("Invalid time series length")

        y = target[self._target_var].values
        logger.debug("y={}".format(y))

        # compute local data
        time_idx = np.where(self._local_ds["time"] == time)[0][0]
        time_slice = slice(time_idx - self._timeseries_weeks + 1, time_idx + 1)
        local_ds = (
            self._local_ds[self._input_vars + self._oci_input_vars]
            .sel(latitude=lat, longitude=lon)
            .isel(time=time_slice)
        )
        local_data = xr.concat(
            [
                local_ds[var_name]
                for var_name in self._input_vars + self._oci_input_vars
            ],
            dim="values",
        )
        local_data = local_data.transpose("values", "time")
        logger.debug("local_data={}".format(local_data.values))

        return local_data.values, y 


    def balanced_sampler(self): 
        logger.info("Creating weighted random sampler")
        gt_threshold_target = 0.3
        le_threshold_target = 0.2
        zero_threshold_target = 0.5
        logger.info("Target proportions (>,<=,0) = {}, {}, {}".format(gt_threshold_target, le_threshold_target, zero_threshold_target))

        gt_threshold_weight = gt_threshold_target / self._gt_threshold_samples_count
        le_threshold_weight = le_threshold_target / self._le_threshold_samples_count
        zero_threshold_weight = zero_threshold_target / self._zero_threshold_samples_count
        
        gt = np.full(self._gt_threshold_samples_count, gt_threshold_weight)
        le = np.full(self._le_threshold_samples_count, le_threshold_weight)
        zero = np.full(self._zero_threshold_samples_count, zero_threshold_weight)
        samples_weights = np.concatenate((gt, le, zero))
        samples_weights = torch.from_numpy(samples_weights).to(device)
        logger.debug("samples_weights = {}".format(samples_weights))

        return WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)


class GRUTransform:
    def __init__(
        self,
        root_dir,
        timesteps,
        target_week,
    ):
        self.root_dir = root_dir
        self._timesteps = timesteps
        self._target_week = target_week
        if target_week < 1 or target_week > 24:
            raise ValueError("Target week is not valid")
        self._local_mean_std_per_feature = torch.load(
            "{}/{}".format(self.root_dir, "mean_std_stats_local.pk")
        )
        logger.debug(
            "Loaded local dataset mean, std={}".format(self._local_mean_std_per_feature)
        )

    @property
    def target_week(self):
        return self._target_week

    @target_week.setter
    def target_week(self, value):
        if value < 1 or value > 24:
            raise ValueError("Target week is not valid")
        self._target_week = value

    def __call__(self, data):
        x, y = data

        features_count = x.shape[0]
        if self._timesteps <= 0 or self._timesteps > x.shape[1]: 
            logger.warning("Invalid timesteps requested, should be in [1,{}]".format(x.shape[1]))
        timesteps = max(1, min(self._timesteps, x.shape[1]))
        local_mean_std = self._local_mean_std_per_feature[:features_count, :]
        local_mean_std = np.transpose(local_mean_std)
        local_mu = local_mean_std[0]
        local_mu = np.repeat(local_mu, timesteps)
        local_mu = np.reshape(local_mu, (features_count, -1))
        local_std = local_mean_std[1]
        local_std = np.repeat(local_std, timesteps)
        local_std = np.reshape(local_std, (features_count, -1))
        x = (x[:,-timesteps:] - local_mu) / local_std
        x = np.nan_to_num(x, nan=-1.0)
        # transpose from feature x times to time x feature
        x = x.transpose()
        x = np.float32(x)
        x = torch.from_numpy(x)

        # label
        y = np.where(y > 0.0, 1, 0)
        y = np.expand_dims(y, axis=1)
        y = y[self._target_week - 1]
        y = torch.from_numpy(y)

        return x, y
