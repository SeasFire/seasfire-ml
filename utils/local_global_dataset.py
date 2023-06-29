import torch
from torch_geometric.data import Dataset, Data
import os
import xarray as xr
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class LocalGlobalDataset(Dataset):
    def __init__(self, root_dir, local_radius, include_oci_variables=True, transform=None):
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
        logger.info("sp_res={}".format(self._sp_res))

        self._samples = torch.load(os.path.join(self.root_dir, "samples.pt"))
        logger.info("Samples={}".format(len(self._samples)))
        self._indices = list(range(len(self._samples)))

        logger.info("Precomputing global graph")
        self._global_latlon_shape = self._metadata["global_latlon_shape"]
        self._global_edge_index = self._get_knn_for_grid(
            self._global_latlon_shape[0], self._global_latlon_shape[1], 3
        )

        logger.info("Precomputing local graph")
        self._max_radius = self._metadata["max_radius"]
        self._radius = local_radius
        logger.info("Using local radius={}".format(self._radius))
        if self._radius > self._max_radius:
            raise ValueError("Max radius is smaller than local radius.")
        self._latlon_shape = (2*self._radius+1, 2*self._radius+1)
        logger.info("Local grid shape={}".format(self._latlon_shape))
        self._edge_index = self._get_knn_for_grid(
            self._latlon_shape[0], self._latlon_shape[1], 3
        )

        logger.info("Loading global dataset")
        with xr.open_dataset(os.path.join(self.root_dir, "global.h5")) as ds:
            self._global_ds = ds.load()
        logger.debug("global_ds={}".format(self._global_ds))

        # Compute global graph features
        for oci_var in self._oci_input_vars:
            self._global_ds[oci_var] = self._global_ds[oci_var].expand_dims(
                dim={
                    "latitude": self._global_ds["latitude"],
                    "longitude": self._global_ds["longitude"],
                }
            )
        logger.debug("global_ds={}".format(self._global_ds))

        logger.info("Precomputing global positions")
        self._global_pos = np.array(
            list(
                map(
                    np.array,
                    self._global_ds.stack(vertex=("latitude", "longitude"))[
                        "vertex"
                    ].values,
                )
            ),
            dtype=np.float32,
        )
        logger.debug("global_pos={}".format(self._global_pos))

        logger.info("Loading area dataset")
        with xr.open_dataset(os.path.join(self.root_dir, "area.h5")) as ds:
            self._area_ds = ds.load()
        logger.debug("area_ds={}".format(self._area_ds))

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
        return len(self._samples)

    @property
    def local_features(self):
        return tuple(self._input_vars + self._oci_input_vars)

    @property
    def global_features(self):
        return tuple(self._input_vars + self._oci_input_vars)

    @property
    def local_global_nodes(self):
        local_nodes = self._latlon_shape[0] * self._latlon_shape[1]
        global_nodes = self._global_latlon_shape[0] * self._global_latlon_shape[1]
        return local_nodes + global_nodes

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
        target = self._ground_truth_ds.sel(
            latitude=lat,
            longitude=lon,
        ).isel(time=time_slice).fillna(0)

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
        lat_slice = slice(lat + self._radius * self._sp_res, lat - self._radius * self._sp_res)
        lon_slice = slice(lon - self._radius * self._sp_res, lon + self._radius * self._sp_res)

        local_ds = (
            self._local_ds[self._input_vars + self._oci_input_vars]
            .sel(latitude=lat_slice, longitude=lon_slice)
            .isel(time=time_slice)
        )
        local_ds = local_ds.transpose("latitude", "longitude", "time")
        local_data = xr.concat(
            [
                local_ds[var_name]
                for var_name in self._input_vars + self._oci_input_vars
            ],
            dim="values",
        )
        local_data = local_data.stack(vertex=("latitude", "longitude"))
        local_data = local_data.transpose("vertex", "values", "time")
        logger.debug("local_data={}".format(local_data))

        local_pos = np.array(
            list(map(np.array, local_data["vertex"].values)), dtype=np.float32
        )
        logger.debug("local_pos={}".format(local_pos))

        # compute global data
        time_idx = np.where(self._global_ds["time"] == time)[0][0]
        time_slice = slice(time_idx - self._timeseries_weeks + 1, time_idx + 1)
        global_ds = self._global_ds.isel(time=time_slice).load()
        global_data = xr.concat(
            [
                global_ds[var_name]
                for var_name in self._input_vars + self._oci_input_vars
            ],
            dim="values",
        )
        global_data = global_data.stack(vertex=("latitude", "longitude"))
        global_data = global_data.transpose("vertex", "values", "time")
        logger.debug("global_data={}".format(global_data))

        # compute area
        area_data = self._area_ds.sel(latitude=lat, longitude=lon).to_array()
        logger.debug("area_data={}".format(area_data))

        return Data(
            x=local_data.values,
            pos=local_pos,
            edge_index=self._edge_index,
            latlon_shape=self._latlon_shape,
            center_lat=lat,
            center_lon=lon,
            center_time=time,
            center_vertex_idx=local_data.values.shape[0] // 2,
            area=area_data.values[0],
            global_x=global_data.values,
            global_pos=self._global_pos,
            global_latlon_shape=self._global_latlon_shape,
            global_edge_index=self._global_edge_index,
            y=y,
        )

    def _get_knn_for_grid(self, lat_dim, lon_dim, k, add_self_loops=True):
        filename = os.path.join(
            self.root_dir,
            "knn_{}_{}_{}_{}.pt".format(lat_dim, lon_dim, k, add_self_loops),
        )
        try:
            knn = torch.load(filename)
            return knn
        except FileNotFoundError:
            pass

        knn = self._generate_knn_for_grid(
            lat_dim, lon_dim, k, add_self_loops=add_self_loops
        )
        torch.save(knn, filename)
        return knn

    def _generate_knn_for_grid(self, lat_dim, lon_dim, k, add_self_loops=True):
        points = np.zeros((lat_dim * lon_dim, 2))
        for i in range(lat_dim):
            for j in range(lon_dim):
                points[i * lon_dim + j] = [i, j]

        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(points)
        _, indices = nbrs.kneighbors(points)

        edges = []
        for i in range(lat_dim * lon_dim):
            for j in indices[i]:
                if i != j:
                    edges.append((i, j))
                    edges.append((j, i))
                elif add_self_loops:
                    edges.append((i, j))

        sources, targets = zip(*edges)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        logger.debug("Computed edge tensor= {}".format(edge_index))

        return edge_index


class LocalGlobalTransform:
    def __init__(
        self,
        root_dir,
        target_week,
        append_position_as_feature=True,
    ):
        self.root_dir = root_dir
        self._target_week = target_week
        if target_week < 1 or target_week > 24:
            raise ValueError("Target week is not valid")
        self._local_mean_std_per_feature = torch.load(
            "{}/{}".format(self.root_dir, "mean_std_stats_local.pk")
        )
        logger.debug(
            "Loaded local dataset mean, std={}".format(self._local_mean_std_per_feature)
        )
        self._global_mean_std_per_feature = torch.load(
            "{}/{}".format(self.root_dir, "mean_std_stats_global.pk")
        )
        logger.debug(
            "Loaded global dataset mean, std={}".format(
                self._global_mean_std_per_feature
            )
        )
        self._append_position_as_feature = append_position_as_feature

    @property
    def target_week(self):
        return self._target_week

    @target_week.setter
    def target_week(self, value):
        if value < 1 or value > 24:
            raise ValueError("Target week is not valid")
        self._target_week = value

    def __call__(self, data):
        # local graph features
        features_count = data.x.shape[1]
        local_mean_std = self._local_mean_std_per_feature[:features_count,:]
        local_mean_std = np.transpose(local_mean_std)
        local_mu = local_mean_std[0]
        local_mu = np.repeat(local_mu, data.x.shape[2])
        local_mu = np.reshape(local_mu, (features_count, -1))
        local_std = local_mean_std[1]
        local_std = np.repeat(local_std, data.x.shape[2])
        local_std = np.reshape(local_std, (features_count, -1))
        for i in range(0, data.x.shape[0]):
            data.x[i, :, :] = (data.x[i, :, :] - local_mu) / local_std
        data.x = np.nan_to_num(data.x, nan=-1.0)

        if self._append_position_as_feature:
            latlon = np.transpose(data.pos)
            lat = latlon[0]
            lon = latlon[1]
            cos_lat = np.reshape(
                np.repeat(np.cos(lat * np.pi / 180), data.x.shape[2]),
                (-1, data.x.shape[2]),
            )
            sin_lat = np.reshape(
                np.repeat(np.sin(lat * np.pi / 180), data.x.shape[2]),
                (-1, data.x.shape[2]),
            )
            cos_lon = np.reshape(
                np.repeat(np.cos(lon * np.pi / 180), data.x.shape[2]),
                (-1, data.x.shape[2]),
            )
            sin_lon = np.reshape(
                np.repeat(np.sin(lon * np.pi / 180), data.x.shape[2]),
                (-1, data.x.shape[2]),
            )
            pos = np.stack((cos_lat, sin_lat, cos_lon, sin_lon), axis=1)
            data.x = np.concatenate((data.x, pos), axis=1)

        data.x = torch.from_numpy(data.x)

        # global graph features
        global_features_count = data.global_x.shape[1]
        global_mean_std = self._global_mean_std_per_feature[:global_features_count,:]
        global_mean_std = np.transpose(global_mean_std)
        global_mu = global_mean_std[0]
        global_mu = np.repeat(global_mu, data.global_x.shape[2])
        global_mu = np.reshape(global_mu, (global_features_count, -1))
        global_std = global_mean_std[1]
        global_std = np.repeat(global_std, data.global_x.shape[2])
        global_std = np.reshape(global_std, (global_features_count, -1))
        for i in range(0, data.global_x.shape[0]):
            data.global_x[i, :, :] = (data.global_x[i, :, :] - global_mu) / global_std
        data.global_x = np.nan_to_num(data.global_x, nan=-1.0)

        if self._append_position_as_feature:
            latlon = np.transpose(data.global_pos)
            lat = latlon[0]
            lon = latlon[1]
            cos_lat = np.reshape(
                np.repeat(np.cos(lat * np.pi / 180), data.global_x.shape[2]),
                (-1, data.global_x.shape[2]),
            )
            sin_lat = np.reshape(
                np.repeat(np.sin(lat * np.pi / 180), data.global_x.shape[2]),
                (-1, data.global_x.shape[2]),
            )
            cos_lon = np.reshape(
                np.repeat(np.cos(lon * np.pi / 180), data.global_x.shape[2]),
                (-1, data.global_x.shape[2]),
            )
            sin_lon = np.reshape(
                np.repeat(np.sin(lon * np.pi / 180), data.global_x.shape[2]),
                (-1, data.global_x.shape[2]),
            )
            pos = np.stack((cos_lat, sin_lat, cos_lon, sin_lon), axis=1)
            data.global_x = np.concatenate((data.global_x, pos), axis=1)

        data.global_x = torch.from_numpy(data.global_x)

        # label
        y = np.where(data.y > 0.0, 1, 0)
        y = np.expand_dims(y, axis=1)
        y = y[self._target_week - 1]
        data.y = torch.from_numpy(y)

        return data
