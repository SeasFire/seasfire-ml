import torch
from torch_geometric.data import Dataset, Data
import os
import xarray as xr
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class LocalGlobalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # load self._indices
        local_filenames = [
            entry for entry in os.listdir(self.root_dir) if entry.startswith("local_")
        ]
        global_filenames = [
            entry for entry in os.listdir(self.root_dir) if entry.startswith("global_")
        ]
        ground_truth_filenames = [
            entry
            for entry in os.listdir(self.root_dir)
            if entry.startswith("ground_truth_")
        ]

        if len(local_filenames) != len(global_filenames) or len(
            ground_truth_filenames
        ) != len(global_filenames):
            raise ValueError("Missing dataset files")

        # initializing substrings
        sub1 = "_"
        sub2 = "."

        self._indices = [
            filename[filename.index(sub1) + len(sub1) : filename.index(sub2)]
            for filename in local_filenames
        ]

        self._metadata = torch.load(os.path.join(self.root_dir, "metadata.pt"))
        logger.debug("Metadata={}".format(self._metadata))

    @property
    def len(self):
        return len(self._indices)

    def get(self, idx: int) -> Data:
        # load datasets
        with xr.open_dataset(
            os.path.join(self.root_dir, "local_{}.hd5".format(idx))
        ) as ds:
            local_ds = ds.load()
        with xr.open_dataset(
            os.path.join(self.root_dir, "global_{}.hd5".format(idx))
        ) as ds:
            global_ds = ds.load()
        with xr.open_dataset(
            os.path.join(self.root_dir, "ground_truth_{}.hd5".format(idx))
        ) as ds:
            ground_truth_ds = ds.load()
        with xr.open_dataset(
            os.path.join(self.root_dir, "area_{}.hd5".format(idx))
        ) as ds:
            area_ds = ds.load()    
        sample_metadata = torch.load(
            os.path.join(self.root_dir, "metadata_{}.pt".format(idx))
        )    

        # compute ground truth features
        target_var = self._metadata["target_var"]
        y = ground_truth_ds[target_var].values
        logger.debug("y={}".format(y))

        # compute local graph features
        input_vars = self._metadata["input_vars"]
        logger.debug("Found input vars={}".format(input_vars))
        oci_input_vars = self._metadata["oci_input_vars"]
        logger.debug("Found oci input vars={}".format(oci_input_vars))

        for oci_var in oci_input_vars:
            local_ds[oci_var] = local_ds[oci_var].expand_dims(
                dim={
                    "latitude": local_ds["latitude"],
                    "longitude": local_ds["longitude"],
                }
            )

        local_data = xr.concat([local_ds[var_name] for var_name in input_vars+oci_input_vars], dim="values")
        local_data = local_data.stack(vertex=("latitude", "longitude"))
        local_data = local_data.transpose("vertex", "values", "time")
        logger.debug("local_data={}".format(local_data))

        local_pos = np.array(list(map(np.array, local_data["vertex"].values)))
        logger.debug("local_pos={}".format(local_pos))

        # Compute global graph features
        for oci_var in oci_input_vars:
            global_ds[oci_var] = global_ds[oci_var].expand_dims(
                dim={
                    "latitude": global_ds["latitude"],
                    "longitude": global_ds["longitude"],
                }
            )
        global_data = xr.concat([global_ds[var_name] for var_name in input_vars+oci_input_vars], dim="values")
        global_data = global_data.stack(vertex=("latitude", "longitude"))
        global_data = global_data.transpose("vertex", "values", "time")
        logger.debug("global_data={}".format(global_data))

        global_pos = np.array(list(map(np.array, global_data["vertex"].values)))
        logger.debug("global_pos={}".format(global_pos))

        area_data = area_ds.to_array()
        logger.debug("area_data={}".format(area_data))

        #logger.info("test={}".format(test.to_array().values.shape))
        # local_ds = local_ds.transpose('latitude', 'longitude', 'time')
        # logger.info("local_ds[\"oci_censo\"]={}".format(local_ds["oci_censo"]))
        # local_ds["oci_censo"] = local_ds["oci_censo"].expand_dims(dim={"latitude": local_ds["latitude"]})
        # logger.info("test={}".format(local_ds["oci_censo"]))
        # local_ds = (
        #     local_ds
        #     .expand_dims(dim={"time": sample_region.time}, axis=0)
        # )

        latlon_shape=self._metadata["local_latlon_shape"]
        edge_index = self._get_knn_for_grid(latlon_shape[0], latlon_shape[1], 3)

        global_latlon_shape=self._metadata["global_latlon_shape"]
        global_edge_index = self._get_knn_for_grid(global_latlon_shape[0], global_latlon_shape[1], 3)

        return Data(
            x = local_data.values,
            pos = local_pos,
            edge_index=edge_index,
            latlon_shape=latlon_shape,
            center_lat=sample_metadata["center_lat"],
            center_lon=sample_metadata["center_lon"],
            center_time=sample_metadata["center_time"],
            center_vertex_idx = local_data.values.shape[0] // 2,
            area = area_data.values[0],
            global_x = global_data.values,
            global_pos = global_pos,
            global_latlon_shape=global_latlon_shape,
            global_edge_index=global_edge_index,
            y = y,
        )

    def _get_knn_for_grid(self, lat_dim, lon_dim, k, add_self_loops=True): 
        filename = os.path.join(self.root_dir, "knn_{}_{}_{}_{}.pt".format(lat_dim, lon_dim, k, add_self_loops))
        try: 
            knn = torch.load(filename)
            return knn
        except FileNotFoundError: 
            pass
        
        knn = self._generate_knn_for_grid(lat_dim, lon_dim, k, add_self_loops=add_self_loops)
        torch.save(knn, filename)

    def _generate_knn_for_grid(self, lat_dim, lon_dim, k, add_self_loops=True): 
        points = np.zeros((lat_dim * lon_dim, 2))
        for i in range(lat_dim): 
            for j in range(lon_dim):
                points[i*lon_dim+j] = [i, j]

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
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
        logger.info(
            "Loaded local dataset mean, std={}".format(self._local_mean_std_per_feature)
        )
        self._global_mean_std_per_feature = torch.load(
            "{}/{}".format(self.root_dir, "mean_std_stats_global.pk")
        )
        logger.info(
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
        local_mean_std = np.transpose(self._local_mean_std_per_feature)
        local_mu = local_mean_std[0]
        local_mu = np.repeat(local_mu, data.x.shape[2])
        local_mu = np.reshape(local_mu, (data.x.shape[1],-1))
        local_std = local_mean_std[1]
        local_std = np.repeat(local_std, data.x.shape[2])
        local_std = np.reshape(local_std, (data.x.shape[1],-1))
        for i in range(0, data.x.shape[0]):
            data.x[i, :, :] = (data.x[i, :, :] - local_mu) / local_std
        data.x = np.nan_to_num(data.x, nan=-1.0)
        data.x = torch.from_numpy(data.x)

        # TODO: add position
        # if self._append_position_as_feature:
        #     local_positions = data.local_pos.unsqueeze(2).expand(
        #         -1, -1, data.local_x.shape[2]
        #     )
        #     data.local_x = torch.cat((data.local_x, local_positions), dim=1)

        # global graph features
        global_mean_std = np.transpose(self._global_mean_std_per_feature)
        global_mu = global_mean_std[0]
        global_mu = np.repeat(global_mu, data.global_x.shape[2])
        global_mu = np.reshape(global_mu, (data.global_x.shape[1],-1))
        global_std = global_mean_std[1]
        global_std = np.repeat(global_std, data.global_x.shape[2])
        global_std = np.reshape(global_std, (data.global_x.shape[1],-1))
        for i in range(0, data.global_x.shape[0]):
            data.global_x[i, :, :] = (data.global_x[i, :, :] - global_mu) / global_std
        data.global_x = np.nan_to_num(data.global_x, nan=-1.0)
        data.global_x = torch.from_numpy(data.global_x)

        # TODO: add position
        # if self._append_position_as_feature:
        #     global_positions = data.global_pos.unsqueeze(2).expand(
        #         -1, -1, data.global_x.shape[2]
        #     )
        #     data.global_x = torch.cat((data.global_x, global_positions), dim=1)

        # label
        y = np.where(data.y > 0.0, 1, 0)
        y = np.expand_dims(y, axis=1)
        y = y[self._target_week - 1]
        data.y = torch.from_numpy(y)

        return data
