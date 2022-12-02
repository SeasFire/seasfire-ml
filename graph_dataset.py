import torch
from torch_geometric.data import Dataset
import os
import xarray as xr
from scale_dataset import *


class GraphDataset(Dataset):
    def __init__(self, root_dir, transform, task):
        """
        Desc
        ----
        A graph dataset. Each graph is a pt file.
        Dataset consists of multiple pt files that are stored in the root_dir.

        Args
        ----
        root_dir: str
            * whole path where all graphs (pt files) were stored.

        Returns
        ----
        All the torch_geometric.data.Data objects (graphs)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.task = task

    def __len__(self):
        return len([entry for entry in os.listdir(self.root_dir)])

    def __getitem__(self, idx):
        graph = torch.load(self.root_dir + f"graph_{idx}.pt")

        if self.task == 'binary':
            graph.y = torch.where(graph.y>5000, 1., 0.)
        elif self.task == 'regression':
            graph.y = graph.y / 1000.0

        if self.transform is not None:
            graph.x = self.transform.transform(graph.x)
            graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        # print(graph.x.shape)
        # graph.pos = torch.cos(graph.pos)
        # print(graph.pos)

        return graph
