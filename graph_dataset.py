import torch
from torch_geometric.data import Dataset
import os
from scale_dataset import *


class GraphDataset(Dataset):
    def __init__(self, root_dir, transforms):
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
        self.transforms = transforms

    def __len__(self):
        return len([entry for entry in os.listdir(self.root_dir)])

    def __getitem__(self, idx):
        graph = torch.load(self.root_dir + f"graph_{idx}.pt")

        graph.y = graph.y / 1000.0

        if self.transforms is not None:
            graph.x = self.transforms.transform(graph.x)
            graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        # graph.pos = torch.cos(graph.pos)
        # print(graph.pos)
        # graph.x = torch.cat((graph.x[:,:], graph.pos[:,:2]), axis = 1)

        return graph
