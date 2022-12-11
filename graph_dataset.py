import torch
from torch_geometric.data import Dataset
import os
import xarray as xr
from scale_dataset import *


class GraphDataset(Dataset):
    def __init__(self, root_dir, transform, task, shuffle=True): #, drop_last=True
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
        graph = torch.load(os.path.join(self.root_dir, "graph_{}.pt".format(idx)))

        ## Define label
        if self.task == "binary":
            graph.y = torch.where(graph.y > 0.0, 1.0, 0.0)
        elif self.task == "regression":
            graph.y = graph.y / 1000.0

        ## Standardize features
        if self.transform is not None:
            graph.x = self.transform.transform(graph.x)
            graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        ## Add cosine of position to features
        pos_graph = []
        # graph.pos = torch.cos(graph.pos)
        
        for i in range(0, graph.x.shape[0]):
            cosine_pos_features = torch.cos(graph.pos[i,:]).unsqueeze(1)
            cosine_pos_features = cosine_pos_features.expand(cosine_pos_features.shape[0], graph.x.shape[2])
            sine_pos_features = torch.cos(graph.pos[i,:]).unsqueeze(1)
            sine_pos_features = sine_pos_features.expand(sine_pos_features.shape[0], graph.x.shape[2])
            
            graph_features = graph.x[i,:,:]
            
            pos_graph.append(torch.cat((graph_features, cosine_pos_features, sine_pos_features), axis = 0))
        
        graph.x = torch.stack(pos_graph, dim=0)
        # print(graph.x.shape)
        return graph
