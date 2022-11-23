import torch 
from torch_geometric.data import Dataset
import os
from scale_dataset import *


class GraphLoader(Dataset):
    def __init__(self, root_dir, transforms): #, csv_file
        '''
        Desc
        ----
        Loads dataset of graphs. Each graph is a pt file.
        Dataset consists of multiple pt files that are stored in the root_dir.

        Args
        ----
        root_dir: str
            * whole path where all graphs (pt files) were stored.
            
        Returns
        ----
        All the torch_geometric.data.Data objects (graphs)
        '''
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len([entry for entry in os.listdir(self.root_dir)])

    def __getitem__(self, idx):
        graph = torch.load(self.root_dir + f'graph_{idx}.pt')
        graph.x = torch.cat((graph.x[:,:4], graph.x[:,5:]), axis = 1)
        #graph.y = graph.y / 73000.0

        number_of_nodes = graph.x.shape[0]

        if self.transforms is not None:
            graph.x[0] = self.transforms.transform(graph.x[0])
            for node_idx in range(0, number_of_nodes):
                graph.x[node_idx] = self.transforms.transform(graph.x[node_idx])
                graph.x[node_idx] = torch.nan_to_num(graph.x[node_idx], nan=-1.0)

        return graph
    
