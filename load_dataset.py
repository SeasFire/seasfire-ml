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
        graph.y = graph.y/1000.0
        graph.x = torch.cat((graph.x[:,:4], graph.x[:,5:]), axis = 1)
        #graph.y = graph.y / 73000.0

        graph.pos[:,:2] = torch.cos(graph.pos[:,:2])
        # print(graph.pos)

        if self.transforms is not None:
            graph = self.transforms.transform(graph)
            graph.x = torch.nan_to_num(graph.x, nan=-1.0)
        
        graph.x = torch.cat((graph.x[:,:], graph.pos[:,:2]), axis = 1)
        
        return graph

