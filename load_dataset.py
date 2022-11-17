import torch 
from torch_geometric.data import Dataset
import os


class LoadData(Dataset):
    def __init__(self, root_dir, transform=None): #, csv_file
        '''
        Desc
        ----
        Loads dataset of graphs. Each graph is a pt file.
        Dataset consists of multiple pt files that are stored in the root_dir.

        Args
        ----
        root_dir: str
            * whole path where all graphs (pt files) stored.
            
        Returns
        ----
        All the torch_geometric.data.Data objects (graphs)
        '''
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([entry for entry in os.listdir(self.root_dir)])

    def __getitem__(self, idx):
        graph = torch.load(self.root_dir + f'/graph_{idx+1}.pt')

        return graph

