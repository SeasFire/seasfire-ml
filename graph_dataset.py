import torch
from torch_geometric.data import Dataset, Data
import os
import logging

logger = logging.getLogger(__name__)

class GraphDataset(Dataset):
    def __init__(self, root_dir, transform):
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

        # load self._indices 
        length = len([entry for entry in os.listdir(self.root_dir)])
        self._indices = []
        for idx in range(length): 
            if os.path.exists(os.path.join(self.root_dir, "graph_{}.pt".format(idx))): 
                self._indices.append(idx)

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        return torch.load(os.path.join(self.root_dir, "graph_{}.pt".format(idx)))
