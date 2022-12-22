import torch
from torch_geometric.data import Dataset, Data
import os
import logging

logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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
        filenames = [entry for entry in os.listdir(self.root_dir)]
        # initializing substrings
        sub1 = "_"
        sub2 = "."
        
        self._indices = [filename[filename.index(sub1) + len(sub1): filename.index(sub2)] for filename in filenames]

        # define number of features per node --> same for all nodes
        graph = torch.load(
            os.path.join(self.root_dir, "graph_{}.pt".format(self._indices[0]))
        )

        if self.transform is not None: 
            t_graph = self.transform(graph)
            if isinstance(t_graph , Data):
                self._num_features = t_graph.x.shape[1]
            else: 
                self._num_features = t_graph[0].shape[1]
        else: 
            self._num_features = graph.x.shape[1]

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the dataset.
        Alias for :py:attr:`~num_node_features`."""
        return self._num_features

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        return self._num_features

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        return torch.load(os.path.join(self.root_dir, "graph_{}.pt".format(idx)))
