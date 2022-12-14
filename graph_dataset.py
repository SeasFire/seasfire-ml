import torch
from torch_geometric.data import Dataset
import os

class GraphDataset(Dataset):
    def __init__(self, root_dir, transform, task, append_position_as_feature=True):
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
        self.append_position_as_feature = append_position_as_feature

    def __len__(self):
        return len([entry for entry in os.listdir(self.root_dir)])

    def __getitem__(self, idx):
        graph = torch.load(os.path.join(self.root_dir, "graph_{}.pt".format(idx)))
        self.graph = graph
        
        # Define label
        if self.task == "binary":
            self.graph.y = torch.where(self.graph.y > 0.0, 1, 0)
            self.graph.y = torch.nn.functional.one_hot(self.graph.y, 2).float()
        elif self.task == "regression":
            self.graph.y = self.graph.y / 1000.0
        else:
            raise ValueError("Invalid task")

        # Standardize features
        if self.transform is not None:
            self.graph.x = self.transform.transform(self.graph.x)
            self.graph.x = torch.nan_to_num(self.graph.x, nan=-1.0)

        # Concatenate positions with features
        if self.append_position_as_feature:
            positions = self.graph.pos.unsqueeze(2).expand(-1, -1, self.graph.x.shape[2])
            self.graph.x = torch.cat((self.graph.x, positions), dim=1)

        return self.graph

    def num_features(self) -> int:
        r"""Returns the number of features per node in the dataset.
        Alias for :py:attr:`~num_node_features`."""
        return self.graph.x.shape[1]
