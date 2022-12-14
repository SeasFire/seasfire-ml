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

        # Define label
        if self.task == "binary":
            graph.y = torch.where(graph.y > 0.0, 1, 0)
            graph.y = torch.nn.functional.one_hot(graph.y, 2).float()
        elif self.task == "regression":
            graph.y = graph.y / 1000.0
        else:
            raise ValueError("Invalid task")

        # Standardize features
        if self.transform is not None:
            graph.x = self.transform.transform(graph.x)
            graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        # Concatenate positions with features
        if self.append_position_as_feature:
            positions = graph.pos.unsqueeze(2)
            positions = positions.expand(-1, -1, graph.x.shape[2])
            graph.x = torch.cat((graph.x, positions), dim=1)

        return graph
