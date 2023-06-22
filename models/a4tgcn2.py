import torch
import torch.nn.functional as F
from torch_geometric.nn import aggr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention2GNN(torch.nn.Module):
    def __init__(
        self,
        tgcn_model,
        node_features,
        hidden_channels,
        periods,
        **kwargs
    ):
        super(Attention2GNN, self).__init__()
        self.periods = periods
        self._base_tgcn = tgcn_model(
            in_channels=node_features,
            out_channels=hidden_channels,
            add_graph_aggregation_layer=False,
            **kwargs
        )
        self.mean_aggr_z = aggr.MeanAggregation()
        self.fc = torch.nn.Linear(hidden_channels[-1], 1)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        readout_batch=None,
    ) -> torch.FloatTensor:
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        for period in range(self.periods):
            H = self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H, readout_batch
            )
        # keep the last
        H_accum = H
            
        # Readout layer
        index = (
            torch.zeros(X.shape[0], dtype=int)
            if readout_batch is None
            else readout_batch
        )
        index = index.to(device)
        H_accum = self.mean_aggr_z(H_accum, index)  # (b,16)

        # h.to(device)
        h = F.relu(H_accum)
        out = self.fc(h)
        return out
