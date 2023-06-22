import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionGNN(torch.nn.Module):
    def __init__(
        self,
        tgcn_model,
        node_features,
        hidden_channels,
        periods,
        **kwargs
    ):
        super(AttentionGNN, self).__init__()
        self.periods = periods
        self._base_tgcn = tgcn_model(
            in_channels=node_features,
            out_channels=hidden_channels,
            add_graph_aggregation_layer=True,
            **kwargs
        )
        self.attention = torch.nn.Parameter(torch.empty(periods, device=device))
        torch.nn.init.uniform_(self.attention)

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
        H_accum = 0
        probs = torch.nn.functional.softmax(self.attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H, readout_batch
            )
        h = F.relu(H_accum)
        out = self.fc(h)
        return out
