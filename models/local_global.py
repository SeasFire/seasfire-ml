import torch
import logging
import torch.nn.functional as F
from .tgcn2 import TGCN2


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalGlobalModel(torch.nn.Module):
    def __init__(
        self,
        local_node_features,
        local_hidden_channels,
        global_node_features,
        global_hidden_channels,
    ):
        super(LocalGlobalModel, self).__init__()
        self.local_gnn = TGCN2(
            in_channels=local_node_features,
            out_channels=local_hidden_channels,
            add_graph_aggregation_layer=False,
        )
        self.global_gnn = TGCN2(
            in_channels=global_node_features,
            out_channels=global_hidden_channels,
            add_graph_aggregation_layer=False,
        )
        output_channels = local_hidden_channels[-1] + global_hidden_channels[-1]
        self.fc = torch.nn.Linear(output_channels, 1)

    def forward(
        self,
        local_x: torch.FloatTensor,
        global_x: torch.FloatTensor,
        local_edge_index: torch.LongTensor,
        global_edge_index: torch.LongTensor,
        local_edge_weight: torch.FloatTensor = None,
        global_edge_weight: torch.FloatTensor = None,
        local_H: torch.FloatTensor = None,
        global_H: torch.FloatTensor = None,
        readout_batch=None,
    ) -> torch.FloatTensor:
        local_out = self.local_gnn(
            local_x, local_edge_index, local_edge_weight, local_H, readout_batch
        )
        global_out = self.global_gnn(
            global_x, global_edge_index, global_edge_weight, global_H, readout_batch
        )

        return local_out, global_out

        # out = self.fc(out[:, -1, :])
        # return out
