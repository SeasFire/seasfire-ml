import torch
import logging
import torch.nn.functional as F
from .tgcn2 import TGCN2
from .attention import Encoder as TransformerEncoder

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalGlobalModel(torch.nn.Module):
    def __init__(
        self,
        local_node_features,
        local_hidden_channels,
        local_timesteps,
        global_node_features,
        global_hidden_channels,
        global_timesteps,
        total_nodes,
    ):
        super(LocalGlobalModel, self).__init__()
        if local_hidden_channels[-1] != global_hidden_channels[-1]:
            raise ValueError("Output embedding of each TGCN should be the same")
        self.total_nodes = total_nodes
        self.local_gnn = TGCN2(
            in_channels=local_node_features,
            out_channels=local_hidden_channels,
            add_graph_aggregation_layer=False,
        )
        self.local_timesteps = local_timesteps
        self.global_gnn = TGCN2(
            in_channels=global_node_features,
            out_channels=global_hidden_channels,
            add_graph_aggregation_layer=False,
        )
        self.global_timesteps = global_timesteps
        self.attention = TransformerEncoder(
            local_hidden_channels[-1],
            total_nodes,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
        )
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool1d(local_hidden_channels[-1])
        self.fc = torch.nn.Linear(total_nodes * local_hidden_channels[-1], 1)

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
        
        local_x = local_x[:,:,-self.local_timesteps:]
        for period in range(self.local_timesteps):
            local_H = self.local_gnn(
                local_x[:, :, period],
                local_edge_index,
                local_edge_weight,
                local_H,
                readout_batch,
            )

        global_x = global_x[:,:,-self.global_timesteps:]
        for period in range(self.global_timesteps):
            global_H = self.global_gnn(
                global_x[:, :, period],
                global_edge_index,
                global_edge_weight,
                global_H,
                readout_batch,
            )

        batch_size = len(torch.unique(readout_batch))

        local_vertex_count = local_H.shape[0] // batch_size
        local_H = local_H.view(batch_size, local_vertex_count, -1)

        global_vertex_count = global_H.shape[0] // batch_size
        global_H = global_H.view(batch_size, global_vertex_count, -1)
        h = torch.cat((local_H, global_H), dim=1)
        h = self.attention(h)
        h = self.global_avg_pooling(h)
        h = h.view(batch_size, -1)
        h = self.fc(h)
        return h.squeeze(1)
