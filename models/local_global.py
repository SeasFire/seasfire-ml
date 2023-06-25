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
        periods,
    ):
        super(LocalGlobalModel, self).__init__()
        self.periods = periods
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
        
        for period in range(self.periods):
            local_H = self.local_gnn(
                local_x[:, :, period], local_edge_index, local_edge_weight, local_H, readout_batch
            )

            global_H = self.global_gnn(
                global_x[:, :, period], global_edge_index, global_edge_weight, global_H, readout_batch
            )

        logger.info("local_H shape = {}".format(local_H.shape))
        logger.info("local_H = {}".format(local_H))
        logger.info("global_H shape = {}".format(global_H.shape))
        logger.info("global_H = {}".format(global_H))

        logger.info("readout_batch_H shape = {}".format(readout_batch.shape))
        logger.info("readout_batch_H = {}".format(readout_batch))

        H = torch.cat((local_H, global_H), dim=1)

        logger.info("H shape = {}".format(H.shape))
        logger.info("H = {}".format(H))

        return H

        # out = self.fc(out[:, -1, :])
        # return out
