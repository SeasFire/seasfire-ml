import torch
import logging
from .conv_lstm import ConvLSTM
from .attention import Encoder as TransformerEncoder
from torch_geometric.nn import MLP
import torch.nn.functional as F

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLstmLocalGlobalModel(torch.nn.Module):
    def __init__(
        self,
        local_height,
        local_width,
        local_features,
        local_hidden_channels,
        local_kernel_size,
        local_layers,
        global_height,        
        global_width,
        global_features,
        global_hidden_channels,
        global_kernel_size,
        global_layers,
        decoder_hidden_channels=None,
        include_global=True,
    ):
        super(ConvLstmLocalGlobalModel, self).__init__()

        if local_hidden_channels[-1] != global_hidden_channels[-1]:
            raise ValueError(
                "Output embedding of each ConvLSTM (local and global) should be the same"
            )

        self.local_conv_lstm = ConvLSTM(
            local_features,
            local_hidden_channels,
            local_kernel_size,
            local_layers,
            True,
            True,
            False,
            dilation=1,
        )
        flattened_channels = local_hidden_channels[-1] * local_height * local_width

        self._include_global = include_global
        if include_global:
            self.global_conv_lstm = ConvLSTM(
                global_features,
                global_hidden_channels,
                global_kernel_size,
                global_layers,
                True,
                True,
                False,
                dilation=1,
            )
            flattened_channels += global_hidden_channels[-1] * global_height * global_width

        if decoder_hidden_channels is None:
            self.decoder = torch.nn.Linear(flattened_channels, 1)
        else:
            self.decoder = MLP([flattened_channels] + decoder_hidden_channels + [1])

    def forward(
        self,
        local_x: torch.FloatTensor,
        global_x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = local_x.shape[0]
        _, local_H = self.local_conv_lstm(
            local_x,
        )
        local_H = local_H[0][0]
        local_H = local_H.view(batch_size, -1)

        if self._include_global:
            _, global_H = self.global_conv_lstm(
                global_x,
            )
            global_H = global_H[0][0]
            global_H = global_H.view(batch_size, -1)

        # logger.info("local_H={}".format(local_H.shape))
        # logger.info("global_H={}".format(global_H.shape))

        if self._include_global:
            h = torch.cat((local_H, global_H), dim=1)
        else:
            h = local_H

        h = self.decoder(h)
        return h.squeeze(1)
