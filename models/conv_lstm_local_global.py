import torch
import logging
from .conv_lstm import ConvLSTM
from .attention import Encoder as TransformerEncoder
from torch_geometric.nn import MLP

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLstmLocalGlobalModel(torch.nn.Module):
    def __init__(
        self,
        local_features,
        local_hidden_channels,
        local_kernel_size,
        local_layers,
        global_features,
        global_hidden_channels,
        global_kernel_size,
        global_layers,
        decoder_hidden_channels=None,
        include_global=True,
    ):
        super(ConvLstmLocalGlobalModel, self).__init__()

        if local_hidden_channels[-1] != global_hidden_channels[-1]:
            raise ValueError("Output embedding of each TGCN (local and global) should be the same")
        
        self.local_conv_lstm = ConvLSTM(
            local_features,
            local_hidden_channels,
            local_kernel_size,
            local_layers,
            True,
            True,
            False,
            dilation=1
        )
        output_channels = local_hidden_channels

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
                dilation=1
            )
            output_channels += global_hidden_channels

        if decoder_hidden_channels is None: 
            self.decoder = torch.nn.Linear(output_channels, 1)
        else: 
            self.decoder = MLP([output_channels] + decoder_hidden_channels + [1])

    def forward(
        self,
        local_x: torch.FloatTensor,
        global_x: torch.FloatTensor,
        readout_batch=None,
    ) -> torch.FloatTensor:
        
        #local_x = local_x[:,:,-self.local_timesteps:]
        _, local_H = self.local_conv_lstm(
            local_x,
        )

        if self._include_global:
            #global_x = global_x[:,:,-self.global_timesteps:]
            _, global_H = self.global_gnn(
                global_x,
            )

        #batch_size = len(torch.unique(readout_batch))

        # local_vertex_count = local_H.shape[0] // batch_size
        # local_H = local_H.view(batch_size, local_vertex_count, -1)

        if self._include_global:
        #     global_vertex_count = global_H.shape[0] // batch_size
        #     global_H = global_H.view(batch_size, global_vertex_count, -1)
            h = torch.cat((local_H, global_H), dim=1)
        else:
            h = local_H

        # h = self.attention(h)
        # h = h.view(batch_size, -1)
        # h = self.decoder(h)
        return h.squeeze(1)
