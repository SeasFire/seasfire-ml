import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import aggr
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TGatConv(torch.nn.Module):
    r"""An implementation of a Temporal Graph Attention Gated Recurrent Cell.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: tuple,
        heads: int = 4,
        concat: bool = False,
        add_self_loops: bool = True,
        add_graph_aggregation_layer: bool = True
    ):
        super(TGatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads 
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.add_graph_aggregation_layer = add_graph_aggregation_layer 

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GATv2Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels[0],
            heads=self.heads,
            concat=self.concat,
            add_self_loops=self.add_self_loops,
        )

        self.conv_z_2 = GATv2Conv(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[1],
            heads=self.heads,
            concat=self.concat,
            add_self_loops=self.add_self_loops,
        )

        if self.add_graph_aggregation_layer:
            self.mean_aggr_z = aggr.MeanAggregation()
        self.linear_z = torch.nn.Linear(2 * self.out_channels[1], self.out_channels[1])

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GATv2Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels[0],
            heads=self.heads,
            concat=self.concat,
            add_self_loops=self.add_self_loops,
        )

        self.conv_r_2 = GATv2Conv(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[1],
            heads=self.heads,
            concat=self.concat,
            add_self_loops=self.add_self_loops,
        )

        if self.add_graph_aggregation_layer:
            self.mean_aggr_r = aggr.MeanAggregation()
        self.linear_r = torch.nn.Linear(2 * self.out_channels[1], self.out_channels[1])

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GATv2Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels[0],
            heads=self.heads,
            concat=self.concat,
            add_self_loops=self.add_self_loops,
        )

        self.conv_h_2 = GATv2Conv(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[1],
            heads=self.heads,
            concat=self.concat,
            add_self_loops=self.add_self_loops,
        )

        if self.add_graph_aggregation_layer:
            self.mean_aggr_h = aggr.MeanAggregation()
        
        self.linear_h = torch.nn.Linear(2 * self.out_channels[1], self.out_channels[1])

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H, readout_batch):
        if H is None:
            dim_0 = (readout_batch.unique(return_counts=True))[0].shape[0]
            H = torch.zeros(dim_0, self.out_channels[1]).to(X.device)  # (b,16)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, readout_batch):
        # layer1
        Z_temp = self.conv_z(X, edge_index, edge_weight)  # (num_nodes, 32)
        Z_temp = Z_temp.relu()
        Z_temp = F.dropout(Z_temp, p=0.3, training=self.training)

        # layer2
        Z_temp = self.conv_z_2(Z_temp, edge_index, edge_weight)  # (num_nodes, 16)
        Z_temp = Z_temp.relu()
        Z_temp = F.dropout(Z_temp, p=0.2, training=self.training)

        if self.add_graph_aggregation_layer:
            # Readout layer
            index = (
                torch.zeros(X.shape[0], dtype=int)
                if readout_batch is None
                else readout_batch
            )
            index = index.to(device)
            Z_temp = self.mean_aggr_z(Z_temp, index)  # (b,16)

        Z = torch.cat([Z_temp, H], axis=1)  # (b, 32)

        Z = self.linear_z(Z)  # (b,16)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, readout_batch):
        # layer1
        R_temp = self.conv_r(X, edge_index, edge_weight)  # (num_nodes, 32)
        R_temp = R_temp.relu()
        R_temp = F.dropout(R_temp, p=0.3, training=self.training)

        # layer2
        R_temp = self.conv_r_2(R_temp, edge_index, edge_weight)  # (num_nodes, 16)
        R_temp = R_temp.relu()
        R_temp = F.dropout(R_temp, p=0.2, training=self.training)

        if self.add_graph_aggregation_layer:
            # Readout layer
            index = (
                torch.zeros(X.shape[0], dtype=int)
                if readout_batch is None
                else readout_batch
            )
            index = index.to(device)
            R_temp = self.mean_aggr_r(R_temp, index)  # (b,16)

        R = torch.cat([R_temp, H], axis=1)  # (b, 32)

        R = self.linear_r(R)  # (b,16)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(
        self, X, edge_index, edge_weight, H, R, readout_batch
    ):
        # layer1
        H_tilde_temp = self.conv_h(X, edge_index, edge_weight)  # (num_nodes, 32)
        H_tilde_temp = H_tilde_temp.relu()
        H_tilde_temp = F.dropout(H_tilde_temp, p=0.3, training=self.training)

        # layer2
        H_tilde_temp = self.conv_h_2(
            H_tilde_temp, edge_index, edge_weight
        ) # (num_nodes, 16)
        H_tilde_temp = H_tilde_temp.relu()
        H_tilde_temp = F.dropout(H_tilde_temp, p=0.2, training=self.training)

        if self.add_graph_aggregation_layer:
            # Readout layer
            index = (
                torch.zeros(X.shape[0], dtype=int)
                if readout_batch is None
                else readout_batch
            )
            index = index.to(device)
            H_tilde_temp = self.mean_aggr_z(H_tilde_temp, index)  # (b,16)

        H_tilde = torch.cat([H_tilde_temp, H * R], axis=1)  # (b, 32)

        H_tilde = self.linear_h(H_tilde)  # (b,16)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        readout_batch=None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H, readout_batch)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, readout_batch)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, readout_batch)
        H_tilde = self._calculate_candidate_state(
            X, edge_index, edge_weight, H, R, readout_batch
        )
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
