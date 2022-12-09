import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class my_TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(my_TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H, readout_batch):
        if H is None:
            # torch.sum((x.unique(return_counts=True))[0])
            H = torch.zeros(2, self.out_channels).to(X.device)
            print("H_set: ", H.shape)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, readout_batch):
        # Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)

        print(X.shape)

        #GCN
        Z_temp = self.conv_z(X, edge_index, edge_weight) # (b, 207, 64)
        
        # Readout layer
        readout_batch = torch.zeros(X.shape[0], dtype=int) if readout_batch is None else readout_batch
        readout_batch = readout_batch.to(device)
        Z = global_mean_pool(Z_temp, readout_batch) #(b,64)
        print(Z.shape)
        Z = torch.cat([Z, H], axis=1) # (b, 64)
        print("Z: ", Z.shape)

        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, readout_batch):
        # R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        
        #GCN
        R_temp = self.conv_r(X, edge_index, edge_weight) # (b, 207, 64)
        
        # Readout layer
        readout_batch = torch.zeros(X.shape[0], dtype=int) if readout_batch is None else readout_batch
        readout_batch = readout_batch.to(device)
        R_temp = global_mean_pool(R_temp, readout_batch) #(b,64)
        
        R = torch.cat([R_temp, H], axis=1) # (b, 64)
        print("r: ", R.shape)
        
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, readout_batch):
        # H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        
        #GCN
        H_tilde_temp = self.conv_h(X, edge_index, edge_weight) # (b, 207, 64)
        
        # Readout layer
        readout_batch = torch.zeros(X.shape[0], dtype=int) if readout_batch is None else readout_batch
        readout_batch = readout_batch.to(device)
        H_tilde_temp = global_mean_pool(H_tilde_temp, readout_batch) #(b,64)
        
        H_tilde = torch.cat([H_tilde_temp, H], axis=1) # (b, 64)
        print("H_tilde: ", H_tilde.shape)
        
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        print("H_final: ", H.shape)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        readout_batch = None,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None
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
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, readout_batch)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
