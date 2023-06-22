import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A3TGCN2(torch.nn.Module):
    r"""A version of A3T-GCN with multiple layers.`_
    Args:
        tgcn_model: Basic TGCN model constructor
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
    """

    def __init__(
        self,
        tgcn_model,
        in_channels: int,
        out_channels: tuple,
        periods: int,
        add_graph_aggregation_layer: bool,
        **kwargs
    ):
        super(A3TGCN2, self).__init__()

        self.periods = periods
        self._base_tgcn = tgcn_model(
            in_channels=in_channels,
            out_channels=out_channels,
            add_graph_aggregation_layer=add_graph_aggregation_layer,
            **kwargs
        )
        
        self.attention = torch.nn.Parameter(torch.empty(periods, device=device))
        torch.nn.init.uniform_(self.attention)

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
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.
        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self.attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H, readout_batch
            )
        return H_accum


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
        self.tgnn = A3TGCN2(
            tgcn_model=tgcn_model,
            in_channels=node_features,
            out_channels=hidden_channels,
            periods=periods,
            add_graph_aggregation_layer=True,
            **kwargs
        )
        self.fc = torch.nn.Linear(hidden_channels[-1], 1)
        self.sigmoid = torch.nn.Sigmoid()

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

        h = self.tgnn(X, edge_index, edge_weight, H, readout_batch)
        h.to(device)
        h = F.relu(h)
        out = self.fc(h)
        out = self.sigmoid(out)
            
        return out
