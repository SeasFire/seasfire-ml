import torch
import torch.nn.functional as F
from .tgcn2 import TGCN2
from .tgatconv import TGatConv


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
        self, tgcn_model, in_channels: int, out_channels: tuple, periods: int, **kwargs
    ):
        super(A3TGCN2, self).__init__()

        self.periods = periods
        self._base_tgcn = tgcn_model(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        output_channels,
        periods,
        learning_rate,
        weight_decay,
        task,
        **kwargs
    ):
        super(AttentionGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell with 2 layers
        self.tgnn = A3TGCN2(
            tgcn_model=tgcn_model,
            in_channels=node_features,
            out_channels=output_channels,
            periods=periods,
            **kwargs
        )
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(output_channels[1], output_channels[2])

        # print(self.parameters)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.task = task

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

        h = self.linear(h)
        if self.task == "binary":
            h = torch.softmax(h, dim=1)

        return h
