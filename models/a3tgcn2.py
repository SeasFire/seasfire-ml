import torch
import torch.nn.functional as F
from .tgcn2 import TGCN2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A3TGCN2(torch.nn.Module):
    r"""A version of A3T-GCN with multiple layers.`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self.attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
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
                X[:, :, period], edge_index, edge_weight, H
            )
        return H_accum


class AttentionGNN(torch.nn.Module):
    def __init__(self, node_features, periods, learning_rate, weight_decay):
        super(AttentionGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell with 2 layers
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(16, 2)

        # print(self.parameters)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x, edge_index, task, readout_batch=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """

        h = self.tgnn(x, edge_index, readout_batch)
        h.to(device)
        h = F.relu(h)

        h = self.linear(h)
        if task == "binary":
            h = torch.softmax(h, dim=1)

        return h
