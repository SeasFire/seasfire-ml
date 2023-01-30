import torch
import torch.nn.functional as F
from torch_geometric.nn import aggr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A4TGCN2(torch.nn.Module):
    r"""A new model.`_
    Args:
        tgcn_model: Basic TGCN model constructor
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
    """

    def __init__(
        self, tgcn_model, in_channels: int, out_channels: tuple, periods: int, aggregator, **kwargs
    ):
        super(A4TGCN2, self).__init__()

        self.periods = periods
        self._base_tgcn = tgcn_model(
            in_channels=in_channels,
            out_channels=out_channels,
            add_graph_aggregation_layer=False,
            **kwargs
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = torch.nn.Parameter(torch.empty(periods, device=device))
        
        if aggregator == "Mean":
            self.mean_aggr_z = aggr.MeanAggregation()
        else:
            self.mean_aggr_z = aggr.set_transformer.SetTransformerAggregation(channels=out_channels[1])
            self.mean_aggr_z.reset_parameters()
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
            res = self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H, readout_batch
            )
            H_accum = H_accum + probs[period] * res

        # Readout layer
        index = (
            torch.zeros(X.shape[0], dtype=int)
            if readout_batch is None
            else readout_batch
        )
        index = index.to(device)
        H_accum = self.mean_aggr_z(H_accum, index)  # (b,16)
        return H_accum


class Attention2GNN(torch.nn.Module):
    def __init__(
        self,
        tgcn_model,
        node_features,
        output_channels,
        periods,
        learning_rate,
        weight_decay,
        task,
        aggregator,
        **kwargs
    ):
        super(Attention2GNN, self).__init__()
        self.tgnn = A4TGCN2(
            tgcn_model=tgcn_model,
            in_channels=node_features,
            out_channels=output_channels,
            periods=periods,
            aggregator=aggregator,
            **kwargs
        )
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(output_channels[1], output_channels[2])

        # print(self.parameters)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate         #, weight_decay=weight_decay
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
