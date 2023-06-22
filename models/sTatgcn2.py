import torch
import torch.nn.functional as F
import logging
from .aggr import set_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

class TransformerAggregationGNN(torch.nn.Module):
    r"""A new model.`_
    Args:
        tgcn_model: Basic TGCN model constructor
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
    """

    def __init__(
        self, 
        tgcn_model, 
        node_features: int,
        hidden_channels: tuple,
        periods: int,        
        **kwargs
    ):
        super(TransformerAggregationGNN, self).__init__()
        self.periods = periods
        self._base_tgcn = tgcn_model(
            in_channels=node_features,
            out_channels=hidden_channels,
            add_graph_aggregation_layer=False,
            **kwargs
        )
        self.aggr = set_transformer.SetTransformerAggregation(channels=hidden_channels[-1], heads=4)
        self.fc = torch.nn.Linear(hidden_channels[-1], 1)

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
        H_accum_list = []
        for period in range(self.periods):
            H_temp = self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H, readout_batch
            )
            H_accum_list.append(H_temp)
        
        H_accum = torch.stack(H_accum_list, dim=1)
        H_accum = torch.flatten(H_accum, end_dim=1)

         # Readout layer
        index = (
            torch.zeros(X.shape[0], dtype=int)
            if readout_batch is None
            else readout_batch
        )
        index = (torch.sort(index.repeat(self.periods)))[0]
        index = index.to(device)

        h = self.aggr(H_accum, index) 
        h = F.relu(h)
        out = self.fc(h)
        return out
