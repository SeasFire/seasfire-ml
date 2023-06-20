import torch
import logging
import torch.nn.functional as F


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
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
        if H is not None: 
            h0 = H
            h0.to(device)
        else: 
            h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)

        out, _ = self.gru(X, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
