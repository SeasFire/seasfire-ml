import torch
import logging
import torch.nn.functional as F


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.FloatTensor,
        h: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        if h is not None:
            h0 = h.to(device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
