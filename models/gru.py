import torch
import torch.nn.functional as F
from torch.nn import GRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(torch.nn.Module):
    def __init__(
        self, node_features, output_channels, periods, learning_rate, weight_decay, task
    ):
        super(GRUModel, self).__init__()
        self.gru = GRU(
            input_size=node_features, hidden_size=output_channels[1], dropout=0.3
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
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """

        h = self.gru(X, H)
        h.to(device)
        h = F.relu(h)

        h = self.linear(h)
        if self.task == "binary":
            h = torch.softmax(h, dim=1)

        return h

