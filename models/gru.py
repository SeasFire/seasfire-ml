import torch
import logging
import torch.nn.functional as F
from torch.nn import GRU


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(torch.nn.Module):
    def __init__(
        self, node_features, output_channels, periods, learning_rate, weight_decay, task
    ):
        super(GRUModel, self).__init__()
        self.gru = GRU(
            input_size=node_features, hidden_size=output_channels[1], batch_first=True
        )

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(output_channels[1], output_channels[2])

        # print(self.parameters)
        # self.optimizer = torch.optim.Adam(
        #     self.parameters(), lr=learning_rate, weight_decay=weight_decay
        # )
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, maximize=False)
        logger.info("Optimizer={}".format(self.optimizer))
        
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

        out, h = self.gru(X, H)
        h.to(device)
        out.to(device)
        out = F.relu(out[:,-1,:])

        out = self.linear(out)
        if self.task == "binary":
            out = torch.softmax(out, dim=1)

        return out
