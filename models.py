import argparse
import json
import time
import logging
import os
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from my_A3TGCN import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class AttentionGNN(torch.nn.Module):
    def __init__(self, node_features, periods, learning_rate, weight_decay):
        super(AttentionGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = my_A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
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
        if task == 'binary':
            h = torch.softmax(h, dim=1)
            
        return h

