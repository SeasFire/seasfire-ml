import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN


import argparse
import json
import time
import logging
import os
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric_temporal import GConvLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, learning_rate, weight_decay, task):
        super(GCN, self).__init__()
        self.task = task

        self.conv1 = GCNConv(input_size, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, out_channels=hidden_channels)

        self.conv3 = GCNConv(hidden_channels - 16, out_channels=hidden_channels - 16)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        if(self.task == 'regression'):
            self.lin2 = Linear(hidden_channels, 1)
        elif(self.task == 'binary'):
            self.lin2 = Linear(hidden_channels, 1)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x, edge_index, batch=None):

        # Node embedding
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)

        # Readout layer
        batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
        batch = batch.to(device)
        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)

        # Final classifier
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        x = x.relu()

        return x


class AttentionGNN(torch.nn.Module):
    def __init__(self, node_features, periods, learning_rate, weight_decay):
        super(AttentionGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, 1)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x, edge_index, batch=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        
        
        h = self.tgnn(x, edge_index)
        h.to(device)
        h = F.relu(h)

        # Readout layer
        batch = torch.zeros(h.shape[0], dtype=int) if batch is None else batch
        batch = batch.to(device)
        h = global_mean_pool(h, batch)

        h = self.linear(h)
        h = F.relu(h)
        return h

class LstmGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, learning_rate, weight_decay, task):
        super(LstmGCN, self).__init__()
        # Documentation for GConvLSTM:
        # https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gconv_lstm.GConvLSTM
        self.task = task
        
        self.recurrent_1 = GConvLSTM(node_features, 16, K=5)
        # self.recurrent_2 = GConvLSTM(32, 16, 4)

        if(self.task == 'regression'):
            self.lin1 = Linear(16, 1)
        elif(self.task == 'binary'):
            self.lin1 = Linear(16, 2)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x, edge_index, edge_weight=None):

        print(x.shape)
        print(edge_index.shape)

        # Process the sequence of graphs with our 2 GConvLSTM layers
        h = None
        c = None

        h, c = self.recurrent_1(x, edge_index, edge_weight, H=h, C=c) 
        print("what")

        # Feed hidden state output of first layer to the 2nd layer
        # h2, c2 = self.recurrent_2(h1, edge_index, H=h2, C=c2)
        # print("is")

        # Use the final hidden state output of 2nd recurrent layer for input to classifier
        x = F.relu(h)
        print("going")

        x = F.dropout(x, training=self.training)
        print("on")

        x = self.lin1(x)
        print("?????????????????????")
        x = x.relu()

        return x
        # return F.log_softmax(x, dim=1)


