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
from torch_geometric.nn import global_mean_pool


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, learning_rate, weight_decay):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        
        self.conv1 = GCNConv(
            input_size, hidden_channels)
        
        self.conv2 = GCNConv(
            hidden_channels, out_channels = hidden_channels)
        
        self.lin = Linear(hidden_channels, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def forward(self, x, edge_index, batch = None):
        
        # Node embedding 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Readout layer
        batch = torch.zeros(x.shape[0],dtype=int) if batch is None else batch
        batch = batch.to(device)
        x = global_mean_pool(x, batch)
        

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
    
        return x

