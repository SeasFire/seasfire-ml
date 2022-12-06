import torch
import numpy as np


class StandardScaling:
    def __init__(self, model):
        self._model = model

    def fit(self, graphs):
        self.graphs = graphs
        self.mean_std_tuples = []
        temp = []

        for feature_idx in range(0, self.graphs[0].shape[1]):
            if self._model == "AttentionGNN":
                temp = np.concatenate(
                    [graph[:, feature_idx, :] for graph in self.graphs]
                )
            elif self._model == "GCN":
                temp = np.concatenate([graph[:, feature_idx] for graph in self.graphs])

            self.mean_std_tuples.append(tuple((np.nanmean(temp), np.nanstd(temp))))

        return self.mean_std_tuples

    def transform(self, graph):
        self.graph = graph

        tmp = list(zip(*self.mean_std_tuples))
        mu = torch.Tensor(list(tmp[0]))
        # print(mu.shape)
        std = torch.Tensor(list(tmp[1]))
        # print(std)

        if self._model == "AttentionGNN":
            mu = mu.unsqueeze(1) 
            mu = mu.expand(mu.shape[0], graph.shape[2])
            std = std.unsqueeze(1) 
            std = std.expand(std.shape[0], graph.shape[2])
            for i in range(0, self.graph.shape[0]):
                self.graph[i, :, :] = (self.graph[i, :, :] - mu) / std
        elif self._model == "GCN":
            self.graph = (self.graph - mu) / std

        return self.graph
