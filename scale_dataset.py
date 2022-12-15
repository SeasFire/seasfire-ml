import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StandardScaling:
    def __init__(self, model, task, append_position_as_feature=True):
        self._model = model
        self._mean_std_tuples = None
        self._task = task 
        self._append_position_as_feature = append_position_as_feature

    def fit(self, graphs):
        mean_std_tuples = []
        temp = []

        for feature_idx in range(0, graphs[0].shape[1]):
            if self._model == "AttentionGNN":
                temp = np.concatenate(
                    [graph[:, feature_idx, :] for graph in graphs]
                )
            else: 
                raise ValueError("Invalid model")

            mean_std_tuples.append(tuple((np.nanmean(temp), np.nanstd(temp))))

        self._mean_std_tuples = mean_std_tuples
        return self._mean_std_tuples

    def __call__(self, graph):
        tmp = list(zip(*self._mean_std_tuples))
        mu = torch.Tensor(list(tmp[0]))
        # print(mu.shape)
        std = torch.Tensor(list(tmp[1]))
        # print(std)

        if self._model == "AttentionGNN":
            mu = mu.unsqueeze(1) 
            mu = mu.expand(mu.shape[0], graph.x.shape[2])
            std = std.unsqueeze(1) 
            std = std.expand(std.shape[0], graph.x.shape[2])
            for i in range(0, graph.x.shape[0]):
                graph.x[i, :, :] = (graph.x[i, :, :] - mu) / std
        else: 
            raise ValueError("Invalid model")                

        # Define label
        if self._task == "binary":
            graph.y = torch.where(graph.y > 0.0, 1, 0)
            graph.y = torch.nn.functional.one_hot(graph.y, 2).float()
        elif self._task == "regression":
            graph.y = graph.y / 1000.0
        else:
            raise ValueError("Invalid task")

        graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        # Concatenate positions with features
        if self._append_position_as_feature:
            positions = graph.pos.unsqueeze(2).expand(-1, -1, graph.x.shape[2])
            graph.x = torch.cat((graph.x, positions), dim=1)

        return graph
