import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GraphNormalize:
    def __init__(
        self, model, task, target_month, mean_std_per_feature, append_position_as_feature=True
    ):
        self._model = model
        self._mean_std_tuples = None
        self._task = task
        self._target_month = target_month
        self._mean_std_per_feature = mean_std_per_feature
        self._append_position_as_feature = append_position_as_feature

    def __call__(self, graph):
        tmp = list(zip(*self._mean_std_per_feature))
        mu = torch.Tensor(list(tmp[0]))
        std = torch.Tensor(list(tmp[1]))

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
            if graph.y.shape[0] == 1:
                graph.y = torch.where(graph.y > 0.0, 1, 0)
                graph.y = torch.nn.functional.one_hot(graph.y, 2).float()    
            elif graph.y.shape[0] > 1:
                y = (graph.y)[self._target_month-1]
                y = torch.where(y > 0.0, 1, 0)
                y = torch.nn.functional.one_hot(y, 2).float()
                graph.y = y.unsqueeze(0)

        elif self._task == "regression":
            # graph.y = graph.y / 1000.0
            if graph.y.shape[0] == 1:
                graph.y = (graph.y * 100) / (717448.7552)    
            elif graph.y.shape[0] > 1:
                y = (graph.y)[self._target_month-1]
                y = (y * 100) / (717448.7552) 
                graph.y = y.unsqueeze(0)
            
        else:
            raise ValueError("Invalid task")

        graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        # Concatenate positions with features
        if self._append_position_as_feature:
            positions = graph.pos.unsqueeze(2).expand(-1, -1, graph.x.shape[2])
            graph.x = torch.cat((graph.x, positions), dim=1)

        return graph


class ToCentralNodeAndNormalize:
    def __init__(
        self, model, task, target_month, mean_std_per_feature, append_position_as_feature=True
    ):
        self._model = model
        self._mean_std_tuples = None
        self._task = task
        self._target_month = target_month
        self._mean_std_per_feature = mean_std_per_feature
        self._append_position_as_feature = append_position_as_feature

    def __call__(self, graph):
        tmp = list(zip(*self._mean_std_per_feature))
        mu = torch.Tensor(list(tmp[0]))
        std = torch.Tensor(list(tmp[1]))

        if self._model == "GRU":
            # keep only first node of graph
            # assume that it is the central node
            graph.x = graph.x[0,:,:]
            mu = mu.unsqueeze(1)
            mu = mu.expand(-1, graph.x.shape[1])
            std = std.unsqueeze(1)
            std = std.expand(-1, graph.x.shape[1])
            graph.x = (graph.x - mu) / std
        else:
            raise ValueError("Invalid model")

        # Define label
        if self._task == "binary":
            if graph.y.shape[0] == 1:
                graph.y = torch.where(graph.y > 0.0, 1, 0)
                graph.y = torch.nn.functional.one_hot(graph.y, 2).float()    
            elif graph.y.shape[0] > 1:
                y = (graph.y)[self._target_month-1]
                y = torch.where(y > 0.0, 1, 0)
                y = torch.nn.functional.one_hot(y, 2).float()
                graph.y = y.unsqueeze(0)
        elif self._task == "regression":
            # graph.y = graph.y / 1000.0
            if graph.y.shape[0] == 1:
                graph.y = (graph.y * 100) / (717448.7552)    
            elif graph.y.shape[0] > 1:
                y = (graph.y)[self._target_month-1]
                y = (y * 100) / (717448.7552) 
                graph.y = y.unsqueeze(0)
        else:
            raise ValueError("Invalid task")

        graph.x = torch.nan_to_num(graph.x, nan=-1.0)

        # Concatenate positions with features
        if self._append_position_as_feature:
            graph.pos = graph.pos[0,:]
            positions = graph.pos.unsqueeze(1).expand(-1, graph.x.shape[1])
            graph.x = torch.cat((graph.x, positions), dim=0)

        graph.x = graph.x.permute(1,0)
        graph.y = graph.y.squeeze(0)
        return graph.x, graph.y
