import torch
import numpy as np


class StandardScaling:
    def __init__(self, model):
        self._model = model
        self._mean_std_tuples = None

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

    def __call__(self, features):
        tmp = list(zip(*self._mean_std_tuples))
        mu = torch.Tensor(list(tmp[0]))
        # print(mu.shape)
        std = torch.Tensor(list(tmp[1]))
        # print(std)

        if self._model == "AttentionGNN":
            mu = mu.unsqueeze(1) 
            mu = mu.expand(mu.shape[0], features.shape[2])
            std = std.unsqueeze(1) 
            std = std.expand(std.shape[0], features.shape[2])
            for i in range(0, features.shape[0]):
                features[i, :, :] = (features[i, :, :] - mu) / std
        else: 
            raise ValueError("Invalid model")                

        return features
