import torch
import numpy as np



class StandardScaling():
    def __init__(self):
        pass

    def fit(self, graphs):
        self.graphs = graphs
        self.mean_std_tuples = []
        
        for feature_idx in range (0, self.graphs[0].shape[1]):
       
            temp = np.concatenate([graph[:, feature_idx] for graph in self.graphs])
            self.mean_std_tuples.append(
                tuple((np.nanmean(temp), np.nanstd(temp))))

        return self.mean_std_tuples

    def transform(self, node):
        self.node = node

        tmp = list(zip(*self.mean_std_tuples))
        mu = torch.Tensor(list(tmp[0]))
        std = torch.Tensor(list(tmp[1]))

        self.node = (self.node - mu) /std
        return self.node

    # def fit_transform(self, X):
    #     self.fit(X)
    #     self.transform(self.X)
    #     return self.X_scaled
