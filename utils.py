import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_mean_std_per_feature(dataset): 
    logger.debug("Computing mean-std per feature for dataset")

    mean_std_tuples = []

    for feature_idx in range(dataset.num_node_features): 
        temp = np.concatenate(
            [graph.x[:, feature_idx, :] for graph in dataset]
        )
        mean_std_tuples.append(tuple((np.nanmean(temp), np.nanstd(temp))))

    return mean_std_tuples

