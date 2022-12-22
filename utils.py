import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)


def compute_mean_std_per_feature(dataset, cache_filename=None):
    logger.debug("Computing mean-std per feature for dataset")

    if cache_filename is not None:
        try:
            mean_std_tuples = torch.load(cache_filename)
            logger.info(
                "Loading dataset mean-std status from {}".format(cache_filename)
            )
            return mean_std_tuples
        except:
            pass

    mean_std_tuples = []

    for feature_idx in range(dataset.num_node_features):
        temp = np.concatenate([graph.x[:, feature_idx, :] for graph in dataset])
        mean_std_tuples.append(tuple((np.nanmean(temp), np.nanstd(temp))))

    if cache_filename is not None:
        logger.info("Caching dataset mean-std status to {}".format(cache_filename))
        torch.save(mean_std_tuples, cache_filename)

    return mean_std_tuples
