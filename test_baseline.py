#!/usr/bin/env python3

import logging
import argparse
from tqdm import tqdm

import os
import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from utils import (
    LocalGlobalDataset,
    LocalGlobalTransform
)

from models import BaselineModel

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, loader, model_name):
    logger.info("Starting Test")

    with torch.no_grad():
        metrics = [
            Accuracy(task="binary").to(device),
            Recall(task="binary").to(device),
            F1Score(task="binary").to(device),
            AveragePrecision(task="binary").to(device),
            AUROC(task="binary").to(device),
            StatScores(task="binary").to(device)
        ]

        for _, data in enumerate(tqdm(loader)):
            lat = data.center_lat
            lon = data.center_lon
            time = data.center_time

            y = data.y
            preds = model((lat, lon, time))

            preds = preds.gt(0.0).float()
            y = y.gt(0.0)

            for metric in metrics:
                metric.update(preds, y)

        result = "{}".format(model_name)
        for metric, metric_name in zip(
            metrics, ["Accuracy", "Recall", "F1Score", "Average Precision (AUPRC)", "AUROC", "Stats"]
        ):
            metric_value = metric.compute()
            logger.info("| Test {}: {}".format(metric_name, metric_value))
            result += ",{}".format(metric_value)
            metric.reset()

        logger.info(result)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    model_name = "baseline"

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    if args.log_file is None:
        log_file = "{}/{}.test.logs".format(args.out_dir, model_name)
    else: 
        log_file = args.log_file
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("Using model={}".format(model_name))
    logger.info("Using target week={}".format(args.target_week))

    dataset = LocalGlobalDataset(
        root_dir=args.test_path,
        target_week=1,
        local_radius=2,
        local_k=2,
        global_k=2,        
        include_local_oci_variables=False,
        include_global_oci_variables=False,
        include_global=False,
        transform=LocalGlobalTransform(args.test_path, False, True),
    )

    logger.info("Dataset length: {}".format(len(dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    loader = torch_geometric.loader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BaselineModel(args.cube_path, False, args.method)

    test(model=model, loader=loader, model_name=model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "--cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="../1d_SeasFireCube.zarr",
        help="Cube path",
    )    
    parser.add_argument(
        "-t",
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="data.24/test",
        help="Test set path",
    )
    parser.add_argument(
        "--method",
        metavar="KEY",
        type=str,
        action="store",
        dest="method",
        default="mean",
        help="Method to calculate the baseline results (mean or majority)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="batch_size",
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--target-week",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_week",
        default=1,
        help="Target week",
    )
    parser.add_argument(
        "--out-dir",
        metavar="KEY",
        type=str,
        action="store",
        dest="out_dir",
        default="runs",
        help="Default output directory",
    )        
    parser.add_argument(
        "--log-file",
        metavar="KEY",
        type=str,
        action="store",
        dest="log_file",
        default=None,
        help="Filename to output all logs",
    )        
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    main(args)
