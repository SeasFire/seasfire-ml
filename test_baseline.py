#!/usr/bin/env python3

import logging
import argparse
from tqdm import tqdm

import os
import torch
import torch_geometric
from sklearn.metrics import average_precision_score
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
        all_y_true = []
        all_y_score = []

        for _, data in enumerate(tqdm(loader)):
            lat = data.center_lat
            lon = data.center_lon
            time = data.center_time

            y_true = data.y
            y_score = model((lat, lon, time))

            y_true = y_true.gt(0.0).detach().cpu().numpy()
            y_true = 1*y_true
            logger.debug("y_true={}".format(y_true))
            all_y_true.extend(y_true)
            
            y_score = y_score.detach().cpu().numpy()
            logger.debug("y_score={}".format(y_score))
            all_y_score.extend(y_score)
 
        avg_precision = average_precision_score(all_y_true, all_y_score)
        logger.info("{} Test AveragePrecision (AUPRC): {}".format(model_name, avg_precision))


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    model_name = "baseline_{}".format(args.method)
    logger.info("Using model={}".format(model_name))

    if args.log_file is None:
        log_file = "{}/{}.test.logs".format(args.out_dir, model_name)
    else: 
        log_file = args.log_file
    logger.addHandler(logging.FileHandler(log_file))



    dataset = LocalGlobalDataset(
        root_dir=args.test_path,
        target_week=1,
        local_radius=2,
        local_k=9,
        global_k=9,        
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

    model = BaselineModel(args.cube_path, args.load_cube_in_memory, args.method)

    test(model=model, loader=loader, model_name=model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "--cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="../seasfire_1deg.zarr",
        help="Cube path",
    )    
    parser.add_argument(
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="data/test",
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
        "--batch-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="batch_size",
        default=32,
        help="Batch size",
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
    parser.add_argument(
        "--load-cube-in-memory", dest="load_cube_in_memory", action="store_true"
    )
    parser.add_argument(
        "--no-load-cube-in-memory", dest="load_cube_in_memory", action="store_false"
    )
    parser.set_defaults(load_cube_in_memory=False)            
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    main(args)
