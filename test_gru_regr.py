#!/usr/bin/env python3
import logging
import argparse
from tqdm import tqdm

import resource 

import os
import torch
import torch_geometric
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from models import (
    GRUModel,
)
from utils import (
    GRUDataset,
    GRUTransform
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, model_name):
    logger.info("Starting Test")

    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)

    with torch.no_grad():
        model.eval()

        metrics = [
            ("Mean Absolute Error", MeanAbsoluteError().to(device)),
            ("Mean Squared Error", MeanSquaredError().to(device)),
            ("R2Score", R2Score().to(device)),
        ]

        predictions = []
        labels = []

        for _, data in enumerate(tqdm(loader)):

            x = data[0].to(device)
            y = data[1].to(device)

            preds = model(x)
            y = torch.log10(1 + y)
            probs = torch.sigmoid(preds)

            for _, metric in metrics:
                metric.update(probs, y)

            preds_cpu = preds.cpu()
            y_cpu = y.float().cpu()
            predictions.append(preds_cpu)
            labels.append(y_cpu)
            del preds 
            del y

        loss = criterion(torch.cat(predictions), torch.cat(labels))
        logger.info(f"| Test Loss: {loss}")

        result = "{}".format(model_name)
        for metric_name, metric in metrics:
            metric_value = metric.compute()
            logger.info("| Test {}: {}".format(metric_name, metric_value))
            result += ",{}".format(metric_value)
            metric.reset()

        logger.info(result)


def build_model_name(args): 
    model_type = "gru-regr"
    target = "target-{}".format(args.target_week)
    oci = "oci-1" if args.include_oci_variables else "oci-0"
    timesteps = "time-l{}".format(args.timesteps)
    return "{}_{}_{}_{}".format(model_type, target, oci, timesteps)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    model_name = build_model_name(args)
    logger.info("Using model={}".format(model_name))
    logger.info("Using timesteps={}".format(args.timesteps))    
    if args.log_file is None:
        log_file = "{}/{}.test.logs".format(args.out_dir, model_name)
    else: 
        log_file = args.log_file
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("Using target week={}".format(args.target_week))

    dataset = GRUDataset(
        root_dir=args.test_path,
        target_week=args.target_week,
        include_oci_variables=args.include_oci_variables,
        transform=GRUTransform(args.test_path, args.timesteps),
    )

    logger.info("Dataset length: {}".format(len(dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    model = GRUModel(
        len(dataset.local_features),
        args.hidden_channels[0],
        num_layers=2,
        output_size=1,
        dropout=0.1
    )
    model.load_state_dict(torch.load(args.model_path))

    loader = torch_geometric.loader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test(model=model, loader=loader, model_name=model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="data.36/test",
        help="Test set path",
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="model_path",
        default=None,
        help="Path to load the trained model from",
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
        "--hidden-channels",
        metavar="KEY",
        type=str,
        action="store",
        dest="hidden_channels",
        default="64",
        help="Hidden channels for layers of the GRU",
    )    
    parser.add_argument(
        "--timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="timesteps",
        default=24,
        help="Time steps in the past",
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
        "--num-workers",
        metavar="KEY",
        type=int,
        action="store",
        dest="num_workers",
        default=12,
        help="Num workers",
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
    parser.add_argument(
        "--include-oci-variables", dest="include_oci_variables", action="store_true"
    )
    parser.add_argument(
        "--no-include-oci-variables", dest="include_oci_variables", action="store_false"
    )
    parser.set_defaults(include_oci_variables=False)

    args = parser.parse_args()

    args.hidden_channels = [int(x) for x in args.hidden_channels.split(",")]

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))    

    #torch.multiprocessing.set_start_method('spawn') 
    main(args)
