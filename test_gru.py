#!/usr/bin/env python3
import logging
import argparse
from tqdm import tqdm

import os
import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from utils import (
    GRUDataset,
    GRUTransform
)


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, criterion, model_name):
    logger.info("Starting Test")

    model = model.to(device)

    with torch.no_grad():
        model.eval()

        metrics = [
            Accuracy(task="binary").to(device),
            Recall(task="binary").to(device),
            F1Score(task="binary").to(device),
            AveragePrecision(task="binary").to(device),
            AUROC(task="binary").to(device),
            StatScores(task="binary").to(device)
        ]

        predictions = []
        labels = []

        for _, data in enumerate(tqdm(loader)):

            x = data[0].to(device)
            y = data[1].to(device)

            preds = model(x)

            predictions.append(preds)
            labels.append(y.float())

            probs = torch.sigmoid(preds)

            logger.debug("probs = {}".format(probs))
            logger.debug("y = {}".format(y.float()))

            for metric in metrics:
                metric.update(probs, y)

        loss = criterion(torch.cat(predictions), torch.cat(labels))
        logger.info(f"| Test Loss: {loss}")

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

    model_info = torch.load(args.model_path)
    model = model_info["model"]
    criterion = model_info["criterion"]
    model_name = model_info["name"]

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
    logger.info("Using timesteps={}".format(args.timesteps))

    dataset = GRUDataset(
        root_dir=args.test_path,
        include_oci_variables=args.include_oci_variables,
        transform=GRUTransform(args.test_path, args.timesteps, args.target_week),
    )

    logger.info("Dataset length: {}".format(len(dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    loader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test(model=model, loader=loader, criterion=criterion, model_name=model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t",
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="data/test",
        help="Test set path",
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="model_path",
        default="LocalGlobal_target1.pt",
        help="Path to load the trained model from",
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
        "-lt",
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

    #torch.multiprocessing.set_start_method('spawn') 

    args = parser.parse_args()
    main(args)
