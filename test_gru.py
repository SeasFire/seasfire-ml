#!/usr/bin/env python3
import logging
import argparse
from tqdm import tqdm

import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from utils import (
    GRUDataset,
    GRUTransform,
)


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, criterion):
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

        for metric, metric_name in zip(
            metrics, ["Accuracy", "Recall", "F1Score", "Average Precision (AUPRC)", "AUROC", "Stats"]
        ):
            logger.info("| Test {}: {}".format(metric_name, metric.compute()))
            metric.reset()


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

    logger.info("Using model={}".format(model_name))
    logger.info("Using target week={}".format(args.target_week))

    dataset = GRUDataset(
        root_dir=args.test_path,
        include_oci_variables=args.include_oci_variables,
        transform=GRUTransform(args.test_path, args.target_week),
    )

    logger.info("Dataset length: {}".format(len(dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    loader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test(model=model, loader=loader, criterion=criterion)


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
    main(args)
