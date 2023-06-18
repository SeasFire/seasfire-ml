#!/usr/bin/env python3
import logging
import argparse
from tqdm import tqdm

import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score
from graph_dataset import GraphDataset
from torch_geometric.data import Data


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, criterion, task):
    logger.info("Starting test")

    model = model.to(device)

    with torch.no_grad():
        model.eval()

        metrics = [
            Accuracy(task="multiclass", num_classes=2).to(device),
            F1Score(task="multiclass", num_classes=2, average="macro").to(device),
            AveragePrecision(task="multiclass", num_classes=2).to(device),
            AUROC(task="multiclass", num_classes=2).to(device),
        ]

        predictions = []
        labels = []

        for _, data in enumerate(tqdm(loader)):

            if isinstance(data, Data):
                data = data.to(device)
                x = data.x
                y = data.y
                edge_index = data.edge_index
                batch = data.batch
            else:
                x = data[0].to(device)
                y = data[1].to(device)
                edge_index = None
                batch = None

            preds = model(x, edge_index, None, None, batch)
            if task == "regression":
                y = y.unsqueeze(1)

            predictions.append(preds)
            labels.append(y)

            if task == "binary":
                y_class = torch.argmax(y, dim=1)
                for metric in metrics:
                    metric.update(preds, y_class)

        loss = criterion(torch.cat(labels), torch.cat(predictions))
        logger.info(f"| Test Loss: {loss}")

        if task == "binary":
            for metric, metric_name in zip(
                metrics, ["Accuracy", "F1Score", "Average Precision", "AUROC"]
            ):
                temp = metric.compute()
                logger.info(f"| Test {metric_name}: {(temp):.4f}")
                metric.reset()
        elif task == "regression":
            raise ValueError("Not yet supported")
        else:
            raise ValueError("Not yet supported")


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
    transform = model_info["transform"]
    model_name = model_info["name"]

    logger.info("Using model={}".format(model_name))
    logger.info("Using target week={}".format(args.target_week))

    if transform.target_week != args.target_week:
        logger.warning(
            "Model has been trained with target week {} instead of {}".format(
                transform.target_week, args.target_week
            )
        )
    transform.target_week = args.target_week

    if model_name in [
        "AttentionGNN-TGCN2",
        "AttentionGNN-TGatConv",
        "Attention2GNN-TGCN2",
        "Attention2GNN-TGatConv",
        "Transformer_Aggregation-TGCN2",
    ]:
        loader_class = torch_geometric.loader.DataLoader
    elif model_name == "GRU":
        loader_class = torch.utils.data.DataLoader
    else:
        raise ValueError("Invalid model")

    dataset = GraphDataset(
        root_dir=args.test_path,
        transform=transform,
    )

    logger.info("Test dataset length: {}".format(len(dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    loader = loader_class(
        dataset,
        batch_size=args.batch_size,
    )

    test(model=model, loader=loader, criterion=criterion, task=args.task)


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
        default="best_GRU_t1.pt",
        help="Path to load the trained model from",
    )
    parser.add_argument(
        "--task",
        metavar="KEY",
        type=str,
        action="store",
        dest="task",
        default="binary",
        help="Model task",
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
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    main(args)
