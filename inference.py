#!/usr/bin/env python3
import logging
import argparse
import pickle as pkl
from tqdm import tqdm

import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score
from graph_dataset import GraphDataset
from torch_geometric.data import Data


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, criterion, task):

    model = model.to(device)

    with torch.no_grad():
        model.eval()

        test_metrics_dict = {"Preds": [], "Target": []}
        test_metrics = [
            Accuracy(task="multiclass", num_classes=2).to(device),
            F1Score(task="multiclass", num_classes=2, average="macro").to(device),
            AveragePrecision(task="multiclass", num_classes=2).to(device),
            AUROC(task="multiclass", num_classes=2).to(device),
        ]

        test_predictions = []
        test_labels = []

        for data in tqdm(loader):
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

            test_predictions.append(preds)
            test_labels.append(y)

            if task == "binary":
                y_class = torch.argmax(y, dim=1)
                for metric in test_metrics:
                    metric.update(preds, y_class)

        test_loss = criterion(torch.cat(test_labels), torch.cat(test_predictions))
        logger.info(f" | Test Loss: {test_loss}")

        # #Count zeros and ones in predictions and true values
        # pred_new = torch.argmax(torch.cat(test_predictions), dim=1)
        # print("Count zeros and ones in y_pred: ", torch.bincount(pred_new))
        # y_new = torch.argmax(torch.cat(test_labels), dim=1)
        # print("Count zeros and ones in y_true: ", torch.bincount(y_new))

        if task == "binary":
            for metric, metric_name in zip(
                test_metrics, ["Accuracy", "F1Score", "Average Precision", "AUROC"]
            ):
                temp = metric.compute()
                logger.info(f"| Test {metric_name}: {(temp):.4f}")
                metric.reset()
            test_metrics_dict["Preds"].append(
                (torch.argmax(torch.cat(test_predictions), dim=1))
                .cpu()
                .detach()
                .numpy()
            )
            test_metrics_dict["Target"].append(
                torch.cat(test_labels).cpu().detach().numpy()
            )
        elif task == "regression":
            test_metrics_dict["Preds"].append(
                torch.cat(test_predictions).cpu().detach().numpy()
            )
            test_metrics_dict["Target"].append(
                torch.cat(test_labels).cpu().detach().numpy()
            )

        with open("test_metrics.pkl", "wb") as file:
            pkl.dump(test_metrics_dict, file)


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.FileHandler("logs.log"))

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

    logger.info("Using target month={}".format(args.target_month))
    transform.target_month = args.target_month

    test_dataset = GraphDataset(
        root_dir=args.test_path,
        transform=transform,
    )

    if model_name in [
        "AttentionGNN-TGCN2",
        "AttentionGNN-TGatConv",
        "Attention2GNN-TGCN2",
        "Attention2GNN-TGatConv",
    ]:
        loader_class = torch_geometric.loader.DataLoader
    elif model_name == "GRU":
        loader_class = torch.utils.data.DataLoader
    else:
        raise ValueError("Invalid model")

    loader = loader_class(
        test_dataset,
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
        default="data25/data/test",
        help="Test set path",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        metavar="KEY",
        type=str,
        action="store",
        dest="model_name",
        default="Attention2GNN-TGCN2",
        help="Model name",
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="model_path",
        default="36_binary_attention_model.pt",
        help="Path to save the trained model",
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
        "--target-month",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_month",
        default=1,
        help="Target month",
    )
    args = parser.parse_args()
    main(args)
