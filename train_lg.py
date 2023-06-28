#!/usr/bin/env python3
import logging
import os
import argparse
import numpy as np
import pickle as pkl
import random
from tqdm import tqdm

import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from torch.optim import lr_scheduler
from models import (
    LocalGlobalModel,
)
from utils import (
    LocalGlobalDataset,
    LocalGlobalTransform,
)


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    logger.info("Using random seed={}".format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(model, train_loader, epochs, val_loader, target_week):
    logger.info("Starting training for {} epochs".format(epochs))

    train_metrics_dict = {
        "Accuracy": [],
        "Recall": [],
        "F1Score": [],
        "AveragePrecision (AUPRC)": [],
        "AUROC": [],
        "StatsScore": [],
        "Loss": [],
    }
    val_metrics_dict = {
        "Accuracy": [],
        "Recall": [],
        "F1Score": [],
        "AveragePrecision (AUPRC)": [],
        "AUROC": [],
        "StatsScore": [],
        "Loss": [],
    }

    pos_weight = torch.FloatTensor([1.0]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_metrics = [
        Accuracy(task="binary").to(device),
        Recall(task="binary").to(device),
        F1Score(task="binary").to(device),
        AveragePrecision(task="binary").to(device),
        AUROC(task="binary").to(device),
        StatScores(task="binary").to(device),
    ]

    val_metrics = [
        Accuracy(task="binary").to(device),
        Recall(task="binary").to(device),
        F1Score(task="binary").to(device),
        AveragePrecision(task="binary").to(device),
        AUROC(task="binary").to(device),
        StatScores(task="binary").to(device),
    ]

    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=args.learning_rate,
        rho=0.9,
        eps=1e-06,
        weight_decay=args.weight_decay,
        foreach=None,
        maximize=False,
    )
    logger.info("Optimizer={}".format(optimizer))
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0, verbose=True
    )
    logger.info("LR scheduler={}".format(scheduler))
    model = model.to(device)
    current_max_avg = 0

    for epoch in range(1, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))

        model.train()

        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for _, data in enumerate(tqdm(train_loader)):
            # logger.info("Data={}".format(data))

            data = data.to(device)
            local_x = data.x
            global_x = data.global_x
            local_edge_index = data.edge_index
            global_edge_index = data.global_edge_index
            y = data.y
            batch = data.batch

            preds = model(
                local_x,
                global_x,
                local_edge_index,
                global_edge_index,
                None,
                None,
                None,
                None,
                batch,
            )

            train_predictions.append(preds)
            train_labels.append(y.float())


            train_loss = criterion(preds, y.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            probs = torch.sigmoid(preds)

            #logger.info("preds = {}".format((probs > 0.5).float()))
            #logger.info("y = {}".format(y.float()))

            for metric in train_metrics:
                metric.update(probs, y)

        # Validation
        logger.info("Epoch {} Validation".format(epoch))

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            model.eval()

            for _, data in enumerate(tqdm(val_loader)):

                data = data.to(device)
                local_x = data.x
                global_x = data.global_x
                local_edge_index = data.edge_index
                global_edge_index = data.global_edge_index
                y = data.y
                batch = data.batch

                preds = model(
                    local_x,
                    global_x,
                    local_edge_index,
                    global_edge_index,
                    None,
                    None,
                    None,
                    None,
                    batch,
                )

                val_predictions.append(preds)
                val_labels.append(y.float())

                probs = torch.sigmoid(preds)

                # logger.info("preds = {}".format((probs > 0.5).float()))
                # logger.info("y = {}".format(y.float()))

                for metric in val_metrics:
                    metric.update(probs, y)

        train_loss = criterion(torch.cat(train_predictions), torch.cat(train_labels))
        val_loss = criterion(torch.cat(val_predictions), torch.cat(val_labels))

        logger.info("| Train Loss: {:.4f}".format(train_loss))

        for metric, key in zip(train_metrics, train_metrics_dict.keys()):
            temp = metric.compute()
            logger.info("| Train " + key + ": {}".format(temp))
            train_metrics_dict[key].append(temp.cpu().detach().numpy())
            metric.reset()

        logger.info("| Val Loss: {:.4f}".format(val_loss))

        for metric, key in zip(val_metrics, val_metrics_dict.keys()):
            temp = metric.compute()
            logger.info("| Val " + key + ": {}".format(temp))
            val_metrics_dict[key].append(temp.cpu().detach().numpy())
            metric.reset()

        if val_metrics_dict["AveragePrecision"][epoch - 1] > current_max_avg:
            current_max_avg = val_metrics_dict["AveragePrecision"][epoch - 1]
            logger.info("Found new best model in epoch {}".format(epoch))
            logger.info(
                "Saving best model as best_LocalGlobal_target{}.pt".format(target_week)
            )
            torch.save(
                {
                    "model": model,
                    "criterion": criterion,
                    "name": "LocalGlobal",
                },
                "best_LocalGlobal_target{}.pt".format(target_week),
            )

        train_metrics_dict["Loss"].append(train_loss.cpu().detach().numpy())
        val_metrics_dict["Loss"].append(val_loss.cpu().detach().numpy())
        scheduler.step()

    logger.info("Saving model as LocalGlobal_target{}.pt".format(target_week))
    torch.save(
        {
            "model": model,
            "criterion": criterion,
            "name": "LocalGlobal",
        },
        "LocalGlobal_target{}.pt".format(target_week),
    )
    with open("train_metrics.pkl", "wb") as file:
        pkl.dump(train_metrics_dict, file)
    with open("val_metrics.pkl", "wb") as file:
        pkl.dump(val_metrics_dict, file)


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.FileHandler("logs.log"))

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    set_seed(42)

    train_dataset = LocalGlobalDataset(
        root_dir=args.train_path,
        transform=LocalGlobalTransform(args.train_path, args.target_week, args.append_pos_as_features),
    )
    val_dataset = LocalGlobalDataset(
        root_dir=args.val_path,
        transform=LocalGlobalTransform(args.val_path, args.target_week, args.append_pos_as_features),
    )

    logger.info("Train dataset length: {}".format(len(train_dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    train_loader = torch_geometric.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = torch_geometric.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    model = LocalGlobalModel(
        len(train_dataset.local_features) + 4 if args.append_pos_as_features else 0,
        args.hidden_channels,
        len(train_dataset.global_features) + 4 if args.append_pos_as_features else 0,
        args.hidden_channels,
        train_dataset.local_global_nodes,
        args.timesteps
    )

    train(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        target_week=args.target_week,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t",
        "--train-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="train_path",
        default="data/train",
        help="Train set path",
    )
    parser.add_argument(
        "-v",
        "--val-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="val_path",
        default="data/val",
        help="Validation set path",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="batch_size",
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--hidden-channels",
        metavar="KEY",
        type=str,
        action="store",
        dest="hidden_channels",
        default="32,32",
        help="Hidden channels for layer 1 and layer 2 of GCN",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="KEY",
        type=int,
        action="store",
        dest="epochs",
        default=100,
        help="Epochs",
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
        "-ts",
        "--timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="timesteps",
        default=24,
        help="Time steps in the past",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="KEY",
        type=float,
        action="store",
        dest="learning_rate",
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        metavar="KEY",
        type=float,
        action="store",
        dest="weight_decay",
        default=0.05,
        help="Weight decay",
    )
    parser.add_argument(
        "--append-pos-as-features", dest="append_pos_as_features", action="store_true"
    )
    parser.add_argument(
        "--no-append-pos-as-features", dest="append_pos_as_features", action="store_false"
    )
    parser.set_defaults(append_pos_as_features=True)    
    args = parser.parse_args()

    args.hidden_channels = args.hidden_channels.split(",")
    if len(args.hidden_channels) != 2:
        raise ValueError("Expected hidden channels to be a list of two elements")
    args.hidden_channels = (int(args.hidden_channels[0]), int(args.hidden_channels[1]))
    main(args)
