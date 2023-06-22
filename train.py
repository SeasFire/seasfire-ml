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
from torch_geometric.data import Data
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from torch.optim import lr_scheduler
from models import (
    AttentionGNN,
    GRUModel,
    TGatConv,
    TGCN2,
    Attention2GNN,
    TransformerAggregationGNN,
)
from utils import GraphDataset, GraphNormalize, ToCentralNodeAndNormalize, compute_mean_std_per_feature


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


def train(model, train_loader, epochs, val_loader, task, model_name, transform):
    logger.info("Starting training for {} epochs".format(epochs))

    train_metrics_dict = {
        "Accuracy": [],
        "Recall": [],
        "F1Score": [],
        "AveragePrecision": [],
        "AUROC": [],
        "StatsScore": [],
        "Loss": [],
    }
    val_metrics_dict = {
        "Accuracy": [],
        "Recall": [],
        "F1Score": [],
        "AveragePrecision": [],
        "AUROC": [],
        "StatsScore": [],
        "Loss": [],
    }

    if task == "binary":
        pos_weight = torch.FloatTensor([2.0]).to(device)
        criterion = torch.nn.BCELoss(weight=pos_weight)

        train_metrics = [
            Accuracy(task="binary").to(device),
            Recall(task="binary").to(device),
            F1Score(task="binary").to(device),
            AveragePrecision(task="binary").to(device),
            AUROC(task="binary").to(device),
            StatScores(task="binary").to(device)
        ]

        val_metrics = [
            Accuracy(task="binary").to(device),
            Recall(task="binary").to(device),
            F1Score(task="binary").to(device),
            AveragePrecision(task="binary").to(device),
            AUROC(task="binary").to(device),
            StatScores(task="binary").to(device)
        ]
    elif task == "regression":
        criterion = torch.nn.MSELoss()

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
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
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

            train_predictions.append(preds)
            train_labels.append(y.float())
            # logger.info("preds = {}".format(preds))
            # logger.info("y = {}".format(y.float()))
            
            train_loss = criterion(preds, y.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if task == "binary":
                #y_class = torch.argmax(y, dim=1)
                for metric in train_metrics:
                    #metric.update(preds, y_class)
                    metric.update(preds, y)

        # Validation
        logger.info("Epoch {} Validation".format(epoch))

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            model.eval()

            for _, data in enumerate(tqdm(val_loader)):
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

                val_predictions.append(preds)
                val_labels.append(y.float())

                if task == "binary":
                    #y_class = torch.argmax(y, dim=1)
                    for metric in val_metrics:
                        #metric.update(preds, y_class)
                        metric.update(preds, y)

        train_loss = criterion(torch.cat(train_labels), torch.cat(train_predictions))
        val_loss = criterion(torch.cat(val_labels), torch.cat(val_predictions))

        logger.info("| Train Loss: {:.4f}".format(train_loss))

        if task == "binary":
            for metric, key in zip(train_metrics, train_metrics_dict.keys()):
                temp = metric.compute()
                logger.info("| Train " + key + ": {}".format(temp))
                train_metrics_dict[key].append(temp.cpu().detach().numpy())
                metric.reset()

        logger.info("| Val Loss: {:.4f}".format(val_loss))

        if task == "binary":
            for metric, key in zip(val_metrics, val_metrics_dict.keys()):
                temp = metric.compute()
                logger.info("| Val " + key + ": {}".format(temp))
                val_metrics_dict[key].append(temp.cpu().detach().numpy())
                metric.reset()


            if val_metrics_dict["AveragePrecision"][epoch - 1] > current_max_avg:
                current_max_avg = val_metrics_dict["AveragePrecision"][epoch - 1]
                logger.info("Found new best model in epoch {}".format(epoch))
                logger.info("Saving best model as best_{}_target{}.pt".format(model_name, transform.target_week))
                torch.save({
                    "model": model,
                    "criterion": criterion,
                    "transform": transform,
                    "name": model_name,
                }, "best_{}_target{}.pt".format(model_name, transform.target_week))


        train_metrics_dict["Loss"].append(train_loss.cpu().detach().numpy())
        val_metrics_dict["Loss"].append(val_loss.cpu().detach().numpy())
        scheduler.step()

    logger.info("Saving model as {}_target{}.pt".format(model_name, transform.target_week))
    torch.save(
        {
            "model": model,
            "criterion": criterion,
            "transform": transform,
            "name": model_name,
        },
        "{}_target{}.pt".format(model_name, transform.target_week)
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

    logger.info("Extracting dataset statistics")
    mean_std_per_feature = compute_mean_std_per_feature(
        GraphDataset(root_dir=args.train_path),
        cache_filename="dataset_mean_std_cached_stats.pk",
    )
    logger.info("Statistics: {}".format(mean_std_per_feature))
    logger.info("Using model: {}".format(args.model_name))
    if args.model_name in [
        "AttentionGNN-TGCN2",
        "AttentionGNN-TGatConv",
        "Attention2GNN-TGCN2",
        "Attention2GNN-TGatConv",
        "Transformer_Aggregation-TGCN2",
    ]:
        loader_class = torch_geometric.loader.DataLoader
        transform = GraphNormalize(
            args.model_name,
            task=args.task,
            target_week=args.target_week,
            mean_std_per_feature=mean_std_per_feature,
            append_position_as_feature=True,
        )
    elif args.model_name == "GRU":
        loader_class = torch.utils.data.DataLoader
        transform = ToCentralNodeAndNormalize(
            args.model_name,
            task=args.task,
            target_week=args.target_week,
            mean_std_per_feature=mean_std_per_feature,
            use_first_number_of_variables=10,
            append_position_as_feature=True,
        )
    else:
        raise ValueError("Invalid model")

    train_dataset = GraphDataset(
        root_dir=args.train_path,
        transform=transform,
    )
    logger.info("Train dataset length: {}".format(len(train_dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    train_loader = loader_class(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_dataset = GraphDataset(
        root_dir=args.val_path,
        transform=transform,
    )
    val_loader = loader_class(val_dataset, batch_size=args.batch_size)

    logger.info("Building model {}".format(args.model_name))

    if args.model_name == "AttentionGNN-TGCN2":
        model = AttentionGNN(
            TGCN2,
            train_dataset.num_node_features,
            args.hidden_channels,
            args.timesteps,
        ).to(device)
    elif args.model_name == "AttentionGNN-TGatConv":
        model = AttentionGNN(
            TGatConv,
            train_dataset.num_node_features,
            args.hidden_channels,
            args.timesteps,
        ).to(device)
    elif args.model_name == "Attention2GNN-TGCN2":
        model = Attention2GNN(
            TGCN2,
            train_dataset.num_node_features,
            args.hidden_channels,
            args.timesteps,
        ).to(device)
    elif args.model_name == "Attention2GNN-TGatConv":
        model = Attention2GNN(
            TGatConv,
            train_dataset.num_node_features,
            args.hidden_channels,
            args.timesteps,
        ).to(device)
    elif args.model_name == "Transformer_Aggregation-TGCN2":
        model = TransformerAggregationGNN(
            TGCN2,
            train_dataset.num_node_features,
            args.hidden_channels,
            args.timesteps,
        ).to(device)
    elif args.model_name == "GRU":
        model = GRUModel(
            train_dataset.num_node_features,
            args.gru_hidden_size,
            1,
            1,
        )
    else:
        raise ValueError("Invalid model")

    train(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        task=args.task,
        model_name=args.model_name,
        transform=transform,
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
        "-m",
        "--model-name",
        metavar="KEY",
        type=str,
        action="store",
        dest="model_name",
        default="AttentionGNN-TGCN2",
        help="Model name",
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
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--hidden-channels",
        metavar="KEY",
        type=str,
        action="store",
        dest="hidden_channels",
        default="64,64",
        help="Hidden channels for layer 1 and layer 2 of GCN",
    )
    parser.add_argument(
        "--gru-hidden-size",
        metavar="KEY",
        type=int,
        action="store",
        dest="gru_hidden_size",
        default=128,
        help="GRU hidden size",
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
    args = parser.parse_args()

    args.hidden_channels = args.hidden_channels.split(',')
    if len(args.hidden_channels) != 2: 
        raise ValueError("Expected hidden channels to be a list of two elements")
    args.hidden_channels = (int(args.hidden_channels[0]), int(args.hidden_channels[1]))
    main(args)

#./train.py --train-path "weekly_data/train/" --val-path "weekly_data/val/" --model-name "GRU" --model-path "gru.pt" --batch-size 128 --target-week 5 --learning-rate 5e-4
