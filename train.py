#!/usr/bin/env python3

import torch
from tqdm import tqdm
import logging
import os
import torch_geometric
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

from models import AttentionGNN
from graph_dataset import GraphDataset
from transforms import GraphNormalize
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score
from utils import compute_mean_std_per_feature

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


def train(model, train_loader, epochs, val_loader, task):
    current_max_avg = 0
    best_model = model

    optimizer = model.optimizer

    if task == "binary":
        criterion = torch.nn.CrossEntropyLoss()

        accuracy_train = Accuracy(task="multiclass", num_classes=2).to(device)
        F1_score_train = F1Score(task="multiclass", num_classes=2).to(device)
        avprc_train = AveragePrecision(task="multiclass", num_classes=2).to(device)
        auroc_train = AUROC(task="multiclass", num_classes=2).to(device)

        accuracy_val = Accuracy(task="multiclass", num_classes=2).to(device)
        F1_score_val = F1Score(task="multiclass", num_classes=2).to(device)
        avprc_val = AveragePrecision(task="multiclass", num_classes=2).to(device)
        auroc_val = AUROC(task="multiclass", num_classes=2).to(device)
    elif task == "regression":
        criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))

        model.train()

        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for _, data in enumerate(tqdm(train_loader)):
            data = data.to(device)

            preds = model(data.x, data.edge_index, None, None, data.batch)
            y = data.y
            if task == "regression":
                y = y.unsqueeze(1)

            train_predictions.append(preds)
            train_labels.append(y)
            train_loss = criterion(y, preds)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if task == "binary":
                y_class = torch.argmax(y, dim=1)
                accuracy_train.update(preds, y_class)
                F1_score_train.update(preds, y_class)
                avprc_train.update(preds, y_class)
                auroc_train.update(preds, y_class)

        # Validation
        logger.info("Epoch {} Validation".format(epoch))

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            model.eval()

            for _, data in enumerate(tqdm(val_loader)):
                data = data.to(device)

                preds = model(data.x, data.edge_index, None, None, data.batch)
                y = data.y
                if task == "regression":
                    y = y.unsqueeze(1)

                val_predictions.append(preds)
                val_labels.append(y)

                if task == "binary":
                    y_class = torch.argmax(y, dim=1)
                    accuracy_val.update(preds, y_class)
                    F1_score_val.update(preds, y_class)
                    avprc_val.update(preds, y_class)
                    auroc_val.update(preds, y_class)

        train_loss = criterion(torch.cat(train_labels), torch.cat(train_predictions))
        val_loss = criterion(torch.cat(val_labels), torch.cat(val_predictions))

        logger.info("| Train Loss: {:.4f}".format(train_loss))
        logger.info("| Val Loss: {:.4f}".format(val_loss))

        if task == "binary":
            logger.info("| Train Accuracy: {:.4f}".format(accuracy_train.compute()))
            logger.info("| Train F1_score: {:.4f}".format(F1_score_train.compute()))
            logger.info(
                "| Train Average precision: {:.4f}".format(avprc_train.compute())
            )
            logger.info("| Train AUROC: {:.4f}".format(auroc_train.compute()))

            logger.info("| Val Accuracy: {:.4f}".format(accuracy_val.compute()))
            logger.info("| Val F1_score: {:.4f}".format(F1_score_val.compute()))
            logger.info("| Val Average precision: {:.4f}".format(avprc_val.compute()))
            logger.info("| Val AUROC: {:.4f}".format(auroc_val.compute()))

            if avprc_val.compute() > current_max_avg:
                best_model = model

            accuracy_train.reset()
            F1_score_train.reset()
            avprc_train.reset()
            auroc_train.reset()

            accuracy_val.reset()
            F1_score_val.reset()
            avprc_val.reset()
            auroc_val.reset()

    return model, best_model, criterion


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.FileHandler("logs.log"))

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    set_seed(32)

    logger.info("Extracting dataset statistics")
    mean_std_per_feature = compute_mean_std_per_feature(
        GraphDataset(root_dir=args.train_path)
    )
    logger.info("Statistics: {}".format(mean_std_per_feature))

    transform = GraphNormalize(
        args.model_name,
        task=args.task,
        mean_std_per_feature=mean_std_per_feature,
        append_position_as_feature=True,
    )

    train_dataset = GraphDataset(
        root_dir=args.train_path,
        transform=transform,
    )
    logger.info("Train dataset length: {}".format(len(train_dataset)))

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_dataset = GraphDataset(
        root_dir=args.val_path,
        transform=transform,
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=args.batch_size
    )

    logger.info("Building model {}".format(args.model_name))

    if args.task == "binary":
        linear_out_channels = 2
        args.hidden_channels = args.hidden_channels + (linear_out_channels,)
    elif args.task == "regression":
        linear_out_channels = 1
        args.hidden_channels = args.hidden_channels + (linear_out_channels,)

    if args.model_name == "AttentionGNN":
        num_features = train_dataset.num_node_features
        timesteps = args.timesteps
        model = AttentionGNN(
            num_features,
            args.hidden_channels,
            timesteps,
            args.learning_rate,
            args.weight_decay,
            task=args.task,
        ).to(device)
    else:
        raise ValueError("Invalid model")

    logger.info("Starting training")
    model, best_model, criterion = train(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        task=args.task,
    )

    logger.info("Saving model as {}".format(args.model_path))
    model_info = {
        "model": model,
        "criterion": criterion,
        "transform": transform,
        "name": args.model_name,
    }

    best_model_info = {
        "model": model,
        "criterion": criterion,
        "transform": transform,
        "name": args.model_name,
    }

    # Save the entire model to PATH
    torch.save(model_info, args.model_path)

    # Save the entire best model to PATH
    torch.save(best_model_info, "best_" + args.model_path)


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
        "--model_name",
        metavar="KEY",
        type=str,
        action="store",
        dest="model_name",
        default="AttentionGNN",
        help="Model name",
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="model_path",
        default="binary_attention_model.pt",
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
        "--hidden-channels",
        metavar="KEY",
        type=tuple,
        action="store",
        dest="hidden_channels",
        default=(32, 16),
        help="Hidden channels for layer 1 and layer 2 of GCN",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="KEY",
        type=int,
        action="store",
        dest="epochs",
        default=50,
        help="Epochs",
    )
    parser.add_argument(
        "-ts",
        "--timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="timesteps",
        default=12,
        help="Time steps in the past",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="KEY",
        type=float,
        action="store",
        dest="learning_rate",
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        metavar="KEY",
        type=float,
        action="store",
        dest="weight_decay",
        default=5e-4,
        help="Weight decay",
    )
    args = parser.parse_args()
    main(args)
