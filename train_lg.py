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


def train(model, train_loader, epochs, val_loader, model_name, out_dir):
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

    # pos_weight = torch.FloatTensor([1.0]).to(device)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss()

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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    logger.info("Optimizer={}".format(optimizer))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-4
    )
    # scheduler = lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epochs, eta_min=0, verbose=True
    # )
    logger.info("LR scheduler={}".format(scheduler))
    model = model.to(device)
    iters = len(train_loader)

    current_max_avg = 0

    for epoch in range(1, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))
        logger.info("Current lr={}".format(scheduler.get_last_lr()))

        model.train()

        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for i, data in enumerate(tqdm(train_loader)):
            # logger.info("Data={}".format(data))

            data = data.to(device)
            local_x = data.x
            global_x = data.get("global_x")
            local_edge_index = data.edge_index
            global_edge_index = data.get("global_edge_index")
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

            train_loss = criterion(preds, y.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            scheduler.step(epoch - 1 + i / iters)

            probs = torch.sigmoid(preds)
            # logger.info("preds = {}".format((probs > 0.5).float()))
            # logger.info("y = {}".format(y.float()))
            for metric in train_metrics:
                metric.update(probs, y)

            preds_cpu = preds.cpu()
            y_cpu = y.float().cpu()
            train_predictions.append(preds_cpu)
            train_labels.append(y_cpu)
            del preds
            del y

        # Validation
        logger.info("Epoch {} Validation".format(epoch))

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            model.eval()

            for _, data in enumerate(tqdm(val_loader)):
                data = data.to(device)
                local_x = data.x
                global_x = data.get("global_x")
                local_edge_index = data.edge_index
                global_edge_index = data.get("global_edge_index")
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

                probs = torch.sigmoid(preds)
                # logger.info("preds = {}".format((probs > 0.5).float()))
                # logger.info("y = {}".format(y.float()))
                for metric in val_metrics:
                    metric.update(probs, y)

                preds_cpu = preds.cpu()
                y_cpu = y.float().cpu()
                val_predictions.append(preds_cpu)
                val_labels.append(y_cpu)
                del preds
                del y

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

        save_model(model, criterion, "last", model_name, out_dir)
        if (
            val_metrics_dict["AveragePrecision (AUPRC)"][epoch - 1] > current_max_avg
            and epoch > 10
        ):
            current_max_avg = val_metrics_dict["AveragePrecision (AUPRC)"][epoch - 1]
            logger.info("Found new best model in epoch {}".format(epoch))
            save_model(model, criterion, "best", model_name, out_dir)

        train_metrics_dict["Loss"].append(train_loss.cpu().detach().numpy())
        val_metrics_dict["Loss"].append(val_loss.cpu().detach().numpy())
        # scheduler.step()

    with open("train_metrics.pkl", "wb") as file:
        pkl.dump(train_metrics_dict, file)
    with open("val_metrics.pkl", "wb") as file:
        pkl.dump(val_metrics_dict, file)


def build_model_name(args):
    model_type = "local-global" if args.include_global else "local"
    target = "target-{}".format(args.target_week)
    local_radius = "radius-{}".format(args.local_radius)

    oci = "oci-l{}-g{}".format(
        "1" if args.include_local_oci_variables else "0",
        "1" if args.include_global_oci_variables and args.include_global else "0",
    )

    if args.include_global:
        timesteps = "time-l{}-g{}".format(args.local_timesteps, args.global_timesteps)
    else:
        timesteps = "time-l{}-g0".format(args.local_timesteps)
    return "{}_{}_{}_{}_{}".format(model_type, target, oci, local_radius, timesteps)


def save_model(model, criterion, model_type, model_name, out_dir):
    filename = "{}/{}.{}_model.pt".format(out_dir, model_name, model_type)
    logger.info("Saving model as {}".format(filename))
    torch.save(
        {
            "model": model,
            "criterion": criterion,
            "name": model_name,
        },
        filename,
    )


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    if args.log_file is None:
        model_name = build_model_name(args)
        log_file = "{}/{}.train.logs".format(args.out_dir, model_name)
    else:
        log_file = args.log_file
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    set_seed(42)

    train_dataset = LocalGlobalDataset(
        root_dir=args.train_path,
        local_radius=args.local_radius,
        local_k=args.local_k,
        global_k=args.global_k,
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=LocalGlobalTransform(
            args.train_path,
            args.target_week,
            args.include_global,
            args.append_pos_as_features,
        ),
    )

    num_samples = None
    if args.batches_per_epoch is not None:
        num_samples = args.batches_per_epoch * args.batch_size
        logger.info("Will sample {} samples".format(num_samples))
    train_balanced_sampler = train_dataset.balanced_sampler(num_samples=num_samples)

    val_dataset = LocalGlobalDataset(
        root_dir=args.val_path,
        local_radius=args.local_radius,
        local_k=args.local_k,
        global_k=args.global_k,
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=LocalGlobalTransform(
            args.val_path,
            args.target_week,
            args.include_global,
            args.append_pos_as_features,
        ),
    )

    logger.info("Train dataset length: {}".format(len(train_dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_balanced_sampler,
        num_workers=args.num_workers,
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = LocalGlobalModel(
        len(train_dataset.local_features) + 4 if args.append_pos_as_features else 0,
        args.hidden_channels,
        args.local_timesteps,
        train_dataset.local_nodes,
        len(train_dataset.global_features) + 4 if args.append_pos_as_features else 0,
        args.hidden_channels,
        args.global_timesteps,
        train_dataset.global_nodes,
        args.decoder_hidden_channels,
        args.include_global,
    )

    train(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        model_name=build_model_name(args),
        out_dir=args.out_dir,
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
        default="64,64",
        help="Hidden channels for layer 1 and layer 2 of GCN",
    )
    parser.add_argument(
        "--decoder-hidden-channels",
        metavar="KEY",
        type=str,
        action="store",
        dest="decoder_hidden_channels",
        default="256,64",
        help="Hidden channels for decoder layers",
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
        "--batches-per-epoch",
        metavar="KEY",
        type=int,
        action="store",
        dest="batches_per_epoch",
        default=None,
        help="Batches per epoch.",
    )
    parser.add_argument(
        "--local-radius",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_radius",
        default=7,
        help="Local radius",
    )
    parser.add_argument(
        "--local-k",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_k",
        default=2,
        help="Local k for knn graph.",
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
        "-lt",
        "--local-timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_timesteps",
        default=24,
        help="Time steps in the past for the local part",
    )
    parser.add_argument(
        "-gt",
        "--global-timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="global_timesteps",
        default=24,
        help="Time steps in the past for the global part",
    )
    parser.add_argument(
        "--global-k",
        metavar="KEY",
        type=int,
        action="store",
        dest="global_k",
        default=2,
        help="Global k for knn graph.",
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
        default=0.03,
        help="Weight decay",
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
    parser.add_argument("--include-global", dest="include_global", action="store_true")
    parser.add_argument(
        "--no-include-global", dest="include_global", action="store_false"
    )
    parser.set_defaults(include_global=True)
    parser.add_argument(
        "--append-pos-as-features", dest="append_pos_as_features", action="store_true"
    )
    parser.add_argument(
        "--no-append-pos-as-features",
        dest="append_pos_as_features",
        action="store_false",
    )
    parser.set_defaults(append_pos_as_features=True)
    parser.add_argument(
        "--include-global-oci-variables",
        dest="include_global_oci_variables",
        action="store_true",
    )
    parser.add_argument(
        "--no-include-global-oci-variables",
        dest="include_global_oci_variables",
        action="store_false",
    )
    parser.set_defaults(include_global_oci_variables=True)
    parser.add_argument(
        "--include-local-oci-variables",
        dest="include_local_oci_variables",
        action="store_true",
    )
    parser.add_argument(
        "--no-include-local-oci-variables",
        dest="include_local_oci_variables",
        action="store_false",
    )
    parser.set_defaults(include_local_oci_variables=True)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()

    args.hidden_channels = args.hidden_channels.split(",")
    if len(args.hidden_channels) != 2:
        raise ValueError("Expected hidden channels to be a list of two elements")
    args.hidden_channels = (int(args.hidden_channels[0]), int(args.hidden_channels[1]))

    args.decoder_hidden_channels = [
        int(x) for x in args.decoder_hidden_channels.split(",")
    ]

    # torch.multiprocessing.set_start_method('spawn')

    main(args)
