#!/usr/bin/env python3
import logging
import os
import argparse
import numpy as np
import random
from tqdm import tqdm

import resource

import torch
import torch_geometric
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim import lr_scheduler
from models import (
    LocalGlobalModel,
)
from utils import (
    LocalGlobalDataset,
    LocalGlobalTransform,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from collections import defaultdict


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSLE10Loss(torch.nn.Module):
    def __init__(self):
        super(MSLE10Loss, self).__init__()

    def forward(self, y_pred, y_true):
        # Calculate MSLE
        msle_loss = torch.nn.functional.mse_loss(torch.log10(y_pred + 1), torch.log10(y_true + 1))
        return msle_loss


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    epochs,
    val_loader,
    model_name,
    out_dir,
    cur_epoch=1,
    best_so_far=None,
):
    logger.info("Starting training for {} epochs".format(epochs))

    criterion = MSLE10Loss()

    train_metrics = [
        ("MeanAbsoluteError", MeanAbsoluteError().to(device)),
        ("MeanSquaredError", MeanSquaredError().to(device)),
        ("R2Score", R2Score().to(device)),
    ]
    train_history = defaultdict(lambda: [])

    val_metrics = [
        ("MeanAbsoluteError", MeanAbsoluteError().to(device)),
        ("MeanSquaredError", MeanSquaredError().to(device)),
        ("R2Score", R2Score().to(device)),
    ]
    val_history = defaultdict(lambda: [])

    logger.info("Optimizer={}".format(optimizer))
    logger.info("LR scheduler={}".format(scheduler))

    model = model.to(device)
    iters = len(train_loader)

    for epoch in range(cur_epoch, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))

        model.train()

        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for i, data in enumerate(tqdm(train_loader)):
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

            preds = torch.relu(preds)
            train_loss = criterion(preds, y.float())

            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if torch.any(torch.isnan(preds)):
                logger.warning("Nan value found in prediction!")

            if i == 0:
                logger.info("preds = {}, y = {}".format(preds, y.float()))

            for _, metric in train_metrics:
                metric.update(preds, y)

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
                preds = torch.relu(preds)

                for _, metric in val_metrics:
                    metric.update(preds, y)

                preds_cpu = preds.cpu()
                y_cpu = y.float().cpu()
                val_predictions.append(preds_cpu)
                val_labels.append(y_cpu)
                del preds
                del y

        train_loss = criterion(torch.cat(train_predictions), torch.cat(train_labels))
        val_loss = criterion(torch.cat(val_predictions), torch.cat(val_labels))

        logger.info("| Train Loss: {:.4f}".format(train_loss))
        train_history["Loss"].append(train_loss.cpu().detach().numpy())

        for metric_name, metric in train_metrics:
            metric_value = metric.compute()
            train_history[metric_name].append(metric_value.cpu().detach().numpy())
            logger.info("| Train {}: {}".format(metric_name, metric_value))
            metric.reset()

        logger.info("| Val Loss: {:.4f}".format(val_loss))
        val_history["Loss"].append(val_loss.cpu().detach().numpy())        

        for metric_name, metric in val_metrics:
            metric_value = metric.compute()
            val_history[metric_name].append(metric_value.cpu().detach().numpy())
            logger.info("| Val {}: {}".format(metric_name, metric_value))
            metric.reset()

        save_checkpoint(
            model, epoch, best_so_far, optimizer, scheduler, model_name, out_dir
        )
        save_model(model, "last", model_name, out_dir)

        if epoch > 10 and (
            best_so_far is None
            or val_history["MeanSquaredError"][-1] < best_so_far
        ):
            best_so_far = val_history["MeanSquaredError"][-1]
            logger.info("Found new best model in epoch {}".format(epoch))
            save_model(model, "best", model_name, out_dir)

        scheduler.step(val_loss)


def build_model_name(args):
    model_type = "local-global-regr" if args.include_global else "local-regr"
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


def save_model(model, model_type, model_name, out_dir):
    filename = "{}/{}.{}_model.pt".format(out_dir, model_name, model_type)
    logger.info("Saving model as {}".format(filename))
    torch.save(model.state_dict(), filename)


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    model_name = build_model_name(args)
    if args.log_file is None:
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
        target_week=args.target_week,
        local_radius=args.local_radius,
        local_k=args.local_k,
        global_k=args.global_k,
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=LocalGlobalTransform(
            args.train_path,
            args.include_global,
            args.append_pos_as_features,
        ),
    )

    num_samples = None
    if args.batches_per_epoch is not None:
        num_samples = args.batches_per_epoch * args.batch_size
        logger.info("Will sample {} samples".format(num_samples))
    train_balanced_sampler = train_dataset.balanced_sampler(num_samples=num_samples, targets=[0.75, 0.25])

    val_dataset = LocalGlobalDataset(
        root_dir=args.val_path,
        target_week=args.target_week,
        local_radius=args.local_radius,
        local_k=args.local_k,
        global_k=args.global_k,
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=LocalGlobalTransform(
            args.val_path,
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
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model_state_dict = None
    optimizer_state_dict = None
    scheduler_state_dict = None
    cur_epoch = 1
    best_so_far = None
    if args.from_checkpoint:
        checkpoint = load_checkpoint(model_name=model_name, out_dir=args.out_dir)
        if checkpoint is not None:
            (
                model_state_dict,
                optimizer_state_dict,
                scheduler_state_dict,
                cur_epoch,
                best_so_far,
            ) = checkpoint
            cur_epoch += 1

    if model_state_dict is None and args.pretrained_model_path is not None:
        model_state_dict = torch.load(args.pretrained_model_path, map_location=device)

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

    if model_state_dict is not None:
        # Important to set the device here, otherwise we get a strange exception.
        # See https://discuss.pytorch.org/t/loading-a-model-runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least-two-devices-cuda-0-and-cpu/143897
        model.to(device)
        model.load_state_dict(model_state_dict)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.05, patience=10, cooldown=10, verbose=True
    )
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        epochs=args.epochs,
        best_so_far=best_so_far,
        val_loader=val_loader,
        model_name=model_name,
        out_dir=args.out_dir,
        cur_epoch=cur_epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "--train-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="train_path",
        default="data/train",
        help="Train set path",
    )
    parser.add_argument(
        "--val-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="val_path",
        default="data/val",
        help="Validation set path",
    )
    parser.add_argument(
        "--pretrained-model-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="pretrained_model_path",
        default=None,
        help="Path to load the a pretrained model from",
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
        default="64,64",
        help="Hidden channels for layer 1 and layer 2 of GCN",
    )
    parser.add_argument(
        "--decoder-hidden-channels",
        metavar="KEY",
        type=str,
        action="store",
        dest="decoder_hidden_channels",
        default="64,64",
        help="Hidden channels for decoder layers",
    )
    parser.add_argument(
        "--epochs",
        metavar="KEY",
        type=int,
        action="store",
        dest="epochs",
        default=1000,
        help="Epochs",
    )
    parser.add_argument(
        "--batches-per-epoch",
        metavar="KEY",
        type=int,
        action="store",
        dest="batches_per_epoch",
        default=500,
        help="Batches per epoch.",
    )
    parser.add_argument(
        "--local-radius",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_radius",
        default=2,
        help="Local radius.",
    )
    parser.add_argument(
        "--local-k",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_k",
        default=9,
        help="Local k for how many nearest neighbors in spatial graph.",
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
        "--local-timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_timesteps",
        default=24,
        help="Time steps in the past for the local part",
    )
    parser.add_argument(
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
        default=9,
        help="Global k for how many nearest neighbors in spatial graph.",
    )
    parser.add_argument(
        "--learning-rate",
        metavar="KEY",
        type=float,
        action="store",
        dest="learning_rate",
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        metavar="KEY",
        type=float,
        action="store",
        dest="weight_decay",
        default=0.00001,
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
    parser.add_argument(
        "--from-checkpoint", dest="from_checkpoint", action="store_true"
    )
    parser.add_argument(
        "--no-from-checkpoint", dest="from_checkpoint", action="store_false"
    )
    parser.set_defaults(from_checkpoint=False)
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

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # torch.multiprocessing.set_start_method('spawn')

    main(args)
