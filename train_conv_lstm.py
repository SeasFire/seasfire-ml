#!/usr/bin/env python3
import logging
import os
import argparse
from tqdm import tqdm

import resource 

import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from torch.optim import lr_scheduler
from models import (
    ConvLstmLocalGlobalModel,
)
from utils import (
    ConvLstmDataset,
    ConvLstmTransform,
    load_checkpoint, 
    save_checkpoint,
    set_seed,
)


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, scheduler, train_loader, epochs, val_loader, model_name, out_dir, cur_epoch=1, best_so_far=None):
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

    logger.info("Optimizer={}".format(optimizer))
    logger.info("LR scheduler={}".format(scheduler))

    model = model.to(device)
    iters = len(train_loader)

    for epoch in range(cur_epoch, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))
        logger.info("Current lr={}".format(scheduler.get_last_lr()))

        model.train()

        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for i, data in enumerate(tqdm(train_loader)):

            local_x = data["x"].to(device)
            global_x = data.get("global_x")
            if global_x is not None: 
                global_x = global_x.to(device)
            y = data["y"].to(device)

            preds = model(
                local_x,
                global_x,
            )
            y = y.gt(0.0)

            train_loss = criterion(preds, y.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            scheduler.step(epoch - 1 + i / iters)

            probs = torch.sigmoid(preds)
            logger.debug("probs = {}".format(probs))
            if torch.any(torch.isnan(probs)): 
                logger.warning("Nan value found in prediction!")
            logger.debug("preds = {}".format((probs > 0.5).float()))
            logger.debug("y = {}".format(y.float()))
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

                local_x = data["x"].to(device)
                global_x = data.get("global_x")
                if global_x is not None: 
                    global_x = global_x.to(device)
                y = data["y"].to(device)

                preds = model(
                    local_x,
                    global_x,
                )
                y = y.gt(0.0)

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

        save_checkpoint(model, epoch, best_so_far, optimizer, scheduler, model_name, out_dir)
        save_model(model, "last", model_name, out_dir)

        if (
            epoch > 10
            and (best_so_far is None or val_metrics_dict["AveragePrecision (AUPRC)"][-1] > best_so_far)
            and val_metrics_dict["F1Score"][-1] > 0.5
        ):
            best_so_far = val_metrics_dict["AveragePrecision (AUPRC)"][-1]
            logger.info("Found new best model in epoch {}".format(epoch))
            save_model(model, "best", model_name, out_dir)

        train_metrics_dict["Loss"].append(train_loss.cpu().detach().numpy())
        val_metrics_dict["Loss"].append(val_loss.cpu().detach().numpy())
        # scheduler.step()


def build_model_name(args):
    model_type = "conv-lstm-local-global" if args.include_global else "conv-lstm-local"
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
    torch.save(model.state_dict(),filename)



def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    model_name=build_model_name(args)
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

    train_dataset = ConvLstmDataset(
        root_dir=args.train_path,
        target_week=args.target_week,
        local_radius=args.local_radius,
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=ConvLstmTransform(
            args.train_path,
            args.include_global,
            args.append_pos_as_features,
        ),
    )

    num_samples = None
    if args.batches_per_epoch is not None:
        num_samples = args.batches_per_epoch * args.batch_size
        logger.info("Will sample {} samples".format(num_samples))
    train_balanced_sampler = train_dataset.balanced_sampler(num_samples=num_samples)

    val_dataset = ConvLstmDataset(
        root_dir=args.val_path,
        target_week=args.target_week,
        local_radius=args.local_radius,
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=ConvLstmTransform(
            args.val_path,
            args.include_global,
            args.append_pos_as_features,
        ),
    )

    logger.info("Train dataset length: {}".format(len(train_dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_balanced_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
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
            model_state_dict, optimizer_state_dict, scheduler_state_dict, cur_epoch, best_so_far = checkpoint
            cur_epoch += 1

    model = ConvLstmLocalGlobalModel(
        train_dataset.latlon_shape[0],
        train_dataset.latlon_shape[1],
        len(train_dataset.local_features) + (4 if args.append_pos_as_features else 0),
        args.hidden_channels,
        [(3,3),(3,3)],
        len(args.hidden_channels),
        train_dataset.global_latlon_shape[0],
        train_dataset.global_latlon_shape[1],
        len(train_dataset.global_features) + (4 if args.append_pos_as_features else 0),
        args.hidden_channels,
        [(3,3),(3,3)],
        len(args.hidden_channels),
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

    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=3
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
        help="Hidden channels ",
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
        default=3,
        help="Local radius.",
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
        default=36,
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
        default=0.000001,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-workers",
        metavar="KEY",
        type=int,
        action="store",
        dest="num_workers",
        default=16,
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
    parser.set_defaults(include_global=False)
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
    parser.set_defaults(include_global_oci_variables=False)
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
    parser.set_defaults(include_local_oci_variables=False)
    parser.add_argument("--from-checkpoint", dest="from_checkpoint", action="store_true")
    parser.add_argument(
        "--no-from-checkpoint", dest="from_checkpoint", action="store_false"
    )
    parser.set_defaults(from_checkpoint=False)    
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()

    args.hidden_channels = [int(x) for x in args.hidden_channels.split(",")]
    args.decoder_hidden_channels = [
        int(x) for x in args.decoder_hidden_channels.split(",")
    ]

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # torch.multiprocessing.set_start_method('spawn')

    main(args)
