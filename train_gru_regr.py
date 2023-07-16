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
    GRUModel
)
from utils import (
    GRUDataset,
    GRUTransform,
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


def train(model, optimizer, scheduler, train_loader, epochs, val_loader, model_name, out_dir, cur_epoch=1, best_so_far=None):
    logger.info("Starting training for {} epochs".format(epochs))

    train_metrics_dict = {
        "MeanAbsoluteError": [],
        "MeanSquaredError": [], 
        "R2Score": [],
        "Loss": [],
    }
    val_metrics_dict = {
        "MeanAbsoluteError": [],
        "MeanSquaredError": [], 
        "R2Score": [],
        "Loss": [],
    }

    criterion = torch.nn.MSELoss()

    train_metrics = [
        MeanAbsoluteError().to(device),
        MeanSquaredError().to(device),
        R2Score().to(device),
    ]

    val_metrics = [
        MeanAbsoluteError().to(device),
        MeanSquaredError().to(device),
        R2Score().to(device),
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

            x = data[0].to(device)
            y = data[1].to(device)

            y = torch.log10(1 + y)

            preds = model(x)

            train_loss = criterion(preds, y.float())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            scheduler.step(epoch - 1 + i / iters)

            if torch.any(torch.isnan(preds)): 
                logger.warning("Nan value found in prediction!")            
            # logger.info("preds = {}".format(preds))
            # logger.info("y = {}".format(y.float()))
            for metric in train_metrics:
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

                x = data[0].to(device)
                y = data[1].to(device)

                y = torch.log10(1 + y)

                preds = model(x)

                # logger.info("preds = {}".format(preds))
                # logger.info("y = {}".format(y.float()))
                for metric in val_metrics:
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
            and (best_so_far is None or val_metrics_dict["MeanSquaredError"][-1] < best_so_far)
        ):
            best_so_far = val_metrics_dict["MeanSquaredError"][-1]
            logger.info("Found new best model in epoch {}".format(epoch))
            save_model(model, "best", model_name, out_dir)

        train_metrics_dict["Loss"].append(train_loss.cpu().detach().numpy())
        val_metrics_dict["Loss"].append(val_loss.cpu().detach().numpy())
        #scheduler.step()

def build_model_name(args): 
    model_type = "gru-regr"
    target = "target-{}".format(args.target_week)
    oci = "oci-1" if args.include_oci_variables else "oci-0"
    timesteps = "time-l{}".format(args.timesteps)
    return "{}_{}_{}_{}".format(model_type, target, oci, timesteps)

def save_model(model, model_type, model_name, out_dir):
    filename = "{}/{}.{}_model.pt".format(out_dir, model_name, model_type)
    logger.info("Saving model as {}".format(filename))
    torch.save(model.state_dict(),filename)

def save_checkpoint(model, epoch, best_so_far, optimizer, scheduler, model_name, out_dir):
    filename = "{}/{}.checkpoint.pt".format(out_dir, model_name)
    logger.info("Saving checkpoint as {}".format(filename))

    torch.save(model.state_dict(), "{}/{}.checkpoint.model.pt".format(out_dir, model_name))
    torch.save(optimizer.state_dict(), "{}/{}.checkpoint.optimizer.pt".format(out_dir, model_name))
    torch.save(scheduler.state_dict(), "{}/{}.checkpoint.scheduler.pt".format(out_dir, model_name))
    torch.save(epoch, "{}/{}.checkpoint.epoch.pt".format(out_dir, model_name))
    torch.save(best_so_far, "{}/{}.checkpoint.best_so_far.pt".format(out_dir, model_name))

def load_checkpoint(model_name, out_dir):
    epoch = 0
    if os.path.exists("{}/{}.checkpoint.epoch.pt".format(out_dir, model_name)): 
        epoch = torch.load("{}/{}.checkpoint.epoch.pt".format(out_dir, model_name))

    model_state_dict = None
    if os.path.exists("{}/{}.checkpoint.model.pt".format(out_dir, model_name)): 
        model_state_dict = torch.load("{}/{}.checkpoint.model.pt".format(out_dir, model_name), map_location=device)

    optimizer_state_dict = None
    if os.path.exists("{}/{}.checkpoint.optimizer.pt".format(out_dir, model_name)): 
        optimizer_state_dict = torch.load("{}/{}.checkpoint.optimizer.pt".format(out_dir, model_name), map_location=device)

    scheduler_state_dict = None
    if os.path.exists("{}/{}.checkpoint.scheduler.pt".format(out_dir, model_name)): 
        scheduler_state_dict = torch.load("{}/{}.checkpoint.scheduler.pt".format(out_dir, model_name), map_location=device)

    best_so_far = None
    if os.path.exists("{}/{}.checkpoint.best_so_far.pt".format(out_dir, model_name)): 
        best_so_far = torch.load("{}/{}.checkpoint.best_so_far.pt".format(out_dir, model_name))    

    return (model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_so_far)

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

    train_dataset = GRUDataset(
        root_dir=args.train_path,
        target_week=args.target_week,
        include_oci_variables=args.include_oci_variables,
        transform=GRUTransform(args.train_path, args.timesteps),
    )
    num_samples = None
    if args.batches_per_epoch is not None: 
        num_samples = args.batches_per_epoch * args.batch_size
        logger.info("Will sample {} samples".format(num_samples))
    train_balanced_sampler = train_dataset.balanced_sampler(num_samples=num_samples)    
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

    val_dataset = GRUDataset(
        root_dir=args.val_path,
        target_week=args.target_week,        
        include_oci_variables=args.include_oci_variables,
        transform=GRUTransform(args.val_path, args.timesteps),
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
            model_state_dict, optimizer_state_dict, scheduler_state_dict, cur_epoch, best_so_far = checkpoint
            cur_epoch += 1

    model = GRUModel(
        len(train_dataset.local_features),
        args.hidden_channels[0],
        num_layers=2,
        output_size=1,
        dropout=0.1
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
        optimizer, T_0=50, T_mult=1
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
    parser = argparse.ArgumentParser(description="Train Gru Regression")
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
        default="64",
        help="Hidden channels for layers of the GRU",
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
        "--target-week",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_week",
        default=1,
        help="Target week",
    )
    parser.add_argument(
        "--timesteps",
        metavar="KEY",
        type=int,
        action="store",
        dest="timesteps",
        default=24,
        help="Time steps in the past",
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
        default=0.001,
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
    parser.add_argument(
        "--include-oci-variables", dest="include_oci_variables", action="store_true"
    )
    parser.add_argument(
        "--no-include-oci-variables", dest="include_oci_variables", action="store_false"
    )
    parser.set_defaults(include_oci_variables=False)
    parser.add_argument("--from-checkpoint", dest="from_checkpoint", action="store_true")
    parser.add_argument(
        "--no-from-checkpoint", dest="from_checkpoint", action="store_false"
    )
    parser.set_defaults(from_checkpoint=False)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")    
    args = parser.parse_args()

    args.hidden_channels = [int(x) for x in args.hidden_channels.split(",")]

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    #torch.multiprocessing.set_start_method('spawn')

    main(args)
