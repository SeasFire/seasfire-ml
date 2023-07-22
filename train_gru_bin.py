#!/usr/bin/env python3
import logging
import os
import argparse
from tqdm import tqdm
import resource 
import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall
from torch.optim import lr_scheduler
from models import (
    GRUModel
)
from utils import (
    GRUDataset,
    GRUTransform,
    load_checkpoint, 
    save_checkpoint,
    set_seed,
)
from collections import defaultdict

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(model, optimizer, scheduler, train_loader, epochs, val_loader, model_name, out_dir, bins, cur_epoch=1, best_so_far=None):
    logger.info("Starting training for {} epochs".format(epochs))

    criterion = torch.nn.CrossEntropyLoss()

    num_classes = len(bins)+1
    train_metrics = [
        ("Accuracy", Accuracy(task="multiclass", num_classes=num_classes).to(device)),
        ("Recall",Recall(task="multiclass", num_classes=num_classes).to(device)),
        ("F1Score",F1Score(task="multiclass", num_classes=num_classes).to(device)),
        ("AveragePrecision (AUPRC)",AveragePrecision(task="multiclass", num_classes=num_classes, average=None).to(device)),
        ("AveragePrecision (AUPRC) weighted",AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted").to(device)),
        ("AUROC",AUROC(task="multiclass", num_classes=num_classes).to(device)),
        ("StatScores",StatScores(task="multiclass", num_classes=num_classes).to(device)),
    ]
    train_history = defaultdict(lambda: [])

    val_metrics = [
        ("Accuracy", Accuracy(task="multiclass", num_classes=num_classes).to(device)),
        ("Recall", Recall(task="multiclass", num_classes=num_classes).to(device)),
        ("F1Score", F1Score(task="multiclass", num_classes=num_classes).to(device)),
        ("AveragePrecision (AUPRC)", AveragePrecision(task="multiclass", num_classes=num_classes, average=None).to(device)),
        ("AveragePrecision (AUPRC) weighted",AveragePrecision(task="multiclass", num_classes=num_classes, average="weighted").to(device)),
        ("AUROC", AUROC(task="multiclass", num_classes=num_classes).to(device)),
        ("StatScores", StatScores(task="multiclass", num_classes=num_classes).to(device)),
    ]
    val_history = defaultdict(lambda: [])

    logger.info("Optimizer={}".format(optimizer))
    logger.info("LR scheduler={}".format(scheduler))
    
    model = model.to(device)
    iters = len(train_loader)

    bins = torch.tensor(bins, dtype=torch.float, device=device)

    for epoch in range(cur_epoch, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))

        model.train()

        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for i, data in enumerate(tqdm(train_loader)):

            x = data[0].to(device)
            y = data[1].to(device)

            preds = model(x)
            
            if torch.any(torch.isnan(preds)):
                logger.warning("Nan value found in prediction!")
            
            y = torch.bucketize(y, bins).squeeze(dim=1)

            if i == 0: 
                preds_probs = torch.softmax(preds, dim=1)
                preds_classes = torch.argmax(preds_probs, dim=1)
                logger.info("preds_classes = {}, y = {}".format(preds_classes, y))

            train_loss = criterion(preds, y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            for _, metric in train_metrics:
                metric.update(preds, y)

            preds_cpu = preds.cpu()
            y_cpu = y.cpu()
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

                preds = model(x)
                y = torch.bucketize(y, bins).squeeze(dim=1)

                for _, metric in val_metrics:
                    metric.update(preds, y)
 
                preds_cpu = preds.cpu()
                y_cpu = y.cpu()
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

        save_checkpoint(model, epoch, best_so_far, optimizer, scheduler, model_name, out_dir)
        save_model(model, "last", model_name, out_dir)

        if (
            epoch > 10
            and (best_so_far is None or val_history["AveragePrecision (AUPRC) weighted"][-1] > best_so_far)
            and val_history["F1Score"][-1] > 0.5
        ):
            best_so_far = val_history["AveragePrecision (AUPRC) weighted"][-1]
            logger.info("Found new best model in epoch {}".format(epoch))
            save_model(model, "best", model_name, out_dir)

        scheduler.step(val_loss)


def build_model_name(args): 
    model_type = "gru-bins"
    target = "target-{}".format(args.target_week)
    oci = "oci-1" if args.include_oci_variables else "oci-0"
    timesteps = "time-l{}".format(args.timesteps)
    return "{}_{}_{}_{}".format(model_type, target, oci, timesteps)

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
        output_size=len(args.bins)+1,
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
        bins=args.bins,
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
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--hidden-channels",
        metavar="KEY",
        type=str,
        action="store",
        dest="hidden_channels",
        default="100",
        help="Hidden channels for layers of the GRU",
    )
    parser.add_argument(
        "--bins",
        metavar="KEY",
        type=str,
        action="store",
        dest="bins",
        default="0,100,1000",
        help="Bins to use",
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
        default=0.0001,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-workers",
        metavar="KEY",
        type=int,
        action="store",
        dest="num_workers",
        default=8,
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
    args.bins = [int(x) for x in args.bins.split(",")]

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    #torch.multiprocessing.set_start_method('spawn')

    main(args)
