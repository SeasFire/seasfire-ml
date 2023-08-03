#!/usr/bin/env python3
import logging
import argparse
from tqdm import tqdm

import resource 

import os
import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, StatScores, Recall

from models import (
    LocalGlobalModel,
)
from utils import (
    LocalGlobalDataset,
    LocalGlobalTransform,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, model_name):
    logger.info("Starting Test")

    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)

    with torch.no_grad():
        model.eval()

        metrics = [
            ("Accuracy", Accuracy(task="binary").to(device)),
            ("Recall", Recall(task="binary").to(device)),
            ("F1Score", F1Score(task="binary").to(device)),
            ("AveragePrecision (AUPRC)", AveragePrecision(task="binary").to(device)),
            ("AUROC", AUROC(task="binary").to(device)),
            ("StatScores", StatScores(task="binary").to(device))
        ]

        predictions = []
        labels = []

        for _, data in enumerate(tqdm(loader)):

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
            y = y.gt(0.0)
            probs = torch.sigmoid(preds)

            for _, metric in metrics:
                metric.update(probs, y)

            preds_cpu = preds.cpu()
            y_cpu = y.float().cpu()
            predictions.append(preds_cpu)
            labels.append(y_cpu)
            del preds 
            del y

        loss = criterion(torch.cat(predictions), torch.cat(labels))
        logger.info(f"| Test Loss: {loss}")

        result = "{}".format(model_name)
        for metric_name, metric  in metrics:
            metric_value = metric.compute()
            logger.info("| Test {}: {}".format(metric_name, metric_value))
            result += ",{}".format(metric_value)
            metric.reset()
            
        logger.info(result)


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


def main(args):
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    if not os.path.exists(args.out_dir):
        logger.info("Creating output folder {}".format(args.out_dir))
        os.makedirs(args.out_dir)

    model_name = build_model_name(args)
    logger.info("Using model={}".format(model_name))    
    if args.log_file is None:
        log_file = "{}/{}.test.logs".format(args.out_dir, model_name)
    else: 
        log_file = args.log_file
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("Using target week={}".format(args.target_week))

    dataset = LocalGlobalDataset(
        root_dir=args.test_path,
        target_week=args.target_week,
        local_radius=args.local_radius,
        local_k=args.local_k,
        global_k=args.global_k,        
        include_global=args.include_global,
        include_local_oci_variables=args.include_local_oci_variables,
        include_global_oci_variables=args.include_global_oci_variables,
        transform=LocalGlobalTransform(args.test_path, args.include_global, args.append_pos_as_features),
    )

    logger.info("Dataset length: {}".format(len(dataset)))
    logger.info("Using batch size={}".format(args.batch_size))

    model = LocalGlobalModel(
        len(dataset.local_features) + (4 if args.append_pos_as_features else 0),
        args.hidden_channels,
        args.local_timesteps,
        dataset.local_nodes,
        len(dataset.global_features) + (4 if args.append_pos_as_features else 0),
        args.hidden_channels,
        args.global_timesteps,
        dataset.global_nodes,
        args.decoder_hidden_channels,
        args.include_global,
    )
    model.load_state_dict(torch.load(args.model_path))

    loader = torch_geometric.loader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test(model=model, loader=loader, model_name=model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
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
        default=None,
        help="Path to load the trained model from",
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
        default="256,64",
        help="Hidden channels for decoder layers",
    )    
    parser.add_argument(
        "--local-radius",
        metavar="KEY",
        type=int,
        action="store",
        dest="local_radius",
        default=2,
        help="Local radius",
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
        "--global-k",
        metavar="KEY",
        type=int,
        action="store",
        dest="global_k",
        default=9,
        help="Global k for how many nearest neighbors in spatial graph.",
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
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)
    parser.add_argument(
        "--include-global", dest="include_global", action="store_true"
    )
    parser.add_argument(
        "--no-include-global", dest="include_global", action="store_false"
    )
    parser.set_defaults(include_global=True)        
    parser.add_argument(
        "--append-pos-as-features", dest="append_pos_as_features", action="store_true"
    )
    parser.add_argument(
        "--no-append-pos-as-features", dest="append_pos_as_features", action="store_false"
    )
    parser.set_defaults(append_pos_as_features=True)
    parser.add_argument(
        "--include-local-oci-variables", dest="include_local_oci_variables", action="store_true"
    )
    parser.add_argument(
        "--no-include-local-oci-variables", dest="include_local_oci_variables", action="store_false"
    )
    parser.set_defaults(include_local_oci_variables=False)
    parser.add_argument(
        "--include-global-oci-variables", dest="include_global_oci_variables", action="store_true"
    )
    parser.add_argument(
        "--no-include-global-oci-variables", dest="include_global_oci_variables", action="store_false"
    )
    parser.set_defaults(include_global_oci_variables=False)    

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

    #torch.multiprocessing.set_start_method('spawn') 

    main(args)
