#!/usr/bin/env python3
import logging
import argparse
from tqdm import tqdm

import resource

import os
import torch
import torch_geometric
import xarray as xr
import numpy as np

from models import (
    LocalGlobalModel,
)
from utils import (
    LocalGlobalDataset,
    LocalGlobalTransform,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_map(cube, model, loader, output_path):
    logger.info("Starting Inference")

    model = model.to(device)

    logger.info("{}".format(cube["gwis_ba"]))

    gwis_ba = cube["gwis_ba"]
    gwis_ba_prediction = xr.zeros_like(gwis_ba, dtype=np.float32)

    with torch.no_grad():
        model.eval()

        for _, data in enumerate(tqdm(loader)):
            data = data.to(device)

            local_x = data.x
            global_x = data.get("global_x")
            local_edge_index = data.edge_index
            global_edge_index = data.get("global_edge_index")
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
            probs_cpu = probs.cpu()

            center_lat = data.center_lat[0]
            center_lon = data.center_lon[0]
            center_time = data.center_time[0]
            prob_value = probs_cpu[0]

            gwis_ba_prediction.loc[
                dict(latitude=center_lat, longitude=center_lon, time=center_time)
            ] = prob_value

    dataset = xr.Dataset(
        {
            "gwis_ba_prediction": gwis_ba_prediction,
        },
        attrs={"description": "A dataset with our gwis_ba prediction"},
    )

    dataset.to_netcdf(output_path)


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
        transform=LocalGlobalTransform(
            args.test_path, args.include_global, args.append_pos_as_features
        ),
    )

    logger.info("Dataset length: {}".format(len(dataset)))

    model = LocalGlobalModel(
        len(dataset.local_features) + 4 if args.append_pos_as_features else 0,
        args.hidden_channels,
        args.local_timesteps,
        dataset.local_nodes,
        len(dataset.global_features) + 4 if args.append_pos_as_features else 0,
        args.hidden_channels,
        args.global_timesteps,
        dataset.global_nodes,
        args.decoder_hidden_channels if not args.include_global else None,
        args.include_global,
    )
    model.load_state_dict(torch.load(args.model_path))

    loader = torch_geometric.loader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    cube = xr.open_zarr(args.cube_path, consolidated=False)
    create_map(cube=cube, model=model, loader=loader, output_path=args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "--cube-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="cube_path",
        default="../seasfire_1deg.zarr",
        help="Cube path",
    )
    parser.add_argument(
        "--output-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="output_path",
        default="gwis_ba_pred.nc",
        help="Output path for predictions",
    )    
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
        default=4,
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
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)
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
