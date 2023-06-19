######### COMMENTS ###########
# NEEDS FIX FOR DATES #
# RUNS ONLY FOR BATCH SIZE 1 #
# python3 inference_map.py --test-path "weekly_data/test/" --model-name "AttentionGNN-TGCN2" --model-path "gru.pt" #

#!/usr/bin/env python3
import numpy
import logging
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch_geometric
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score
from graph_dataset import GraphDataset
from torch_geometric.data import Data


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, model_name, loader, criterion, task, date):

    model = model.to(device)

    with torch.no_grad():
        model.eval()

        test_metrics_dict = {"Preds": [], "Target": []}
        test_metrics = [
            Accuracy(task="multiclass", num_classes=2).to(device),
            F1Score(task="multiclass", num_classes=2, average="macro").to(device),
            AveragePrecision(task="multiclass", num_classes=2).to(device),
            AUROC(task="multiclass", num_classes=2).to(device),
        ]

        test_predictions = []
        test_labels = []

        preds_map_dict = {}
        true_map_dict = {}

        for data in tqdm(loader):
            if isinstance(data, Data):
                data = data.to(device)
                x = data.x
                y = data.y
                time = data.center_time
                lon = data.center_lon
                lat = data.center_lat
                edge_index = data.edge_index
                batch = data.batch
            else:
                x = data[0].to(device)
                y = data[1].to(device)
                edge_index = None
                batch = None

            ############# PRINT PREDICTION MAP FOR ONE SPECIFIC TIME PERIOD ################
            preds = model(x, edge_index, None, None, batch)
            if model_name != 'GRU':
                dict_preds = torch.argmax(preds, dim=1)
                dict_y = torch.argmax(y, dim=1)
                
                for i in range(0, len(dict_y)):
                    if data.center_time[i] == numpy.datetime64(date):
                        preds_map_dict[((data.center_lon[i]).item(),(data.center_lat[i]).item())] = (dict_preds[i].item())
                        true_map_dict[((data.center_lon[i]).item(),(data.center_lat[i]).item())] = (dict_y[i].item())

            ############# END OF PRINTING PREDICTION MAP FOR ONE SPECIFIC TIME PERIOD ################

            if task == "regression":
                y = y.unsqueeze(1)

            test_predictions.append(preds)
            test_labels.append(y)

            if task == "binary":
                y_class = torch.argmax(y, dim=1)
                for metric in test_metrics:
                    metric.update(preds, y_class)

        ################ SAVE PREDICTIONS IN PKL FILE #######################
        if model_name != 'GRU':
            with open('preds_map_dict.pkl', 'wb') as handle:
                pkl.dump(preds_map_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

            with open('preds_map_dict.pkl', 'rb') as handle:
                preds_map_dict = pkl.load(handle)
            
            with open('true_map_dict.pkl', 'wb') as handle:
                pkl.dump(true_map_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

            with open('true_map_dict.pkl', 'rb') as handle:
                true_map_dict = pkl.load(handle)

        # ################ END OF SAVING PREDICTIONS IN PKL FILE #######################

        # ################## PRINT PREDICTIONS IN MAP ##########################
        if model_name != 'GRU':
            lon_list = [-24.5, -23.5, -22.5, -21.5, -20.5, -19.5, -18.5, -17.5, -16.5, -15.5, -14.5, -13.5, -12.5,
                        -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5,
                        4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                        20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5,
                        35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5]

            lat_list = [36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5,
                        51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 
                        66.5, 67.5, 68.5, 69.5, 70.5, 71.5]

            preds_map = np.empty([75,36])
            true_map = np.empty([75,36])

            for i, lon in enumerate(lon_list):
                for j, lat in enumerate(lat_list):
                    if (lon,lat) in list(preds_map_dict.keys()):
                        preds_map[i,j] = (preds_map_dict[(lon,lat)])
                        true_map[i,j] = (true_map_dict[(lon,lat)])
                    else:
                        preds_map[i,j] = 0
                        true_map[i,j] = 0
                    
            plt.figure()
            #subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1,2) 
            # use the created array to output our multiple images
            axarr[0].imshow(preds_map)
            axarr[1].imshow(true_map)
            plt.savefig("preds2.png")
            plt.show()
        ################## END OF PRINTING PREDICTIONS IN MAP ######################

        test_loss = criterion(torch.cat(test_labels), torch.cat(test_predictions))
        logger.info(f" | Test Loss: {test_loss}")

        if task == "binary":
            for metric, metric_name in zip(
                test_metrics, ["Accuracy", "F1Score", "Average Precision", "AUROC"]
            ):
                temp = metric.compute()
                logger.info(f"| Test {metric_name}: {(temp):.4f}")
                metric.reset()
            test_metrics_dict["Preds"].append(
                (torch.argmax(torch.cat(test_predictions), dim=1))
                .cpu()
                .detach()
                .numpy()
            )
            test_metrics_dict["Target"].append(
                torch.cat(test_labels).cpu().detach().numpy()
            )
        elif task == "regression":
            test_metrics_dict["Preds"].append(
                torch.cat(test_predictions).cpu().detach().numpy()
            )
            test_metrics_dict["Target"].append(
                torch.cat(test_labels).cpu().detach().numpy()
            )

        with open("test_metrics.pkl", "wb") as file:
            pkl.dump(test_metrics_dict, file)


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.FileHandler("logs.log"))

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    model_info = torch.load(args.model_path)
    model = model_info["model"]
    criterion = model_info["criterion"]
    transform = model_info["transform"]
    model_name = model_info["name"]
    
    logger.info("Using model={}".format(model_name))

    logger.info("Using target week={}".format(args.target_week))
    transform.target_week = args.target_week

    test_dataset = GraphDataset(
        root_dir=args.test_path,
        transform=transform,
    )

    if model_name in [
        "AttentionGNN-TGCN2",
        "AttentionGNN-TGatConv",
        "Attention2GNN-TGCN2",
        "Attention2GNN-TGatConv",
        "Transformer_Aggregation-TGCN2",
    ]:
        loader_class = torch_geometric.loader.DataLoader
    elif model_name == "GRU":
        loader_class = torch.utils.data.DataLoader
    else:
        raise ValueError("Invalid model")

    loader = loader_class(
        test_dataset,
        batch_size=args.batch_size,
    )

    date  = args.date + "T00:00:00.000000000"
    test(model=model, model_name=model_name, loader=loader, criterion=criterion, task=args.task, date = date)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t",
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="data25/data/test",
        help="Test set path",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        metavar="KEY",
        type=str,
        action="store",
        dest="model_name",
        default="Transformer_Aggregation-TGCN2",
        help="Model name",
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="model_path",
        default="5_binary_attention_model.pt",
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
        default=1024,
        help="Batch size",
    )
    parser.add_argument(
        "--target-week",
        metavar="KEY",
        type=int,
        action="store",
        dest="target_week",
        default=4,
        help="Target week",
    )
    parser.add_argument(
        "--date",
        metavar="KEY",
        type=str,
        action="store",
        dest="date",
        default="2020-01-01",
        help="Predicted date",
    )
    args = parser.parse_args()
    main(args)

#python3 inference.py --test-path "weekly_data/test/" --model-name "GRU" --model-path "4_weeks_best_gru.pt"
#python3 inference.py --test-path "weekly_data/test/" --model-name "AttentionGNN-TGCN2" --model-path "random_best_gru.pt" 