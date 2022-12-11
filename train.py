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

from models import GCN, AttentionGNN, LstmGCN
from graph_dataset import GraphDataset
from scale_dataset import StandardScaling

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


def train(model, train_loader, epochs, val_loader, batch_size, task):
    optimizer = model.optimizer

    criterion = 0
    if task == 'binary':
        criterion = torch.nn.BCELoss()
    elif task == 'regression':
        # criterion = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        logger.info("Starting Epoch {}".format(epoch))

        model.train()
    
        logger.info("Epoch {} Training".format(epoch))

        train_predictions = []
        train_labels = []

        for _, data in enumerate(tqdm(train_loader)):
            
            data = data.to(device)

            preds = model(data.x, data.edge_index, task, data.batch)
            y = data.y.unsqueeze(1)

            train_predictions.append(preds)
            train_labels.append(y)

            train_loss = criterion(y, preds)
            print(train_loss)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if task == 'binary':
                preds = preds.to("cpu").reshape(-1).detach().numpy().round()
                print("Preds: ",preds)

                y = y.to("cpu").reshape(-1).detach().numpy()
                print("Y: ", y)

                correct = (preds == y).astype(int)
                print(correct)
                acc = correct.sum() / correct.shape[0]
                print(f'Accuracy: {acc:.4f}')


            # train_loss = criterion(y, preds)

        # # Validation
        # logger.info("Epoch {} Validation".format(epoch))
        # val_predictions = []
        # val_labels = []
        # with torch.no_grad():
        #     model.eval()

        #     for _, data in enumerate(tqdm(val_loader)):
        #         data = data.to(device)

        #         preds = model(data.x, data.edge_index, data.batch)
        #         y = data.y.unsqueeze(1)

        #         val_predictions.append(preds)
        #         val_labels.append(y)

        #         # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        #         # acc = int(correct) / int(data.test_mask.sum())
        #         # print(f'Accuracy: {acc:.4f}')

        train_loss = criterion(torch.cat(train_labels), torch.cat(train_predictions))
        # val_loss = criterion(torch.cat(val_labels), torch.cat(val_predictions))

        print(f"Epoch {epoch} | Train Loss: {train_loss}") # + f" | Val Loss: {val_loss}")
    #     # print(train_labels)
    #     # print(train_predictions)
        plt.figure(figsize=(24, 15))
        x_axis = torch.arange(
            0, (torch.cat(train_predictions).to("cpu").detach().numpy()).shape[0]
        )
        plt.scatter(
            x_axis,
            torch.cat(train_predictions).to("cpu").detach().numpy(),
            linestyle="dotted",
            color="b",
        )
        plt.scatter(
            x_axis,
            torch.cat(train_labels).to("cpu").numpy(),
            linestyle="dotted",
            color="r",
        )
        # plt.show()
        plt.savefig("logs/" + str(epoch) + ".png")

    return model, criterion


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.FileHandler("logs.log"))

    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.info("Torch cuda version: {}".format(torch.version.cuda))

    set_seed()


    logger.info("Extracting dataset statistics")
    scaler = StandardScaling(args.model_name)
    graphs = []
    number_of_train_samples = len(os.listdir(args.train_path))
    for idx in range(0, number_of_train_samples):
        graph = torch.load(os.path.join(args.train_path, "graph_{}.pt".format(idx)))
        graphs.append(graph.x)
    mean_std_tuples = scaler.fit(graphs)
    logger.info("Statistics: {}".format(mean_std_tuples))

    train_dataset = GraphDataset(
        root_dir=args.train_path, transform=scaler, task=args.task
    )
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=args.batch_size
    )

    val_dataset = GraphDataset(root_dir=args.val_path, transform=scaler, task=args.task)
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=args.batch_size
    )

    logger.info("Building model {}".format(args.model_name))
    if args.model_name == "AttentionGNN":
        num_node_features = train_dataset.num_node_features + 2
        timesteps = args.timesteps
        model = AttentionGNN(
            num_node_features, timesteps, args.learning_rate, args.weight_decay
        ).to(device)
    elif args.model_name == "LstmGCN":
        num_node_features = 10
        model = LstmGCN(
            num_node_features,
            args.hidden_channels,
            args.learning_rate,
            args.weight_decay,
            args.task,
        ).to(device)
    elif args.model_name == "GCN":
        num_node_features = train_dataset.num_node_features
        model = GCN(
            num_node_features,
            args.hidden_channels,
            args.learning_rate,
            args.weight_decay,
            args.task,
        ).to(device)
    else:
        raise ValueError("Invalid model")

    logger.info("Starting training")
    model, criterion = train(
        model, train_loader, args.epochs, val_loader, args.batch_size, args.task
    )

    logger.info("Saving model as {}".format(args.model_path))
    model_info = {
        "model": model,
        "criterion": criterion,
        "scaler": scaler,
        "name": args.model_name,
    }

    ## Save the entire model to PATH
    # torch.save(model_info, args.model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t" "--train-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="train_path",
        default="data/train",
        help="Train set path",
    )
    parser.add_argument(
        "-v" "--val-path",
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
        default="attention_model.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--task",
        metavar="KEY",
        type=str,
        action="store",
        dest="task",
        default="regression",
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
        "-ch",
        "--hidden-channels",
        metavar="KEY",
        type=int,
        action="store",
        dest="hidden_channels",
        default=64,
        help="Hidden channels",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="KEY",
        type=int,
        action="store",
        dest="epochs",
        default=1,
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
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        metavar="KEY",
        type=float,
        action="store",
        dest="weight_decay",
        default=5e-5,
        help="Weight decay",
    )
    args = parser.parse_args()
    main(args)
