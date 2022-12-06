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
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train(model, train_loader, epochs, val_loader, batch_size):

    optimizer = model.optimizer
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(0, epochs + 1)):
        model.train()
        optimizer.zero_grad()

        train_predictions = []
        train_labels = []

        val_predictions = []
        val_labels = []

        for data in train_loader:
            data = data.to(device)

            preds = []
            preds = model(data.x, data.edge_index, data.batch)
            y = data.y.unsqueeze(1)

            train_predictions.append(preds)
            train_labels.append(y)

            train_loss = criterion(y, preds)
            train_loss.backward()

            optimizer.step()

        # Validation
        with torch.no_grad():
            model.eval()

            for data in val_loader:
                data = data.to(device)

                preds = model(data.x, data.edge_index, data.batch)
                y = data.y.unsqueeze(1)

                val_predictions.append(preds)
                val_labels.append(y)

                # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                # acc = int(correct) / int(data.test_mask.sum())
                # print(f'Accuracy: {acc:.4f}')

        train_loss = criterion(torch.cat(train_labels), torch.cat(train_predictions))
        val_loss = criterion(torch.cat(val_labels), torch.cat(val_predictions))

        print(f"Epoch {epoch} | Train Loss: {train_loss}" + f" | Val Loss: {val_loss}")
        # print(train_labels)
        # print(train_predictions)
        plt.figure(figsize=(24, 15))
        x_axis = torch.arange(0, (torch.cat(train_predictions).to('cpu').detach().numpy()).shape[0])
        plt.scatter(x_axis, torch.cat(train_predictions).to('cpu').detach().numpy(), linestyle = 'dotted', color='b')
        plt.scatter(x_axis, torch.cat(train_labels).to('cpu').numpy(), linestyle = 'dotted', color='r')
        # plt.show()
        plt.savefig('logs/' + str(epoch)+'.png')

    return model, criterion

def main(args):
    FileOutputHandler = logging.FileHandler("logs.log")
    logger.addHandler(FileOutputHandler)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    set_seed()

    scaler = StandardScaling(args.model_name)

    graphs = []
    number_of_train_samples = len(os.listdir(args.train_path))
    for idx in range(0, number_of_train_samples):
        graph = torch.load(args.train_path + f"/graph_{idx}.pt")
        graphs.append(graph.x)
    mean_std_tuples = scaler.fit(graphs)
    # print(mean_std_tuples)

    train_dataset = GraphDataset(root_dir=args.train_path, transform=scaler, task=args.task)
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=args.batch_size
    )

    val_dataset = GraphDataset(root_dir=args.val_path, transform=scaler, task=args.task)
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=args.batch_size
    )

    model = None
    
    if args.model_name == "AttentionGNN":
        num_node_features = train_dataset.num_node_features
        timesteps = args.timesteps
        model = AttentionGNN(num_node_features, timesteps, args.learning_rate, args.weight_decay
        ).to(device)
    elif args.model_name == "LstmGCN":
        num_node_features = 10
        model = LstmGCN(num_node_features, args.hidden_channels, args.learning_rate, args.weight_decay, args.task
        ).to(device)
    elif args.model_name == "GCN":
        num_node_features = train_dataset.num_node_features
        model = GCN(num_node_features, args.hidden_channels, args.learning_rate, args.weight_decay, args.task
        ).to(device)
    else:
        raise ValueError("Invalid model")

    model, criterion = train(model, train_loader, args.epochs, val_loader, args.batch_size)

    model_info = {
        'model': model,
        'criterion': criterion,
        'scaler': scaler,
        'name': args.model_name
    }

    ## Save the entire model to PATH
    torch.save(model_info, args.model_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t" "--train-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="train_path",
        default="dataset/train/",
        help="Train set path",
    )
    parser.add_argument(
        "-v" "--val-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="val_path",
        default="dataset/val/",
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
        default=1,
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
        default=5,
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
