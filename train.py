#!/usr/bin/env python3

## First draft for GCN model

import torch
from tqdm import tqdm
import logging
import os
import torch_geometric
import argparse

from models import GCN, AttentionGNN, LstmGCN
from graph_dataset import GraphDataset
from scale_dataset import StandardScaling

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_loader, epochs, val_loader, batch_size):

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

        for data in data_loader:
            data = data.to(device)
            # preds = [model(data.x[:,:,i], data.edge_index) for i in range(0, 12)]
            preds = model(data.x, data.edge_index)
            # preds = model(data.x, data.edge_index, data.batch)
            pred = preds[12]
            pred = pred.unsqueeze(1)
            y = data.y.unsqueeze(1)

            print(pred)
            print(y)

            train_predictions.append(pred)
            train_labels.append(y)

            train_loss = criterion(y, pred)
            train_loss.backward()

            optimizer.step()

        # Validation
        # with torch.no_grad():
        #     model.eval()

        #     for data in val_loader:
        #         data = data.to(device)

        #         preds = model(data.x, data.edge_index, data.batch)
        #         y = data.y.unsqueeze(1)

        #         val_predictions.append(preds)
        #         val_labels.append(y)

        #         correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        #         acc = int(correct) / int(data.test_mask.sum())
        #         print(f'Accuracy: {acc:.4f}')

        train_loss = criterion(torch.cat(train_labels), torch.cat(train_predictions))
        # val_loss = criterion(torch.cat(val_labels), torch.cat(val_predictions))

        print(f"Epoch {epoch} | Train Loss: {train_loss}") #f" | Val Loss: {val_loss}")

        # x_axis = torch.arange(0, (torch.cat(train_predictions).to('cpu').detach().numpy()).shape[0])

        # plt.scatter(x_axis, torch.cat(train_predictions).to('cpu').detach().numpy(), linestyle = 'dotted', color='b')
        # plt.scatter(x_axis, torch.cat(train_labels).to('cpu').numpy(), linestyle = 'dotted', color='r')
        # # plt.show()
        # plt.savefig('logs/' + str(epoch)+'.png')


def main(args):
    FileOutputHandler = logging.FileHandler("logs.log")
    logger.addHandler(FileOutputHandler)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    scaler = StandardScaling(args.model)

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
    
    print(args.model)
    if args.model == "AttentionGNN":
        # input_size = 250
        model = AttentionGNN(10, 12, args.learning_rate, args.weight_decay
        ).to(device)
    elif args.model == "LstmGCN":
        input_size = 10
        model = LstmGCN(input_size, args.hidden_channels, args.learning_rate, args.weight_decay, args.task
        ).to(device)
    elif args.model == "GCN":
        input_size = train_dataset.num_node_features
        model = GCN(input_size, args.hidden_channels, args.learning_rate, args.weight_decay, args.task
        ).to(device)
    else:
        raise ValueError("Invalid model")

    train(model, train_loader, args.epochs, val_loader, args.batch_size)


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
        "--model",
        metavar="KEY",
        type=str,
        action="store",
        dest="model",
        default="AttentionGNN",
        help="Model name",
    )
    parser.add_argument(
        "-t",
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
        default=10,
        help="Epochs",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="KEY",
        type=float,
        action="store",
        dest="learning_rate",
        default=5e-5,
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
