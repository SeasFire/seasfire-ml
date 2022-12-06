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

def test(model, test_loader, model_name, criterion):
    # Validation
    with torch.no_grad():
        model.eval()
        val_predictions = []
        val_labels = []

        for data in test_loader:
            data = data.to(device)

            if model_name == "AttentionGNN":
                preds = model(data.x, data.edge_index)

                ## Care only for central node
                preds = preds[int(data.x.shape[0]/2)]
                preds = preds.unsqueeze(1)
            elif model_name == "GCN":
                preds = model(data.x, data.edge_index, data.batch)
            # elif model_name == "LstmGCN":
            #     preds = [model(data.x[:,:,i], data.edge_index) for i in range(0, 12)]
            
            y = data.y.unsqueeze(1)

            val_predictions.append(preds)
            val_labels.append(y)

            # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            # acc = int(correct) / int(data.test_mask.sum())
            # print(f'Accuracy: {acc:.4f}')

    val_loss = criterion(torch.cat(val_labels), torch.cat(val_predictions))

    print(f" | Val Loss: {val_loss}")
    # print(train_labels)
    # print(train_predictions)
    plt.figure(figsize=(24, 15))
    x_axis = torch.arange(0, (torch.cat(val_predictions).to('cpu').detach().numpy()).shape[0])
    plt.scatter(x_axis, torch.cat(val_predictions).to('cpu').detach().numpy(), linestyle = 'dotted', color='b')
    plt.scatter(x_axis, torch.cat(val_labels).to('cpu').numpy(), linestyle = 'dotted', color='r')
    # plt.show()
    plt.savefig('logs1/' + 'test'+'.png')


def main(args):
    FileOutputHandler = logging.FileHandler("logs.log")
    logger.addHandler(FileOutputHandler)

    logger.debug("Torch version: {}".format(torch.__version__))
    logger.debug("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        logger.debug("Torch cuda version: {}".format(torch.version.cuda))

    model_info = torch.load(args.model_path)
    model = model_info['model']
    criterion = model_info['criterion']
    scaler = model_info['scaler']
    model_name = model_info['name']

    test_dataset = GraphDataset(root_dir=args.test_path, transform=scaler, task=args.task)
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=args.batch_size
    )
    
    test(model, test_loader, model_name, criterion)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t",
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="dataset/val/",
        help="Test set path",
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

    args = parser.parse_args()
    main(args)