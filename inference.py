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

from models import AttentionGNN
from graph_dataset import GraphDataset
from transforms import GraphNormalize
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, loader, criterion, task):
    with torch.no_grad():
        model.eval()

        test_metrics_dict = {"Accuracy":[],"F1Score":[],"AveragePrecision":[], "AUROC":[], "Loss":[]}
        test_metrics = [
            Accuracy(task="multiclass", num_classes=2).to(device),
            F1Score(task = 'multiclass', num_classes = 2, average='macro').to(device),
            AveragePrecision(task="multiclass", num_classes = 2).to(device),
            AUROC(task="multiclass", num_classes = 2).to(device)
        ]

        test_predictions = []
        test_labels = []

        for data in tqdm(loader):
            data = data.to(device)

            preds = model(data.x, data.edge_index, None, None, data.batch)
            y = data.y

            test_predictions.append(preds)
            test_labels.append(y)

            if task == 'binary':
                y_class = torch.argmax(y, dim=1)
                for metric in test_metrics: 
                    metric.update(preds, y_class)
                
        test_loss = criterion(torch.cat(test_labels), torch.cat(test_predictions))
        print(f" | Test Loss: {test_loss}")

        if task == 'binary':
            for metric, key in zip(test_metrics,test_metrics_dict.keys()): 
                temp = metric.compute()
                print(f'Test {key}: {(temp):.4f}')
                test_metrics_dict[key].append(temp.cpu().detach().numpy())
                metric.reset()
        
        test_metrics_dict["Loss"].append(test_loss.cpu().detach().numpy())

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
    transform = model_info['transform']
    model_name = model_info['name']

    test_dataset = GraphDataset(
        root_dir=args.test_path, 
        transform=transform,)
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=args.batch_size
    )
    
    test(model=model, loader=test_loader, criterion=criterion, task=args.task)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t",
        "--test-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="data/test/",
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
        default="best_binary_attention_model.pt",
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
        default=64,
        help="Batch size",
    )

    args = parser.parse_args()
    main(args)
