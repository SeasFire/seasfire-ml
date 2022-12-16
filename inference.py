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
from transforms import StandardScaling
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_loader, criterion, task):
    with torch.no_grad():
        model.eval()

        metrics = [
            Accuracy(task="multiclass", num_classes=2).to(device),
            F1Score(task = 'binary', num_classes = 2).to(device),
            AveragePrecision(task="multiclass", num_classes = 2).to(device),
            AUROC(task="multiclass", num_classes = 2).to(device)
        ]

        test_predictions = []
        test_labels = []

        for data in tqdm(test_loader):
            data = data.to(device)

            preds = model(data.x, data.edge_index, task, data.batch)
            y = data.y

            test_predictions.append(preds)
            test_labels.append(y)

            if task == 'binary':
                y_class = torch.argmax(y, dim=1)
                for metric in metrics: 
                    metric.update(preds, y_class)
                
        test_loss = criterion(torch.cat(test_labels), torch.cat(test_predictions))
        print(f" | Test Loss: {test_loss}")

        if task == 'binary':
            for metric in metrics: 
                print(f'Test {metric.__class__.__name}: {(metric.compute()):.4f}')
                metric.reset()

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
    
    test(model, test_loader, criterion, args.task)

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
        default="best_temp_binary_attention_model.pt",
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
        default=16,
        help="Batch size",
    )

    args = parser.parse_args()
    main(args)
