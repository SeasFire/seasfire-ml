import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def plot_train_metrics(train_metrics, val_metrics, task): 
    x_axis_len = len(train_metrics['Loss'])
    x_axis = range(1, x_axis_len+1)

    if task == 'regression':
        plt.figure()
        plt.plot(x_axis, train_metrics['Loss'], label='Train Loss', color='red')
        plt.plot(x_axis, val_metrics['Loss'], label='Validation Loss', color='blue')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.xticks(np.arange(0, x_axis_len, 2))
        
        plt.legend(loc='best')
        plt.savefig('Loss' + '.png')
    elif task == 'binary': 
        for key in train_metrics.keys():
            plt.figure()
            plt.plot(x_axis, train_metrics[key], label='Train ' + key, color='red')
            plt.plot(x_axis, val_metrics[key], label='Validation ' + key, color='blue')
        
            plt.title('Training and Validation ' + key)
            plt.xlabel('Epochs')
            plt.ylabel(key)
        
            plt.xticks(np.arange(0, x_axis_len, 2))
            
            plt.legend(loc='best')
            plt.savefig(key + '.png')

def plot_test_metrics(test_metrics):
    # x_axis_len = test_metrics['Target'][0].shape[0]
    # x_axis = range(1, x_axis_len+1)

    # y_axis_target = test_metrics['Target'][0].reshape(-1)
    # y_axis_preds = test_metrics['Preds'][0].reshape(-1)

    # plt.figure(figsize=(40,6))
    # plt.plot(x_axis[:100], y_axis_target[:100], 'o', label='Test', color='red')
    # plt.plot(x_axis[:100], y_axis_preds[:100], 'o', label='Validation Loss', color='blue')

    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Labels')

    # plt.xticks(np.arange(0, x_axis_len, 2))
    
    # plt.legend(loc='best')
    # plt.savefig('Test_Loss' + '.png')

def main(args):

    if args.split == 'train':
        train_metrics = pkl.load(open(args.train_path, 'rb'))
        val_metrics = pkl.load(open(args.val_path, 'rb'))

        plot_train_metrics(train_metrics, val_metrics, args.task)
    # else:
    #     test_metrics = pkl.load(open(args.test_path, 'rb'))
    #     plot_test_metrics(test_metrics)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument(
        "-t",
        "--train-pkl-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="train_path",
        default="train_metrics.pkl",
        help="Train pkl file path",
    )
    parser.add_argument(
        "-v",
        "--val-pkl-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="val_path",
        default="val_metrics.pkl",
        help="Val pkl file path",
    )
    parser.add_argument(
        "--test-pkl-path",
        metavar="PATH",
        type=str,
        action="store",
        dest="test_path",
        default="test_metrics.pkl",
        help="Test pkl file path",
    )
    parser.add_argument(
        "--split",
        metavar="KEY",
        type=str,
        action="store",
        dest="split",
        default="test",
        help="Split: train, test",
    )

    args = parser.parse_args()
    main(args)
