import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def plot_target_vs_preds(metrics, split):
    target_array = np.array(metrics["Target"])
    target_array = np.reshape(target_array, (target_array.shape[0], target_array.shape[1]))
    target_array = np.reshape(target_array, (-1))

    preds_array = np.array(metrics["Preds"])
    preds_array = np.reshape(preds_array, (preds_array.shape[0], preds_array.shape[1]))
    preds_array = np.reshape(preds_array, (-1))

    print(target_array.shape, preds_array.shape)

    x_axis_len = target_array.shape[0]
    x_axis = range(1, x_axis_len+1)

    print(x_axis_len)
    print(x_axis)

    plt.figure(figsize=(40,6))
    plt.plot(x_axis, target_array, 'o', label='Target', color='red')
    plt.plot(x_axis, preds_array, 'o', label='Predictions', color='blue')

    plt.title('Target vs Predictions')
    plt.xlabel('Samples')
    plt.ylabel('Values')

    plt.xticks(np.arange(0, x_axis_len, 20), rotation=90)
    
    plt.legend(loc='best')
    plt.savefig('target_vs_preds_' + split + '.png')

def plot_train_metrics(train_metrics, val_metrics, task): 
    if task == 'regression':
        # Loss
        x_axis_len = len(train_metrics['Loss'])
        x_axis = range(1, x_axis_len+1)

        plt.figure(figsize=(20,10))
        plt.plot(x_axis, train_metrics['Loss'], label='Train Loss', color='red')
        plt.plot(x_axis, val_metrics['Loss'], label='Validation Loss', color='blue')

        plt.title('Training and Validation Loss', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)

        plt.xticks(np.arange(0, x_axis_len, 20), fontsize=14)
        
        plt.legend(loc='best', fontsize=18)
        plt.savefig('Loss' + '.png')
        
        #Target-Preds
        #train
        plot_target_vs_preds(train_metrics, split='train')
        #val
        plot_target_vs_preds(val_metrics, split='val')
    elif task == 'binary': 
        for key in train_metrics.keys():
            plt.figure(figsize=(20,10))
            plt.plot(x_axis, train_metrics[key], 'o',label='Train ' + key, color='red')
            plt.plot(x_axis, val_metrics[key], 'o',label='Validation ' + key, color='blue')
        
            plt.title('Training and Validation ' + key, fontsize=18)
            plt.xlabel('Epochs', fontsize=18)
            plt.ylabel(key, fontsize=18)
        
            plt.xticks(np.arange(0, x_axis_len, 20), fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc='best', fontsize=18)
            plt.savefig(key + '.png')



def main(args):

    if args.split == 'train':
        train_metrics = pkl.load(open(args.train_path, 'rb'))
        val_metrics = pkl.load(open(args.val_path, 'rb'))

        plot_train_metrics(train_metrics, val_metrics, args.task)
    else:
        test_metrics = pkl.load(open(args.test_path, 'rb'))
        plot_target_vs_preds(test_metrics, split=args.split)

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
    parser.add_argument(
        "--task",
        metavar="KEY",
        type=str,
        action="store",
        dest="task",
        default="regression",
        help="binary or regression",
    )

    args = parser.parse_args()
    main(args)
