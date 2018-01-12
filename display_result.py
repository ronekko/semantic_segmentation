# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 20:53:43 2018

@author: sakurai
"""

from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable


if __name__ == '__main__':
    # Load and print the hyper parameters
    print()
    with Path('hyper_params.yaml').open() as f:
        hp = yaml.load(f)
        print('## Hyper parameters')
        for name, value in hp.items():
            print('{}: {}'.format(name, value))

    # Print the best score
    best_scores = np.load('best_scores.npz')
    best_epoch = int(best_scores['best_epoch'])
    best_train = best_scores['best_train']
    best_val = best_scores['best_val']
    best_test = best_scores['best_test']
    print()
    print('## Best scores (at epoch {}):'.format(best_epoch))
    print('\t\tLoss\tPAcc\tmIoU\tmCPAcc')
    print('[train]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(*best_train))
    print('[valid]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(*best_val))
    print('[test]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(*best_test))
    print()

    # Plot the learning curves
    logs = np.load('logs.npz')
    # np.savez(f, train_log=train_log, val_log=val_log, test_log=test_log)
    best_epoch = int(best_scores['best_epoch'])
    best_train = best_scores['best_train']
    best_val = best_scores['best_val']
    best_test = best_scores['best_test']

    eval_interval = hp['eval_interval']
    num_epochs = len(logs.items()[0][1][0])
    epochs = np.arange(0, num_epochs * eval_interval, eval_interval)
    plt.subplot(1, 2, 1)
    for split in ['train', 'val', 'test']:
        key = split + '_log'
        label = split + '_loss'
        plt.plot(epochs, logs[key][0], label=label)
    plt.title('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    for split in ['train', 'val', 'test']:
        key = split + '_log'
        label = split + '_loss'
        plt.plot(epochs, logs[key][1], label=label)
    plt.title('Pixel accuracy')
    plt.ylim(0.6, 1.0)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.subplot(1, 2, 1)
    for split in ['train', 'val', 'test']:
        key = split + '_log'
        label = split + '_loss'
        plt.plot(epochs, logs[key][2], label=label)
    plt.title('Average of class-wise IoUs')
    plt.ylim(0.4, 1.0)
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    for split in ['train', 'val', 'test']:
        key = split + '_log'
        label = split + '_loss'
        plt.plot(epochs, logs[key][3], label=label)
    plt.title('Average of class-wise pixel accuracies')
    plt.ylim(0.6, 1.0)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()