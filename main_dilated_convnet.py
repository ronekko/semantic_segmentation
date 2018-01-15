# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:38:00 2018

@author: sakurai
"""

from types import SimpleNamespace

import chainer

import common
from dilated_convnet import DilatedConvNet


if __name__ == '__main__':
    hp = SimpleNamespace()

    # Parameters for network
    hp.num_classes = 11

    # Parameters for optimization
    hp.num_epochs = 350
    hp.batch_size = 10
    hp.lr_init = 1e-3
    hp.optimizer = chainer.optimizers.Adam
    hp.weight_decay = 5e-4
    hp.lr_decrease_rate = 1.0
    hp.epochs_decrease_lr = []
    hp.shuffle = True

    # Parameters for experiment
    hp.device = 0
    hp.eval_interval = 5

    model = DilatedConvNet(hp.num_classes)
    result = common.train_eval(model, hp)
    best_model, best_epoch, best_train, best_val, best_test = result[:5]
    train_log, val_log, test_log = result[5:]

    # Print
    print()
    print('Best scores (at epoch {}):'.format(best_epoch))
    print('\t\tLoss\tPAcc\tmIoU\tmCPAcc')
    print('[train]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(*best_train))
    print('[valid]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(*best_val))
    print('[test]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(*best_test))
    print(hp)
    print()
