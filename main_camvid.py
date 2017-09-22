# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:51:02 2017

@author: sakurai
"""
import contextlib
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import chainer
import chainer.functions as F
from chainer import cuda, optimizers
from chainer.iterators import SerialIterator
from chainer.dataset import concat_examples
from chainer.datasets import TransformDataset, TupleDataset
from chainercv.links import SegNetBasic
from chainercv.datasets import CamVidDataset
from chainercv.evaluations import (
    eval_semantic_segmentation, calc_semantic_segmentation_iou)

from dilated_convnet import DilatedConvNet
from full_resolution_resnet import FullResolutionResNet
from segnet import SegNet
from enet import ENet
from enet_pre_activation import ENetPreActivation


def as_tuple_dataset(dataset_class, device=-1, **kwargs):
    arrays = concat_examples(dataset_class(**kwargs), device)
    xp = cuda.get_array_module(*arrays)
    arrays = [xp.ascontiguousarray(a) for a in arrays]
    return TupleDataset(*arrays)


def random_flip_transform(in_data):
    img, label = in_data
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        label = label[:, ::-1]
    return img, label


def no_transform(in_data):
    return in_data


@contextlib.contextmanager
def inference_mode():
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            yield


def evaluate(net, dataset, batch_size, class_weight):
    xp = net.xp
    device = int(cuda.get_device_from_array(
        next(next(net.children()).params()).data))

    n_class = len(class_weight)
    losses = []
    confusion = xp.zeros((n_class, n_class), np.int64)
    with inference_mode():
        for batch in tqdm(SerialIterator(dataset, batch_size, False, False),
                          total=len(dataset)/batch_size):
            x, t = concat_examples(batch, device)
            y = net(x)
            losses.append(F.softmax_cross_entropy(
                y, t, class_weight=class_weight, ignore_label=-1).data)
            y = y.data.argmax(1)
            mask = t >= 0  # ground truth label -1 is ignored
            confusion += xp.bincount(
                n_class * t[mask] + y[mask],
                minlength=n_class**2).reshape((n_class, n_class))
            del y

    loss_avg = cuda.to_cpu(xp.stack(losses).mean())
    loss_avg = np.asscalar(loss_avg)
    confusion = cuda.to_cpu(confusion)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)
    scores = {'iou': iou, 'miou': np.nanmean(iou),
              'pixel_accuracy': pixel_accuracy,
              'class_accuracy': class_accuracy,
              'mean_class_accuracy': np.nanmean(class_accuracy)}
    return loss_avg, scores


def calc_weight(dataset_train, num_classes):
    n_cls_pixels = np.zeros((num_classes,))
    n_img_pixels = np.zeros((num_classes,))

    for img, label in dataset_train:
        for cls_i in np.unique(label):
            if cls_i == -1:
                continue
            n_cls_pixels[cls_i] += np.sum(label == cls_i)
            n_img_pixels[cls_i] += label.size
    freq = n_cls_pixels / n_img_pixels
    median_freq = np.median(freq)

    class_weight = median_freq / freq
    return class_weight


if __name__ == '__main__':
    p = SimpleNamespace()
    p.num_classes = 11
    p.device = 0
    p.shuffle = True
    p.num_epochs = 350
    p.batch_size = 12
    p.learning_rate = 1e-3
    p.weight_decay = 1e-50
    p.eval_interval = 5

    ds_train = CamVidDataset(split='train')
    class_weight = calc_weight(ds_train, p.num_classes)
    ds_train = TransformDataset(as_tuple_dataset(CamVidDataset, split='train'),
                                no_transform)
    ds_val = as_tuple_dataset(CamVidDataset, split='val')
    ds_test = as_tuple_dataset(CamVidDataset, split='test')

    xp = np if p.device < 0 else cuda.cupy
    class_weight = xp.asarray(class_weight, np.float32)
#    net = SegNet(p.num_classes).to_gpu()
#    net = FullResolutionResNet(p.num_classes).to_gpu()
#    net = DilatedConvNet(p.num_classes).to_gpu()
#    net = ENet(p.num_classes).to_gpu()
    net = ENetPreActivation(p.num_classes).to_gpu()

#    optimizer = optimizers.MomentumSGD(p.learning_rate)
    optimizer = optimizers.Adam(p.learning_rate)
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=p.weight_decay))

    epochs = []
    train_losses = []
    val_losses = []
    test_losses = []
    train_paccs = []  # pacc = pixel accurecy
    val_paccs = []
    test_paccs = []
    train_mcaccs = []  # mcacc = mean class accuracy
    val_mcaccs = []
    test_mcaccs = []
    train_mious = []  # miou = mean intersection-over-union
    val_mious = []
    test_mious = []

    time_origin = time.time()
    for epoch in range(p.num_epochs):
        time_begin = time.time()
        it_train = SerialIterator(ds_train, p.batch_size, False, p.shuffle)
        for batch in tqdm(it_train):
            x, t = concat_examples(batch, p.device)

            y = net(x)
            loss = F.softmax_cross_entropy(y, t, class_weight=class_weight,
                                           ignore_label=-1)
            net.cleargrads()
            loss.backward()
            optimizer.update()
            del loss
            del y
            net.cleargrads()

        if epoch % p.eval_interval == p.eval_interval - 1:
            epochs.append(epoch)
            loss, scores = evaluate(net, ds_train, p.batch_size, class_weight)
            train_losses.append(loss)
            train_paccs.append(scores['pixel_accuracy'])
            train_mcaccs.append(scores['mean_class_accuracy'])
            train_mious.append(scores['miou'])
            loss, scores = evaluate(net, ds_val, p.batch_size, class_weight)
            val_losses.append(loss)
            val_paccs.append(scores['pixel_accuracy'])
            val_mcaccs.append(scores['mean_class_accuracy'])
            val_mious.append(scores['miou'])
            loss, scores = evaluate(net, ds_test, p.batch_size, class_weight)
            test_losses.append(loss)
            test_paccs.append(scores['pixel_accuracy'])
            test_mcaccs.append(scores['mean_class_accuracy'])
            test_mious.append(scores['miou'])

        time_end = time.time()
        epoch_time = time_end - time_begin
        total_time = time_end - time_origin
        print('# {}, time: {:0.2f} (total {:0.1f}) [s]'.format(
            epoch, epoch_time, total_time))

        if epoch % p.eval_interval == p.eval_interval - 1:
            # Plots
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_losses, label='train_loss')
            plt.plot(epochs, val_losses, label='val_loss')
            plt.plot(epochs, test_losses, label='test_loss')
            plt.title('Loss')
            plt.grid()
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_paccs, label='train_pacc')
            plt.plot(epochs, val_paccs, label='val_pacc')
            plt.plot(epochs, test_paccs, label='test_pacc')
            plt.title('Pixel accuracy')
            plt.ylim(0.6, 1.0)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_mious, label='train_miou')
            plt.plot(epochs, val_mious, label='val_miou')
            plt.plot(epochs, test_mious, label='test_miou')
            plt.title('Average of class-wise IoUs')
            plt.ylim(0.4, 1.0)
            plt.grid()
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_mcaccs, label='train_mcacc')
            plt.plot(epochs, val_mcaccs, label='val_mcacc')
            plt.plot(epochs, test_mcaccs, label='test_mcacc')
            plt.title('Average of class-wise pixel accuracies')
            plt.ylim(0.6, 1.0)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Print
            print('\t\tLoss\tPAcc\tmIoU\tmCPAcc')
            print('[train]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(
                train_losses[-1], train_paccs[-1],
                train_mious[-1], train_mcaccs[-1]))
            print('[valid]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(
                val_losses[-1], val_paccs[-1],
                val_mious[-1], val_mcaccs[-1]))
            print('[test]\t\t{:1.3}\t{:1.3}\t{:1.3}\t{:1.3}'.format(
                test_losses[-1], test_paccs[-1],
                test_mious[-1], test_mcaccs[-1]))
            print(p)
            print()
