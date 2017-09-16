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

from full_resolution_resnet import FullResolutionResNet


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
    losses = []
    accs = []
    with inference_mode():
        for batch in tqdm(SerialIterator(dataset, batch_size, False, False),
                          total=len(dataset)/batch_size):
            x, t = concat_examples(batch, device)
            y = net(x)
            losses.append(F.softmax_cross_entropy(
                y, t, class_weight=class_weight, ignore_label=-1).data)
            accs.append(F.accuracy(y, t, ignore_label=-1).data)
    loss_avg = cuda.to_cpu(xp.stack(losses).mean())
    acc_avg = cuda.to_cpu(xp.stack(accs).mean())
    return loss_avg, acc_avg


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
    p.batch_size = 6
    p.learning_rate = 1e-1
    p.weight_decay = 1e-5
    p.eval_interval = 5

    ds_train = CamVidDataset(split='train')
    class_weight = calc_weight(ds_train, p.num_classes)
    ds_train = TransformDataset(as_tuple_dataset(CamVidDataset, split='train'),
                                no_transform)
    ds_val = as_tuple_dataset(CamVidDataset, split='val')
    ds_test = as_tuple_dataset(CamVidDataset, split='test')

    xp = np if p.device < 0 else cuda.cupy
    class_weight = xp.asarray(class_weight, np.float32)
#    net = SegNetBasic(p.num_classes).to_gpu()
    net = FullResolutionResNet(p.num_classes).to_gpu()
    optimizer = optimizers.MomentumSGD(p.learning_rate)
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=p.weight_decay))

    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
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
            loss, acc = evaluate(net, ds_train, p.batch_size, class_weight)
            train_losses.append(loss)
            train_accs.append(acc)
            loss, acc = evaluate(net, ds_val, p.batch_size, class_weight)
            val_losses.append(loss)
            val_accs.append(acc)
            loss, acc = evaluate(net, ds_test, p.batch_size, class_weight)
            test_losses.append(loss)
            test_accs.append(acc)

        time_end = time.time()
        epoch_time = time_end - time_begin
        total_time = time_end - time_origin
        print("# {}, time: {} ({})".format(epoch, epoch_time, total_time))

        if epoch % p.eval_interval == p.eval_interval - 1:
            # Plots
            plt.plot(epochs, train_losses, label='train_loss')
            plt.plot(epochs, val_losses, label='val_loss')
            plt.plot(epochs, test_losses, label='test_loss')
            plt.title('Loss')
            plt.grid()
            plt.legend()
            plt.show()

            plt.plot(epochs, train_accs, label='train_acc')
            plt.plot(epochs, val_accs, label='val_acc')
            plt.plot(epochs, test_accs, label='test_acc')
            plt.title('Accuracy')
            plt.ylim(0.6, 1.0)
            plt.grid()
            plt.legend()
            plt.show()

            # Print
            print('[Train] loss:', train_losses[-1])
            print('[Train]  acc:', train_accs[-1])
            print('[Valid] loss:', val_losses[-1])
            print('[Valid]  acc:', val_accs[-1])
            print('[Test]  loss:', test_losses[-1])
            print('[Test]   acc:', test_accs[-1])
            print(p)
            print()

            # TODO: Use chainercv.evaluations.eval_semantic_segmentation