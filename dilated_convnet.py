# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:05:34 2017

@author: sakurai

"Multi-Scale Context Aggregation by Dilated Convolutions"
https://arxiv.org/abs/1511.07122v3
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

from full_resolution_resnet import CBR


class DcBR(chainer.Chain):
    '''
    Sequence of DilatedConv-Batchnorm-ReLU.
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 dilate=1, nobias=True, initialW=None, initial_bias=None):
        super(DcBR, self).__init__(
            dilated_conv=L.DilatedConvolution2D(
                in_channels, out_channels, ksize=ksize, stride=stride, pad=pad,
                dilate=dilate, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias),
            bn=L.BatchNormalization(out_channels))

    def __call__(self, x):
        return F.relu(self.bn(self.dilated_conv(x)))


class DilatedConvNet(chainer.Chain):
    def __init__(self, n_class=None):

        c1, c2, c3, c4, c5, c6, c7 = 30, 30, 40, 40, 40, 50, 50
        super(DilatedConvNet, self).__init__()
        with self.init_scope():
            self.conv1 = CBR(None, c1, 3, pad=1)
            self.conv2 = CBR(c1, c2, 3, pad=1)
            self.dilated_conv3 = DcBR(c2, c3, 3, pad=2, dilate=2)
            self.dilated_conv4 = DcBR(c3, c4, 3, pad=4, dilate=4)
            self.dilated_conv5 = DcBR(c4, c5, 3, pad=8, dilate=8)
            self.dilated_conv6 = DcBR(c5, c6, 3, pad=16, dilate=16)
            self.conv7 = CBR(c6, c7, 3, pad=1)
            self.conv_out = L.Convolution2D(c7, n_class, 1)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.dilated_conv3(h))
        h = F.relu(self.dilated_conv4(h))
        h = F.relu(self.dilated_conv5(h))
        h = F.relu(self.dilated_conv6(h))
        h = F.relu(self.conv7(h))
        return self.conv_out(h)


if __name__ == '__main__':
    pass