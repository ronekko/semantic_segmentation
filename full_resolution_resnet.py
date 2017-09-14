# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:22:42 2017

@author: sakurai

"Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes"
https://arxiv.org/abs/1611.08323
"""

import matplotlib.pyplot as plt
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class FullResolutionResNet(chainer.Chain):
    def __init__(self, num_classes):
#        chs = [48, 96, 192, 384, 384, 192, 192, 96, 48]
        chs = [24, 48, 96, 192, 192, 96, 96, 48, 24]
        z_ch = 16
        super(FullResolutionResNet, self).__init__(
                cbr_in=CBR(3, chs[0], 5, pad=2),
                ru_seq_in=RUSequence(chs[0], [chs[0]] * 3),
                conv_in=L.Convolution2D(chs[0], z_ch, 1),
                frru_seq_1=FRRUSequence(chs[0], z_ch, [chs[1]] * 3, 1),
                frru_seq_2=FRRUSequence(chs[1], z_ch, [chs[2]] * 4, 2),
                frru_seq_3=FRRUSequence(chs[2], z_ch, [chs[3]] * 2, 3),
                frru_seq_4=FRRUSequence(chs[3], z_ch, [chs[4]] * 2, 4),
                frru_seq_5=FRRUSequence(chs[4], z_ch, [chs[5]] * 2, 3),
                frru_seq_6=FRRUSequence(chs[5], z_ch, [chs[6]] * 2, 2),
                frru_seq_7=FRRUSequence(chs[6], z_ch, [chs[7]] * 2, 1),
                conv_out=L.Convolution2D(chs[7] + z_ch, chs[8], 1),
                ru_seq_out=RUSequence(chs[8], [chs[8]] * 3),
                conv_class=L.Convolution2D(chs[8], num_classes, 1)
            )

    def __call__(self, x):
        h = self.cbr_in(x)
        h = self.ru_seq_in(h)

        ysize0 = h.shape[2:]
        z = self.conv_in(h)
        y = F.max_pooling_2d(h, 2)
        y, z = self.frru_seq_1(y, z)
        ysize1 = y.shape[2:]
        y = F.max_pooling_2d(y, 2)
        y, z = self.frru_seq_2(y, z)
        ysize2 = y.shape[2:]
        y = F.max_pooling_2d(y, 2)
        y, z = self.frru_seq_3(y, z)
        ysize3 = y.shape[2:]
        y = F.max_pooling_2d(y, 2)
        y, z = self.frru_seq_4(y, z)

        y = F.unpooling_2d(y, 2, outsize=ysize3)
        y, z = self.frru_seq_5(y, z)
        y = F.unpooling_2d(y, 2, outsize=ysize2)
        y, z = self.frru_seq_6(y, z)
        y = F.unpooling_2d(y, 2, outsize=ysize1)
        y, z = self.frru_seq_7(y, z)
        y = F.unpooling_2d(y, 2, outsize=ysize0)

        h = F.concat((y, z))
        h = self.conv_out(h)
        h = self.ru_seq_out(h)
        return self.conv_class(h)


class RUSequence(chainer.ChainList):
    '''
    Sequence of `ResidualUnit`s.
    '''
    def __init__(self, in_channels, out_channels=(48, 48, 48)):
        in_ch = in_channels
        units = []
        for out_ch in out_channels:
            units.append(RU(in_ch, out_ch))
            in_ch = out_ch
        super(RUSequence, self).__init__(*units)

    def __call__(self, x):
        for unit in self:
            x = unit(x)
        return x


class FRRUSequence(chainer.ChainList):
    '''
    Sequence of `FRRU`s.
    '''
    def __init__(self, y_in_ch, z_in_ch, y_out_ch, pooling_level):
        in_ch = y_in_ch
        units = []
        for out_ch in y_out_ch:
            units.append(FRRU(in_ch, z_in_ch, out_ch, pooling_level))
            in_ch = out_ch
        super(FRRUSequence, self).__init__(*units)

    def __call__(self, y, z):
        for unit in self:
            y, z = unit(y, z)
        return y, z


class RU(chainer.Chain):
    '''
    Residual unit.
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = in_channels
        super(RU, self).__init__(
            cbr_1=CBR(in_channels, mid_channels, 3, pad=1),
            cbr_2=CBR(mid_channels, out_channels, 3, pad=1))

    def __call__(self, x):
        return x + self.cbr_2(self.cbr_1(x))


class FRRU(chainer.Chain):
    def __init__(self, y_in_ch, z_in_ch, y_out_ch, pooling_level):
        self.y_in_ch = y_in_ch
        self.z_in_ch = z_in_ch
        super(FRRU, self).__init__(
            cbr_1=CBR(y_in_ch + z_in_ch, y_out_ch, 3, pad=1),
            cbr_2=CBR(y_out_ch, y_out_ch, 3, pad=1),
            conv=L.Convolution2D(y_out_ch, z_in_ch, 1))
        self.level = pooling_level

    def __call__(self, y, z):
        pooling_size = 2 ** self.level
        hz = F.max_pooling_2d(z, pooling_size)
        hy = F.concat((y, hz))
        y = self.cbr_2(self.cbr_1(hy))
        hz = self.conv(y)
        z = z + F.unpooling_2d(hz, pooling_size, outsize=z.shape[2:])
        return y, z


class CBR(chainer.Chain):
    '''
    Sequence of Conv-Batchnorm-ReLU.
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=True, initialW=None, initial_bias=None, **kwargs):
        super(CBR, self).__init__(
            conv=L.Convolution2D(
                in_channels, out_channels, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                **kwargs),
            bn=L.BatchNormalization(out_channels))

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))


if __name__ == '__main__':
    use_gpu = True
    shape = 6, 3, 360, 480
    B, C, H, W = shape

    xp = np if not use_gpu else chainer.cuda.cupy
    x = xp.random.randn(*shape).astype('f')
    t = xp.random.randint(-1, 12, (B, H, W)).astype(xp.int32)

    net = FullResolutionResNet()
    if use_gpu:
        net.to_gpu()
    y = net(x)
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
