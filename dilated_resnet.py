# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:11:21 2018

@author: sakurai

"Dilated Residual Networks",
https://arxiv.org/abs/1705.09914
"""

import chainer
import chainer.functions as F
import chainer.links as L


class DilatedResnetA(chainer.Chain):
    '''
    Reversible Residual Network.

    Args:
        n (int):
            Number of units in each group.
    '''

    def __init__(self, n_classes, n=2, use_bottleneck=False):
        super(DilatedResnetA, self).__init__(
            conv2=L.Convolution2D(3, 64, 7, pad=3, stride=2),
            stage3=ResnetStage(n, 64, 64, 1, 1, use_bottleneck),
            stage4=ResnetStage(n, 64, 128, 2, 1, use_bottleneck),
            stage5=ResnetStage(n, 128, 256, 1, 2, use_bottleneck),
            stage6=ResnetStage(n, 256, 512, 1, 4, use_bottleneck),
            bn_out=L.BatchNormalization(512),
            conv_out=L.Convolution2D(512, n_classes, 1)
        )

    def __call__(self, x):
        B, C, H, W = x.shape
        x = self.conv2(x)  # (n, 3, 360, 480) -> (n, 64, 180, 240)
        x = F.average_pooling_2d(x, 2)  # (n, 64, 180, 240) -> (n, 64, 90, 120)
        x = self.stage3(x)  # (n, 64, 90, 120) -> (n, 64, 90, 120)
        x = self.stage4(x)  # (n, 64, 90, 120) -> (n, 128, 45, 60)
        x = self.stage5(x)  # (n, 128, 45, 60) -> (n, 256, 45, 60)
        x = self.stage6(x)  # (n, 256, 45, 60) -> (n, 512, 45, 60)
        x = self.bn_out(x)
        x = F.relu(x)
        x = self.conv_out(x)
        x = F.resize_images(x, (H, W))
        return x


#class DilatedResnetB(chainer.Chain):
#    '''
#    Reversible Residual Network.
#
#    Args:
#        n (int):
#            Number of units in each group.
#    '''
#
#    def __init__(self, n_classes, use_bottleneck=False):
#        super(DilatedResnet, self).__init__(
#            conv2=L.Convolution2D(3, 64, 7, pad=3, stride=2),
#            stage3=ResnetStage(2,  64, 1, 1, use_bottleneck),
#            stage4=ResnetStage(2, 128, 2, 1, use_bottleneck),
#            stage5=ResnetStage(2, 256, 1, 2, use_bottleneck),
#            stage6=ResnetStage(2, 512, 1, 4, use_bottleneck),
#            bn_out=L.BatchNormalization(512),
#            conv_out=L.Convolution2D(512, n_classes, 1)
#        )
#
#    def __call__(self, x):
#        B, C, H, W = x.shape
#        x = self.conv2(x)
#        x = F.average_pooling_2d(x, 2)
#        x = self.stage3(x)
#        x = self.stage4(x)
#        x = self.stage5(x)
#        x = self.stage6(x)
#        x = self.bn_out(x)
#        x = F.relu(x)
#        x = self.conv_out(x)
#        x = F.resize_images(x, (H, W))
#        return x


class ResnetStage(chainer.ChainList):
    '''Reversible sequence of `ResnetUnit`s.
    '''
    def __init__(self, n_blocks, ch_in, ch_out, stride, dilate,
                 use_bottleneck=True):
        if use_bottleneck:
            block_class = ResnetBottleneckBlock
        else:
            block_class = ResnetBlock
        n_blocks = n_blocks - 1
        blocks = [block_class(ch_in, ch_out, stride, dilate=1,
                              transition=True)]
        blocks += [block_class(ch_out, ch_out, 1, dilate)
                   for i in range(n_blocks)]
        super(ResnetStage, self).__init__(*blocks)

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class ResnetBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, stride, dilate, transition=False):
        pad = 1 + dilate - 1
        super(ResnetBlock, self).__init__(
            brc1=BRCChain(ch_in, ch_out, 3, pad=pad, stride=stride,
                          dilate=dilate),
            brc2=BRCChain(ch_out, ch_out, 3, pad=pad, dilate=dilate))
        self.ch_out = ch_out
        self.stride = stride
        self.transition = transition

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        if self.transition:
            x = avgpool_and_extend_channels(x, self.ch_out, self.stride)
        return x + h


class ResnetBottleneckBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, stride, dilate, transition=False):
        bottleneck = ch_out // 4
        pad = 1 + dilate - 1
        super(ResnetBottleneckBlock, self).__init__(
            brc1=BRCChain(ch_in, bottleneck, 1, pad=0),
            brc2=BRCChain(
                bottleneck, bottleneck, 3, pad=pad, dilate=dilate),
            brc3=BRCChain(bottleneck, ch_out, 1, pad=0))
        self.ch_out = ch_out
        self.stride = stride
        self.transition = transition

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        h = self.brc3(h)
        if self.transition:
            x = avgpool_and_extend_channels(x, self.ch_out, self.stride)
        return x + h


class BRCChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    Convolution2D (a.k.a. pre-activation unit).
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, decay=0.9,
                 **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(BRCChain, self).__init__(
            bn=L.BatchNormalization(in_ch, decay=decay),
            conv=L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias, **kwargs))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = self.conv(h)
        return y


def extend_channels(x, out_ch):
    '''Extends channels (i.e. depth) of the input BCHW tensor x by zero-padding
    if out_ch is larger than the number of channels of x, otherwise returns x.
    '''
    b, in_ch, h, w = x.shape
    if in_ch == out_ch:
        return x
    elif in_ch > out_ch:
        raise ValueError('out_ch must be larger than x.shape[1].')

    xp = chainer.cuda.get_array_module(x)
    filler_shape = (b, out_ch - in_ch, h, w)
    filler = xp.zeros(filler_shape, x.dtype)
    return F.concat((x, filler), axis=1)


def avgpool_and_extend_channels(x, ch_out=None, ksize=2):
    if ksize > 1:
        x = F.average_pooling_2d(x, ksize)
    if ch_out is None:
        ch_in = x.shape[1]
        ch_out = ch_in * 2
    return extend_channels(x, ch_out)


if __name__ == '__main__':
    import numpy as np
    from chainer import cuda

    use_gpu = True
    xp = np if not use_gpu else cuda.cupy

    B, C, H, W = 12, 3, 360, 480
    n_class = 11
    x = xp.random.randn(B, C, H, W).astype('f')

    net = DilatedResnetA(n_class).to_gpu()
    y = net(x)
    assert y.shape == (B, n_class, H, W), 'y.shape = {}\nexpected: {}'.format(
        y.shape, (B, n_class, H, W))
