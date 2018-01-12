# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:13:49 2017

@author: sakurai

A variant with pre-activation units of
"ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation",
https://arxiv.org/abs/1606.02147
"""

import chainer
import chainer.functions as F
import chainer.links as L


class ENetPreActivation(chainer.Chain):
    def __init__(self, n_class):
        ch_init, ch1, ch2, ch3, ch4, ch5 = 16, 128, 256, 256, 128, 32
        super(ENetPreActivation, self).__init__(
            initial=InitialBlock(ch_init),
            stage1=Stage1(ch_init, ch1),
            stage2=Stage2(ch1, ch2),
            stage3=Stage3(ch2, ch3),
            stage4=Stage4(ch3, ch4),
            stage5=Stage5(ch4, ch5),
            bn=L.BatchNormalization(ch5),
            prelu=L.PReLU(),
            fullconv=UpConv(ch5, n_class, 1))

    def __call__(self, x):
        h = self.initial(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.stage5(h)
        h = self.bn(h)
        h = self.prelu(h)
        return self.fullconv(h)


class InitialBlock(chainer.Chain):
    def __init__(self, ch_out=16):
        ch_conv = ch_out - 3
        super(InitialBlock, self).__init__(
            bn=L.BatchNormalization(3),
            conv=L.Convolution2D(3, ch_conv, 3, stride=2, pad=1, nobias=True))

    def __call__(self, x):
        x = self.bn(x)
        return F.concat((F.max_pooling_2d(x, 2), self.conv(x)))


class Stage1(chainer.ChainList):
    def __init__(self, ch_in=16, ch_out=64):
        super(Stage1, self).__init__()
        self.add_link(BottleneckModule(ch_in, ch_out, downsample=True))
        for i in range(4):
            self.add_link(BottleneckModule(ch_out, ch_out))

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class Stage2(chainer.ChainList):
    def __init__(self, ch_in=64, ch_out=128):
        super(Stage2, self).__init__()
        self.add_link(BottleneckModule(ch_in, ch_out, downsample=True))  # 2.0
        self.add_link(BottleneckModule(ch_out, ch_out))                   # 2.1
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=2))         # 2.2
        self.add_link(BottleneckModule(ch_out, ch_out, asymmetric=True))  # 2.3
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=4))         # 2.4
        self.add_link(BottleneckModule(ch_out, ch_out))                   # 2.5
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=8))         # 2.6
        self.add_link(BottleneckModule(ch_out, ch_out, asymmetric=True))  # 2.7
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=16))        # 2.8

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class Stage3(chainer.ChainList):
    def __init__(self, ch_in=128, ch_out=128):
        super(Stage3, self).__init__()
        # stage 3 has no downsampleing module
        self.add_link(BottleneckModule(ch_in, ch_out))                    # 3.1
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=2))         # 3.2
        self.add_link(BottleneckModule(ch_out, ch_out, asymmetric=True))  # 3.3
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=4))         # 3.4
        self.add_link(BottleneckModule(ch_out, ch_out))                   # 3.5
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=8))         # 3.6
        self.add_link(BottleneckModule(ch_out, ch_out, asymmetric=True))  # 3.7
        self.add_link(BottleneckModule(ch_out, ch_out, dilate=16))        # 3.8

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class Stage4(chainer.ChainList):
    def __init__(self, ch_in=128, ch_out=64):
        super(Stage4, self).__init__()
        self.add_link(BottleneckModule(ch_in, ch_out, upsample=True))  # 4.0
        self.add_link(BottleneckModule(ch_out, ch_out))                # 4.1
        self.add_link(BottleneckModule(ch_out, ch_out))                # 4.2

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class Stage5(chainer.ChainList):
    def __init__(self, ch_in=64, ch_out=16):
        super(Stage5, self).__init__()
        self.add_link(BottleneckModule(ch_in, ch_out, upsample=True))  # 5.0
        self.add_link(BottleneckModule(ch_out, ch_out))                # 5.1

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class BottleneckModule(chainer.Chain):
    def __init__(self, in_ch, out_ch, dilate=1,
                 downsample=False, upsample=False, asymmetric=False):
        assert not (downsample and upsample)

        bottleneck_ch = out_ch // 4
        if downsample:
            bprc1 = BPreluC(in_ch, bottleneck_ch, 2, stride=2)
        elif upsample:
            bprc1 = BPreluCUp(in_ch, bottleneck_ch, 1, upscale=2)
        else:
            bprc1 = BPreluC(in_ch, bottleneck_ch, 1)
        ksize2 = 5 if asymmetric else 3
        pad2 = 2 if asymmetric else 1
        super(BottleneckModule, self).__init__(
            bprc1=bprc1,
            bprc2=BPreluC(bottleneck_ch, bottleneck_ch, ksize2, pad=pad2,
                          dilate=dilate, asymmetric=asymmetric),
            bprc3=BPreluC(bottleneck_ch, out_ch, 1))

        if upsample:
            self.add_link('upconv', UpConv(in_ch, out_ch, 1, 2))

        self.out_ch = out_ch
        self.downsample = downsample
        self.upsample = upsample

    def __call__(self, x):
        h = self.bprc1(x)
        h = self.bprc2(h)
        h = self.bprc3(h)
        if self.downsample:
            x = F.max_pooling_2d(x, 2)
            x = extend_channels(x, self.out_ch)
        if self.upsample:
            x = self.upconv(x)
        return x + h


class BPreluC(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, PReLU and
    Convolution2D. Note that the default value of nobias is True.
    Note that if `dilated` > 1 then `pad` argument is ignored and the `pad`
    parameter for the DilatedConvolution2D is set to `dilate` .
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 dilate=1, upsample=False, asymmetric=False, nobias=True,
                 initialW=None, initial_bias=None, **kwargs):
        in_ch, out_ch = in_channels, out_channels
        if asymmetric:
            conv = AsymmetricConvolution2D(
                in_ch, out_ch, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                **kwargs)
        elif dilate > 1:
            conv = L.DilatedConvolution2D(
                in_ch, out_ch, ksize=ksize, stride=stride, pad=dilate,
                dilate=dilate, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias)
        else:
            conv = L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                   pad=pad, nobias=nobias, initialW=initialW,
                                   initial_bias=initial_bias, **kwargs)
        super(BPreluC, self).__init__(
            bn=L.BatchNormalization(in_ch),
            prelu=L.PReLU(),
            conv=conv)

    def __call__(self, x):
        return self.conv(self.prelu(self.bn(x)))


class BPreluCUp(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, PReLU and
    Convolution2D. Note that the default value of nobias is True.
    Note that if `dilated` > 1 then `pad` argument is ignored and the `pad`
    parameter for the DilatedConvolution2D is set to `dilate` .
    '''
    def __init__(self, in_channels, out_channels, ksize=None, upscale=2,
                 stride=1, pad=0, nobias=True, initialW=None,
                 initial_bias=None, **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(BPreluCUp, self).__init__(
            bn=L.BatchNormalization(in_ch),
            prelu=L.PReLU(),
            upconv=UpConv(
                in_ch, out_ch, ksize=ksize, upscale=upscale, stride=stride,
                pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias, **kwargs))

    def __call__(self, x):
        return self.upconv(self.prelu(self.bn(x)))


class ConvBn(chainer.Chain):
    '''
    Note that the default value of nobias for conv is True.
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=True, initialW=None, initial_bias=None, **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(ConvBn, self).__init__(
            conv=L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias, **kwargs),
            bn=L.BatchNormalization(out_ch))

    def __call__(self, x):
        return self.bn(self.conv(x))


class UpConv(chainer.Chain):
    '''
    A link to upsample the input by conv + depth2space.
    '''
    def __init__(self, in_channels, out_channels, ksize=None, upscale=2,
                 stride=1, pad=0, nobias=True, initialW=None,
                 initial_bias=None, **kwargs):
        in_ch = in_channels
        out_ch = out_channels * upscale * upscale
        super(UpConv, self).__init__(
            conv=L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias, **kwargs))
        self.upscale = upscale

    def __call__(self, x):
        h = self.conv(x)
        return F.depth2space(h, self.upscale)


class AsymmetricConvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(AsymmetricConvolution2D, self).__init__(
            conv1=L.Convolution2D(
                in_ch, in_ch, ksize=(ksize, 1), stride=stride, pad=(pad, 0),
                nobias=False, initialW=initialW, initial_bias=initial_bias,
                **kwargs),
            conv2=L.Convolution2D(
                in_ch, out_ch, ksize=(1, ksize), stride=stride, pad=(0, pad),
                nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                **kwargs))

    def __call__(self, x):
        return self.conv2(self.conv1(x))


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


if __name__ == '__main__':
    import numpy as np
    from chainer import cuda

    use_gpu = True
    xp = np if not use_gpu else cuda.cupy

    B, C, H, W = 2, 3, 360, 480
    n_class = 11
    x = xp.random.randn(B, C, H, W).astype('f')

    net = ENetPreActivation(n_class).to_gpu()
    y = net(x)
    assert y.shape == (B, n_class, H, W), 'y.shape = {}'.format(y.shape)