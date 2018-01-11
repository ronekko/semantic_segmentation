from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.transforms import resize
from chainercv.utils import download_model


def _without_cudnn(f, x):
    with chainer.using_config('use_cudnn', 'never'):
        return f.apply((x,))[0]


class SegNet(chainer.Chain):

    """
    chainercv.links.SegNetBasic with different network size.
    """

    _models = {
        'camvid': {
            'n_class': 11,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.2/segnet_camvid_2017_05_28.npz'
        }
    }

    def __init__(self, n_class=None, pretrained_model=None, initialW=None):
        if n_class is None:
            if pretrained_model not in self._models:
                raise ValueError(
                    'The n_class needs to be supplied as an argument.')
            n_class = self._models[pretrained_model]['n_class']

        if initialW is None:
            initialW = chainer.initializers.HeNormal()

        c1, c2, c3 = 64, 128, 256
        d4, d3, d2 = c3, c2, c1
        c4 = 512
        d1 = 64
        ksize = 7

        pad = ksize // 2
        super(SegNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, c1, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv1_bn = L.BatchRenormalization(c1, initial_beta=0.001)
            self.conv2 = L.Convolution2D(
                c1, c2, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv2_bn = L.BatchRenormalization(c2, initial_beta=0.001)
            self.conv3 = L.Convolution2D(
                c2, c3, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv3_bn = L.BatchRenormalization(c3, initial_beta=0.001)
            self.conv4 = L.Convolution2D(
                c3, c4, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv4_bn = L.BatchRenormalization(c4, initial_beta=0.001)
            self.conv_decode4 = L.Convolution2D(
                c4, d4, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv_decode4_bn = L.BatchRenormalization(
                d4, initial_beta=0.001)
            self.conv_decode3 = L.Convolution2D(
                d4, d3, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv_decode3_bn = L.BatchRenormalization(
                d3, initial_beta=0.001)
            self.conv_decode2 = L.Convolution2D(
                d3, d2, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv_decode2_bn = L.BatchRenormalization(
                d2, initial_beta=0.001)
            self.conv_decode1 = L.Convolution2D(
                d2, d1, ksize, 1, pad, nobias=True, initialW=initialW)
            self.conv_decode1_bn = L.BatchRenormalization(
                d1, initial_beta=0.001)
            self.conv_classifier = L.Convolution2D(
                d1, n_class, 1, 1, 0, initialW=initialW)

        self.n_class = n_class

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def _upsampling_2d(self, x, pool):
        if x.shape != pool.indexes.shape:
            min_h = min(x.shape[2], pool.indexes.shape[2])
            min_w = min(x.shape[3], pool.indexes.shape[3])
            x = x[:, :, :min_h, :min_w]
            pool.indexes = pool.indexes[:, :, :min_h, :min_w]
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(
            x, pool.indexes, ksize=(pool.kh, pool.kw),
            stride=(pool.sy, pool.sx), pad=(pool.ph, pool.pw), outsize=outsize)

    def __call__(self, x):
        """Compute an image-wise score from a batch of images

        Args:
            x (chainer.Variable): A variable with 4D image array.

        Returns:
            chainer.Variable:
            An image-wise score. Its channel size is :obj:`self.n_class`.

        """
        p1 = F.MaxPooling2D(2, 2)
        p2 = F.MaxPooling2D(2, 2)
        p3 = F.MaxPooling2D(2, 2)
        p4 = F.MaxPooling2D(2, 2)
        h = F.local_response_normalization(x, 5, 1, 1e-4 / 5., 0.75)
        f1 = self.conv1(x)
        h = _without_cudnn(p1, F.relu(self.conv1_bn(f1)))
        f2 = self.conv2(h)
        h = _without_cudnn(p2, F.relu(self.conv2_bn(f2)))
        f3 = self.conv3(h)
        h = _without_cudnn(p3, F.relu(self.conv3_bn(f3)))
        f4 = self.conv4(h)
        h = _without_cudnn(p4, F.relu(self.conv4_bn(f4)))
        h = self._upsampling_2d(h, p4)
        h = self.conv_decode4_bn(self.conv_decode4(h))
        h = self._upsampling_2d(h, p3)
        h = self.conv_decode3_bn(self.conv_decode3(h))
        h = self._upsampling_2d(h, p2)
        h = self.conv_decode2_bn(self.conv_decode2(h))
        h = self._upsampling_2d(h, p1)
        h = self.conv_decode1_bn(self.conv_decode1(h))
        score = self.conv_classifier(h)
        return score

    def predict(self, imgs):
        """Conduct semantic segmentations from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.

        Returns:
            list of numpy.ndarray:

            List of integer labels predicted from each image in the input \
            list.

        """
        labels = []
        for img in imgs:
            C, H, W = img.shape
            with chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                score = self.__call__(x)[0].data
            score = chainer.cuda.to_cpu(score)
            if score.shape != (C, H, W):
                dtype = score.dtype
                score = resize(score, (H, W)).astype(dtype)

            label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels
