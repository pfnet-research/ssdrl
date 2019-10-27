import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
from source.batch_normalization import BatchNormalization as MyBN


def _downsample(x, op='ave'):
    if op == 'ave':
        # Downsample (Mean Avg Pooling with 2x2 kernel
        return F.average_pooling_2d(x, 2)
    elif op == 'max':
        # Downsample (Mean Avg Pooling with 2x2 kernel
        return F.max_pooling_2d(x, 2)
    else:
        raise NotImplementedError


class SVHNClassifier(chainer.Chain):
    def __init__(self, ch=128, n_classes=10, activation='relu', bn=True):
        w = chainer.initializers.GlorotUniform()
        self.n_classes = n_classes
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'lrelu':
            self.act = lambda x: F.leaky_relu(x, 0.1)
        elif activation == 'elu':  # ELU option because the assumption in DRL requires smoothness
            self.act = F.elu
        else:
            raise NotImplementedError

        super(SVHNClassifier, self).__init__()
        self.bn = bn
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c0_2 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch, ch * 2, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch * 2, ch * 2, 3, 1, 1, initialW=w)
            self.c1_2 = L.Convolution2D(ch * 2, ch * 2, 3, 1, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch * 2, ch * 4, 3, 1, 0, initialW=w)
            self.c2_1 = L.Convolution2D(ch * 4, ch * 2, 1, 1, 0, initialW=w)
            self.c2_2 = L.Convolution2D(ch * 2, ch, 1, 1, 0, initialW=w)
            self.l3 = L.Linear(ch, n_classes, initialW=w)

            self.b0_0 = MyBN(ch) if self.bn else lambda x: x
            self.b0_1 = MyBN(ch) if self.bn else lambda x: x
            self.b0_2 = MyBN(ch) if self.bn else lambda x: x
            self.b1_0 = MyBN(ch * 2) if self.bn else lambda x: x
            self.b1_1 = MyBN(ch * 2) if self.bn else lambda x: x
            self.b1_2 = MyBN(ch * 2) if self.bn else lambda x: x
            self.b2_0 = MyBN(ch * 4) if self.bn else lambda x: x
            self.b2_1 = MyBN(ch * 2) if self.bn else lambda x: x
            self.b2_2 = MyBN(ch) if self.bn else lambda x: x
            self.b3 = MyBN(n_classes) if self.bn else lambda x: x

    def __call__(self, x, y=None, return_feature=False, **kwargs):
        h = self.act(self.b0_0(self.c0_0(x)))
        h = self.act(self.b0_1(self.c0_1(h)))
        h = self.act(self.b0_2(self.c0_2(h)))
        h = _downsample(h, 'max')
        h = F.dropout(h, 0.5)
        h = self.act(self.b1_0(self.c1_0(h)))
        h = self.act(self.b1_1(self.c1_1(h)))
        h = self.act(self.b1_2(self.c1_2(h)))
        h = _downsample(h, 'max')
        h = F.dropout(h, 0.5)
        h = self.act(self.b2_0(self.c2_0(h)))
        h = self.act(self.b2_1(self.c2_1(h)))
        h = self.act(self.b2_2(self.c2_2(h)))
        h = F.mean(h, axis=(2, 3))
        h = self.b3(self.l3(h))
        return h
