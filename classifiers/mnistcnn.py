import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np


class MNISTClassifier(chainer.Chain):
    def __init__(self, ch=64, n_classes=10, activation='relu', bn=False):
        w = chainer.initializers.GlorotUniform()
        self.n_classes = n_classes
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'elu':  # ELU option because the assumption in DRL requires smoothness
            self.act = F.elu
        else:
            raise NotImplementedError

        super(MNISTClassifier, self).__init__()
        self.bn = bn
        with self.init_scope():
            self.c0 = L.Convolution2D(1, ch, 4, 2, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(ch * 2, ch * 4, 4, 2, 1, initialW=w)
            self.l3 = L.Linear(ch * 4, n_classes, initialW=w)

            self.b0 = L.BatchNormalization(ch) if self.bn else lambda x: x
            self.b1 = L.BatchNormalization(ch * 2) if self.bn else lambda x: x
            self.b2 = L.BatchNormalization(ch * 4) if self.bn else lambda x: x

    def __call__(self, x, y=None, return_feature=False, **kwargs):
        h = self.act(self.b0(self.c0(x)))
        h = self.act(self.b1(self.c1(h)))
        h = self.act(self.b2(self.c2(h)))
        h = F.mean(h, axis=(2, 3))
        h = self.l3(h)
        return h
