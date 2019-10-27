import os

import numpy as np
from PIL import Image
import chainer
from chainer import cuda
from chainer.dataset import dataset_mixin
import scipy.misc
from datasets.simple_image_dataset import SimpleImageDataset

ORGSIZE = 28


class MNISTDataset(SimpleImageDataset):
    def __init__(self, size=28, train=True, dequantize=False, resize_method='bilinear'):
        data_train, data_test = chainer.datasets.get_mnist(ndim=3, withlabel=True, scale=255)
        data = data_train if train else data_test
        super(MNISTDataset, self).__init__(data, size=size,
                                           resize_method=resize_method, dequantize=dequantize)


class MNISTSSDataset():
    def __init__(self, size=28, train=True, dequantize=False, resize_method='bilinear', N_l=1000, seed=1234,
                 N_valid=None):
        data, _ = chainer.datasets.get_mnist(ndim=3, withlabel=True, scale=255)
        rng = np.random.RandomState(seed=seed)
        randix = rng.permutation(len(data))
        data_l = list()
        data_ul = list()

        for i in range(N_l):
            x, y = data[randix[i]]
            data_l.append([x, y])
        if N_valid:
            if N_valid > N_l:
                raise ValueError('N_valid should be larger than N_l')
            if train:
                data_l = data_l[:N_valid]
            else:
                data_l = data_l[N_valid:]
        for i in range(N_l, len(data)):
            x, y = data[randix[i]]
            data_ul.append([x, y])

        for i in range(len(data_ul)):
            data_ul[i][1] = - 1  # remove label information from the unlabeled dataset
        self.dataset_l = SimpleImageDataset(data_l, size=size, resize_method=resize_method, dequantize=dequantize)
        self.dataset_ul = SimpleImageDataset(data_ul, size=size, resize_method=resize_method, dequantize=dequantize)
