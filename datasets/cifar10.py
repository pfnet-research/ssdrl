import os, sys

import numpy as np
from PIL import Image
import chainer
from chainer import cuda
from chainer.dataset import dataset_mixin
import scipy.misc

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))

from scipy import linalg

ORGSIZE = 32
SCALE = 255


class SimpleImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, data, size, resize_method):
        self.data = data
        self.size = size
        self.resize_method = resize_method

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        image = self.data[i][0]
        label = self.data[i][1]

        if self.size != image.shape[1] or self.size != image.shape[2]:
            image = scipy.misc.imresize(image.transpose(1, 2, 0),
                                        [self.size, self.size], self.resize_method).transpose(2, 0, 1)
        return (image, label)


def _apply_zca_whitening(data, zca_path):
    print("Apply ZCA whitenning")
    stats = np.load(zca_path)
    components, mean = stats['components'], stats['mean']
    X = np.array([x for x, _ in data], dtype=np.float32)
    orgshape = X.shape
    X = np.reshape(X, (len(X), -1))
    X = np.dot(X - mean, components.T).reshape(orgshape)
    ndata = []
    for i in range(len(data)):
        _, y = data[i]
        ndata.append([X[i], y])
    return ndata


class CIFAR10Dataset(SimpleImageDataset):
    def __init__(self, size=32, train=True, resize_method='bilinear',
                 zca_path='./datasets/cifar10_zca/zca.npz'):
        data_train, data_test = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=SCALE)
        data = data_train if train else data_test
        data = _apply_zca_whitening(data, zca_path) if zca_path is not None else data
        super(CIFAR10Dataset, self).__init__(data, size=size, resize_method=resize_method)


class CIFAR10SSDataset():
    def __init__(self, size=32, train=True, resize_method='bilinear', N_l=4000, seed=1234,
                 include_labeled_in_unlabeled=False, zca_path='./datasets/cifar10_zca/zca.npz'):
        data_train, data_test = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=SCALE)
        data = data_train if train else data_test
        rng = np.random.RandomState(seed=seed)
        randix = rng.permutation(len(data))
        data_l = list()
        data_ul = list()
        for i in range(N_l):
            x, y = data[randix[i]]
            data_l.append([x, y])
        if include_labeled_in_unlabeled:
            for i in range(0, len(data)):
                x, y = data[randix[i]]
                data_ul.append([x, y])
        else:
            for i in range(N_l, len(data)):
                x, y = data[randix[i]]
                data_ul.append([x, y])

        for i in range(len(data_ul)):
            data_ul[i][1] = - 1  # remove label information from the unlabeled dataset
        data_l = _apply_zca_whitening(data_l, zca_path) if zca_path is not None else data_l
        data_ul = _apply_zca_whitening(data_ul, zca_path) if zca_path is not None else data_ul
        self.dataset_l = SimpleImageDataset(data_l, size=size, resize_method=resize_method)
        self.dataset_ul = SimpleImageDataset(data_ul, size=size, resize_method=resize_method)


def _ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


if __name__ == '__main__':
    data_train, _ = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=SCALE)
    X = np.asarray([x for x, _ in data_train], dtype=np.float32)
    components, mean, _ = _ZCA(X.reshape(len(X), -1))
    np.savez('zca.npz', components=components, mean=mean)
