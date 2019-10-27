import os

import numpy as np
from PIL import Image
import chainer
from chainer import cuda
from chainer.dataset import dataset_mixin
import scipy.misc


class SimpleImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, data, size, resize_method, dequantize, division=True):
        self.data = data
        self.size = size
        self.resize_method = resize_method
        self.dequantize = dequantize
        self.division = division

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        image = self.data[i][0]
        label = self.data[i][1]

        image = np.asarray(image, np.uint8)
        if self.size != image.shape[1] or self.size != image.shape[2]:
            image = scipy.misc.imresize(image.transpose(1, 2, 0),
                                        [self.size, self.size], self.resize_method).transpose(2, 0, 1)
        image = np.array(image / 128. - 1., np.float32) if self.division else image
        if self.dequantize and self.division:
            image += np.random.uniform(size=image.shape, low=0., high=1. / 128).astype(np.float32)

        return (image, label)
