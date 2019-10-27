# Collection of transports costs c(z,z')
import chainer
import numpy as np
import chainer.functions as F


def cost_fun(x1, y1, x2, y2, type='square', reduce='mean'):
    if type == 'square':
        return square(x1, x2, reduce)
    elif type == 'L2':
        return L2(x1, x2, reduce)
    elif type == 'L1':
        return L1(x1, x2, reduce)
    else:
        raise NotImplementedError


def square(x1, x2, reduce='mean'):
    axis = tuple([int(i) for i in np.arange(1, len(x1.shape))])
    ret = 0.5 * F.sum(F.square(x1 - x2), axis)
    if reduce == 'mean':
        return F.mean(ret)
    elif reduce == 'sum':
        return F.sum(ret)
    elif reduce == 'no':
        return ret
    else:
        raise NotImplementedError


def L2(x1, x2, reduce='mean'):
    axis = tuple([int(i) for i in np.arange(1, len(x1.shape))])
    ret = F.sqrt(F.sum(F.square(x1 - x2), axis) + 1e-6)
    if reduce == 'mean':
        return F.mean(ret)
    elif reduce == 'sum':
        return F.sum(ret)
    elif reduce == 'no':
        return ret
    else:
        raise NotImplementedError


def L1(x1, x2, reduce='mean'):
    axis = tuple([int(i) for i in np.arange(1, len(x1.shape))])
    ret = F.sum(F.absolute_error(x1, x2), axis)
    if reduce == 'mean':
        return F.mean(ret)
    elif reduce == 'sum':
        return F.sum(ret)
    elif reduce == 'no':
        return ret
    else:
        raise NotImplementedError
