# Collection of loss l(z)
import chainer
import chainer.functions as F
from chainer import cuda


def loss_fun(y, y_true, type='cross_entropy', reduce='mean'):
    if type == 'cross_entropy':
        if y.shape[1] == 1 or len(y.shape) == 1:
            return sigmoid_cross_entropy(y, y_true, reduce=reduce)
        else:
            return softmax_cross_entropy(y, y_true, reduce=reduce)
    elif type == 'kl':
        # both y and y_true should be logit (= the output before softmax activation)
        return kl(y, y_true, reduce=reduce)
    else:
        raise NotImplementedError


def sigmoid_cross_entropy(y, y_true, reduce='mean'):
    if reduce == 'mean' or reduce == 'sum':
        ret = F.sigmoid_cross_entropy(y, y_true, reduce='mean')
        if reduce == 'sum':
            return len(y) * ret
        else:
            return ret
    elif reduce == 'no':
        return F.sigmoid_cross_entropy(y, y_true, reduce='no')
    else:
        raise NotImplementedError


def softmax_cross_entropy(y, y_true, reduce='mean'):
    if reduce == 'mean' or reduce == 'sum':
        ret = F.softmax_cross_entropy(y, y_true, reduce='mean')
        if reduce == 'sum':
            return len(y) * ret
        else:
            return ret
    elif reduce == 'no':
        return F.softmax_cross_entropy(y, y_true, reduce='no')
    else:
        raise NotImplementedError


def kl_binary(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p_logit = F.concat([p_logit, xp.zeros(p_logit.shape, xp.float32)], 1)
    q_logit = F.concat([q_logit, xp.zeros(q_logit.shape, xp.float32)], 1)
    return kl_categorical(p_logit, q_logit)


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return _kl


def kl(p_logit, q_logit, reduce='mean'):
    if p_logit.shape[1] == 1:
        ret = kl_binary(p_logit, q_logit)
    else:
        ret = kl_categorical(p_logit, q_logit)
    if reduce == 'mean':
        return F.mean(ret)
    elif reduce == 'sum':
        return F.sum(ret)
    elif reduce == 'no':
        return ret
    else:
        raise NotImplementedError
