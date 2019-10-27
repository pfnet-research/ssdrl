import chainer
from chainer import Variable
import chainer.functions as F
import numpy as np
import copy
from losses import loss_fun
from transport_costs import cost_fun
from attacks import wrm_attack


def get_batch(iterator, xp):
    batch = iterator.next()
    batchsize = len(batch)
    x = []
    y = []
    for j in range(batchsize):
        _x = batch[j][0]
        _y = batch[j][1]
        if isinstance(_x, (list, tuple)):
            for k in range(len(_x)):
                x.append(np.asarray(_x[k]).astype("f"))
                y.append(np.asarray(_y[k]).astype(np.int32))
        else:
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
    x = xp.asarray(x)
    y = xp.asarray(y, dtype=xp.int32)
    return Variable(x), Variable(y)


def validation_loss_and_acc(cls, iterator, loss_type=None, n=50000):
    @chainer.training.make_extension()
    def _evaluate(trainer):
        iterator.reset()
        losses = []
        accs = []
        for i in range(0, n, iterator.batch_size):
            x, y = get_batch(iterator, cls.xp)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                logit = cls(x)
                loss = loss_fun(logit, y, loss_type)
                acc = F.accuracy(logit, y)
                losses.append(chainer.cuda.to_cpu(loss.array))
                accs.append(chainer.cuda.to_cpu(acc.array))
        chainer.reporter.report({
            'val_loss': np.mean(np.asarray(losses)),
            'val_acc': np.mean(np.asarray(accs))
        })

    return _evaluate


def adversarial_validation_loss_and_acc(cls, iterator, steps=5, gamma=1.0, alpha=1.0,
                                        loss_type=None, c_type=None,
                                        clip_x=False, n=50000):
    @chainer.training.make_extension()
    def _evaluate(trainer):
        iterator.reset()
        phis = []
        losses = []
        accs = []
        phis_0 = []
        phis_1 = []
        phis_2 = []
        with chainer.using_config('train', False):
            for _ in range(0, n, iterator.batch_size):
                x, y = get_batch(iterator, cls.xp)
                x_adv, _phis = wrm_attack(cls=cls, x=x, y=y, gamma=gamma, steps=steps,
                                   loss_type=loss_type, c_type=c_type, alpha=alpha, clip_x=clip_x, return_phis=True)
                logit = cls(x_adv)
                loss = loss_fun(logit, y, loss_type, reduce='mean')
                cost = cost_fun(x1=x_adv, y1=y, x2=x, y2=y, type=c_type, reduce='mean')
                phi = loss - gamma * cost
                acc = F.accuracy(logit, y)
                phis_0.append(chainer.cuda.to_cpu(_phis[0]))
                phis_1.append(chainer.cuda.to_cpu(_phis[1]))
                phis_2.append(chainer.cuda.to_cpu(_phis[2]))
                phis.append(chainer.cuda.to_cpu(phi.array))
                losses.append(chainer.cuda.to_cpu(loss.array))
                accs.append(chainer.cuda.to_cpu(acc.array))

        chainer.reporter.report({
            'val_phi': np.mean(np.asarray(phis)),
            'val_phi_0': np.mean(np.asarray(phis_0)),
            'val_phi_1': np.mean(np.asarray(phis_1)),
            'val_phi_2': np.mean(np.asarray(phis_2)),
            'adv_val_loss': np.mean(np.asarray(losses)),
            'adv_val_acc': np.mean(np.asarray(accs))
        })

    return _evaluate
