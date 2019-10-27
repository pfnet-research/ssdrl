import chainer
from chainer import Variable
import chainer.functions as F
import numpy as np
import copy
from losses import loss_fun
from transport_costs import cost_fun
from chainer import cuda


def _normalize(d, xp, norm_type='L2'):
    if norm_type == 'L2':
        return d / (xp.sqrt(xp.sum(d ** 2, axis=tuple(range(1, len(d.shape))), keepdims=True)) + 1e-6)
    elif norm_type == 'max':
        return xp.sign(d)


def _projection(x, x_t, eps, xp, norm_type='L2'):
    x_diff = x_t - x
    return x + eps * _normalize(x_diff, xp, norm_type=norm_type)


def wrm_loss(cls, x, y=None, gamma=1., steps=5.0, loss_type='cross_entropy', c_type='sqaure', alpha=1.0, clip_x=True,
             use_true_label=True):
    _y = y if use_true_label else None
    x_adv = wrm_attack(cls=cls, x=x, y=_y, steps=steps, loss_type=loss_type, c_type=c_type,
                       gamma=gamma, alpha=alpha, clip_x=clip_x)
    logit = cls(x_adv)
    return loss_fun(logit, y, loss_type)


def wrm_attack(cls, x, y=None, steps=5.0, loss_type='cross_entropy', c_type='sqaure',
               gamma=1., alpha=1.0, clip_x=True, return_phis=False):
    xp = cls.xp
    x_org = copy.deepcopy(x)
    _alpha = alpha / gamma
    if return_phis:
        phis = []
    for t in range(steps):
        logit = cls(x)
        if y is None:
            y = F.argmax(logit, axis=1)
        loss = loss_fun(logit, y, loss_type, reduce='sum')
        cost = cost_fun(x1=x, y1=y, x2=x_org, y2=y, type=c_type, reduce='sum')
        phi = loss - gamma * cost
        # print(xp.mean(phi.array), xp.mean(xp.sum((x.array - x_org.array) ** 2, axis=(1, 2, 3))))
        if return_phis:
            phis.append(phi.array)
        grad = chainer.grad([phi], [x])[0]
        lr = _alpha / (t + 1)
        x = x + lr * grad.array
        if clip_x:
            x = F.clip(x, -1., 1.)
    if return_phis:
        return x, phis
    else:
        return x


def virtual_adversarial_loss(cls, x, steps, loss_type='kl', eps=1.0, xi=1e-6, stop_grad=True, clip_x=True):
    logit = cls(x)
    x_adv = virtual_adversarial_attack(cls=cls, x=x, steps=steps,
                                       loss_type=loss_type, eps=eps, xi=xi, logit=logit, clip_x=clip_x)
    logit_adv = cls(x_adv)
    if stop_grad:
        logit = logit.array
    return loss_fun(logit, logit_adv, type=loss_type)


def virtual_adversarial_attack(cls, x, steps=1, loss_type='kl', eps=2.0, xi=1e-6, logit=None, clip_x=True):
    xp = cuda.get_array_module(x.array)
    if logit is None:
        logit = cls(x)
    x_org = copy.deepcopy(x)
    for t in range(steps):
        # Apply 1 step virtual adversarial attack and multiple projected gradient descent
        d = _normalize(xp.random.normal(size=x.shape), xp)
        x_d = x + xi * d
        logit_d = cls(x_d)
        kl_loss = loss_fun(logit, logit_d, type=loss_type)
        grad = chainer.grad([kl_loss], [x_d])[0]
        d = _normalize(grad.array, xp)
        x = x + eps * d
        x = Variable(_projection(x_org.array, x.array, eps, xp))
        if clip_x:
            x = F.clip(x, -1., 1.)
    return x


def adversarial_loss(cls, x, y, steps=1, loss_type='cross_entropy', eps=2.0, clip_x=True, use_true_label=True,
                     norm_type='L2'):
    _y = y if use_true_label else None
    x_adv = adversarial_attack(cls=cls, x=x, y=_y, steps=steps, eps=eps,
                               loss_type=loss_type, clip_x=clip_x, norm_type=norm_type)
    logit_adv = cls(x_adv)
    loss = loss_fun(logit_adv, y, type=loss_type)
    return loss


def adversarial_attack(cls, x, y=None, steps=1, loss_type='cross_entropy', eps=2.0, clip_x=True, norm_type='L2',
                       alpha=None):
    # you can prevent from label leaking by setting y to None
    xp = cuda.get_array_module(x.array)
    if alpha is None:
        alpha = eps

    x_org = copy.deepcopy(x)
    for t in range(steps):
        logit = cls(x)
        if y is None:
            y = F.argmax(logit, 1)
        loss = loss_fun(logit, y, type=loss_type)
        grad = chainer.grad([loss], [x])[0]
        d = _normalize(grad.array, xp, norm_type=norm_type)
        x = x + alpha * d
        x = Variable(_projection(x_org.array, x.array, eps, xp, norm_type=norm_type))
        if clip_x:
            x = F.clip(x, -1., 1.)
    return x


def sswrm_loss(cls, x, steps, loss_type, c_type, gamma, alpha, lamb, clip_x=True, max_classes=None):
    """
    Calculation of R_{SSAR} in Eq.(10).
    :param cls: classifier model which takes input x and output logit of the probability
    :param x: input
    :param steps: the number of SGD updates for the evaluation of 'max z' in Eq.(9) in the paper
    :param loss_type: fitting loss function, default is cross entropy function
    :param c_type: transport cost function, denoted as c in the paper.
    :param gamma: denoted as γ in Eq.(6,7). This is the dual parameter to epsilion parameter of Wasserstein Ball.
    :param alpha: learning rate of the SGD for the evaluation of 'max z' in Eq.(9)
    :param lamb: denoted as λ in Eq.(5). This is the parameter which determines performing optimistic or pesimistic SSDRL.
    :param clip_x: clip x into the range of [-1, 1].
    :param max_classes: the number of classes used for the evaluation of R_{SSAR} in Eq.(10). This is set to 1 when the fast-SSDRL algorithm is performed.
    :return: R_{SSAR} value in Eq.(10).
    """
    xp = cls.xp
    _alpha = alpha / gamma
    if max_classes is None:
        max_classes = cls.n_classes
    logit = cls(x)
    _y = xp.argsort(logit.array, 1)  # Note: xp.argsort sorts an array with ascending order by default
    if lamb < 0:  # optimisitic
        _y = _y[:, ::-1]
    x_org = x
    xs = []
    for k in range(max_classes):
        x = copy.deepcopy(x_org)
        y = _y[:, k]
        for t in range(steps):
            logit = cls(x)
            loss = loss_fun(logit, y, loss_type, reduce='sum')
            cost = cost_fun(x1=x, y1=y, x2=x_org, y2=y, type=c_type, reduce='sum')
            phi = loss - gamma * cost
            grad = chainer.grad([phi], [x])[0]
            lr = _alpha / (t + 1)
            x += lr * grad.array
            if clip_x:
                x = F.clip(x, -1., 1.)
        xs.append(x.array)
    losses = []
    weights = []
    for k in range(max_classes):
        x = xs[k]
        y = _y[:, k]
        logit = cls(x)
        loss = loss_fun(logit, y, loss_type, reduce='no')
        losses.append(loss)
        cost = cost_fun(x1=x, y1=y, x2=x_org, y2=y, type=c_type, reduce='no')
        phi = loss - gamma * cost
        weights.append(lamb * phi.array)
    weights = xp.stack(weights).transpose()
    weights = F.exp(F.log_softmax(weights)).array
    losses = F.transpose(F.stack(losses))
    loss_ul = F.mean(F.sum(weights * losses, 1))
    return loss_ul
