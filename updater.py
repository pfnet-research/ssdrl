import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from transport_costs import cost_fun
from losses import loss_fun
import copy
import attacks


class Updater(chainer.training.StandardUpdater):
    def __init__(self, method='wrm', loss_type='cross_entropy', c_type='square', vat_loss_type='kl', gamma=1, lamb=None,
                 steps=5, alpha=1., clip_x=True, eps=1.0, use_true_label=True, *args, **kwargs):
        self.classifier = kwargs.pop('classifier')
        self.method = method
        self.loss_type = loss_type  # Loss
        self.c_type = c_type  # Transport cost
        self.vat_loss_type = vat_loss_type  # Loss for vat loss
        self.gamma = gamma  # coeff between loss and transport cost
        self.lamb = lamb  # Entorpy coeff
        self.steps = steps  # internal optimization steps
        self.alpha = alpha  # internal optimization learning rate hyperparam
        self.steps = steps  # internal optimization steps
        self.eps = eps  # norm of adversarial perturbation
        self.use_true_label = use_true_label
        self.clip_x = clip_x  # if True, apply [-1,1] clipping after adding adversarial perturbation
        self.max_classes = kwargs.pop('max_classes') if 'max_classes' in kwargs else None
        super(Updater, self).__init__(*args, **kwargs)

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x = Variable(xp.asarray(x))
        y = Variable(xp.asarray(y, dtype=xp.int32))
        return x, y

    def get_batch_unlabeled(self, xp):
        batch = self.get_iterator('unlabeled').next()
        batchsize = len(batch)
        x = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
        x = Variable(xp.asarray(x))
        return x

    def update_core(self):
        xp = self.classifier.xp
        optimizer = self.get_optimizer('main')
        x, y = self.get_batch(xp)
        # The vanilla training
        if self.method == 'mle':
            logit = self.classifier(x)
            loss_l = loss_fun(logit, y, self.loss_type)
            chainer.reporter.report({'loss_l': loss_l})
            loss_total = loss_l
        # Distributionally robust learning
        elif self.method == 'drl':
            _ = self.classifier(x)  # Update batch stats
            with chainer.using_config('user_update_bn_stats', False):
                loss_l = attacks.wrm_loss(cls=self.classifier, x=x, y=y, gamma=self.gamma, steps=self.steps,
                                          loss_type=self.loss_type, c_type=self.c_type, alpha=self.alpha,
                                          clip_x=self.clip_x, use_true_label=self.use_true_label)
            chainer.reporter.report({'loss_l': loss_l})
            loss_total = loss_l
        # Semi-supervised distributionally robust learning
        elif self.method == 'ssdrl':
            _ = self.classifier(x)  # Update batch stats
            with chainer.using_config('user_update_bn_stats', False):
                loss_l = attacks.wrm_loss(cls=self.classifier, x=x, y=y, gamma=self.gamma, steps=self.steps,
                                          loss_type=self.loss_type, c_type=self.c_type, alpha=self.alpha,
                                          clip_x=self.clip_x,
                                          use_true_label=self.use_true_label)
                chainer.reporter.report({'loss_l': loss_l})
                x_ul = self.get_batch_unlabeled(xp=xp)
                loss_ul = attacks.sswrm_loss(cls=self.classifier, x=x_ul, steps=self.steps,
                                             loss_type=self.loss_type, c_type=self.c_type,
                                             gamma=self.gamma, alpha=self.alpha, lamb=self.lamb,
                                             clip_x=self.clip_x, max_classes=self.max_classes)
            chainer.reporter.report({'loss_ul': loss_ul})
            loss_total = loss_l + loss_ul
        # Adversarial training
        elif self.method == 'at':
            logit = self.classifier(x)
            with chainer.using_config('user_update_bn_stats', False):
                loss_l = loss_fun(logit, y, self.loss_type)
                chainer.reporter.report({'loss_l': loss_l})
                loss_adv = attacks.adversarial_loss(cls=self.classifier, x=x, y=y, steps=self.steps,
                                                    loss_type=self.loss_type,
                                                    eps=self.eps, clip_x=self.clip_x,
                                                    use_true_label=self.use_true_label)
            chainer.reporter.report({'loss_adv': loss_adv})
            loss_total = loss_l + loss_adv
        # Virtual adversarial training
        elif self.method == 'vat':
            logit = self.classifier(x)
            with chainer.using_config('user_update_bn_stats', False):
                loss_l = loss_fun(logit, y, self.loss_type)
                chainer.reporter.report({'loss_l': loss_l})
                loss_adv = attacks.adversarial_loss(cls=self.classifier, x=x, y=y, steps=self.steps,
                                                    loss_type=self.loss_type,
                                                    eps=self.eps, clip_x=self.clip_x,
                                                    use_true_label=self.use_true_label)
                chainer.reporter.report({'loss_adv': loss_adv})
                x_ul = self.get_batch_unlabeled(xp=xp)
                loss_vadv = attacks.virtual_adversarial_loss(cls=self.classifier, x=x_ul, steps=self.steps,
                                                             loss_type=self.vat_loss_type,
                                                             eps=self.eps, clip_x=self.clip_x)
            chainer.reporter.report({'loss_vadv': loss_vadv})
            loss_total = loss_l + loss_vadv
        else:
            raise NotImplementedError
        chainer.reporter.report({'loss_total': loss_total})
        self.classifier.cleargrads()
        loss_total.backward()
        optimizer.update()
