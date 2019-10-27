import os, sys, time
import shutil
import yaml

import argparse
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))

import source.yaml_utils as yaml_utils
from extentions import validation_loss_and_acc, adversarial_validation_loss_and_acc
import json
import functools
import warnings
from chainer import functions as F
from losses import loss_fun
from transport_costs import cost_fun
from attacks import wrm_attack
from extentions import get_batch


def create_result_dir(result_dir, config_path, config):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    open(os.path.join(result_dir, os.path.basename(config_path)), 'w').write(
        yaml.dump(config, default_flow_style=False))
    copy_to_result_dir(
        config['models']['classifier']['fn'], result_dir)
    copy_to_result_dir(
        config['dataset']['dataset_fn'], result_dir)
    copy_to_result_dir(
        config['updater']['fn'], result_dir)


def load_models(config):
    cls_conf = config['models']['classifier']
    cls = yaml_utils.load_model(cls_conf['fn'], cls_conf['name'], cls_conf['args'])
    return cls


def load_dataset_eval(config):
    dataset = yaml_utils.load_module(config.dataset_eval['dataset_fn'],
                                     config.dataset_eval['dataset_name'])
    return dataset(**config.dataset_eval['args'])


def main(config, args):
    out = args.results_dir
    create_result_dir(out, args.config_path, config)
    config = yaml_utils.Config(config)
    if args.gpu > -1:
        chainer.cuda.get_device_from_id(args.gpu).use()
    cls = load_models(config)
    chainer.serializers.load_npz(args.snapshot, cls)
    if args.gpu > -1:
        cls.to_gpu()
    dataset = load_dataset_eval(config)
    # Iterator
    iterator = chainer.iterators.SerialIterator(dataset, config.batchsize, shuffle=False, repeat=False)

    phis = []
    losses = []
    accs = []
    losses_adv = []
    accs_adv = []
    phis_0 = []
    phis_1 = []
    phis_2 = []
    results = {}
    for gamma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        with chainer.using_config('train', False):
            iterator.reset()
            xp = cls.xp
            for _ in range(0, len(iterator.dataset), iterator.batch_size):
                print(_)
                x, y = get_batch(iterator, cls.xp)
                x_adv, _phis = wrm_attack(cls=cls, x=x, y=y, gamma=gamma, steps=config.steps,
                                          loss_type=config.loss_type, c_type=config.c_type, alpha=config.alpha,
                                          clip_x=config.clip_x, return_phis=True)
                logit = cls(x_adv)
                loss = loss_fun(logit, y, config.loss_type, reduce='mean')
                cost = cost_fun(x1=x_adv, y1=y, x2=x, y2=y, type=config.c_type, reduce='mean')
                phi = loss - gamma * cost
                acc = F.accuracy(logit, y)
                phis_0.append(chainer.cuda.to_cpu(_phis[0]))
                phis_1.append(chainer.cuda.to_cpu(_phis[1]))
                phis_2.append(chainer.cuda.to_cpu(_phis[2]))
                phis.append(chainer.cuda.to_cpu(phi.array))
                losses_adv.append(chainer.cuda.to_cpu(loss.array))
                accs_adv.append(chainer.cuda.to_cpu(acc.array))

                logit = cls(x)
                loss = loss_fun(logit, y, config.loss_type, reduce='mean')
                acc = F.accuracy(logit, y)
                losses.append(chainer.cuda.to_cpu(loss.array))
                accs.append(chainer.cuda.to_cpu(acc.array))
        results[gamma] = {
            'phi_0': str(np.mean(phis_0)),
            'phi_1': str(np.mean(phis_1)),
            'phi_2': str(np.mean(phis_2)),
            'phi': str(np.mean(phis)),
            'val_loss': str(np.mean(losses)),
            'val_acc': str(np.mean(accs)),
            'adv_val_loss': str(np.mean(losses_adv)),
            'adv_val_acc': str(np.mean(accs_adv)),
        }
    import json
    r = json.dumps(results)
    loaded_r = json.loads(r)
    print(loaded_r)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        f.write(r)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-w', '--warning', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config_path))

    for attr in args.attrs:
        module, new_value = attr.split('=')
        keys = module.split('.')
        target = functools.reduce(dict.__getitem__, keys[:-1], config)
        target[keys[-1]] = yaml.load(new_value)
    print(config)
    return config, args


if __name__ == '__main__':
    config, args = parse_args()
    if not args.warning:
        # Ignore warnings
        warnings.simplefilter('ignore')
    main(config, args)
