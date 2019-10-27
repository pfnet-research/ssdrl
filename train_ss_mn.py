import os, sys, time
import shutil
import yaml

import argparse
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))

import source.yaml_utils as yaml_utils
import multiprocessing
from extentions import validation_loss_and_acc, adversarial_validation_loss_and_acc
import json
import functools
import warnings


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


def make_optimizer(model, alpha=0.001, beta1=0.9, beta2=0.999):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def main(config, args):
    out = args.results_dir
    create_result_dir(out, args.config_path, config)
    config = yaml_utils.Config(config)
    chainer.cuda.get_device_from_id(0).use()
    print("init")

    classifier = load_models(config)
    classifier.to_gpu()

    # Optimizer
    opt = make_optimizer(
        classifier, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    dataset = yaml_utils.load_dataset(config)
    dataset_l = dataset.dataset_l
    dataset_ul = dataset.dataset_ul
    # Iterator
    multiprocessing.set_start_method('forkserver')
    iterator_l = chainer.iterators.SerialIterator(dataset_l, config.batchsize)
    iterator_ul = chainer.iterators.SerialIterator(dataset_ul, config.batchsize_ul)
    iterators = {'main': iterator_l, 'unlabeled': iterator_ul}
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'classifier': classifier,
        'iterator': iterators,
        'optimizer': opt,

    })
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    report_keys = ["loss_total", "loss_l", "loss_adv", "loss_ul", "loss_ul_separated",
                   "loss_vadv", "val_phi", "val_phi_0", "val_phi_1", "val_phi_2", "val_loss", "adv_val_loss", "acc",
                   "val_acc", "adv_val_acc"]

    # Set up logging
    trainer.extend(extensions.snapshot(), trigger=(config.snapshot_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        classifier, classifier.__class__.__name__ + '_{.updater.iteration}.npz'),
        trigger=(config.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(config.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
    # Eval dataset and iterator
    dataset_eval = load_dataset_eval(config)
    eval_iter = chainer.iterators.SerialIterator(dataset_eval, 100, shuffle=False)
    trainer.extend(validation_loss_and_acc(classifier, eval_iter, loss_type=config.loss_type, n=5000),
                   trigger=(config.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(adversarial_validation_loss_and_acc(classifier, eval_iter,
                                                       steps=config.steps, gamma=config.gamma, alpha=config.alpha,
                                                       loss_type=config.loss_type, c_type=config.c_type,
                                                       clip_x=config.clip_x, n=5000),
                   trigger=(config.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=config.display_interval))

    trainer.extend(extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                          (config.iteration_decay_start, config.iteration), opt))

    if args.snapshot:
        print("Resume training with snapshot:{}".format(args.snapshot))
        chainer.serializers.load_npz(args.snapshot, trainer)

    print("start training")
    trainer.run()


def parse_args():
    parser = argparse.ArgumentParser()
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
