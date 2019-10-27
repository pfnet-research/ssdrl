#!/bin/bash
input=#path to the .txt file which includes filenames of the snapshot
while IFS= read -r var; do
     python3 -u jobs/train_and_eval/calc_adv_loss_drl.py --gpu=0 --config=configs/mnistss_drl.yml --snapshot=mnist_models/${var}/MNISTClassifier_50000.npz --jobid=${var} --attr batchsize=100 alpha=1. steps=15
done < "$input"

input=#path to the .txt file which includes filenames of the snapshot
while IFS= read -r var; do
     python3 -u jobs/train_and_eval/calc_adv_loss_drl.py --gpu=0 --config=configs/cifar10ss_drl.yml --snapshot=cifar10_models/${var}/CIFAR10Classifier_48000.npz --jobid=${var} --attr batchsize=100 alpha=1. steps=15
done < "$input"

input=#path to the .txt file which includes filenames of the snapshot
while IFS= read -r var; do
     python3 -u jobs/train_and_eval/calc_adv_loss_drl.py --gpu=0 --config=configs/svhnss_drl.yml --snapshot=svhn_models/${var}/SVHNClassifier_48000.npz --jobid=${var} --attr batchsize=100 alpha=0.5 steps=15
done < "$input"

