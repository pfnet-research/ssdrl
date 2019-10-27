# SSDRL: Semi-Supervised Distributionally Robust Learning
Reproducing code for the paper: \
Robustness to Adversarial Perturbations in Learning from Incomplete Data\
Amir Najafi, Shin-ichi Maeda, Masanori Koyama, and Takeru Miyato\
Thirty-third Conference on Neural Information Processing Systems, (2019).\
(arXiv version: https://arxiv.org/abs/1905.13021 \
 poter PDF: https://drive.google.com/file/d/1HDG4KWYiZtCZwWLV13OtbCk0XSW9GwJv/view?usp=sharing)

## Requirements:
- python 3.5, CUDA, cuDNN
- Install python libraries listed in `requirements.txt` (or run `pip install -r requirements.txt`)

## The core implementation of SSDRL:
The proposed loss is implemented as function `sswrm_loss`, which starts from line 122 of `attacks.py`.
Please see this file if you want to take a closer look at the implementation.

## The scripts used in our experiments:
List of bash scripts for the training:
- on MNIST: `jobs/train_eval/execute_mnist.sh`
- on SVHN: `jobs/train_eval/execute_svhn.sh`
- on CIFAR10: `jobs/train_eval/execute_cifar10.sh` 

List of bash scripts for the adversarial evaluations:
- WRM attack(https://arxiv.org/abs/1710.10571): `jobs/train_eval/evaluation_drl.sh`
- PGM attack(https://arxiv.org/abs/1706.06083): `jobs/train_eval/evaluation_pgm_l2.sh`

Below are example commands for each training and evaluation.
### Training 
#### with SSDRL
- on MNIST
```
results_dir=./results # path to the directory you want to save the outputs
gamma=0.1 # denoted as γ in Eq.(6,7). This is the dual parameter to epsilion parameter of Wasserstein Ball.
lamb=-1.0 # denoted as λ in Eq.(5). This is the parameter which determines performing optimistic or pesimistic SSDRL.
python3 -u train_ss_mn.py --config=configs/mnistss_ssdrl.yml --results_dir=${results_dir} \
            --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=${lamb} dataset.args.seed=1234 \
            alpha=1.0 updater.args.alpha=1.0
```
#### with Fast-SSDRL
- on MNIST
```
results_dir=./results # path to the directory you want to save the outputs
gamma=0.1 # The best hyper-parameter on clean examples, denoted as γ_1 in the paper
#gamma=0.5 # The best hyper-parameter for WRM attacks, denoted as γ_2 in the paper
#gamma=0.02 # The best hyper-parameter for PGM attacks, denoted as γ_3 in the paper
python3 -u train_ss_mn.py --config=configs/mnistss_ssdrl_fast.yml --results_dir=${results_dir} \
            --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=-1 dataset.args.seed=1234 \
            alpha=1.0 updater.args.alpha=1.0
```

- on SVHN
```
results_dir=./results # path to the directory you want to save the outputs
gamma=20.0 # γ_1
#gamma=20.0 # γ_2
#gamma=10.0 # γ_3
python3 -u train_ss_mn.py --config=configs/svhnss_ssdrl_fast.yml --results_dir=${results_dir} \
            --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=-1 dataset.args.seed=1234 \
            alpha=1.0 updater.args.alpha=1.0
```

- on CIFAR-10
```
results_dir=./results # path to the directory you want to save the outputs
gamma=5.0 # γ_1
#gamma=1.0 # γ_2
#gamma=0.5 # γ_3
python3 -u train_ss_mn.py --config=configs/cifar10ss_ssdrl_fast.yml --results_dir=${results_dir} \
            --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=-1 dataset.args.seed=1234 \
            alpha=1.0 updater.args.alpha=1.0
```



### Evaluation with WRM on MNIST
```
python3 -u jobs/train_and_eval/calc_adv_loss_drl.py --gpu=0 --config=configs/mnistss_drl.yml \
           --snapshot=${results_dir}/MNISTClassifier_50000.npz \
           --attr batchsize=100 alpha=1.0 steps=15
```
 
