for seed in 2018 2019 2020; do
    alpha=1.0
    for gamma in 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0; do
        # DRL
        python3 -u train_ss_mn.py --config=configs/mnistss_drl.yml \
        --attr gamma=${gamma} updater.args.gamma=${gamma} dataset.args.seed=${seed} alpha=${alpha} updater.args.alpha=${alpha}
        # SSDRL
        for lamb in -10.0 -1.0 0. 1.0; do
            python3 -u train_ss_mn.py --config=configs/mnistss_ssdrl.yml \
            --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=${lamb} dataset.args.seed=${seed} \
            alpha=${alpha} updater.args.alpha=${alpha}
        done
        # SSDRL with max_classes=1
        python3 -u train_ss_mn.py --config=configs/mnistss_ssdrl_fast.yml \
        --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=-1 dataset.args.seed=${seed} \
        updater.args.max_classes=1 alpha=${alpha} updater.args.alpha=${alpha}
    done
    # pseudo labeling
    python3 -u train_ss_mn.py --config=configs/mnistss_ssdrl_fast.yml \
    --attr gamma=1 updater.args.gamma=1 updater.args.lamb=-1 dataset.args.seed=${seed} \
    updater.args.max_classes=1 updater.args.steps=1 updater.args.alpha=0

    for eps in 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0; do
        # VAT
        python3 -u train_ss_mn.py --config=configs/mnistss_vat.yml \
        --attr gamma=1 updater.args.gamma=1 updater.args.eps=${eps} dataset.args.seed=${seed}
    done
done

