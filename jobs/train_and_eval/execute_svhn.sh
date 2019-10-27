for seed in 2018 2019 2020; do
    alpha=0.5
    for gamma in 0.2 0.5 1.0 2.0 5.0 10.0 20.0; do
        for steps in 2 3 5; do
            # DRL
            python3 -u train_ss_mn.py --config=configs/svhnss_drl.yml \
            --attr gamma=${gamma} updater.args.gamma=${gamma} dataset.args.seed=${seed} alpha=${alpha} updater.args.alpha=${alpha} updater.args.steps=${steps}

            # SSDRL with max_classes=1
            python3 -u train_ss_mn.py --config=configs/svhnss_ssdrl_fast.yml \
            --attr gamma=${gamma} updater.args.gamma=${gamma} updater.args.lamb=-1 dataset.args.seed=${seed} \
            updater.args.max_classes=1 alpha=${alpha} updater.args.alpha=${alpha} updater.args.steps=${steps}
        done
    done

    # pseudo labeling
    python3 -u train_ss_mn.py --config=configs/svhnss_ssdrl_fast.yml \
    --attr gamma=1 updater.args.gamma=1 updater.args.lamb=-1 dataset.args.seed=${seed} \
    updater.args.max_classes=1 updater.args.steps=1 updater.args.alpha=0

    for eps in 0.5 1.0 2.0 5.0 10.0 20.0 50.0; do
        # VAT
        python3 -u train_ss_mn.py --config=configs/svhnss_vat.yml \
        --attr gamma=1 updater.args.gamma=1 updater.args.eps=${eps} dataset.args.seed=${seed}
    done
done
