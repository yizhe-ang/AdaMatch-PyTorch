#!/bin/bash

cd ..

SOURCE=svhn
TARGET=usps
SEED=1

python train.py \
    --root data \
    --seed ${SEED} \
    --trainer AdaMatch \
    --source-domains ${SOURCE} \
    --target-domains ${TARGET} \
    --dataset-config-file configs/datasets/digit5.yaml \
    --config-file configs/trainers/digit5.yaml \
    --output-dir output/${SOURCE}_${TARGET}_${SEED} \
    --wandb
