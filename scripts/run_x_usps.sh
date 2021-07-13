#!/bin/bash

cd ..

SEED=1

SOURCES=(mnist mnist_m svhn syn)
TARGET=usps

for SOURCE in ${SOURCES[@]}; do
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
done
