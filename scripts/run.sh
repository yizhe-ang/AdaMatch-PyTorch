#!/bin/bash

cd ..

DATA_DIR=$1
OUTPUT_DIR=$2

SEED=1

SOURCE=mnist
TARGET=svhn

python train_wandb.py \
    --root ${DATA_DIR} \
    --seed ${SEED} \
    --trainer AdaMatch \
    --source-domains ${SOURCE} \
    --target-domains ${TARGET} \
    --dataset-config-file configs/datasets/digit5.yaml \
    --config-file configs/trainers/digit5.yaml \
    --output-dir ${OUTPUT_DIR}/${SOURCE}_${TARGET}_noflip_${SEED} \
    --wandb