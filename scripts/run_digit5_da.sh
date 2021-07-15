#!/bin/bash

cd ..

DATA_DIR=$1
OUTPUT_DIR=$2

SEED=1

SOURCES=(mnist mnist_m svhn syn usps)
TARGETS=(mnist mnist_m svhn syn usps)

for SOURCE in ${SOURCES[@]}; do
    for TARGET in ${TARGETS[@]}; do

        if [ ${SOURCE} != ${TARGET} ]
        then
            python train.py \
                --root ${DATA_DIR} \
                --seed ${SEED} \
                --trainer AdaMatch \
                --source-domains ${SOURCE} \
                --target-domains ${TARGET} \
                --dataset-config-file configs/datasets/digit5.yaml \
                --config-file configs/trainers/digit5.yaml \
                --output-dir ${OUTPUT_DIR}/${SOURCE}_${TARGET}_${SEED}
        fi

    done
done
