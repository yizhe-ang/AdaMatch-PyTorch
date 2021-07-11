#!/bin/bash

cd ..

python train.py \
    --root data \
    --seed 1 \
    --trainer AdaMatch \
    --source-domains mnist \
    --target-domains svhn \
    --dataset-config-file configs/datasets/digit5.yaml \
    --config-file configs/trainers/digit5.yaml \
    --output-dir output/1
