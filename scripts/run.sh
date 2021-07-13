#!/bin/bash

cd ..

python train.py \
    --root data \
    --seed 1 \
    --trainer AdaMatch \
    --source-domains usps \
    --target-domains mnist \
    --dataset-config-file configs/datasets/digit5.yaml \
    --config-file configs/trainers/digit5.yaml \
    --output-dir output/1 \
    --wandb
