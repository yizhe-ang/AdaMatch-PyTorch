# AdaMatch-PyTorch
Unofficial PyTorch Implementation of [AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732), using [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

## Results

### Evaluation Results
Unsupervised Domain Adaptation (UDA) experiments are run on the DigitFive dataset. An experiment is run for each source-target domain pair (totaling up to 20 experiments).

Test results are directly taken from the final checkpoint at the end of training.

Comparison of this implementation's results to Table 127 in the paper (**this implementation** / original paper):

|          | mnist           | mnistm          | svhn            | syndigit        | usps            |     | Avg             |
| -------- | --------------- | --------------- | --------------- | --------------- | --------------- | --- | --------------- |
| mnist    | -               | **97.9** / 99.2 | **82.1** / 96.9 | **99.2** / 99.7 | **96.7** / 97.8 |     | **94.0** / 98.4 |
| mnistm   | **99.1** / 99.4 | -               | - / 96.9        | - / 99.7        | **98.1** / 97.8 |     | - / 98.5        |
| svhn     | - / 99.3        | - / 98.9        | -               | - / 99.6        | **86.0** / 90.4 |     | - / 97.0        |
| syndigit | - / 99.4        | - / 99.0        | - / 97.0        | -               | **94.4** / 95.8 |     | - / 97.8        |
| usps     | **99.0** / 99.3 | - / 98.9        | - / 96.6        | - / 94.9        | -               |     | - / 97.4        |
|          |                 |                 |                 |                 |                 |     |                 |
| Avg      | - / 99.4        | - / 99.0        | - / 96.8        | - / 98.5        | **93.8** / 95.5 |     | - / 97.8        |
### Differences from Paper
- Training hyperparameters
  - Each experiment is only trained for 1000 epochs, while the paper trains for much longer.
- Augmentations
  - For strong augmentations, this implementation uses RandAugment + CutOut, while the paper uses CTAugment.

## Installation
Follow the instructions to install a Dassl.pytorch conda environment [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

Download the Digit-5 dataset as per the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#digit-5).

## Usage
The main AdaMatch training logic is implemented in the `forward_backward` method in [`trainers/adamatch.py`](trainers/adamatch.py)

The main entry point for running an experiment is `train.py`, after specifying the configuration files `dataset-config-file`, `config-file` for hyperparameters:
```
python train.py \
  --root data_dir \
  --seed 42 \
  --trainer AdaMatch \
  --source-domains mnist \
  --target-domains svhn \
  --dataset-config-file configs/datasets/digit5.yaml \
  --config-file configs/trainers/digit5.yaml \
  --output-dir output_dir/mnist_svhn_42
```

TensorBoard logs are stored in the specified output directory `output-dir`.

To reproduce the UDA DigitFive results above, you can run the `run_digit5_da.sh` script, specifying the data and output directories:
```
cd scripts
bash run_digit5_da.sh data_dir output_dir
```
