# AdaMatch-PyTorch
Unofficial PyTorch Implementation of [AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732), using [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

With reference to the official implementation at https://github.com/google-research/adamatch.

## Results

### Evaluation Results
Unsupervised Domain Adaptation (UDA) experiments are run on the DigitFive dataset. An experiment is run for each source-target domain pair (totaling up to 20 experiments).

Test results are directly taken from the final checkpoint at the end of training (the paper takes the median over the last ten checkpoints).

Comparison of this implementation's results to Table 127 in the paper (**this implementation** / original paper):

|              | mnist           | mnistm          | svhn            | syndigit        | usps            |     | Avg             |
| ------------ | --------------- | --------------- | --------------- | --------------- | --------------- | --- | --------------- |
| **mnist**    | -               | **98.4** / 99.2 | **91.9** / 96.9 | **99.5** / 99.7 | **99.7** / 97.8 |     | **97.4** / 98.4 |
| **mnistm**   | **99.7** / 99.4 | -               | **93.2** / 96.9 | **99.2** / 99.7 | **99.5** / 97.8 |     | **97.9** / 98.5 |
| **svhn**     | **98.4** / 99.3 | **95.5** / 98.9 | -               | **99.0** / 99.6 | **98.5** / 90.4 |     | **97.9** / 97.0 |
| **syndigit** | **99.7** / 99.4 | **98.1** / 99.0 | **95.6** / 97.0 | -               | **99.2** / 95.8 |     | **98.2** / 97.8 |
| **usps**     | **99.6** / 99.3 | **98.2** / 98.9 | **92.5** / 96.6 | **97.9** / 94.9 | -               |     | **97.1** / 97.4 |
|              |                 |                 |                 |                 |                 |     |                 |
| **Avg**      | **99.4** / 99.4 | **97.6** / 99.0 | **93.3** / 96.8 | **98.9** / 98.5 | **99.2** / 95.5 |     | **97.7** / 97.8 |

Unable to match the results as of yet. If you spot any mistakes in my implementation, **feel free to submit a PR :)**
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
