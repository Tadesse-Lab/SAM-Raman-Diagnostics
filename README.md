# Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics

This repository (currently a work in progress) will contain modules and instructions for replicating and extending experiments featured in our paper.

## Setup Repository and Requirements
Clone repository 
```
git clone https://github.com/Tadesse-Lab/SAM-Raman-Diagnostics.git
```
Install requirements:
```
pip install -r requirements.txt
```
## Train
If you want to train the model from scratch, here is a sample training run:
```
python3 train.py --optimizer SAM --epochs 100 --spectra_dir 'SPECTRA_DIR' --label_dir 'LABEL_DIR' --spectra_interval SPECTRA_INTERVAL
```
Full CLI:
```
usage: train.py [-h] [--epochs EPOCHS] [--spectra_dir SPECTRA_DIR] [--label_dir LABEL_DIR]
                [--list_trials LIST_TRIALS] [--spectra_interval SPECTRA_INTERVAL] [--splits_dir SPLITS_DIR]
                [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--in_channels IN_CHANNELS]
                [--layers LAYERS] [--hidden_size HIDDEN_SIZE] [--block_size BLOCK_SIZE]
                [--n_classes N_CLASSES] [--input_dim INPUT_DIM] [--rho RHO] [--momentum MOMENTUM]
                [--weight_decay WEIGHT_DECAY] [--label_smoothing LABEL_SMOOTHING] [--activation ACTIVATION]
                [--optimizer OPTIMIZER] [--base_optimizer BASE_OPTIMIZER] [--seed SEED] [--shuffle SHUFFLE]
                [--save]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Total number of epochs.
  --spectra_dir SPECTRA_DIR
                        Directory to spectra.
  --label_dir LABEL_DIR
                        Directory to labels.
  --list_trials LIST_TRIALS
                        Specified trials to run.
  --spectra_interval SPECTRA_INTERVAL
                        Specified patient intervals for clinical significance.
  --splits_dir SPLITS_DIR
                        Directory to data splits.
  --batch_size BATCH_SIZE
                        Batch size for the training and validation loops.
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer.
  --in_channels IN_CHANNELS
                        Number of input channels.
  --layers LAYERS       Number of layers in the model.
  --hidden_size HIDDEN_SIZE
                        Size of the hidden layer in the model.
  --block_size BLOCK_SIZE
                        Block size for the model.
  --n_classes N_CLASSES
                        Number of classes in the classification task.
  --input_dim INPUT_DIM
                        Input dimension for the model.
  --rho RHO             Rho value for the optimizer.
  --momentum MOMENTUM   Momentum value for the optimizer.
  --weight_decay WEIGHT_DECAY
                        Weight decay value for the optimizer.
  --label_smoothing LABEL_SMOOTHING
                        Use 0.0 for no label smoothing.
  --activation ACTIVATION
                        Type of activation to use in ResNet.
  --optimizer OPTIMIZER
                        optimizer to be used.
  --base_optimizer BASE_OPTIMIZER
                        base optimizer to be used.
  --seed SEED           Initialization seed.
  --shuffle SHUFFLE     Shuffle training set.
  --save                Save results.
```

## Inference
If you want to run inference on your trained model, here is a sample run:
```
python3 inference.py --spectra_dir 'SPECTRA_DIR' --label_dir 'LABEL_DIR' --spectra_interval SPECTRA_INTERVAL --param_dir 'PARAM_DIR' --weight_dir 'LABEL_DIR'
```
Full CLI:
```
usage: inference.py [-h] [--spectra_dir SPECTRA_DIR] [--label_dir LABEL_DIR]
                    [--spectra_interval SPECTRA_INTERVAL] [--weight_dir WEIGHT_DIR] [--param_dir PARAM_DIR]
                    [--seed SEED] [--shuffle SHUFFLE] [--save]

options:
  -h, --help            show this help message and exit
  --spectra_dir SPECTRA_DIR
                        Directory to spectra.
  --label_dir LABEL_DIR
                        Directory to labels.
  --spectra_interval SPECTRA_INTERVAL
                        Specified patient intervals for clinical significance.
  --weight_dir WEIGHT_DIR
                        Directory containing model weight(s).
  --param_dir PARAM_DIR
                        Directory containing model parameters.
  --seed SEED           Initialization seed.
  --shuffle SHUFFLE     Shuffle training set.
  --save                Save results.
```

## Citation
If you find anything in our paper or repository useful, please consider citing:
```
@inproceedings{
    zareno2024sharpnessaware,
    title={Sharpness-Aware Minimization ({SAM}) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics},
    author={Kaitlin Zareno and Jarett Dewbury and Siamak Sorooshyari and Hossein Mobahi and Loza F. Tadesse},
    booktitle={5th Workshop on practical ML for limited/low resource settings},
    year={2024},
    url={https://openreview.net/forum?id=k6FDRRRddZ}
}
```
