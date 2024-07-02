# Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics

![[Results on clinical dataset](https://github.com/Jdewbury/SAM-Raman-Diagnostics/src/SAM-Raman-Diagnostics-1.png)](https://github.com/Jdewbury/SAM-Raman-Diagnostics/blob/main/src/SAM-Raman-Diagnostic-Result.png)
<br><br>
Antimicrobial resistance is expected to claim 10 million lives per year by 2050, and resource-limited regions are most affected. Raman spectroscopy is a novel pathogen diagnostic approach promising rapid and portable antibiotic resistance testing within a few hours, compared to days when using gold standard methods. However, current algorithms for Raman spectra analysis 1) are unable to generalize well on limited datasets across diverse patient populations and 2) require increased complexity due to the necessity of non-trivial pre-processing steps, such as feature extraction, which are essential to mitigate the low-quality nature of Raman spectral data. In this work, we address these limitations using Sharpness-Aware Minimization (SAM) to enhance model generalization across a diverse array of hyperparameters in clinical bacterial isolate classification tasks. We demonstrate that SAM achieves accuracy improvements of up to $10.5\%$ on a single split, and an increase in average accuracy of $2.7\%$ across all splits in spectral classification tasks over the traditional optimizer, Adam. These results display the capability of SAM to advance the clinical application of AI-powered Raman spectroscopy tools.
<br><br>
This repository contains modules and instructions for replicating and extending experiments featured in our paper:
- A training script `./train.py` to train the 1-D ResNet architecture on spectral data
- An evaluation script `./inference.py` to evaluate the trained network on test data

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
python3 train.py --optimizer SAM --epochs 100 --spectra_dir 'PATH_TO_SPECTRA' --label_dir 'PATH_TO_LABEL' --spectra_interval SPECTRA_INTERVAL
```
Spectra and label directories also support multiple paths to combine datasets together, as long as spectra_intervals align. For example:
```
python3 train.py --optimizer SAM --epochs 100 --spectra_dir 'PATH_TO_SPECTRA1' 'PATH_TO_SPECTRA2' --label_dir 'PATH_TO_LABEL1' 'PATH_TO_LABEL2' --spectra_interval SPECTRA_INTERVAL1 SPECTRA_INTERVAL2
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
python3 inference.py --spectra_dir 'PATH_TO_SPECTRA' --label_dir 'PATH_TO_LABEL' --spectra_interval SPECTRA_INTERVAL --param_dir 'PATH_TO_MODEL_PARAMS' --weight_dir 'PATH_TO_TRAINED_WEIGHTS'
```
Full CLI:
```
usage: inference.py [-h] [--spectra_dir SPECTRA_DIR] [--label_dir LABEL_DIR]
                    [--spectra_interval SPECTRA_INTERVAL] [--weight_dir WEIGHT_DIR] [--param_dir PARAM_DIR]
                    [--seed SEED] [--save]

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
  --save                Save results.
```

## Experimental Replication
To replicate the experiments in the paper, the clinical data can be downloaded [here](https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&e=1&dl=0) and saved into a `spectral_data` folder (or adjust filepaths accordingly). 

Across 10 selected seeds for n=5 trials, run:
```
for activation in selu relu gelu; do
    for optimizer in SAM Adam; do
            python3 train.py --optimizer $optimizer --base_optimizer Adam --epochs 100 --activation $activation --seed SEED
    done
done
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
