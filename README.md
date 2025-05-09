# Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics

![Results on clinical dataset](https://github.com/Jdewbury/SAM-Raman-Diagnostics/blob/main/src/SAM-Raman-Diagnostic-Result.png)
<br><br>
Antimicrobial resistance is expected to claim 10 million lives per year by 2050, and resource-limited regions are most affected. Raman spectroscopy is a novel pathogen diagnostic approach promising rapid and portable antibiotic resistance testing within a few hours, compared to days when using gold standard methods. However, current algorithms for Raman spectra analysis 1) are unable to generalize well on limited datasets across diverse patient populations and 2) require increased complexity due to the necessity of non-trivial pre-processing steps, such as feature extraction, which are essential to mitigate the low-quality nature of Raman spectral data. In this work, we address these limitations using Sharpness-Aware Minimization (SAM) to enhance model generalization across a diverse array of hyperparameters in clinical bacterial isolate classification tasks. We demonstrate that SAM achieves accuracy improvements of up to $10.5\%$ on a single split, and an increase in average accuracy of $2.7\%$ across all splits in spectral classification tasks over the traditional optimizer, Adam. These results display the capability of SAM to advance the clinical application of AI-powered Raman spectroscopy tools.
<br><br>
This repository contains modules and instructions for replicating and extending experiments featured in our paper:
- A sample experiment script [SAMRaman.py](SAMRaman.py) to train and evaluate the 1-D ResNet architecture on spectral data

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

## Setup Repository and Requirements
Clone repository 
```
git clone https://github.com/Tadesse-Lab/SAM-Raman-Diagnostics.git
```
Install requirements:
```
pip install -r requirements.txt
```
## Training and Evaluation
If you want to train the model from scratch and evaluate it on a test set, here is a sample training run:
```
python3 SAMRaman.py --optimizer SAM --spectra_dirs SPECTRA_DIRS ... --label_dirs LABEL_DIRS ... --spectra_intervals SPECTRA_INTERVALS ...
```
Spectra and label directories support multiple paths to combine datasets together. Please ensure that the spectra_intervals align with the order
of the paths.

You can modify the default arguments in the [config](utils/config.py) file, or specify CLI arguments during inference.

Full CLI:
```
usage: SAMRaman.py [-h] [--spectra_dirs SPECTRA_DIRS [SPECTRA_DIRS ...]] [--label_dirs LABEL_DIRS [LABEL_DIRS ...]]
                   [--spectra_intervals SPECTRA_INTERVALS [SPECTRA_INTERVALS ...]] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE]    
                   [--seed SEED] [--model {resnet1d}] [--in_channels IN_CHANNELS] [--layers LAYERS] [--hidden_size HIDDEN_SIZE] [--block_size BLOCK_SIZE]
                   [--num_classes NUM_CLASSES] [--input_dim INPUT_DIM] [--activation {relu,selu,gelu}] [--optimizer {sam,asam,adam,sgd}]
                   [--base_optimizer {adam,sgd}] [--rho RHO] [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM] [--label_smoothing LABEL_SMOOTHING]
                   [--scheduler {step,cosine,none}] [--lr_step LR_STEP] [--output_dir OUTPUT_DIR] [--experiment_name EXPERIMENT_NAME]
                   [--early_stopping_patience EARLY_STOPPING_PATIENCE]

options:
  -h, --help            show this help message and exit
  --spectra_dirs SPECTRA_DIRS [SPECTRA_DIRS ...]
                        Directories containing spectral data files. Default: ['data/spectral_data/X_2018clinical.npy', 'data/spectral_data/X_2019clinical.npy']   
  --label_dirs LABEL_DIRS [LABEL_DIRS ...]
                        Directories containing label data files. Default: ['data/spectral_data/y_2018clinical.npy', 'data/spectral_data/y_2019clinical.npy']      
  --spectra_intervals SPECTRA_INTERVALS [SPECTRA_INTERVALS ...]
                        Specified patient intervals for clinical significance. Default: [400, 100]
  --batch_size BATCH_SIZE
                        Batch size for the training and validation loops. Default: 16
  --epochs EPOCHS       Total number of training epochs. Default: 200
  --learning_rate LEARNING_RATE
                        Initial learning rate for training. Default: 0.001
  --seed SEED           Seed to use for reproducibility. Default: 42
  --model {resnet1d}    Model architecture to be used for training. Default: resnet
  --in_channels IN_CHANNELS
                        Number of input channels. Default: 64
  --layers LAYERS       Number of layers in the model. Default: 6
  --hidden_size HIDDEN_SIZE
                        Size of the hidden layer in the model. Default: 100
  --block_size BLOCK_SIZE
                        Block size for the model. Default: 2
  --num_classes NUM_CLASSES
                        Number of classes in the classification task. Default: 5
  --input_dim INPUT_DIM
                        Input dimension for the model. Default: 1000
  --activation {relu,selu,gelu}
                        Activation function to use for model. Default: selu
  --optimizer {sam,asam,adam,sgd}
                        Optimizer to use for training. Default: sam
  --base_optimizer {adam,sgd}
                        Base optimizer for SAM/ASAM. Default: adam
  --rho RHO             Rho value for SAM/ASAM optimizers. Default: 0.05
  --weight_decay WEIGHT_DECAY
                        Weight decay for optimizer. Default: 0.0005
  --momentum MOMENTUM   Momentum value for SGD. Default: 0.9
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing factor. Default: 0.1
  --scheduler {step,cosine,none}
                        Learning rate scheduler to use. Default: step
  --lr_step LR_STEP     Learning rate step size. Default: 0.2
  --output_dir OUTPUT_DIR
                        Output directory to save training results. Default: results
  --experiment_name EXPERIMENT_NAME
                        Name of experiment. Default: sam-raman
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs without improvement for early stopping. Default: 15
```

## Experimental Replication
To replicate the experiments in the paper, the clinical data can be downloaded [here](https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&e=1&dl=0) and saved into a `data/spectral_data` folder (or adjust filepaths accordingly). 

Across 10 selected seeds for n=5 trials, run:
```
for activation in selu relu gelu; do
    for optimizer in sam adam; do
            python3 SAMRaman.py --optimizer $optimizer --base_optimizer adam --epochs 100 --activation $activation --seed SEED
    done
done
```
