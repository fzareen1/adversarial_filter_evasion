# adversarial_filter

The script expects the following folder structure:
adversarial_filter_master/
├── attack_lib.py
├── evasion_attack_V1.py
├── models.py
├── poisoning_attack.py
├── pre-training.py
├── model
│   └── target/ERN/EEGNet/within/0/0/model.pt
├── result/
│   ├── log/
│   │   └── evasion/
│   └── npz/
│       └── evasion/
└── utils/
    ├── batch_linalg.py
    ├── data_loader.py
    ├── EEGLayers.py
    ├── pytorch_utils.py
    └── tempCodeRunnerFile.py

# Root Files:

**models.py:** Contains the model definitions (e.g., EEGNet, DeepConvNet, ShallowConvNet, etc.).
**pre-training.py:** Script for pre-training the models.
**evasion_attack_V1.py:** Main script for the evasion attack experiment.
**poisoning_attack.py:** Script for performing poisoning attacks.
**attack_lib.py:** Contains utility functions and classes used across different attack scripts.
**utils/data_loader.py:**
Contains functions to load and split the ERN EEG data (including ERNLoad and split).

# EEG_data/ERN/
**process_training.m:**
This MATLAB script processes raw CSV files containing EEG data according to the following steps:
1. Down-sampling:
    The data is down-sampled from the original sampling rate (assumed to be 200 Hz) to 128 Hz.
2. Filtering:
    A 4th order Butterworth band-pass filter (1-40 Hz) is applied using zero-phase filtering to clean the EEG signals.
3. Time Window Extraction:
    The script extracts the time window from 0 to 1.3 seconds for each trial.
4. Normalization:
    Each EEG channel is z-score normalized independently.
5. Saving Processed Data:
    Processed data for each subject is saved in a separate .mat file. Subjects are 0-indexed (e.g., s0.mat, s1.mat, …, s15.mat).

The script expects CSV files with filenames such as Data_S01_Sess01.csv, Data_S02_Sess01.csv,
To run the script: 
1. Start MATLAB and set your working directory to the location of this script.
2. The script requires MATLAB (R2018a or later is recommended) with the Signal Processing Toolbox (for functions such as butter, filtfilt, and resample).

# utils/data_loader.py:
This script provides helper functions for loading and splitting EEG data from MATLAB .mat files. It is designed to work with ERN, MI4C, EPFL, BNCI datasets, though only the ERN loader is currently active. The remaining functions are provided as templates (currently commented out) and can be enabled or modified as needed.


# pre-training.py:
Automated Pretraining for ERN EEGNet (within-subject) across subjects and random seeds.

For each subject (0-indexed), the processed ERN data is loaded using an 80/20 split (by ERNLoad),
then further split (75% training, 25% validation) for model selection.
Before training, labels are remapped: any label < 0 is set to 1, and all others to 0.
Each subject’s model is trained for a fixed number of epochs (default: 50) with a learning rate of 5e-4 and batch size 128.
This entire process is repeated for a number of random seeds (default: 10).

The pretrained model for each subject and seed is saved to:
  model/target/ERN/EEGNet/within/<subject>/<seed>/model.pt

Usage:
    python pretraining.py --subjects 16 --seeds 10 --epochs 50 --batch_size 128 --lr 0.0005 --device cpu

# evasion_attack_V1.py: 
The script performs the following tasks:

**Model Loading**: Loads a pre-trained EEG classifier (e.g., EEGNet, DeepConvNet, or ShallowConvNet) from a specified directory.
**Data Loading & Preprocessing:** Imports EEG data from supported datasets (e.g., ERN, MI4C, EPFL), splits the data into training, validation, and test sets, and remaps the labels (e.g., mapping negative values to class 1 and non-negative to class 0).
**Adversarial Filter Optimization:** Uses a spatial filtering layer (or optionally a filter layer) that is fine-tuned adversarially. The optimization objective is to maximize the classification loss (i.e., fool the model) while regularizing the perturbation so that the altered input stays close to the original.
**Evaluation:** Evaluates both the original and the adversarially perturbed inputs using metrics such as accuracy and Balanced Classification Accuracy (BCA).
**Visualization:** Plots example EEG signals comparing the original and adversarial (poisoned) samples.
**Result Logging & Saving:** Logs detailed results for each subject and repeat of the experiment and saves aggregated results as a NumPy .npz file.

**Only used EEGNet Dataset in this phase of the experiments. Other datasets have been commented out.**