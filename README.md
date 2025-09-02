# I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series

Tools and scripts to convert WAY-EEG-GAL EEG recordings into ML-ready tensors and event-aligned sequences for forecasting motor behaviour.
Goal: forecast upcoming motor behaviour from EEG time series.

## Dataset
We use the publicly available [WAY-EEG-GAL dataset](https://www.nature.com/articles/sdata201447) described in the paper:  **Luciw, M., Jarocka, E., & Edin, B. (2014).** 
Open source dataset with 32-channel EEG recorded while 12 participants performed 3,936 grasp-and-lift trials with unpredictable changes in weight and surface friction. Event times (e.g. LEDOn/LEDOff, contacts, lift-off) are provided per trial. EEG target sampling rate is 500 Hz.

Why make it ML-friendly: the release is MATLAB-centric and multi-modal. We standardize EEG into neural component matrices and simple, event-locked windows that match deep-learning input conventions.

Download links are listed in the WAY-EEG-GAL data descriptor (figshare archive, CC BY 4.0).

## Conversion from MATLAB to JSON

The original WAY-EEG-GAL data are provided as MATLAB .mat files. To make them easier to handle in Python, we converted them into structured JSON:

1. **Session files** (HS_P*_S*.mat, WS_P*_S*.mat) → JSON with EEG, EMG, KIN, ENV, MISC sections, each containing signals, channel names, and sampling rates.
2. **Marker files** (P*_AllLifts.mat) → JSON with columns and data tables storing event information.

This keeps the original sampling rates and provides a consistent, ML-friendly format for downstream preprocessing.

## Preprocessing
1. ### Band-pass filtering and channel selection

The first step is handled by bandpass_filter.py. Here we start from the original per-series JSON files (HS_P*_S*.json) that contain the EEG recordings. From these, only the 14 fronto-central channels are retained, as they are most relevant for motor behaviour prediction. A zero-phase Butterworth band-pass filter (0.5–40 Hz) is then applied to remove slow drifts and high-frequency noise while preserving the EEG signal of interest. The output of this stage is a processed JSON file (HS_P*_S*_processed.json) that contains the cleaned EEG data stored under EEG.filtered_data.

2. ### ICA and ICLabel for neural components

The second step is performed by ica.py. It takes the processed JSON from the previous stage as input. The script constructs an MNE Raw object, assigns the standard 10–20 montage, and applies an average reference. An Independent Component Analysis (ICA) is then fitted to decompose the EEG into statistically independent sources. These components are automatically classified using ICLabel, which identifies whether they correspond to brain activity, eye movements, muscle noise, or other artifacts. Only the components labeled as “brain” are kept, and the resulting neural component matrix is saved as a NumPy array (data/HS_P1_S{run}_eeg.npy), with dimensions (components × time). 

3. ### Event-aligned sequence extraction

The final step is implemented in sequences.py. This script combines the neural component matrices with the event marker file (P1_AllLifts.json), which contains trial information and event times such as StartTime/LEDOn. For each trial, the script extracts two windows: a past window of 1000 samples (≈2 seconds) before LEDOn and a future window of 1500 samples (≈3 seconds) after LEDOn, assuming a 500 Hz sampling rate. These pairs of past and future segments are collected across trials and stored as a Python list of tuples. The final dataset is saved as train_sequences.pkl, which contains all the (past, future) arrays needed to train forecasting model.

### Assumptions and notes

Sampling rate: 500 Hz for EEG (as in the data descriptor). Adjust in sequences.py if your conversion differs.

Event alignment: windows are centered on LEDOn as defined in the dataset (StartTime, LEDOn columns).

Scope: this repo processes EEG only; other modalities (EMG, kinematics, forces) are present in the dataset but not used here.


## Citing WAY-EEG-GAL

If you use this repo or derived data, please cite the data descriptor:
Luciw, Jarocka & Edin (2014). Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction. Scientific Data 1:140047.

The dataset is released under Creative Commons Attribution 4.0 (CC BY 4.0).

## Acknowledgements

This work grew out of the Brainhack project “I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series.” We thank the WAY-EEG-GAL authors and the WAY project for making the data and utilities publicly available.
