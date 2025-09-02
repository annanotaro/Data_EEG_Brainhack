# WAY-EEG-GAL Utilities — Brainhack Rome 2025

This repository contains tools and scripts to convert EEG recordings from the WAY-EEG-GAL dataset ([Luciw et al., 2014](https://www.nature.com/articles/sdata201447)) into machine learning-ready tensors that contain event-aligned EEG sequences. The resulting dataset was used to forecast motor behaviour at [Brainhack Rome 2025](https://brainhackrome.github.io/) (project repository [here](https://github.com/matteo-d-m/brainhack-rome-forecasting)) and in related follow-ups. 

## Dataset
We use the publicly available [WAY-EEG-GAL dataset](https://figshare.com/collections/WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376) described in [Luciw et al., 2014](https://www.nature.com/articles/sdata201447). This is an open source dataset containing 32-channels EEG recordings from 12 participants performing 3936 grasp-and-lift trials under unpredictable changes of object weight and surface friction. The EEG sampling rate is 500 Hz.

The original dataset is MATLAB-centric and multi-modal, including both EEG and non-EEG (e.g., EMG) files. The code in this repository standardises the EEG files into neural component matrices and simple, event-locked windows that match deep learning input conventions and can be easily used with common Python libraries.

### Conversion from MATLAB to JSON

The original WAY-EEG-GAL dataset consists of MATLAB `.mat` files. To make them easier to handle in Python, we converted those files into two classes of JSON files:

1. **Session files:** JSON files with EEG, EMG, KIN, ENV, and MISC sections — each one containing signals, channel names, and sampling rates. These files correspond to `{HS_P*_S*, WS_P*_S*}.mat` files in the original dataset
2. **Marker files:** JSON files with columns and data tables storing event information. These files correspond to `P*_AllLifts.mat` in the original dataset 

This keeps the original sampling rates and provides a consistent, machine learning-friendly format for downstream processing.

## Data Preprocessing
### 1. Band-pass filtering and channel selection

The first preprocessing step is handled by `bandpass_filter.py`. Starting from the original per-series JSON files (`{HS_P*_S*.json}`) that contain the EEG recordings, the code retains 14 fronto-central EEG channels of interest and filters the corresponding signals with a zero-phase Butterworth band-pass filter (0.5–40 Hz) to remove slow drifts and high-frequency noise. The output of this stage is a processed JSON file (`HS_P*_S*_processed.json`) that contains the cleaned EEG data, stored in the `EEG.filtered_data` field.

### 2. ICA and ICLabel for neural components

The second preprocessing step is performed by `ica.py`. Starting from the JSON file produced by `bandpass_filter.py`, the code in `ica.py` constructs an MNE Raw object, assigns the standard 10–20 montage, and applies an average reference. Subsequently, it fits an Independent Component Analysis (ICA) model to decompose the EEG into statistically independent components. The resulting components are automatically classified as either brain, ocular, muscular, or other using the ICLabel classifier ([Pion-Tonachini et al., 2019](https://www.sciencedirect.com/science/article/pii/S1053811919304185)). Subsequently, non-brain components are discarded and brain components are saved as NumPy arrays ({data/HS_P1_S{run}_eeg.npy}) of dimension `(components × time)`. 

### 3.  Event-aligned sequence extraction

The third (and last) preprocessing step is implemented in `sequences.py`. This script combines the neural component matrices with the event marker file (`P1_AllLifts.json`), which contains trial information and event times such as `StartTime/LEDOn`. For each trial, the script extracts two windows: a past window of 1000 samples (≈2 seconds) before LEDOn and a future window of 1500 samples (≈3 seconds) after LEDOn, assuming a 500 Hz sampling rate. These pairs of past and future segments are collected across trials and stored as a Python list of tuples. The final dataset is saved as `train_sequences.pkl`, which contains all the (past, future) arrays needed to train a forecasting model.

### Assumptions and notes

- Sampling rate: 500 Hz for EEG (as in the data descriptor). Adjust in `sequences.py` if needed.
- Event alignment: windows are centered on LEDOn as defined in the original dataset (`StartTime, LEDOn` columns)
- Scope: the code in this repo processes EEG data only. Other data (EMG, kinematics, forces) are present in the dataset but not used here


## Citing WAY-EEG-GAL

If you use WAY-EEG-GAL data, please cite the data descriptor:
Luciw, Jarocka & Edin (2014). Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction. Scientific Data 1:140047.

The WAY-EEG-GAL dataset is released under a Creative Commons Attribution 4.0 license ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/deed.en)).

## Acknowledgements

This work grew out of the Brainhack project [I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series](https://github.com/matteo-d-m/brainhack-rome-forecasting). We thank the WAY-EEG-GAL authors and the WAY project for making the data and utilities publicly available.
