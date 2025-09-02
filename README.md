### I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series

Tools and scripts to convert WAY-EEG-GAL EEG recordings into ML-ready tensors and event-aligned sequences for forecasting motor behaviour.

Goal: forecast upcoming motor behaviour from EEG time series.
Pipeline: band-pass + channel selection → ICA + ICLabel → keep neural components → slice (past, future) windows around LEDOn.

# Dataset (WAY-EEG-GAL)

Open source dataset with 32-channel EEG recorded while 12 participants performed 3,936 grasp-and-lift trials with unpredictable changes in weight and surface friction. Event times (e.g. LEDOn/LEDOff, contacts, lift-off) are provided per trial. EEG target sampling rate is 500 Hz.

Why make it ML-friendly: the release is MATLAB-centric and multi-modal. We standardize EEG into neural component matrices and simple, event-locked windows that match deep-learning input conventions.

Download links are listed in the WAY-EEG-GAL data descriptor (figshare archive, CC BY 4.0).

Repository structure
bandpass_filter.py   # Band-pass (0.5–40 Hz) + select 14 channels
ica.py               # ICA + ICLabel → keep “brain” components 
sequences.py         # Slice (past, future) windows around LEDOn 
np.py                # Optional QC plotting of saved arrays

Installation
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install numpy scipy mne mne-icalabel matplotlib

Processing pipeline
Each step consumes the output of the previous step.

Band-pass + channel selection
Run bandpass_filter.py.

Input: HS_P*_S*.json (per-series JSON).

Keeps the 14 fronto-central channels.

Applies a zero-phase Butterworth 0.5–40 Hz band-pass.

Output: HS_P*_S*_processed.json with EEG.filtered_data.

ICA + ICLabel → neural components
Run ica.py.

Input: *_processed.json.

Sets standard 10–20 montage and average reference; fits ICA; labels components with ICLabel; retains only “brain” components.

Output: data/HS_P1_S{run}_neural_components.npy (components × time).

Event-aligned windows
Run sequences.py.

Inputs: data/*_neural_components.npy and P1_AllLifts.json.

Uses StartTime / LEDOn to align events (dataset provides these columns).

Slices past = 1000 samples (≈2 s) and future = 1500 samples (≈3 s) at 500 Hz.

Output: train_sequences.pkl → list of (past, future) arrays.

Quick start
# 1) Band-pass one series (edit the path at the bottom of the script if needed)
python bandpass_filter.py

# 2) Extract neural components for runs 1..9 (default loop in the script)
python ica.py

# 3) Build (past, future) sequences for Participant 1
python sequences.py


Load sequences:

import pickle
with open("train_sequences.pkl","rb") as f:
    seqs = pickle.load(f)   # list of (past, future)
past, future = seqs[0]      # arrays: (components, T_past), (components, T_future)

Inputs and outputs

Input: HS_P*_S*.json (series JSON converted from the public data).

Intermediate: HS_P*_S*_processed.json with EEG.filtered_data.

Components: data/HS_P1_S{run}_neural_components.npy (components × time).

Training windows: train_sequences.pkl (Python list of (past, future) arrays).

Assumptions and notes

Sampling rate: 500 Hz for EEG (as in the data descriptor). Adjust in sequences.py if your conversion differs.

Event alignment: windows are centered on LEDOn as defined in the dataset (StartTime, LEDOn columns).

Scope: this repo processes EEG only; other modalities (EMG, kinematics, forces) are present in the dataset but not used here.

Troubleshooting

No sequences saved: check that data/*_neural_components.npy exist and that the markers JSON matches the participant/run numbers.

Index out of bounds: LEDOn near the start or end of a series can violate window bounds; those trials are skipped by design.

ICLabel missing: pip install mne-icalabel.

Citing WAY-EEG-GAL

If you use this repo or derived data, please cite the data descriptor:
Luciw, Jarocka & Edin (2014). Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction. Scientific Data 1:140047.

The dataset is released under Creative Commons Attribution 4.0 (CC BY 4.0).

Acknowledgements

This work grew out of the Brainhack project “I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series.” We thank the WAY-EEG-GAL authors and the WAY project for making the data and utilities publicly available.