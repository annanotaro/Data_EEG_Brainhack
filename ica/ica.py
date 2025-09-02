import json
import numpy as np
import mne
from mne.preprocessing import ICA 
from mne_icalabel import label_components

def ica(filename):
    with open(json_file, 'r') as f:
        data_json = json.load(f)

    eeg_dict = data_json["EEG"]
    filtered_data = np.array(eeg_dict["filtered_data"])
    fs = eeg_dict["sampling_rate"]
    provided_names = eeg_dict["names"]  

    # 2. Create an MNE Raw object
    raw_data = filtered_data.T  
    n_channels, n_times = raw_data.shape

    # Create default channel names (e.g., "EEG1", "EEG2", ...) for initial info
    default_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=default_names, sfreq=fs, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(raw_data, info)
    print("Raw object info (before renaming):")
    print(raw.info)

    # 3. Rename channels to match provided names
    if len(provided_names) != n_channels:
        raise ValueError("The number of provided channel names does not match the number of channels in the data.")

    rename_mapping = dict(zip(raw.info["ch_names"], provided_names))
    raw.rename_channels(rename_mapping)
    print("Raw object info (after renaming):")
    print(raw.info)

    # Keep only the allowed channels
    allowed_channels = [
        "F3", "Fz", "F4",
        "FC5", "FC1", "FC2", "FC6",
        "C3", "Cz", "C4",
        "CP5", "CP1", "CP2", "CP6",
    ]

    raw.pick_channels(allowed_channels)
    print("Raw object info (after picking allowed channels):")
    print(raw.info)

    # 4. Set Montage and Reference
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.set_eeg_reference('average', projection=False)

    # 5. Fit ICA explicitly on the Raw data
    ica = ICA(n_components=14, random_state=97, max_iter='auto')  # Set n_components to match the number of channels
    ica.fit(raw)
    print("ICA fitted successfully.")

    # 6. Apply ICLabel using mne-icalabel
    labels_dict = label_components(raw, ica, method='iclabel')
    print("ICLabel predicted labels:")
    for i, (label, prob) in enumerate(zip(labels_dict["labels"], labels_dict["y_pred_proba"]), start=1):
        print(f"Label {i}: {label}, Prob: {int(100 * prob)}%")

    # 7. Extract only neural components 
    neural_indices = [i for i, lab in enumerate(labels_dict["labels"]) if lab == "brain"]
    print("Indices of neural components (Brain):", neural_indices)

    sources = ica.get_sources(raw).get_data()  # shape: (n_components, n_times)
    neural_sources = sources[neural_indices, :]
    print("Shape of neural sources:", neural_sources.shape)

    print("Neural components saved to 'HS_P1_S1_neural_components.npy'")

    # 8. Reconstruct EEG using only neural components
    ica.exclude = [i for i in range(ica.n_components_) if i not in neural_indices]
    reconstructed_raw = ica.apply(raw.copy())
    cleaned_eeg = reconstructed_raw.get_data()  # shape: (channels, time points)

    # 9. Normalize EEG per channel (z-score)
    normalized_eeg = (cleaned_eeg - cleaned_eeg.mean(axis=1, keepdims=True)) / cleaned_eeg.std(axis=1, keepdims=True)
    return normalized_eeg

if __name__ == "__main__":
    for i in range(1,10):
        json_file = f'HS_P1_S{i}_processed.json'
        normalized_eeg = ica(json_file)
        np.save(f"HS_P1_S{i}_eeg.npy", normalized_eeg)
        print("Normalized EEG saved to 'HS_P1_S{i}_eeg.npy'")
