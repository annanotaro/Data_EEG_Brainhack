import json
import numpy as np
import mne
from mne.preprocessing import ICA 
from mne_icalabel import label_components
from pathlib import Path

def ica(filename):
    with open(filename, 'r') as f:                     
        data_json = json.load(f)

    eeg_dict = data_json["EEG"]
    filtered_data = np.array(eeg_dict["filtered_data"])
    fs = eeg_dict["sampling_rate"]
    provided_names = eeg_dict["names"]  

    # 2. Create an MNE Raw object (n_channels, n_times)
    raw_data = filtered_data.T
    n_channels, n_times = raw_data.shape
    default_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=default_names, sfreq=fs, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(raw_data, info)

    # 3. Rename channels to match provided names
    if len(provided_names) != n_channels:
        raise ValueError("The number of provided channel names does not match the number of channels in the data.")
    raw.rename_channels(dict(zip(raw.info["ch_names"], provided_names)))

    # Keep only the allowed channels
    allowed_channels = [
        "F3", "Fz", "F4",
        "FC5", "FC1", "FC2", "FC6",
        "C3", "Cz", "C4",
        "CP5", "CP1", "CP2", "CP6",
    ]
    raw.pick_channels(allowed_channels)

    # 4. Montage + reference
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.set_eeg_reference('average', projection=False)

    # 5. ICA
    ica = ICA(n_components=14, random_state=97, max_iter='auto')
    ica.fit(raw)

    # 6. ICLabel → neural indices
    labels_dict = label_components(raw, ica, method='iclabel')
    neural_indices = [i for i, lab in enumerate(labels_dict["labels"]) if lab == "brain"]

    # 7. Neural component matrix (components x time)
    sources = ica.get_sources(raw).get_data()          # (n_components, n_times)
    neural_sources = sources[neural_indices, :]        

    return neural_sources                               

if __name__ == "__main__":
    out_dir = Path("data"); out_dir.mkdir(exist_ok=True)
    for i in range(1, 10):
        json_file = f'HS_P1_S{i}_processed.json'
        neural_sources = ica(json_file)
        np.save(out_dir / f"HS_P1_S{i}_eeg.npy", neural_sources)
        print(f"Neural components saved to '{(out_dir / f'HS_P1_S{i}_eeg.npy')}'")