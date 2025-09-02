import numpy as np
import json
import os
import re
import pickle

def windows(folder, filename):
    """
    Extract (past, future) windows around LedON from neural component matrices.

    Parameters:
      folder (str): Directory containing *_neural_components.npy files.
      filename (str): JSON file containing marker data.

    Returns:
      list[tuple[np.ndarray, np.ndarray]]: (past, future) arrays per event.
    """
    eeg_dir = folder
    markers_file = filename

    with open(markers_file, 'r') as f:
        marker_data = json.load(f)
    columns = marker_data["columns"]
    data_rows = marker_data["data"]

    def col_idx(col_name):
        return columns.index(col_name)

    all_sequences = []

    # Iterate EEG files (only neural component matrices)
    for eeg_filename in sorted(os.listdir(eeg_dir)):
        if not eeg_filename.endswith('_eeg.npy'):
            continue
        eeg_path = os.path.join(eeg_dir, eeg_filename)

        # Extract run number _S{n}
        m = re.search(r'_S(\d+)', eeg_filename)
        if not m:
            print(f"Could not extract run number from file name {eeg_filename}.")
            continue
        run = int(m.group(1))

        X = np.load(eeg_path)  # components x time

        # Rows for this run
        run_rows = [row for row in data_rows if int(row[col_idx("Run")]) == run]

        for row in run_rows:
            start_time_sec = row[col_idx("StartTime")]
            if start_time_sec is None:
                continue

            led_on_sample = int(start_time_sec * 500)  # fs = 500 Hz

            # 1000 samples past (2 s), 1500 samples future (3 s)
            if led_on_sample - 1000 >= 0 and led_on_sample + 1500 <= X.shape[1]:
                past_window = X[:, led_on_sample - 1000 : led_on_sample]
                future_window = X[:, led_on_sample : led_on_sample + 1500]
                all_sequences.append((past_window, future_window))
            else:
                print(f"Skipping trial in run {run}: LEDOn sample {led_on_sample} out of bounds.")

    return all_sequences

if __name__ == "__main__":
    folder = "data"
    filename = "P1_AllLifts.json"

    all_sequences = windows(folder, filename)
    print(f"Collected {len(all_sequences)} (past, future) pairs total.")
    if all_sequences:
        p0, f0 = all_sequences[0]
        print(f"Example shapes, past: {p0.shape}, future: {f0.shape}")

    with open("train_sequences.pkl", "wb") as f:
        pickle.dump(all_sequences, f)
    print("Saved sequences list → train_sequences.pkl")