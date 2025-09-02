import numpy as np
import json
import os
import re
import pickle

def windows(folder, filename):
    """
    Function to extract EEG data windows based on marker events.
    
    Parameters:
    folder (str): Directory containing EEG files.
    filename (str): JSON file containing marker data.
    
    Returns:
    all_sequences (list): List of tuples containing past and future EEG data windows.
    """
    eeg_dir = folder  
    markers_file = filename  

    # Load the marker file and extract columns and data
    with open(markers_file, 'r') as f:
        marker_data = json.load(f)  
    columns = marker_data["columns"]  
    data_rows = marker_data["data"]  

    # Helper function to get the index of a column by name
    def col_idx(col_name):
        return columns.index(col_name)

    all_sequences = []

    # Iterate through EEG files in the specified directory
    for eeg_filename in os.listdir(eeg_dir):
        if eeg_filename.endswith('.npy'):  # Process only .npy files
            eeg_path = os.path.join(eeg_dir, eeg_filename)  # Full path to the EEG file
            
            # Use regex to extract the run number from the filename
            m = re.search(r'_S(\d+)', eeg_filename)
            if m:
                run = int(m.group(1))  # Extract run number
            else:
                print(f"Could not extract run number from file name {eeg_filename}.")
                continue  

            eeg_data = np.load(eeg_path)
            
            # Filter marker rows corresponding to the current run
            run_rows = [row for row in data_rows if int(row[col_idx("Run")]) == run]

            # Process each marker row for the current run
            for row in run_rows:
                start_time_sec = row[col_idx("StartTime")]  # Get the start time of the event
                if start_time_sec is None:  # Skip rows with no start time
                    continue

                led_on_sec = start_time_sec
                led_on_sample = int(led_on_sec * 500)  

                # Extract past and future windows around the event
                if led_on_sample - 1000 >= 0 and led_on_sample + 1500 <= eeg_data.shape[1]:
                    past_window = eeg_data[:, led_on_sample - 1000 : led_on_sample]
                    future_window = eeg_data[:, led_on_sample : led_on_sample + 1500]

                    all_sequences.append((past_window, future_window))
                else:
                    print(f"Skipping trial in run {run}: LEDOn sample {led_on_sample} out of bounds.")          

    return all_sequences  # Return the list of (past, future) pairs

def compute_laplacian(eeg_data, electrode_idx, neighbors_idx):

    electrode_signal = eeg_data[electrode_idx, :]
    neighbors_signal = np.mean(eeg_data[neighbors_idx, :], axis=0)
    laplacian_signal = electrode_signal - neighbors_signal
    
    return laplacian_signal

def compute_laplacian_windows(all_sequences, channel):
    
    with open("normalization.pkl", "rb") as f:
        norm = pickle.load(f)

    channel_names = norm["channel_names"]
    
    if channel == "Cz":
        idx = channel_names.index("Cz")
        neighbours = ["FC1", "FC2", "CP1", "CP2"]
        neighbors_idx = [channel_names.index(c) for c in neighbours]
    if channel == "C3":
        idx = channel_names.index("C3")
        neighbours = ["FC5", "FC1", "CP1", "CP5"]
        neighbors_idx = [channel_names.index(c) for c in neighbours]
    if channel == "C4":
        idx = channel_names.index("C4")
        neighbours = ["FC2", "FC6", "CP6", "CP2"]
        neighbors_idx = [channel_names.index(c) for c in neighbours]

    laplacian_windows = []

    # Process each sequence in the dataset
    for past, future in all_sequences:
        laplacian_past = compute_laplacian(past, idx, neighbors_idx)
        laplacian_future = compute_laplacian(future, idx, neighbors_idx)
        laplacian_windows.append([laplacian_past, laplacian_future])
    
    return laplacian_windows, channel_names

if __name__ == "__main__":
    folder = "data"
    filename = "P1_AllLifts.json"
    all_sequences = windows(folder, filename)
    print(f"Collected {len(all_sequences)} (past, future) pairs from all EEG series.")
    print(f"Example sequence shape: {all_sequences[0][0].shape}, {all_sequences[0][1].shape}")

    lap_sequences, channel_names = compute_laplacian_windows(all_sequences, "C3")
    print(lap_sequences[0][0].shape, lap_sequences[0][1].shape)
    print(len(lap_sequences))
    print(channel_names)

