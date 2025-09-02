import json
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to the input data.
    
    Parameters:
        data (np.ndarray): 2D array (samples x channels).
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int): Filter order.
        
    Returns:
        np.ndarray: Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def preprocess_eeg(json_filepath, output_filepath=None):
    """
    Loads a JSON file with EEG, EMG, and KIN data, applies a 0.5–40Hz bandpass filter to the EEG data,
    selects only the allowed channels.
    
    Parameters:
        json_filepath (str): Path to the input JSON file.
        output_filepath (str, optional): If provided, writes the updated JSON structure to this file.
        
    Returns:
        dict: Updated JSON structure with additional EEG processing results.
    """
    with open(json_filepath, 'r') as f:
        data_json = json.load(f)
    
    # Extract EEG data and sampling rate
    eeg_data = np.array(data_json["EEG"]["data"])  
    fs = data_json["EEG"]["sampling_rate"]
    
    allowed_channels = [
        "F3", "Fz", "F4",
        "FC5", "FC1", "FC2", "FC6",
        "C3", "Cz", "C4",
        "CP5", "CP1", "CP2", "CP6",
    ]

    channel_names = data_json["EEG"]["names"]  
    indices = [i for i, name in enumerate(channel_names) if name in allowed_channels]
    eeg_data = eeg_data[:, indices]
    data_json["EEG"]["names"] = [channel_names[i] for i in indices]
    
    # Apply bandpass filter (0.5-40Hz) on the EEG data
    filtered_eeg = bandpass_filter(eeg_data, lowcut=0.5, highcut=40, fs=fs, order=5)
    
    # Update JSON structure with filtered data and ICA results
    data_json["EEG"]["filtered_data"] = filtered_eeg.tolist()

    if output_filepath:
        with open(output_filepath, 'w') as f:
            json.dump(data_json, f, indent=4)
    
    return data_json

if __name__ == '__main__':
    input_json = 'ica code\HS_P1_S1.json'     
    output_json = f'HS_P1_S1_processed.json'
    processed_data = preprocess_eeg(input_json, output_json)
    print("Processed EEG data with bandpass filtering saved to:", output_json)