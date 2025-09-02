import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

file_path = Path("data") / "HS_P1_S1_eeg.npy"
eeg = np.load(file_path)
n_channels, n_times = eeg.shape
neural_components = np.load(file_path)

# Print the shape and some summary statistics
print("Shape of neural components:", neural_components.shape)
n_components, n_times = neural_components.shape
print(f"Number of neural components: {n_components}")
print(f"Number of time points per component: {n_times}")
print(neural_components)


# Plot time series of a subset of components 
num_plots = min(n_components, 1)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
if num_plots == 1:
    axes = [axes]

for i in range(num_plots):
    axes[i].plot(neural_components[i, :], color='blue')
    axes[i].set_title(f"Neural Component {i+1} Time Series")
    axes[i].set_ylabel("Amplitude")
axes[-1].set_xlabel("Time Points")
plt.tight_layout()
#plt.show()

# Plot power spectral density for the same subset of components using Welch's method
# Assuming a sampling frequency of 500 Hz (adjust if needed)
fs = 500  
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
if num_plots == 1:
    axes = [axes]

for i in range(num_plots):
    freqs, psd = welch(neural_components[i, :], fs=fs, nperseg=1024)
    axes[i].semilogy(freqs, psd, color='green')
    axes[i].set_title(f"Neural Component {i+1} Power Spectral Density")
    axes[i].set_ylabel("PSD (V²/Hz)")
axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout()
#plt.show()