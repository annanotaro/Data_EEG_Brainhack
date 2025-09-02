import matplotlib.pyplot as plt 
from data import EEGSequenceDataset  
import pickle 
 
with open("train_sequences.pkl", "rb") as f:
    train_sequences = pickle.load(f)

with open("normalization.pkl", "rb") as f:
    norm = pickle.load(f)

channel_means = norm["channel_means"]
channel_stds = norm["channel_stds"]
channel_names = norm["channel_names"]

train = EEGSequenceDataset(train_sequences, normalize=True,
                                   channel_means=channel_means, channel_stds=channel_stds,
                                   channel_names=channel_names)

# Get a sample from the dataset
sample_idx = 0
past, future = train[sample_idx]

# Convert tensors to numpy arrays for plotting
past_np = past.numpy()
future_np = future.numpy()

# Plot the past and future windows for all channels
fig, axes = plt.subplots(2, 1, figsize=(10, 10))  
colors = plt.cm.tab10.colors  

# Plot the past window for all channels
for i, channel_name in enumerate(channel_names):
    axes[0].plot(past_np[i, :], label=channel_name, color=colors[i % len(colors)])
axes[0].set_title("Past Window - 2 seconds before event")
axes[0].set_xlabel("Time (samples)")
axes[0].set_ylabel("Normalized Amplitude")
axes[0].legend(loc="upper right", fontsize="small")

# Plot the future window for all channels
for i, channel_name in enumerate(channel_names):
    axes[1].plot(future_np[i, :], label=channel_name, color=colors[i % len(colors)])
axes[1].set_title("Future Window - 3 seconds after event")
axes[1].set_xlabel("Time (samples)")
axes[1].set_ylabel("Normalized Amplitude")
axes[1].legend(loc="upper right", fontsize="small")

plt.tight_layout()
plt.show()

# Get item from train dataset
item = train[0]
past, future = item
print(item)
print(f"Past shape: {past.shape}, Future shape: {future.shape}")

