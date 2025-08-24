import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = r'E:\AE_RBFN\RBFN_Standard_new1\all_histories.npy'
all_histories = np.load(file_path, allow_pickle=True)

# Convert from ndarray to Python list/dict
if isinstance(all_histories, np.ndarray):
    all_histories = all_histories.item() if all_histories.shape == () else all_histories.tolist()

# Function to average histories with varying epoch lengths
def average_histories(histories):
    keys = histories[0].keys()
    min_len = min(len(h[key]) for h in histories for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss'])
    
    avg_history = {}
    for key in keys:
        truncated = [h[key][:min_len] for h in histories]
        stacked = np.stack(truncated, axis=0)
        avg_history[key] = np.mean(stacked, axis=0)
    
    return avg_history

# Get the final history to plot
if isinstance(all_histories, list):
    history = average_histories(all_histories)
elif isinstance(all_histories, dict):
    history = all_histories
else:
    raise ValueError("Unsupported format in all_histories.npy")

# Plotting
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy', color='blue', marker='o')
plt.plot(history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
plt.title('Standard-RBFN: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss', color='green', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', color='red', marker='o')
plt.title('Standard-RBFN: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
