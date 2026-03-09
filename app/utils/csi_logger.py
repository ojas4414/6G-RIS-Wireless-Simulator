import numpy as np
import os


class CSILogger:
    """
    Logs every (channel, theta, user_pos) tuple to memory and can
    flush to disk as a .npy dataset for offline training.
    """

    def __init__(self):
        self.data = []

    def record(self, h, theta, user_pos):
        entry = {
            "channel":  h,
            "theta":    theta.copy(),
            "user_pos": np.array(user_pos)
        }
        self.data.append(entry)

    def count(self):
        return len(self.data)

    def save(self, filename="dataset/csi_dataset.npy"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, self.data)
        print(f"[CSILogger] Dataset saved: {filename}  ({len(self.data)} samples)")