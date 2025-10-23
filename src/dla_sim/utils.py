# src/dla_sim/utils.py
import numpy as np
import time
import json
import os

def set_seed(seed=0):
    """Set random seed for reproducibility."""
    np.random.seed(seed)

def now_str():
    """Return current timestamp (YYYYMMDD-HHMMSS)."""
    return time.strftime("%Y%m%d-%H%M%S")

def save_cluster(path, occupied, meta=None):
    """Save cluster (bool array) and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        occupied=occupied.astype("uint8"),
        meta=meta or {}
    )

def load_cluster(path):
    """Load cluster file and return array + metadata."""
    data = np.load(path, allow_pickle=True)
    return data["occupied"].astype(bool), data["meta"].item()
