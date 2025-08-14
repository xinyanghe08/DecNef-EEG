import os
import numpy as np
import torch
import random

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    print(f"Setting all seeds to {seed}")

    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_validation_split(dataset, val_ratio=0.2):
    """Create train/validation split"""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    return torch.utils.data.random_split(dataset, [train_size, val_size])