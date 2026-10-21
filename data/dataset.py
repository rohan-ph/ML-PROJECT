"""
PyTorch Dataset classes for SNN violence detection.
Provides dataloaders for training on short clips and evaluating on long streams.
"""

import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class RWFClipDataset(Dataset):
    """
    Dataset for loading preprocessed RWF-2000 clips.
    Loads .npy files and applies temporal windowing and augmentations.
    """
    def __init__(self, root_dir: str, split: str = "train", encoding: str = "diff", 
                 clip_length: int = 64, augment: bool = False):
        self.root_dir = Path(root_dir) / split
        self.encoding = encoding
        self.clip_length = clip_length
        self.augment = augment
        
        self.samples = []
        
        # Load Fight (Label 1)
        fight_dir = self.root_dir / "Fight"
        if fight_dir.exists():
            for p in fight_dir.rglob(f"*_{encoding}.npy"):
                self.samples.append((p, 1))
                
        # Load NonFight (Label 0)
        nonfight_dir = self.root_dir / "NonFight"
        if nonfight_dir.exists():
            for p in nonfight_dir.rglob(f"*_{encoding}.npy"):
                self.samples.append((p, 0))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load numpy array [T, C, H, W]
        try:
            data = np.load(path)
        except Exception as e:
            # Fallback for corrupted files
            data = np.zeros((self.clip_length, 3, 112, 112), dtype=np.float32)
            
        T_total = data.shape[0]
        
        # Temporal crop
        if T_total > self.clip_length:
            if self.augment:
                start_t = random.randint(0, T_total - self.clip_length)
            else:
                start_t = (T_total - self.clip_length) // 2
            data = data[start_t : start_t + self.clip_length]
        elif T_total < self.clip_length:
            # Pad with zeros if too short
            pad = np.zeros((self.clip_length - T_total, *data.shape[1:]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
            
        # Convert to tensor
        tensor_data = torch.from_numpy(data) # [T, C, H, W]
        
        # Spatial Augmentations (applied consistently across all frames in the clip)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                tensor_data = F.hflip(tensor_data)
                
            # Random brightness
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                # Adjust brightness frame by frame or on whole tensor
                # torchvision's adjust_brightness expects [..., C, H, W] which works for [T, C, H, W]
                tensor_data = F.adjust_brightness(tensor_data, brightness_factor)
                
        return tensor_data, label


class SyntheticStreamDataset(Dataset):
    """
    Dataset for yielding frames sequentially from a synthetic stream,
    along with ground truth labels.
    """
    def __init__(self, stream_path: str, annotation_path: str):
        self.stream_path = stream_path
        self.annotation_path = annotation_path
        
        self.data = np.load(stream_path, mmap_mode='r') # Use mmap for large files
        self.T = self.data.shape[0]
        
        with open(annotation_path, 'r') as f:
            self.anno = json.load(f)
            
        # Create per-frame ground truth
        self.gt_labels = np.zeros(self.T, dtype=np.int64)
        for incident in self.anno.get('incidents', []):
            start = incident['start_frame']
            end = incident['end_frame']
            self.gt_labels[start:end+1] = 1
            
    def __len__(self):
        return self.T
        
    def __getitem__(self, idx):
        # Return frame and label
        frame = torch.from_numpy(np.array(self.data[idx])) # [C, H, W]
        label = self.gt_labels[idx]
        return frame, label


def get_dataloaders(config: dict, encoding: str = 'diff'):
    """
    Create and return train, val, and test dataloaders based on config.
    """
    data_dir = config.get("data", {}).get("processed_dir", "d:/projectzz/ML PROJ/snn-evidence/data/processed/rwf2000")
    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("data", {}).get("num_workers", 4)
    clip_length = config.get("data", {}).get("clip_length", 64)
    
    train_dataset = RWFClipDataset(
        root_dir=data_dir, split="train", encoding=encoding, 
        clip_length=clip_length, augment=True
    )
    
    val_dataset = RWFClipDataset(
        root_dir=data_dir, split="val", encoding=encoding, 
        clip_length=clip_length, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None
    )
    
    # We might not have a separate test set in RWF, typically val is used as test or split
    return {"train": train_loader, "val": val_loader, "test": val_loader}

if __name__ == "__main__":
    # Simple test
    print("Dataset module loaded successfully.")
