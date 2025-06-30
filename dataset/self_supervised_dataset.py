import random
from pathlib import Path

import numpy as np
import torch
from tifffile import imread
from torch.utils.data import DataLoader, Dataset

# --- Helper function for Self-Supervised Splitting ---

def split_checkerboard(volume: np.ndarray):
    """
    Splits a volume into two disjoint sets using a checkerboard pattern.
    This is a core component for self-supervised denoising (e.g., Noise2Void).
    """
    # A random choice of which pixels go to the source vs. target
    if random.choice([True, False]):
        source = volume[:, 0::2, 0::2]
        target = volume[:, 1::2, 1::2]
    else:
        source = volume[:, 0::2, 1::2]
        target = volume[:, 1::2, 0::2]
    return source, target

# --- Main Dataset Class ---

class SelfSupervisedDenoisingDataset(Dataset):
    """
    Dataset for self-supervised denoising of a single noisy vEM volume.
    It generates input/target pairs by splitting a noisy patch into two disjoint sets.
    """
    def __init__(self,
                 volume_path: Path,
                 is_training: bool,
                 patch_height: int = 128,
                 patch_width: int = 128,
                 patch_depth: int = 5,
                 num_samples_per_epoch: int = 1000
                ):
        
        super().__init__()
        self.is_training = is_training
        self.num_samples = num_samples_per_epoch if self.is_training else 100

        # Load the single noisy volume. For very large files, consider memory mapping.
        if ~volume_path.is_file():
            volume_path = list(volume_path.glob("*.tif"))[0]
            print(f"Input volume path is dir, use {str(volume_path)}.")
        self.volume = imread(volume_path)
        self.volume_shape = self.volume.shape
        
        # We fetch a patch twice the size in XY, as the checkerboard split
        # will downsample it by 2x in height and width.
        self.fetch_height = patch_height * 2
        self.fetch_width = patch_width * 2
        self.fetch_depth = patch_depth

    def __len__(self):
        return self.num_samples

    def _augment_patch(self, patch: np.ndarray):
        # On-the-fly data augmentation for a single patch
        if random.random() > 0.5: # Flip Z
            patch = patch[::-1].copy()
        if random.random() > 0.5: # Flip Y
            patch = patch[:, ::-1].copy()
        if random.random() > 0.5: # Flip X
            patch = patch[:, :, ::-1].copy()

        # Rotations in XY plane
        rotations = random.randint(0, 3)
        if rotations > 0:
            patch = np.rot90(patch, k=rotations, axes=(1, 2)).copy()
            
        return patch

    def __getitem__(self, index):
        # 1. Get random coordinates for a large patch
        z_start = np.random.randint(0, self.volume_shape[0] - self.fetch_depth)
        y_start = np.random.randint(0, self.volume_shape[1] - self.fetch_height)
        x_start = np.random.randint(0, self.volume_shape[2] - self.fetch_width)
        
        patch_coords = np.s_[z_start : z_start + self.fetch_depth,
                             y_start : y_start + self.fetch_height,
                             x_start : x_start + self.fetch_width]
        
        # 2. Extract a single noisy patch
        noisy_patch = self.volume[patch_coords]

        # 3. Apply data augmentation if in training mode
        if self.is_training:
            noisy_patch = self._augment_patch(noisy_patch)
        
        # 4. Normalize patch to [0, 1]
        noisy_patch = noisy_patch.astype(np.float32) / 255.0

        # 5. Split the noisy patch into a source/target pair for self-supervision
        source_patch, target_patch = split_checkerboard(noisy_patch)
        
        # 6. Convert to PyTorch tensors
        source_tensor = torch.from_numpy(np.ascontiguousarray(source_patch)).unsqueeze(0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_patch)).unsqueeze(0)
        
        return {"source": source_tensor, "target": target_tensor}

# --- Dataloader Function ---

def get_self_supervised_dataloader(volume_path: Path,
                                   mode: str,
                                   patch_size: tuple[int, int, int], # (depth, height, width)
                                   batch_size: int = 1,
                                   num_workers: int = 8,
                                   num_samples_per_epoch: int = 1000,
                                   ):
    is_training = (mode == 'train')
    shuffle = is_training # Shuffle only during training

    patch_depth, patch_height, patch_width = patch_size

    dataset = SelfSupervisedDenoisingDataset(
        volume_path=volume_path,
        is_training=is_training,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_depth=patch_depth,
        num_samples_per_epoch=num_samples_per_epoch
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )