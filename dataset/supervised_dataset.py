import random
from pathlib import Path

import numpy as np
import torch
from tifffile import imread
from torch.utils.data import DataLoader, Dataset

class SupervisedDenoisingDataset(Dataset):
    """
    Dataset for supervised denoising of vEM images.
    It pairs a noisy 'source' volume (fast scan) with a clean 'target' volume (slow scan).
    """
    def __init__(self,
                 slow_scan_path: Path,
                 fast_scan_path: Path,
                 is_training: bool,
                 patch_height: int = 128,
                 patch_width: int = 128,
                 patch_depth: int = 5,
                 use_fast_slow_emulation: bool = False,
                 num_samples_per_epoch: int = 1000
                ):
        
        super().__init__()
        self.is_training = is_training
        self.use_fast_slow_emulation = use_fast_slow_emulation
        self.num_samples = num_samples_per_epoch if self.is_training else 100

        # Load volumes. For very large files, consider memory mapping:
        # self.slow_volume = imread(slow_scan_path, aszarr=True)
        if ~slow_scan_path.is_file():
            slow_scan_path = list(slow_scan_path.glob("cn-*.tif"))[0]
            print(f"Input slow volume path is dir, use {str(slow_scan_path)}.")
        if ~fast_scan_path.is_file():
            fast_scan_path = list(fast_scan_path.glob("n-*.tif"))[0]
            print(f"Input fast volume path is dir, use {str(fast_scan_path)}.")
        self.target_volume = imread(slow_scan_path)
        self.source_volume = imread(fast_scan_path)

        assert self.target_volume.shape == self.source_volume.shape, "Source and target volumes must have the same shape."
        assert self.target_volume.dtype == self.source_volume.dtype, "Source and target volumes must have the same dtype."
        
        self.volume_shape = self.target_volume.shape
        
        # If emulating the fast-slow strategy, we need to fetch a larger patch
        # because it will be downsampled by 2x later.
        if self.use_fast_slow_emulation:
            self.fetch_height = patch_height * 2
            self.fetch_width = patch_width * 2
            self.fetch_depth = patch_depth * 2
        else:
            self.fetch_height = patch_height
            self.fetch_width = patch_width
            self.fetch_depth = patch_depth

    def __len__(self):
        return self.num_samples

    def _augment_patches(self, source, target):
        # On-the-fly data augmentation
        if random.random() > 0.5: # Flip Z
            source, target = source[::-1].copy(), target[::-1].copy()
        if random.random() > 0.5: # Flip Y
            source, target = source[:, ::-1].copy(), target[:, ::-1].copy()
        if random.random() > 0.5: # Flip X
            source, target = source[:, :, ::-1].copy(), target[:, :, ::-1].copy()

        # Rotations in XY plane
        rotations = random.randint(0, 3)
        if rotations > 0:
            source = np.rot90(source, k=rotations, axes=(1, 2)).copy()
            target = np.rot90(target, k=rotations, axes=(1, 2)).copy()
            
        return source, target

    def __getitem__(self, index):
        # 1. Get random coordinates for a patch
        z_start = np.random.randint(0, self.volume_shape[0] - self.fetch_depth)
        y_start = np.random.randint(0, self.volume_shape[1] - self.fetch_height)
        x_start = np.random.randint(0, self.volume_shape[2] - self.fetch_width)
        
        patch_coords = np.s_[z_start : z_start + self.fetch_depth,
                             y_start : y_start + self.fetch_height,
                             x_start : x_start + self.fetch_width]
        
        # 2. Extract patches and normalize
        target_patch = self.target_volume[patch_coords].astype(np.float32) / 255.0
        source_patch = self.source_volume[patch_coords].astype(np.float32) / 255.0

        # 3. (Optional) Emulate the fast-slow acquisition pattern
        if self.use_fast_slow_emulation:
            # This block simulates the 'S-F-F-...-F-S' pattern from a fully sampled volume pair
            # It downsamples the patch by 2x in XY
            slow_subsampled = target_patch[:, 0::2, 0::2].copy()
            
            # The target is an average of neighboring pixels from the original slow patch
            target_patch = (target_patch[:, 0::2, 1::2] + 
                            target_patch[:, 1::2, 0::2] + 
                            target_patch[:, 1::2, 1::2]) / 3.0
            
            # The source is the downsampled fast patch...
            source_patch = source_patch[:, 0::2, 0::2].copy()
            # ...with its first and last frames replaced by slow frames
            source_patch[0] = slow_subsampled[0]
            source_patch[-1] = slow_subsampled[-1]

        # 4. Apply data augmentation if in training mode
        if self.is_training:
            source_patch, target_patch = self._augment_patches(source_patch, target_patch)
        
        # 5. Convert to PyTorch tensors
        # Ensure contiguous memory layout before converting
        source_tensor = torch.from_numpy(np.ascontiguousarray(source_patch)).unsqueeze(0)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_patch)).unsqueeze(0)
        
        return {"source": source_tensor, "target": target_tensor}

def get_supervised_dataloader(slow_scan_path: Path,
                              fast_scan_path: Path,
                              mode: str,
                              patch_size: tuple[int, int, int], # (depth, height, width)
                              batch_size: int = 1,
                              num_workers: int = 8,
                              use_fast_slow_emulation: bool = False,
                              num_samples_per_epoch: int = 1000,
                              ):
    is_training = (mode == 'train')
    
    patch_depth, patch_height, patch_width = patch_size

    dataset = SupervisedDenoisingDataset(
        slow_scan_path=slow_scan_path,
        fast_scan_path=fast_scan_path,
        is_training=is_training,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_depth=patch_depth,
        use_fast_slow_emulation=use_fast_slow_emulation,
        num_samples_per_epoch=num_samples_per_epoch
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )