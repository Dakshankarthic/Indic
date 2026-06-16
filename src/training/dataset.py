import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF
import random

class ManuscriptDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Match images to their corresponding .npz masks
        self.samples = []
        for img_path in sorted(self.images_dir.glob("*.*")):
            if img_path.suffix.lower() in ['.jpg', '.png']:
                mask_path = self.masks_dir / f"{img_path.stem}.npz"
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))
                    
        print(f"Loaded {len(self.samples)} image-mask pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load Image (BGR to RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask (.npz format)
        mask_data = np.load(mask_path)
        mask = mask_data['mask'] # Shape: (H, W, 6)
        
        # Convert pixel values [0, 255] to binary [0.0, 1.0]
        mask = (mask > 127).astype(np.float32)

        # Basic transforms (Resize to 512x512 for training simplicity)
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # HWC to CHW format required by PyTorch
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
        mask = mask.transpose((2, 0, 1))

        # Convert to PyTorch tensors
        image_t = torch.tensor(image)
        mask_t = torch.tensor(mask)

        # Data Augmentations
        if self.transform:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image_t = TF.hflip(image_t)
                mask_t = TF.hflip(mask_t)

            # Random brightness/contrast
            image_t = TF.adjust_brightness(image_t, 1.0 + (random.random() - 0.5) * 0.4)
            image_t = TF.adjust_contrast(image_t, 1.0 + (random.random() - 0.5) * 0.4)

        return image_t, mask_t

if __name__ == '__main__':
    # Test dataset
    ds = ManuscriptDataset(r"D:\indic_challenge\training_data_pseudo\images", r"D:\indic_challenge\training_data_pseudo\masks", transform=True)
    if len(ds) > 0:
        img, mask = ds[0]
        print(f"Image tensor shape: {img.shape}")
        print(f"Mask tensor shape: {mask.shape}")

