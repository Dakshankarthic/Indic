import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
from pathlib import Path

from unet_model import UNet
from dataset import ManuscriptDataset

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        inputs_sig = torch.sigmoid(inputs)       
        
        inputs_sig = inputs_sig.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs_sig * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_sig.sum() + targets.sum() + smooth)  
        
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    base_dir = Path(__file__).resolve().parents[2]
    parser.add_argument('--data_dir', type=str, default=str(base_dir / 'training_data_pseudo'))
    parser.add_argument('--save_dir', type=str, default=str(base_dir / 'models' / 'unet'))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_dir}...")
    full_dataset = ManuscriptDataset(img_dir, mask_dir, transform=True)
    
    if len(full_dataset) == 0:
        print("No training data found. Please run run_pseudo_label_pipeline.py first.")
        return

    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(n_channels=3, n_classes=6, bilinear=False).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = DiceBCELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / "unet_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best model to {save_path}")

if __name__ == "__main__":
    train()
