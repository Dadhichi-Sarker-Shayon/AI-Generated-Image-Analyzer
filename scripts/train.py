#!/usr/bin/env python3
"""
Training script for AI-generated image detection.
Supports GenImage, DRCT, DiffusionDB datasets with timm models.
"""
from __future__ import annotations
import argparse
import os
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
import pandas as pd
import io


class AIImageDataset(Dataset):
    """Dataset for AI vs Real classification."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        self.samples = []
        self.labels = []  # 0 = real, 1 = AI-generated
        
        self._load_samples(max_samples_per_class)
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_samples(self, max_per_class: Optional[int]):
        """Load samples from directory structure:
        root/
          real/
            *.jpg
          ai/
            *.jpg
        Or with generator subdirs:
        root/
          real/
          ai/
            midjourney/
            stable_diffusion/
            dall_e/
            ...
        """
        real_dir = self.root_dir / self.split / "real"
        ai_dir = self.root_dir / self.split / "ai"
        
        real_files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        ai_files = list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
        
        # Also check generator subdirectories
        if ai_dir.exists():
            for gen_dir in ai_dir.iterdir():
                if gen_dir.is_dir():
                    ai_files.extend(list(gen_dir.glob("*.jpg")))
                    ai_files.extend(list(gen_dir.glob("*.png")))
        
        if max_per_class:
            real_files = real_files[:max_per_class]
            ai_files = ai_files[:max_per_class]
        
        self.samples = [(f, 0) for f in real_files] + [(f, 1) for f in ai_files]
        np.random.shuffle(self.samples)
        print(f"Loaded {len(real_files)} real, {len(ai_files)} AI images for {self.split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Skip corrupted images - return black image
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class GeneratorDataset(Dataset):
    """Dataset for model-specific classification (which generator)."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        generator_names: Optional[list[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        self.generator_names = generator_names or ["midjourney", "stable_diffusion", "dall_e", "other"]
        self.gen_to_idx = {name: i for i, name in enumerate(self.generator_names)}
        self.samples = []
        self._load_samples()
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_samples(self):
        ai_dir = self.root_dir / self.split / "ai"
        if not ai_dir.exists():
            return
        
        for gen_dir in ai_dir.iterdir():
            if gen_dir.is_dir():
                gen_name = gen_dir.name.lower()
                if gen_name in self.gen_to_idx:
                    idx = self.gen_to_idx[gen_name]
                    for img_file in gen_dir.glob("*.jpg"):
                        self.samples.append((img_file, idx))
                    for img_file in gen_dir.glob("*.png"):
                        self.samples.append((img_file, idx))
        
        np.random.shuffle(self.samples)
        print(f"GeneratorDataset {self.split}: {len(self.samples)} samples across {len(self.generator_names)} generators")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class ParquetDataset(Dataset):
    """Dataset for loading images from parquet files with image bytes and labels."""
    
    def __init__(
        self,
        parquet_dir: str,
        split: str = "train",
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        self.max_samples_per_class = max_samples_per_class
        
        # Find parquet files
        if split == "train":
            pattern = "train-*.parquet"
        else:
            pattern = "validation-*.parquet"
        
        self.parquet_files = sorted(self.parquet_dir.glob(pattern))
        if not self.parquet_files:
            raise FileNotFoundError(f"No {pattern} files found in {parquet_dir}")
        
        # Load metadata (row counts) for each file
        self.file_offsets = []
        self.file_labels = []
        self.total_samples = 0
        self._build_index(max_samples_per_class)
        
        print(f"ParquetDataset {split}: {self.total_samples} samples from {len(self.parquet_files)} files")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _build_index(self, max_per_class: Optional[int]):
        """Build index of all samples across parquet files."""
        real_count = 0
        ai_count = 0
        max_each = max_per_class if max_per_class else float('inf')
        
        for pf in self.parquet_files:
            df = pd.read_parquet(pf, columns=['label'])
            labels = df['label'].values
            
            # Filter based on max_per_class
            real_mask = (labels == 0)
            ai_mask = (labels == 1)
            
            if max_per_class:
                # Need to track counts per class
                real_indices = np.where(real_mask)[0]
                ai_indices = np.where(ai_mask)[0]
                
                # Take up to max_per_class from each
                real_take = real_indices[:max(0, int(max_each) - real_count)]
                ai_take = ai_indices[:max(0, int(max_each) - ai_count)]
                
                if len(real_take) == 0 and len(ai_take) == 0:
                    continue
                
                keep_indices = np.sort(np.concatenate([real_take, ai_take]))
                offset_start = self.total_samples
                self.file_offsets.append((self.total_samples, pf, keep_indices))
                self.total_samples += len(keep_indices)
                real_count += len(real_take)
                ai_count += len(ai_take)
            else:
                offset_start = self.total_samples
                self.file_offsets.append((offset_start, pf, np.arange(len(labels))))
                self.total_samples += len(labels)
                real_count += int(real_mask.sum())
                ai_count += int(ai_mask.sum())
        
        print(f"Loaded {self.total_samples} samples ({real_count} real, {ai_count} AI)")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which file this index belongs to
        for offset, pf, keep_indices in self.file_offsets:
            if idx < offset + len(keep_indices):
                local_idx = idx - offset
                row_idx = keep_indices[local_idx]
                
                # Load the specific row
                df = pd.read_parquet(pf, columns=['image', 'label'])
                row = df.iloc[row_idx]
                
                label = int(row['label'])
                img_bytes = row['image']['bytes']
                
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                except Exception:
                    # Skip corrupted images - return black image
                    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
                
                if self.transform:
                    img = self.transform(img)
                
                return img, torch.tensor(label, dtype=torch.long)
        
        raise IndexError(f"Index {idx} out of range")


# ============================================================
# Advanced Augmentations and Loss Functions
# ============================================================

class MixUp:
    """MixUp augmentation for regularization."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """CutMix augmentation for regularization."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(images.size(2))
        cy = np.random.randint(images.size(3))
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss."""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        
        with torch.no_grad():
            targets = targets.view(-1, 1)
            smooth_targets = torch.zeros_like(log_probs).scatter_(1, targets, 1 - self.smoothing)
            smooth_targets += self.smoothing / (n_classes - 1)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RandAugment:
    """RandAugment for stronger data augmentation."""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augmentations = [
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    
    def __call__(self, img):
        ops = random.sample(self.augmentations, self.n)
        for op in ops:
            img = op(img)
        return img


def get_advanced_transform(img_size=224, use_randaugment=False):
    train_transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
    
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transforms.Compose(train_transform_list)


# Loss function factory
def get_criterion(loss_type='ce', label_smoothing=0.0, focal_gamma=2.0, focal_alpha=1.0):
    """Get loss criterion based on type."""
    if loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(smoothing=label_smoothing)
    elif loss_type == 'ce':
        if label_smoothing > 0:
            return LabelSmoothingLoss(smoothing=label_smoothing)
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a timm model with custom head."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, accum_steps=1, precision="fp16"):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for step, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Mixed precision
        if precision == "fp16" and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss = loss / accum_steps
            scaler.scale(loss).backward()
        elif precision == "bf16":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()
        
        # Gradient clipping to prevent NaN
        if precision != "fp16" or scaler is None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            if precision == "fp16" and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def train_epoch_with_aug(model, loader, criterion, optimizer, device, scaler=None, accum_steps=1, precision="fp16", mixup=None, cutmix=None, mixup_prob=0.5, cutmix_prob=0.5):
    """Training epoch with MixUp and CutMix support."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for step, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Apply MixUp or CutMix
        use_mixup = mixup is not None and np.random.random() < 0.5
        use_cutmix = cutmix is not None and np.random.random() < 0.5 and not use_mixup
        
        if use_mixup:
            images, labels_a, labels_b, lam = mixup(images, labels)
        elif use_cutmix:
            images, labels_a, labels_b, lam = cutmix(images, labels)
        else:
            labels_a = labels_b = labels
            lam = 1.0
        
        # Mixed precision
        if precision == "fp16" and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                if use_mixup or use_cutmix:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            loss = loss / accum_steps
            scaler.scale(loss).backward()
        elif precision == "bf16":
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                if use_mixup or use_cutmix:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
        else:
            outputs = model(images)
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
        
        # Gradient clipping
        if precision != "fp16" or scaler is None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            if precision == "fp16" and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps * images.size(0)
        
        # For accuracy with MixUp/CutMix, use original labels
        if use_mixup or use_cutmix:
            # Use dominant label for accuracy
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(labels_a).float() + (1 - lam) * preds.eq(labels_b).float()).sum().item()
        else:
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, precision="fp16"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_probs_full = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if precision == "fp16":
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            elif precision == "bf16":
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
            
            total_loss += loss.item() * images.size(0)
            probs_full = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            probs = probs_full[:, 1]
            
            # Check for NaN in probs
            if np.isnan(probs).any():
                probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                probs_full = np.nan_to_num(probs_full, nan=1.0 / probs_full.shape[1])
            
            _, preds = outputs.max(1)
            
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_probs_full.append(probs_full)
    
    acc = correct / total
    if len(set(all_labels)) <= 1:
        auc = 0
    elif outputs.shape[1] > 2:  # multi-class: macro one-vs-rest AUC
        pf = np.concatenate(all_probs_full)
        pf = pf / pf.sum(axis=1, keepdims=True)
        auc = roc_auc_score(all_labels, pf, multi_class="ovr", average="macro", labels=list(range(pf.shape[1])))
    else:
        auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, acc, auc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train AI image detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--model", type=str, default="mobilenetv3_small_100", help="timm model name (efficientnet_b0, mobilenetv3_small_100, resnet50 for 4GB VRAM)")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "generator"], 
                       help="binary: real vs AI; generator: which model")
    parser.add_argument("--generators", nargs="+", default=["midjourney", "stable_diffusion", "dall_e", "other"],
                       help="Generator names for generator task")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="Draw this many samples per epoch (shorter epochs = more frequent checkpoints)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU (8 for 4GB VRAM with accum)")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for no multiprocessing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone initially, train head only (saves VRAM)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--img-size", type=int, default=192, help="Input image size (192 saves VRAM)")
    # New arguments for parquet and checkpoint management
    parser.add_argument("--parquet-format", action="store_true", help="Use parquet format dataset (train-*.parquet, validation-*.parquet)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--early-stopping-metric", type=str, default="auc", choices=["auc", "acc", "loss"], help="Metric for early stopping")
    # New arguments for advanced features
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal", "label_smoothing"], help="Loss function type")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal-alpha", type=float, default=1.0, help="Focal loss alpha")
    parser.add_argument("--use-mixup", action="store_true", help="Enable MixUp augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp alpha parameter")
    parser.add_argument("--use-cutmix", action="store_true", help="Enable CutMix augmentation")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix alpha parameter")
    parser.add_argument("--use-randaugment", action="store_true", help="Enable RandAugment")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--use-focal", action="store_true", help="Use focal loss")
    parser.add_argument("--lr-warmup-epochs", type=int, default=2, help="Learning rate warmup epochs")
    parser.add_argument("--unfreeze-epoch", type=int, default=5, help="Epoch to unfreeze backbone")
    parser.add_argument("--use-reduce-lr", action="store_true", help="Use ReduceLROnPlateau scheduler")
    parser.add_argument("--use-warmup", action="store_true", help="Use learning rate warmup")
    parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying MixUp")
    parser.add_argument("--cutmix-prob", type=float, default=0.5, help="Probability of applying CutMix")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data transforms
    train_transform = get_advanced_transform(
        img_size=args.img_size,
        use_randaugment=args.use_randaugment
    )
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if args.parquet_format:
        # Use parquet dataset
        train_ds = ParquetDataset(args.data_dir, "train", transform=train_transform, max_samples_per_class=args.max_samples)
        val_ds = ParquetDataset(args.data_dir, "val", transform=val_transform, max_samples_per_class=args.max_samples)
        num_classes = 2
    elif args.task == "binary":
        train_ds = AIImageDataset(args.data_dir, "train", transform=train_transform, max_samples_per_class=args.max_samples)
        val_ds = AIImageDataset(args.data_dir, "val", transform=val_transform, max_samples_per_class=args.max_samples)
        num_classes = 2
    else:
        train_ds = GeneratorDataset(args.data_dir, "train", transform=train_transform, generator_names=args.generators)
        val_ds = GeneratorDataset(args.data_dir, "val", transform=val_transform, generator_names=args.generators)
        num_classes = len(args.generators)
    
    # Handle class imbalance - use WeightedRandomSampler for all datasets
    if hasattr(train_ds, 'samples'):
        labels = [s[1] for s in train_ds.samples]
    else:
        # For ParquetDataset, compute labels from the dataset
        # We'll use a balanced sampler since parquet is already balanced
        labels = None
    
    if labels is not None:
        class_counts = np.bincount(labels)
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(sample_weights, args.samples_per_epoch or len(sample_weights), replacement=True)
    else:
        sampler = None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, 
                             num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=(sampler is None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    # Model
    model = create_model(args.model, num_classes, pretrained=True).to(device)
    
    # Freeze backbone if requested (for low VRAM)
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name and "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen, training head only")
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Gradient checkpointing
    if args.grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        print("Gradient checkpointing enabled")
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")
    
    # Loss function
    if args.use_focal:
        criterion = get_criterion('focal', focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha)
    elif args.label_smoothing > 0:
        criterion = get_criterion('label_smoothing', label_smoothing=args.label_smoothing)
    else:
        criterion = get_criterion('ce', label_smoothing=args.label_smoothing)
    
    # Only optimize trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    if args.use_warmup:
        def lr_lambda(epoch):
            if epoch < args.lr_warmup_epochs:
                return float(epoch + 1) / float(max(1, args.lr_warmup_epochs))
            return 0.5 * (1 + np.cos(np.pi * (epoch - args.lr_warmup_epochs) / (args.epochs - args.lr_warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.use_reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision
    use_amp = device.type == "cuda" and args.precision != "fp32"
    scaler = torch.amp.GradScaler('cuda') if use_amp and args.precision == "fp16" else None
    if args.precision == "bf16" and device.type == "cuda":
        print("Using bfloat16 mixed precision")
    elif use_amp:
        print("Using fp16 mixed precision with GradScaler")
    
    # MixUp and CutMix
    mixup = MixUp(alpha=args.mixup_alpha) if args.use_mixup else None
    cutmix = CutMix(alpha=args.cutmix_alpha) if args.use_cutmix else None
    
    # Training loop setup
    best_auc = 0
    best_loss = float('inf')
    best_acc = 0
    epochs_no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    accum_steps = max(1, args.accum_steps)
    effective_batch = args.batch_size * accum_steps
    print(f"Effective batch size: {effective_batch} (batch={args.batch_size} x accum={accum_steps})")
    
    start_epoch = 0
    # Resume logic
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_auc" in checkpoint:
            best_auc = checkpoint["best_auc"]
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            for _ in range(start_epoch):
                scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")
    
    # Initialize MixUp and CutMix
    mixup = MixUp(alpha=args.mixup_alpha) if args.use_mixup else None
    cutmix = CutMix(alpha=args.cutmix_alpha) if args.use_cutmix else None
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Unfreeze backbone at specified epoch
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            print(f"\nUnfreezing backbone at epoch {epoch+1}...")
            for param in model.parameters():
                param.requires_grad = True
            # Recreate optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay
            )
            # Recreate scheduler
            if args.use_warmup:
                def lr_lambda(epoch):
                    if epoch < args.lr_warmup_epochs:
                        return float(epoch + 1) / float(max(1, args.lr_warmup_epochs))
                    return 0.5 * (1 + np.cos(np.pi * (epoch - args.lr_warmup_epochs) / (args.epochs - args.lr_warmup_epochs)))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            elif args.use_reduce_lr:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=3, verbose=True
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            print("Backbone unfrozen, full fine-tuning enabled")
        
        # Training with MixUp/CutMix support
        train_loss, train_acc = train_epoch_with_aug(
            model, train_loader, criterion, optimizer, device, scaler, 
            accum_steps, args.precision, mixup, cutmix, 
            args.mixup_prob, args.cutmix_prob
        )
        val_loss, val_acc, val_auc, preds, labels = evaluate(model, val_loader, criterion, device, args.precision)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
        
        # Determine improvement
        improved = False
        if args.early_stopping_metric == "auc" and val_auc > best_auc:
            best_auc = val_auc
            improved = True
        elif args.early_stopping_metric == "loss" and val_loss < best_loss:
            best_loss = val_loss
            improved = True
        elif args.early_stopping_metric == "acc" and val_acc > best_acc:
            best_acc = val_acc
            improved = True
        
        if improved:
            epochs_no_improve = 0
            # Save best model
            checkpoint_path = Path(args.output_dir) / f"best_{args.model}_{args.task}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_auc": best_auc,
                "best_loss": best_loss,
                "best_acc": best_acc,
                "model_name": args.model,
                "task": args.task,
                "generators": args.generators if args.task == "generator" else None,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        # Save checkpoint at interval
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = Path(args.output_dir) / f"epoch_{epoch+1}_{args.model}_{args.task}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_name": args.model,
                "task": args.task,
                "generators": args.generators if args.task == "generator" else None,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Always save last checkpoint for resume
        last_path = Path(args.output_dir) / f"last_{args.model}_{args.task}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_auc": val_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_auc": best_auc,
            "best_loss": best_loss,
            "best_acc": best_acc,
            "epochs_no_improve": epochs_no_improve,
            "model_name": args.model,
            "task": args.task,
            "generators": args.generators if args.task == "generator" else None,
        }, last_path)
        
        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement")
            break
        
        scheduler.step()
    
    print(f"\nBest validation AUC: {best_auc:.4f}")
    print(f"Best validation Loss: {best_loss:.4f}")
    print(f"Best validation Acc: {best_acc:.4f}")
    
    # Final evaluation with classification report
    val_loss, val_acc, val_auc, preds, labels = evaluate(model, val_loader, criterion, device, args.precision)
    target_names = ["real", "ai"] if args.task == "binary" else args.generators
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))


if __name__ == "__main__":
    main()