#!/usr/bin/env python3
"""
Training script for AI-generated image detection.
Supports GenImage, DRCT, DiffusionDB datasets with timm models.
"""
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm


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
        img = Image.open(path).convert("RGB")
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
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a timm model with custom head."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = outputs.max(1)
            
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    return total_loss / total, acc, auc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train AI image detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "generator"], 
                       help="binary: real vs AI; generator: which model")
    parser.add_argument("--generators", nargs="+", default=["midjourney", "stable_diffusion", "dall_e", "other"],
                       help="Generator names for generator task")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data
    if args.task == "binary":
        train_ds = AIImageDataset(args.data_dir, "train", max_samples_per_class=args.max_samples)
        val_ds = AIImageDataset(args.data_dir, "val", max_samples_per_class=args.max_samples)
        num_classes = 2
    else:
        train_ds = GeneratorDataset(args.data_dir, "train", generator_names=args.generators)
        val_ds = GeneratorDataset(args.data_dir, "val", generator_names=args.generators)
        num_classes = len(args.generators)
    
    # Handle class imbalance
    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    # Model
    model = create_model(args.model, num_classes).to(device)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Training loop
    best_auc = 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_auc, preds, labels = evaluate(model, val_loader, criterion, device)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint_path = Path(args.output_dir) / f"best_{args.model}_{args.task}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "model_name": args.model,
                "task": args.task,
                "generators": args.generators if args.task == "generator" else None,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        scheduler.step()
    
    print(f"\nBest validation AUC: {best_auc:.4f}")
    
    # Final evaluation with classification report
    val_loss, val_acc, val_auc, preds, labels = evaluate(model, val_loader, criterion, device)
    target_names = ["real", "ai"] if args.task == "binary" else args.generators
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))


if __name__ == "__main__":
    main()