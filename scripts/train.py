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
        
        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            if precision == "fp16" and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps * images.size(0)
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
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if precision == "fp16":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            elif precision == "bf16":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
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
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="timm model name (efficientnet_b0, mobilenetv3_small_100, resnet50 for 4GB VRAM)")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "generator"], 
                       help="binary: real vs AI; generator: which model")
    parser.add_argument("--generators", nargs="+", default=["midjourney", "stable_diffusion", "dall_e", "other"],
                       help="Generator names for generator task")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU (4 for 4GB VRAM)")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (lower for less RAM)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone, train head only (saves VRAM)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size (192 saves VRAM)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if args.task == "binary":
        train_ds = AIImageDataset(args.data_dir, "train", transform=train_transform, max_samples_per_class=args.max_samples)
        val_ds = AIImageDataset(args.data_dir, "val", transform=val_transform, max_samples_per_class=args.max_samples)
        num_classes = 2
    else:
        train_ds = GeneratorDataset(args.data_dir, "train", transform=train_transform, generator_names=args.generators)
        val_ds = GeneratorDataset(args.data_dir, "val", transform=val_transform, generator_names=args.generators)
        num_classes = len(args.generators)
    
    # Handle class imbalance
    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, 
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
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
    
    criterion = nn.CrossEntropyLoss()
    # Only optimize trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision
    use_amp = device.type == "cuda" and args.precision != "fp32"
    scaler = torch.cuda.amp.GradScaler() if use_amp and args.precision == "fp16" else None
    if args.precision == "bf16" and device.type == "cuda":
        # bfloat16 doesn't need GradScaler
        print("Using bfloat16 mixed precision")
    elif use_amp:
        print("Using fp16 mixed precision with GradScaler")
    
    # Training loop
    best_auc = 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    accum_steps = max(1, args.accum_steps)
    effective_batch = args.batch_size * accum_steps
    print(f"Effective batch size: {effective_batch} (batch={args.batch_size} x accum={accum_steps})")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accum_steps, args.precision)
        val_loss, val_acc, val_auc, preds, labels = evaluate(model, val_loader, criterion, device, args.precision)
        
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
    val_loss, val_acc, val_auc, preds, labels = evaluate(model, val_loader, criterion, device, args.precision)
    target_names = ["real", "ai"] if args.task == "binary" else args.generators
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names))


if __name__ == "__main__":
    main()