#!/usr/bin/env python
"""
Automated Pretraining for ERN EEGNet (within-subject) across subjects and random seeds.

For each subject (0-indexed), the processed ERN data is loaded using an 80/20 split (by ERNLoad),
then further split (75% training, 25% validation) for model selection.
Before training, labels are remapped: any label < 0 is set to 1, and all others to 0.
Each subject’s model is trained for a fixed number of epochs (default: 50) with a learning rate of 5e-4 and batch size 128.
This entire process is repeated for a number of random seeds (default: 10).

The pretrained model for each subject and seed is saved to:
  model/target/ERN/EEGNet/within/<subject>/<seed>/model.pt

Usage:
    python pretrain_ern_all.py --subjects 16 --seeds 10 --epochs 50 --batch_size 128 --lr 0.0005 --device cpu
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.data_loader import ERNLoad, split
from models import EEGNet

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def remap_labels(y):
    """
    Remap labels to binary:
      - if label < 0, then set to 1 (e.g., bad feedback)
      - else set to 0 (e.g., good feedback)
    """
    return np.where(y < 0, 1, 0)

def train_model_for_subject(subject, seed, epochs, batch_size, lr, device):
    """
    Pretrain EEGNet for one subject with the specified seed.
    Loads the subject’s data (using ERNLoad), splits into training and validation sets,
    remaps labels to binary, trains EEGNet, and returns the state dict of the best model.
    """
    print(f"\nPretraining subject {subject} with seed {seed} ...")
    set_seed(seed)
    
    # Load processed data for the subject (ERNLoad returns 80% training and 20% test)
    x_train, y_train, x_test, y_test = ERNLoad(id=subject, setup='within')
    # Further split training data into training (75%) and validation (25%)
    x_train, y_train, x_val, y_val = split(x_train, y_train, ratio=0.75)
    
    # Remap labels: if label < 0, set to 1; else set to 0.
    y_train = remap_labels(y_train)
    y_val   = remap_labels(y_val)
    
    # Convert numpy arrays to torch tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_val   = torch.from_numpy(x_val).float()
    y_val   = torch.from_numpy(y_val).long()
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    
    # Determine model parameters from data shape.
    # Expected shape: (num_trials, 1, channels, samples)
    n_classes = len(np.unique(y_train.numpy()))
    Chans = x_train.shape[2]    # e.g., 56 channels
    Samples = x_train.shape[3]  # e.g., 2721 samples
    
    # Instantiate EEGNet
    model = EEGNet(n_classes=n_classes,
                   Chans=Chans,
                   Samples=Samples,
                   kernLenght=64,
                   F1=4,
                   D=2,
                   F2=8,
                   dropoutRate=0.25)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        val_loss /= total
        val_acc = correct / total
        
        print(f"Subject {subject} Seed {seed} Epoch [{epoch+1}/{epochs}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the model state if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    return best_model_state

def main():
    parser = argparse.ArgumentParser(description="Automated Pretraining for ERN EEGNet (within-subject)")
    parser.add_argument('--subjects', type=int, default=16,
                        help="Number of subjects to pretrain (default: 16)")
    parser.add_argument('--seeds', type=int, default=10,
                        help="Number of random seeds per subject (default: 10)")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs per run (default: 50)")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: "cpu" or "cuda" (default: cpu)')
    args = parser.parse_args()
    
    device = args.device
    subjects = args.subjects
    seeds = args.seeds
    
    # Loop over each subject and each random seed
    for subj in range(subjects):
        for seed in range(seeds):
            best_model_state = train_model_for_subject(subj, seed, args.epochs, args.batch_size, args.lr, device)
            # Save the pretrained model for this subject and seed
            # Save path: model/target/ERN/EEGNet/within/<subject>/<seed>/model.pt
            model_save_path = os.path.join('model', 'target', 'ERN', 'EEGNet', 'within', str(subj), str(seed))
            os.makedirs(model_save_path, exist_ok=True)
            checkpoint_path = os.path.join(model_save_path, 'model.pt')
            torch.save(best_model_state, checkpoint_path)
            print(f"Subject {subj} Seed {seed}: Pretrained model saved to {checkpoint_path}\n")

if __name__ == '__main__':
    main()
