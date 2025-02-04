import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
from models.eth_spot_1hr_1day import ETHSpot1hr1day
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader, default_collate
import math

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_training_history(history, save_path):
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(3, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='orange')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training History')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot directional accuracy
    plt.subplot(3, 1, 2)
    plt.plot(history['val_dir_acc'], label='Directional Accuracy', color='blue')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate - fixed key name
    plt.subplot(3, 1, 3)
    plt.plot(history['lr'], label='Learning Rate', color='green')  # Changed from 'learning_rate' to 'lr'
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    sequence_vols = []
    
    # Calculate rolling statistics (30 days = 720 hours)
    window_size = 720
    
    # Calculate returns first for proper volatility
    returns = torch.log(data[1:, 3] / data[:-1, 3])
    vol = torch.zeros(len(data))
    
    # Simple volatility calculation
    for i in range(len(data)):
        start_idx = max(0, i - window_size)
        vol[i] = returns[start_idx:i+1].std() * math.sqrt(24*365) if i > 0 else 1e-8
    
    # Create sequences with per-sequence normalization
    start_idx = window_size
    for i in tqdm(range(start_idx, len(data) - seq_length - 1)):
        seq = data[i:i+seq_length]
        
        # Less aggressive normalization
        seq_mean = seq.mean(dim=0, keepdim=True)
        seq_std = seq.std(dim=0, keepdim=True) + 1e-6  # Slightly larger epsilon
        seq = (seq - seq_mean) / seq_std
        
        # Calculate return
        current_price = data[i + seq_length - 1, 3]
        future_price = data[i + seq_length, 3]
        ret = (future_price / current_price - 1) * 100
        
        sequences.append(seq.clone().detach())
        targets.append(ret)
        sequence_vols.append(vol[i + seq_length - 1])
    
    X = torch.stack(sequences)
    y = torch.tensor(targets, dtype=torch.float32)
    vol_tensor = torch.tensor(sequence_vols, dtype=torch.float32)
    
    # Use 85th percentile filtering (slightly more aggressive)
    q075, q925 = torch.quantile(y, torch.tensor([0.075, 0.925]))
    valid_mask = (y >= q075) & (y <= q925)
    print(f"Keeping {valid_mask.sum().item()}/{len(valid_mask)} samples ({valid_mask.sum().item()/len(valid_mask):.1%}) after filtering")
    
    return X[valid_mask], y[valid_mask], vol_tensor[valid_mask]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, choices=['ETH', 'BTC'], default='ETH',
                       help='Token to train on (ETH or BTC)')
    return parser.parse_args()

def train_model(data_path, token='ETH', seq_length=24, epochs=200):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
    
    # Token-specific filenames
    model_save_path = f'best_model_1day_{token.lower()}.pth'
    history_save_path = f'training_history_1day_{token.lower()}.png'
    
    # Model configuration with stability improvements
    config = {
        "seq_length": 24,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.45,
        "learning_rate": 0.0004,
        "weight_decay": 2e-5,
        "batch_size": 256,
        "epochs": 200,
        "gradient_clip": 0.8,
        "attention_heads": 1,
        "bidirectional": False,
        "patience": 25
    }
    
    def load_data(path):
        # Expecting CSV with columns: open, high, low, close, volume
        data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(5))
        return torch.tensor(data, dtype=torch.float32)
    
    print("Loading data...")
    raw_data = load_data(data_path)
    
    # Calculate training data cutoff (2 years total)
    training_hours = 2 * 365 * 24  # 2 years in hours
    training_data = raw_data[:training_hours]
    
    print("Creating sequences...")
    X, y, vol = create_sequences(training_data, config['seq_length'])
    
    # Split into train/val (1.5 years train, 0.5 years val)
    train_hours = int(1.5 * 365 * 24)  # 1.5 years in hours
    split_idx = int(len(X) * (train_hours / training_hours))  # Proportional split
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    vol_train, vol_val = vol[:split_idx], vol[split_idx:]  # Added volatility split
    
    # Add validation set checks
    if len(X_val) == 0:
        raise ValueError("Validation set is empty! Check your data splitting and filtering.")
    
    print(f"\nFinal dataset sizes:")
    print(f"- Training samples: {len(X_train)} (~1.5 years)")
    print(f"- Validation samples: {len(X_val)} (~6 months)")
    
    # Force MPS to use float32
    torch.set_default_dtype(torch.float32)
    
    # Set device
    device = (
        torch.device("mps") 
        if torch.backends.mps.is_available() 
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Data loading
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train, vol_train),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda x: tuple(t.to(device).float() for t in default_collate(x))
    )
    
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val, vol_val),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: tuple(t.to(device).float() for t in default_collate(x))
    )
    
    # Create model
    model = ETHSpot1hr1day(
        seq_length=config["seq_length"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        attention_heads=config["attention_heads"],
        bidirectional=config["bidirectional"]
    ).to(device)
    
    # Modified loss function with volatility weighting
    class VolWeightedHuberLoss(nn.Module):
        def __init__(self, delta=0.1):
            super().__init__()
            self.delta = delta
            
        def forward(self, pred, target, vol_weight):
            base_loss = F.huber_loss(pred.squeeze(), target, delta=self.delta, reduction='none')
            vol_weight = 1 / (1 + vol_weight)  # Inverse volatility weighting
            return (base_loss * vol_weight).mean()
    
    criterion = VolWeightedHuberLoss(delta=0.1).to(device)
    
    # Simple optimizer with constant learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),       # Default betas for stability
        eps=1e-8
    )
    
    # No learning rate scheduling - use constant rate
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,  # Effectively constant LR
        gamma=1.0        # No decay
    )
    
    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'val_dir_acc': []  # Added directional accuracy tracking
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (X_batch, y_batch, vol_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch, vol_batch)
            loss.backward()
            
            # Gradient clipping based on gradient norm
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if grad_norm > config["gradient_clip"]:
                print(f"Warning: gradient norm {grad_norm:.2f} exceeded clip value")
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation with directional accuracy
        model.eval()
        val_loss = 0
        correct_dirs = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch, vol_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch, vol_batch).item()
                
                # Calculate directional accuracy
                pred_dir = (pred.squeeze() > 0).float()
                true_dir = (y_batch > 0).float()
                correct_dirs += (pred_dir == true_dir).sum().item()
                total_samples += len(y_batch)
        
        val_loss /= len(val_loader)
        train_loss = total_loss / len(train_loader)
        dir_acc = (correct_dirs / total_samples) * 100  # as percentage
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['val_dir_acc'].append(dir_acc)
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Dir Acc: {dir_acc:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'dir_acc': dir_acc,
                'history': history
            }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Plot training history
    plot_training_history(history, history_save_path)
    
    return model

if __name__ == "__main__":
    args = parse_args()
    data_path = f'data/{args.token}USDT-1h-2022-08-06--2025-02-03.csv'
    train_model(data_path, token=args.token)