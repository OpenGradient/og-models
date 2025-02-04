import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import math

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
from ray import train
from torch.utils.data import TensorDataset, DataLoader

# Import your model file
from models.eth_spot_1hr_1day import ETHSpot1Day

# Data loader and sequence creator functions (adapted from train_1day.py)
def load_data(path):
    if path is None:
        raise ValueError("data_path cannot be None")
    return torch.tensor(
        np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(5)), 
        dtype=torch.float32
    )

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - 1):
        seq = data[i:i+seq_length]
        # Standardize each sequence
        seq = (seq - seq.mean(axis=0, keepdims=True)) / (seq.std(axis=0, keepdims=True) + 1e-8)
        current_price = data[i + seq_length - 1, 3]
        future_price = data[i + seq_length, 3]
        returns = (future_price / current_price - 1) * 100
        sequences.append(torch.tensor(seq, dtype=torch.float32))
        targets.append(returns)
    X = torch.stack(sequences)
    y = torch.tensor(targets, dtype=torch.float32)
    
    # Remove extreme outliers (keep ~98% of samples)
    q01, q99 = torch.quantile(y, 0.01), torch.quantile(y, 0.99)
    valid_mask = (y > q01) & (y < q99)
    print(f"Keeping {valid_mask.sum().item()/len(valid_mask):.1%} of samples after filtering")
    return X[valid_mask], y[valid_mask]

def train_tune(config, checkpoint_dir=None, data_path=None):
    if data_path is None:
        raise ValueError("data_path must be provided to train_tune")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data (keep on CPU initially)
    print(f"Loading data from {data_path}")
    raw_data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=range(5))
    raw_data = torch.tensor(raw_data, dtype=torch.float32)
    
    # Normalize data more robustly
    print("Creating sequences...")
    sequences = []
    targets = []
    for i in range(len(raw_data) - config['seq_length'] - 1):
        seq = raw_data[i:i+config['seq_length']]
        # More robust normalization
        seq_mean = seq.mean(axis=0, keepdims=True)
        seq_std = seq.std(axis=0, keepdims=True)
        seq_std[seq_std < 1e-3] = 1.0  # Prevent division by very small numbers
        seq = (seq - seq_mean) / seq_std
        
        # Calculate returns with clipping to prevent extreme values
        current_price = raw_data[i + config['seq_length'] - 1, 3]
        future_price = raw_data[i + config['seq_length'], 3]
        returns = torch.clamp((future_price / current_price - 1.0) * 100, -10, 10)  # Clip extreme returns
        
        sequences.append(seq)
        targets.append(returns)
    
    X = torch.stack(sequences)
    y = torch.tensor(targets)
    
    # Split into train/val
    train_size = int(0.75 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets and dataloaders (data stays on CPU)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=4,  # Increased from 2
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Move validation data to device
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    # Enhanced model architecture
    class EnhancedETHSpot1Day(nn.Module):
        def __init__(self, seq_length, hidden_dim, num_layers, dropout, attention_heads=4, bidirectional=True):
            super().__init__()
            self.input_dim = 5  # OHLCV data
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # Bidirectional LSTM
            lstm_dropout = dropout if num_layers > 1 else 0.0  # Fix dropout for single layer
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
                bidirectional=bidirectional
            )
            
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
            
            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Output layers with skip connection
            self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)
            
            # Activation functions
            self.silu = nn.SiLU()
        
        def forward(self, x):
            # LSTM layer
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Self-attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Global average pooling
            pooled = torch.mean(attn_out, dim=1)
            
            # MLP with skip connection
            out = self.fc1(pooled)
            out = self.bn1(out)
            out = self.silu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out, attn_weights
    
    # Create model with config values directly (Ray Tune handles the sampling)
    model = EnhancedETHSpot1Day(
        seq_length=config["seq_length"],
        hidden_dim=config["hidden_dim"],      # Ray Tune provides the sampled value
        num_layers=config["num_layers"],      # Ray Tune provides the sampled value
        dropout=config["dropout"],            # Ray Tune provides the sampled value
        attention_heads=config["attention_heads"],
        bidirectional=config["bidirectional"]
    ).to(device)
    
    # Better initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    criterion = nn.HuberLoss(delta=0.1, reduction='mean')
    
    # Update GradScaler initialization
    scaler = torch.amp.GradScaler('cuda')
    
    # Use CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["learning_rate"] * 0.01
    )
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                pred, _ = model(batch_X)
                pred = pred.squeeze()
                loss = criterion(pred, batch_y)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            val_pred, _ = model(X_val)
            val_pred = val_pred.squeeze()
            val_loss = criterion(val_pred, y_val).item()
        
        metrics = {
            "training_loss": total_loss / max(num_batches, 1),
            "val_loss": val_loss,
            "epoch": epoch
        }
        
        train.report(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    # Convert to absolute path and validate
    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    print(f"Using data file: {data_path}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_path = os.path.abspath(os.path.join("out", f"tune_{timestamp}"))
    os.makedirs(storage_path, exist_ok=True)

    # Early stopping scheduler for efficiency
    scheduler = tune.schedulers.ASHAScheduler(
        max_t=100,
        grace_period=20,  # Increased from 15
        reduction_factor=2,
        brackets=3
    )

    ray.init(
        num_cpus=48,
        num_gpus=4,
        object_store_memory=14 * 1024 * 1024 * 1024,
        _memory=32 * 1024 * 1024 * 1024,
        _system_config={
            "object_spilling_threshold": 0.8,
            "object_store_full_delay_ms": 100,
        }
    )
    
    config = {
        "seq_length": 24,
        "hidden_dim": tune.choice([96, 128, 160]),
        "num_layers": tune.choice([1, 2]),
        "dropout": tune.uniform(0.35, 0.45),
        "learning_rate": tune.loguniform(7e-4, 1.2e-3),
        "weight_decay": tune.loguniform(1e-5, 3e-5),
        "batch_size": 256,
        "epochs": 100,
        "gradient_clip": tune.uniform(0.5, 0.9),
        "attention_heads": tune.choice([1, 2]),
        "bidirectional": True
    }

    analysis = tune.run(
        tune.with_parameters(train_tune, data_path=data_path),
        config=config,
        num_samples=50,
        scheduler=scheduler,
        resources_per_trial={
            "cpu": 3,
            "gpu": 0.2,
            "memory": 3 * 1024 * 1024 * 1024
        },
        max_concurrent_trials=20,
        storage_path=f"file://{storage_path}",
        name=f"tune_{timestamp}",
        keep_checkpoints_num=1,
        metric="val_loss",
        mode="min",
        verbose=2,
        reuse_actors=True,
        checkpoint_freq=5
    )

    # Print best trial details
    best_trial = analysis.best_trial
    print(f"\nBest trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    print(f"Best trial final training loss: {best_trial.last_result['training_loss']}") 