import torch
import torch.nn as nn
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from scipy import stats
from models.eth_spot_1hr_1day import ETHSpot1hr1day, EnhancedFeatureLayer
from datetime import datetime
import os
import torch.nn.functional as F
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, choices=['ETH', 'BTC'], default='ETH',
                       help='Token to evaluate (ETH or BTC)')
    return parser.parse_args()

def convert_model_to_onnx(pth_path, onnx_path, token='ETH'):
    # Match the exact config used in training
    config = {
        "seq_length": 24,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.45,
        "learning_rate": 0.0004,
        "weight_decay": 2e-5,
        "batch_size": 256,
        "gradient_clip": 0.8,
        "attention_heads": 1,
        "bidirectional": False,
        "tcn_channels": 128,
        "tcn_kernel_size": 3,
        "input_dim": 5,      
        "output_dim": 1,     
        "num_channels": [128, 128, 128, 128],
        "fc_sizes": [128, 64, 1],
        "epochs": 200,
        "patience": 25
    }
    
    # Create base model
    model = ETHSpot1hr1day(config)
    checkpoint = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Wrap with normalization
    normalized_model = NormalizedModel(model)
    normalized_model.eval()
    
    # Export with normalization included
    torch.onnx.export(
        normalized_model,
        torch.randn(24, 5),  # Example input
        onnx_path,
        input_names=['input'],
        output_names=['regression_output'],
        dynamic_axes={
            'input': {0: 'sequence_length'},
        },
        opset_version=17,
        do_constant_folding=True
    )
    
    print(f"Model exported to {onnx_path}")
    
    # Verify the model
    ort_session = onnxruntime.InferenceSession(onnx_path)
    test_input = torch.randn(24, 5).numpy().astype(np.float32)
    ort_inputs = {'input': test_input}
    ort_output = ort_session.run(['regression_output'], ort_inputs)[0]
    print(f"Test prediction shape: {ort_output.shape}")

def test_onnx_model(onnx_path='onnx/eth_forecast_1day_model.onnx'):
    """Test the ONNX model with a dummy input"""
    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    
    # Create dummy input (2D: sequence_length x features)
    dummy_input = np.random.randn(24, 5).astype(np.float32)
    
    # Add batch dimension for ONNX
    model_input = np.expand_dims(dummy_input, 0)
    
    # Run inference
    ort_inputs = {'input': model_input}
    price_pred, _ = session.run(['price_prediction', 'volatility'], ort_inputs)
    
    print("\nONNX Model Test Results:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Price prediction: {price_pred[0][0]:.4f}%")
    
    return True

def predict_single_sequence(sequence, onnx_path):
    """Make prediction for a single sequence using ONNX model."""
    # Expect [24,5] input
    if sequence.shape != (24, 5):
        raise ValueError(f"Expected shape (24, 5), got {sequence.shape}")
    
    # Normalize sequence (same as training)
    sequence = sequence.astype(np.float32)
    seq_mean = sequence.mean(axis=0, keepdims=True)
    seq_std = sequence.std(axis=0, keepdims=True)
    seq_std[seq_std < 1e-3] = 1.0  # Prevent division by very small numbers
    sequence = (sequence - seq_mean) / seq_std
    
    # Create ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Run inference
    ort_inputs = {session.get_inputs()[0].name: sequence}
    reg_pred = session.run(['regression_output'], ort_inputs)[0]
    return reg_pred.item()

def predict_sequences_batch(sequences, onnx_path, batch_size=256):
    """Make predictions for multiple sequences using batched ONNX inference."""
    sequences = sequences.astype(np.float32)
    predictions = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        # Normalize each sequence in batch (same as training)
        batch_mean = batch.mean(axis=1, keepdims=True)
        batch_std = batch.std(axis=1, keepdims=True)
        batch_std[batch_std < 1e-3] = 1.0  # Prevent division by very small numbers
        batch = (batch - batch_mean) / batch_std
        
        session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Process each sequence in batch individually
        batch_preds = []
        for seq in batch:
            ort_inputs = {session.get_inputs()[0].name: seq}
            pred = session.run(['regression_output'], ort_inputs)[0]
            batch_preds.append(pred.item())
        
        predictions.extend(batch_preds)
    
    return np.array(predictions)

def evaluate_model(data_path, onnx_path='onnx/eth_forecast_1day_model.onnx'):
    """Evaluate model performance on specified backtest periods"""
    print(f"Loading data from: {data_path}")
    data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=range(5))
    
    training_hours = 2 * 365 * 24  # 2 years (training + validation)
    backtest_hours = 6 * 30 * 24   # 6 months

    # Take backtest data immediately after training period
    backtest_data = data[training_hours:training_hours + backtest_hours]
    
    print(f"\n=== Backtesting on last 6 months ===")
    print(f"Backtest size: {len(backtest_data)} hours")
    
    sequences = []
    actual_returns = []
    timestamps = []
    prices = []
    
    # Create sequences, leaving room for next-day prediction
    for i in range(len(backtest_data) - 24):
        seq = backtest_data[i:i+24]
        current_price = seq[-1, 3]  # Close price
        next_price = backtest_data[i+24, 3]  # Next day's close
        actual_return = (np.log(next_price) - np.log(current_price)) * 100
        
        sequences.append(seq)
        actual_returns.append(actual_return)
        timestamps.append(i + 24)
        prices.append(current_price)
    
    sequences = np.array(sequences)
    print(f"Created {len(sequences)} sequences for evaluation")
    
    # Make predictions in batches
    print("Making predictions...")
    predictions = predict_sequences_batch(sequences, onnx_path)
    
    actual_returns = np.array(actual_returns)
    prices = np.array(prices)
    
    # Store results
    results = {
        'predictions': predictions,
        'actual_returns': actual_returns,
        'prices': prices,
        'timestamps': timestamps
    }
    
    # Print detailed statistics
    print("\nTrading Statistics:")
    print_statistics(predictions, actual_returns)
    
    # Print period-specific info
    start_date = "Period: Last 6 months of data"
    print(f"\n{start_date}")
    print(f"Number of trades: {len(predictions)}")
    print(f"Average prediction magnitude: {np.abs(predictions).mean():.4f}%")
    print(f"Average actual return magnitude: {np.abs(actual_returns).mean():.4f}%")
    
    plot_backtest_results(results)
    return results

def print_statistics(predictions, actual_returns):
    """Calculate and print trading statistics"""
    # Direction accuracy
    direction_accuracy = np.mean(np.sign(predictions) == np.sign(actual_returns)) * 100
    
    # Strategy returns
    position = np.sign(predictions)
    returns = position * actual_returns
    cumulative_returns = np.cumsum(returns)
    
    # Buy & Hold returns
    buy_hold_returns = np.cumsum(actual_returns)
    
    # Win rate
    win_rate = np.mean(returns > 0) * 100
    
    # Profit factor
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns))) if len(negative_returns) > 0 else float('inf')
    
    # Correlation
    correlation = np.corrcoef(predictions, actual_returns)[0, 1]
    
    # Print results
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"Strategy Return: {cumulative_returns[-1]:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_returns[-1]:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Prediction Correlation: {correlation:.3f}")
    
    # Additional metrics
    print(f"Max Drawdown: {calculate_max_drawdown(cumulative_returns):.2f}%")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(returns):.2f}")

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown percentage"""
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / (running_max + 100) * 100
    return np.max(drawdown)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365*24):
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    if len(excess_returns) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)

def plot_backtest_results(results, save_dir='plots', token='ETH'):
    """Plot and save backtest results for each period"""
    plt.ioff()  # Turn off interactive mode
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    strategy_returns = np.cumsum(np.sign(results['predictions']) * results['actual_returns'])
    buy_hold_returns = np.cumsum(results['actual_returns'])
    plt.plot(strategy_returns, label='Strategy', color='blue')
    plt.plot(buy_hold_returns, label='Buy & Hold', color='red')
    plt.title(f'Cumulative Returns - Last 6 months ({token})')
    plt.xlabel('Trade Number')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot prediction vs actual
    plt.subplot(2, 1, 2)
    plt.scatter(results['actual_returns'], results['predictions'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Prediction vs Actual Returns ({token})')
    plt.xlabel('Actual Returns (%)')
    plt.ylabel('Predicted Returns (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/backtest_results_6months_{token.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close('all')

class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [24, 5] -> [1, 24, 5]
            
        # Normalize across sequence dimension (dim=1 now due to batch dim)
        seq_mean = x.mean(dim=1, keepdim=True)
        seq_std = x.std(dim=1, keepdim=True)
        seq_std = torch.where(seq_std < 1e-3, torch.ones_like(seq_std), seq_std)
        x = (x - seq_mean) / seq_std
        
        # Forward through base model
        return self.base_model(x)

class Backtest:
    def __init__(self, model, features, returns, mask=None):
        """
        'returns' should be a Series of daily returns for each pool
        (typically computed from APY as a one-shot snapshot).
        """
        self.model = model
        self.features = features
        self.returns = returns
        self.mask = mask
        
    def run_backtest(self, initial_capital=10000, num_days=30):
        # Get portfolio allocation weights
        weights = self.model(self.features, self.mask)
        weights_np = weights.detach().numpy()
        if self.mask is not None:
            weights_np = weights_np * self.mask.numpy()
            weights_np = weights_np / (np.sum(weights_np) + 1e-6)
        # Compute the portfolio's daily return as the weighted sum of individual pool returns
        portfolio_daily_return = np.sum(weights_np * self.returns.values)
        
        # Simulate portfolio compounding over 'num_days'
        portfolio_value = initial_capital
        portfolio_values = [portfolio_value]
        for day in range(num_days):
            portfolio_value *= (1 + portfolio_daily_return)
            portfolio_values.append(portfolio_value)
        dates = pd.date_range(start=pd.Timestamp.today(), periods=num_days+1, freq='D')
        return pd.Series(portfolio_values, index=dates)

if __name__ == "__main__":
    args = parse_args()
    
    # Load token-specific model
    model_path = f'best_model_1day_{args.token.lower()}.pth'
    onnx_path = f'onnx/{args.token.lower()}_forecast_1day_model.onnx'
    data_path = f'data/{args.token}USDT-1h-2022-08-06--2025-02-03.csv'
    
    # Convert to ONNX
    convert_model_to_onnx(model_path, onnx_path, token=args.token)
    
    # Run backtest
    results = evaluate_model(data_path, onnx_path)
    
    # Plot results
    plot_backtest_results(results, token=args.token)