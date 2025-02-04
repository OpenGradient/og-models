# BTC 1-Hour Price Direction Forecasting Model

## Overview
A deep learning model for predicting BTC price movement direction over 1-hour horizons using OHLCV data. The model uses a TCN architecture with direct price data input.

## Model Performance

### Backtest Results
![6-Month Backtest Results](@plots/backtest_results_6months_btc.png)
- Cumulative strategy returns vs buy & hold
- Directional accuracy visualization
- Trade-by-trade performance

### Training History
![Training History](@plots/training_history_1day_btc.png)
- Loss convergence curves
- Validation metrics
- Model optimization progress

## Key Features
- 1-hour forecasting using 24-hour historical data
- TCN (Temporal Convolutional Network) architecture
- Raw OHLCV input support
- ONNX optimized for production deployment

## Input/Output Specifications

### Input Dimensions
- Shape: (24, 5)
- Type: float32
- Sequence Length: 24 (hours)
- Features per timestep: 5 (OHLCV)

### Example Raw Input
```json
{
  "input": [
    [97994.47, 97994.47, 96543.10, 97700.59, 2756.60172],
    [97798.01, 97977.02, 96573.47, 97020.54, 1618.13720],
    [97020.56, 98022.20, 96150.00, 97994.46, 2148.57890],
    [98273.17, 98628.74, 97650.54, 97798.00, 983.70523],
    [97087.92, 98700.00, 96842.33, 98273.16, 2181.85188],
    [97228.86, 97931.65, 96821.54, 97087.92, 2847.73150],
    [98264.01, 98580.38, 97019.00, 97228.85, 3212.44953],
    [99346.87, 99361.97, 98075.02, 98264.00, 1453.70445],
    [99205.08, 99405.74, 98720.55, 99346.87, 905.88368],
    [98795.89, 99439.05, 98705.88, 99205.08, 1152.42229],
    [98178.00, 98827.18, 98160.00, 98795.88, 1755.25944],
    [98864.60, 99166.19, 98157.00, 98178.00, 1451.82051],
    [99417.07, 99604.00, 98645.00, 98864.60, 1494.73706],
    [99690.78, 99694.72, 99200.00, 99417.06, 979.95157],
    [99831.05, 99864.02, 99352.94, 99690.78, 714.66479],
    [99643.30, 100100.00, 99315.91, 99831.04, 917.86988],
    [100265.15, 100277.54, 99565.60, 99643.30, 864.58592],
    [100259.53, 100515.99, 100094.69, 100265.15, 652.60422],
    [99538.60, 100342.69, 99500.00, 100259.53, 714.22872],
    [99206.85, 99884.00, 99031.19, 99538.60, 1247.36452],
    [100452.16, 100755.93, 99148.91, 99206.86, 2872.03426],
    [101248.38, 101271.32, 100425.00, 100452.16, 711.44742],
    [101243.86, 101456.60, 101165.10, 101248.38, 356.77551],
    [100635.66, 101288.51, 100500.00, 101243.86, 625.08919]
  ]
}
```

Each row contains:
1. Open price (USD)
2. High price (USD)
3. Low price (USD)
4. Close price (USD)
5. Volume (BTC)

### Output
- Shape: (1,)
- Type: float32
- Meaning: Predicted percentage return for next hour
- Example: -0.24 means -0.24% predicted decline

### Example Output Interpretation
```python
predicted_return = -0.24  # Model output
current_price = 42081.19  # Last close price
predicted_price = current_price * (1 + predicted_return/100)
predicted_change = predicted_price - current_price

print(f"Current Price: ${current_price:.2f}")
print(f"Predicted Next Hour: ${predicted_price:.2f}")
print(f"Predicted Change: ${predicted_change:.2f}")
```

## Model Architecture
From backtest.py config:
```python
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
    "input_dim": 5,      # OHLCV
    "output_dim": 1      # Single regression output
}
```

## Model Components
1. **TCN Architecture**:
   - 4 temporal blocks with dilations (2,4,8,16)
   - Kernel size: 3
   - Batch normalization
   - SiLU activation

2. **Output Layer**:
   - Global average pooling with attention
   - Batch normalization
   - Dropout (0.45)
   - Linear projection

## Production Usage
Example prediction code:
```python
def predict_single_sequence(sequence_data, onnx_path):
    """
    Make a single prediction using the ONNX model
    
    Args:
        sequence_data: numpy array of shape (24, 5) containing [open, high, low, close, volume]
        onnx_path: path to the ONNX model file
    
    Returns:
        predicted_return: float, predicted return in percentage
    """
    # Normalize sequence
    sequence = sequence_data.astype(np.float32)
    seq_mean = sequence.mean(axis=0, keepdims=True)
    seq_std = sequence.std(axis=0, keepdims=True)
    seq_std[seq_std < 1e-3] = 1.0
    sequence = (sequence - seq_mean) / seq_std
    
    # Run inference
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {session.get_inputs()[0].name: sequence}
    prediction = session.run(['regression_output'], ort_inputs)[0]
    
    return prediction[0]
```

## Data Format
The model expects OHLCV data in CSV format with columns:
- Open price
- High price
- Low price
- Close price
- Volume

## License
MIT License

## Citation
```
@software{opengradient_btc_spot_forecasting_2025,
    title={OpenGradient BTC 1-Hour Price Forecasting},
    year={2025},
    author={OpenGradient},
    url={https://github.com/OpenGradient/og-models}
}
```