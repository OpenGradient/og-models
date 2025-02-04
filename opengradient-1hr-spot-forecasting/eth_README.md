# ETH 1-Hour Price Direction Forecasting Model

## Overview
A deep learning model for predicting ETH price movement direction over 1-hour horizons using OHLCV data. The model uses a TCN architecture with direct price data input.

## Model Performance

### Backtest Results
![6-Month Backtest Results](@plots/backtest_results_6months_eth.png)
- Cumulative strategy returns vs buy & hold
- Directional accuracy visualization
- Trade-by-trade performance

### Training History
![Training History](@plots/training_history_1day_eth.png)
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
    [3124.13, 3090.25, 3110.01, 13826.8299],
    [3112.04, 3096.33, 3111.29, 9661.4066],
    [3111.43, 3091.1, 3104.47, 11975.132],
    [3112.89, 3052.19, 3064.86, 38775.4157],
    [3080.39, 3050.64, 3052.39, 41132.0816],
    [3088.13, 3052.4, 3087.47, 22373.8455],
    [3101.92, 3082.23, 3087.07, 25861.5024],
    [3087.08, 3061.41, 3081.19, 21005.6924],
    [3082.02, 3003.53, 3015.9, 60130.3662],
    [3037.0, 2951.58, 2972.99, 117492.0522],
    [2990.5, 2916.66, 2928.36, 103529.5935],
    [2984.0, 2894.4, 2958.9, 90011.2213],
    [2987.68, 2943.34, 2950.64, 27865.7672],
    [2956.82, 2896.4, 2907.9, 48138.9116],
    [2937.17, 2750.71, 2875.46, 119539.8168],
    [2875.46, 2778.49, 2869.68, 130303.5685],
    [3110.0, 3124.13, 3090.25, 3110.01],
    [3111.3, 3112.04, 3096.33, 3111.29],
    [3104.48, 3111.43, 3091.1, 3104.47],
    [3064.87, 3112.89, 3052.19, 3064.86],
    [3052.4, 3080.39, 3050.64, 3052.39],
    [3087.47, 3088.13, 3052.4, 3087.47],
    [3087.08, 3101.92, 3082.23, 3087.07],
    [3081.18, 3087.08, 3061.41, 3081.19]
  ]
}
```

Each row contains:
1. Open price (USD)
2. High price (USD)
3. Low price (USD)
4. Close price (USD)
5. Volume (ETH)

### Output
- Shape: (1,)
- Type: float32
- Meaning: Predicted percentage return for next hour
- Example: -0.24 means -0.24% predicted decline

### Example Output Interpretation
```python
predicted_return = -0.24  # Model output
current_price = 3081.19  # Last close price
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
@software{opengradient_eth_spot_forecasting_2025,
    title={OpenGradient ETH 1-Hour Price Forecasting},
    year={2025},
    author={OpenGradient},
    url={https://github.com/OpenGradient/og-models}
}
```