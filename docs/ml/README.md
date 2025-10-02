# Machine Learning Module

The Chronos ML module provides state-of-the-art neural network architectures for time series forecasting, including LSTM, GRU, and Transformer models with GPU acceleration.

## Overview

The ML module integrates deep learning capabilities into Chronos, enabling:

- **Recurrent Networks**: LSTM and GRU for sequential modeling
- **Transformer Architecture**: Self-attention for long-range dependencies
- **GPU Acceleration**: CUDA and Metal support with automatic CPU fallback
- **Model Persistence**: Save and load trained models
- **Interpretability**: Attention analysis and feature importance

## Quick Start

### LSTM Forecasting

```rust
use chronos::ml::{LSTMConfig, create_lstm_forecaster, forecast_with_lstm};

// Configure LSTM
let config = LSTMConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 10,
    dropout_rate: 0.2,
    output_size: 1,
    ..Default::default()
};

// Train forecaster
let forecaster = create_lstm_forecaster(&config, &training_data)?;

// Make predictions
let forecast = forecast_with_lstm(&forecaster, &input_sequence, 5)?;
```

### GRU Forecasting

```rust
use chronos::ml::{GRUConfig, create_gru_forecaster, forecast_with_gru};

// Configure GRU
let config = GRUConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 10,
    output_size: 1,
    ..Default::default()
};

// Train and predict
let forecaster = create_gru_forecaster(&config, &training_data)?;
let forecast = forecast_with_gru(&forecaster, &input_sequence, 5)?;
```

### Transformer Forecasting

```rust
use chronos::ml::{TransformerConfig, create_transformer_forecaster};

// Configure Transformer
let config = TransformerConfig {
    d_model: 128,
    n_heads: 8,
    n_encoder_layers: 6,
    sequence_length: 50,
    ..Default::default()
};

// Train and predict
let forecaster = create_transformer_forecaster(&config, &training_data)?;
let forecast = forecast_with_transformer(&forecaster, &input_sequence, 10)?;

// Analyze attention patterns
let analysis = forecaster.analyze_attention(50);
```

## Architecture Comparison

| Feature | LSTM | GRU | Transformer |
|---------|------|-----|-------------|
| **Training Speed** | Medium | Fast | Medium |
| **Memory Usage** | High | Medium | High |
| **Long Dependencies** | Good | Good | Excellent |
| **Interpretability** | Low | Low | High |
| **Parallelization** | Poor | Poor | Excellent |
| **Parameters** | Most | Fewer | Many |

## Guides

- [LSTM and GRU Guide](lstm_gru_guide.md) - Comprehensive guide to recurrent networks
- [Transformer Guide](transformer_guide.md) - Attention-based forecasting
- [Model Persistence](persistence.md) - Saving and loading models
- [GPU Acceleration](gpu_acceleration.md) - Using CUDA and Metal

## Core Concepts

### Time Series Dataset

All models use `TimeSeriesDataset` for training:

```rust
use chronos::ml::TimeSeriesDataset;

let dataset = TimeSeriesDataset::from_timeseries(
    &time_series,
    sequence_length: 10,
    horizon: 1,
    train_ratio: 0.7,
    val_ratio: 0.15,
)?;
```

### Normalization

Automatic Z-score normalization is applied:

```rust
// Normalization happens automatically during training
let forecaster = create_lstm_forecaster(&config, &data)?;

// Predictions are automatically denormalized
let forecast = forecaster.forecast(&input)?;
```

### Training Configuration

Control training with `TrainingConfig`:

```rust
use chronos::ml::TrainingConfig;

let training_config = TrainingConfig {
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    validation_split: 0.2,
    early_stopping: Some(EarlyStoppingConfig::default()),
    ..Default::default()
};
```

## Device Management

All models support automatic device selection:

```rust
use chronos::ml::Device;

// Automatic selection (prefers GPU)
let forecaster = LSTMForecaster::new(&config, Device::Auto)?;

// Explicit device
let forecaster = LSTMForecaster::new(&config, Device::CUDA(0))?;

// CPU only
let forecaster = LSTMForecaster::new(&config, Device::CPU)?;
```

## Model Persistence

Save and load trained models:

```rust
use chronos::ml::{save_model, load_model, ModelFormat};

// Save model
save_model(&forecaster.network, "model.bin", ModelFormat::Binary)?;

// Load model
let loaded_network = load_model("model.bin", ModelFormat::Binary)?;
```

## Best Practices

### Choosing a Model

- **LSTM**: Best for standard time series with moderate sequences
- **GRU**: Faster training, good for real-time applications
- **Transformer**: Best for long sequences and interpretability needs

### Hyperparameter Tuning

1. Start with default configurations
2. Adjust `sequence_length` based on your data periodicity
3. Tune `hidden_size` for model capacity
4. Use `dropout_rate` to prevent overfitting
5. Increase `num_layers` for complex patterns

### Performance Optimization

- Use GPU for training (5-10x speedup)
- Batch predictions for efficiency
- Cache trained models
- Use GRU if training speed is critical

## Examples

See the `examples/` directory for complete examples:

- `examples/ml/stock_forecasting.rs` - Stock price prediction
- `examples/ml/sensor_forecasting.rs` - IoT sensor prediction
- `examples/ml/attention_analysis.rs` - Transformer interpretability

## API Reference

### Recurrent Networks

- `LSTMConfig` - LSTM configuration
- `GRUConfig` - GRU configuration
- `LSTMForecaster` - LSTM model
- `GRUForecaster` - GRU model
- `create_lstm_forecaster()` - Create trained LSTM
- `create_gru_forecaster()` - Create trained GRU
- `forecast_with_lstm()` - Multi-step LSTM forecast
- `forecast_with_gru()` - Multi-step GRU forecast

### Transformer

- `TransformerConfig` - Transformer configuration
- `TransformerForecaster` - Transformer model
- `MultiHeadAttention` - Attention mechanism
- `PositionalEncoding` - Position encoding
- `create_transformer_forecaster()` - Create trained Transformer
- `forecast_with_transformer()` - Multi-step forecast
- `analyze_attention()` - Extract attention patterns

### Core Types

- `TimeSeriesDataset` - Training dataset
- `NormalizationParams` - Data normalization
- `TrainingConfig` - Training parameters
- `TrainingHistory` - Training metrics
- `Device` - Computation device

## Troubleshooting

### Common Issues

**CUDA not found**
- Ensure CUDA drivers are installed
- Model will automatically fallback to CPU

**Out of memory**
- Reduce `batch_size`
- Reduce `sequence_length`
- Use GRU instead of LSTM
- Enable gradient checkpointing

**Poor accuracy**
- Increase `sequence_length`
- Add more layers
- Reduce `dropout_rate`
- Train for more epochs
- Check data normalization

## Contributing

See the main [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to the ML module.

## License

Same as Chronos project: MIT OR Apache-2.0
