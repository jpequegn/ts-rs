# Transformer Architecture Guide

Comprehensive guide to using Transformer models for time series forecasting with self-attention mechanisms and interpretability features.

## Overview

Transformers use self-attention mechanisms to capture long-range dependencies in sequential data. Unlike LSTM and GRU, they process entire sequences in parallel, making them highly efficient and interpretable for time series forecasting.

### Key Advantages

- **Long-Range Dependencies**: Excellent at capturing patterns across long sequences
- **Parallel Processing**: Faster training than sequential models like LSTM/GRU
- **Interpretability**: Attention weights reveal which time steps influence predictions
- **Scalability**: Performance improves with more data and compute

### When to Use Transformers

**Best For**:
- Long sequences (50+ time steps)
- Complex temporal patterns requiring full sequence context
- Applications requiring interpretability and explainability
- Large datasets with sufficient training examples

**Not Ideal For**:
- Very small datasets (<1000 samples)
- Simple patterns better handled by LSTM/GRU
- Real-time applications with strict latency requirements
- Resource-constrained environments

## Quick Start

### Basic Transformer Forecasting

```rust
use chronos::ml::{TransformerConfig, create_transformer_forecaster, forecast_with_transformer};
use chronos::TimeSeries;

// Prepare time series data
let time_series = TimeSeries::new(
    "sensor_data".to_string(),
    timestamps,
    values
)?;

// Configure Transformer
let config = TransformerConfig {
    d_model: 128,           // Model dimension
    n_heads: 8,             // Attention heads
    n_encoder_layers: 6,    // Transformer layers
    sequence_length: 50,    // Input sequence length
    ..Default::default()
};

// Train forecaster
let forecaster = create_transformer_forecaster(&config, &time_series)?;

// Make predictions
let forecast = forecast_with_transformer(&forecaster, &input_sequence, 10)?;
```

### With Attention Analysis

```rust
// Analyze attention patterns
let attention_analysis = forecaster.analyze_attention(50)?;

println!("Average attention score: {}", attention_analysis.average_attention);
println!("Attention entropy: {}", attention_analysis.entropy);

// Get most important time steps
for (step, importance) in attention_analysis.important_steps.iter().take(5) {
    println!("Step {}: importance {:.4}", step, importance);
}
```

## Configuration Guide

### Transformer Configuration

```rust
pub struct TransformerConfig {
    // Model architecture
    pub d_model: usize,           // Model dimension (64-512)
    pub n_heads: usize,           // Number of attention heads (4-16)
    pub n_encoder_layers: usize,  // Number of transformer layers (2-12)
    pub d_ff: usize,              // Feed-forward dimension (256-2048)

    // Sequence handling
    pub sequence_length: usize,   // Input sequence length (20-200)
    pub output_size: usize,       // Output dimension (usually 1)

    // Training parameters
    pub dropout_rate: f64,        // Dropout for regularization (0.1-0.3)
    pub learning_rate: f64,       // Learning rate (0.0001-0.001)
    pub batch_size: usize,        // Batch size (16-128)
    pub epochs: usize,            // Training epochs (50-300)

    // Attention configuration
    pub attention_dropout: f64,   // Attention dropout (0.0-0.2)
    pub use_causal_mask: bool,    // Prevent future information leakage

    // Positional encoding
    pub pos_encoding_type: PositionalEncodingType,  // Sinusoidal or Learned
    pub max_position: usize,      // Maximum sequence position

    // Data handling
    pub train_ratio: f64,         // Training split (0.6-0.8)
    pub val_ratio: f64,           // Validation split (0.1-0.2)

    // Device
    pub device: Device,           // CPU, CUDA, Metal, Auto
}
```

### Parameter Tuning Guidelines

**Model Dimension** (`d_model`):
- Start with 128 for moderate complexity
- Use 64 for simple patterns or small datasets
- Use 256-512 for complex patterns with large datasets
- Must be divisible by `n_heads`

**Number of Heads** (`n_heads`):
- Start with 8 heads
- Use 4 heads for simple patterns
- Use 16 heads for very complex patterns
- `d_model` must be divisible by `n_heads`

**Number of Layers** (`n_encoder_layers`):
- Start with 4-6 layers
- Use 2-3 layers for simpler patterns
- Use 8-12 layers for very complex patterns
- More layers = more capacity but slower training

**Feed-Forward Dimension** (`d_ff`):
- Typically 4x `d_model` (e.g., d_model=128 → d_ff=512)
- Can reduce to 2x for smaller models
- Can increase to 8x for very complex patterns

**Sequence Length** (`sequence_length`):
- Transformers excel with longer sequences (50-200)
- Minimum recommended: 20 time steps
- Maximum depends on available memory
- Quadratic memory growth with sequence length

**Dropout Rate** (`dropout_rate`):
- Start with 0.1
- Increase to 0.2-0.3 if overfitting
- Attention dropout can be lower (0.0-0.1)

**Learning Rate** (`learning_rate`):
- Start with 0.0001 for large models
- Use 0.001 for smaller models
- Transformers are sensitive to learning rate
- Consider using warmup and decay schedules

## Attention Mechanisms

### Multi-Head Attention

Transformers use multi-head attention to capture different aspects of temporal patterns:

```rust
use chronos::ml::MultiHeadAttention;

// Create multi-head attention layer
let attention = MultiHeadAttention::new(
    d_model: 128,
    n_heads: 8,
    dropout: 0.1,
)?;

// Compute attention
let output = attention.forward(query, key, value)?;
```

### Attention Analysis

```rust
// Analyze which time steps are most important
let analysis = forecaster.analyze_attention(sequence_length)?;

// Attention statistics
println!("Average attention: {}", analysis.average_attention);
println!("Max attention: {}", analysis.max_attention);
println!("Attention entropy: {}", analysis.entropy);  // Higher = more distributed

// Feature importance
for (feature, importance) in &analysis.feature_importance {
    println!("Feature {}: {:.4}", feature, importance);
}

// Important time steps
for (step, score) in &analysis.important_steps {
    println!("Step {}: attention score {:.4}", step, score);
}
```

### Visualizing Attention

```rust
// Get attention weights for a specific input
let attention_weights = forecaster.compute_attention_scores(
    query_len: 50,
    key_len: 50,
    causal_mask: true,  // Prevent looking at future
)?;

// attention_weights[i][j] = attention from query i to key j
for (i, row) in attention_weights.iter().enumerate() {
    println!("Query {}: attention distribution {:?}", i, row);
}
```

## Positional Encoding

Transformers need positional information since they process sequences in parallel.

### Positional Encoding Types

```rust
use chronos::ml::PositionalEncodingType;

// Sinusoidal encoding (default)
let config = TransformerConfig {
    pos_encoding_type: PositionalEncodingType::Sinusoidal,
    ..Default::default()
};

// Learned encoding (trained during model training)
let config = TransformerConfig {
    pos_encoding_type: PositionalEncodingType::Learned,
    ..Default::default()
};

// Temporal encoding (uses actual timestamps)
let config = TransformerConfig {
    pos_encoding_type: PositionalEncodingType::Temporal,
    ..Default::default()
};
```

### Custom Positional Encoding

```rust
use chronos::ml::PositionalEncoding;

// Create custom positional encoding
let pos_encoding = PositionalEncoding::new(
    d_model: 128,
    max_len: 200,
    encoding_type: PositionalEncodingType::Sinusoidal,
)?;

// Apply to input sequence
let encoded = pos_encoding.encode(input)?;
```

## Training Process

### Basic Training

```rust
use chronos::ml::TransformerForecaster;

// Create forecaster
let mut forecaster = TransformerForecaster::new(&config, Device::Auto)?;

// Train on time series data
forecaster.train(&time_series)?;

// Access training history
let history = forecaster.training_history();
println!("Training loss: {:?}", history.train_losses);
println!("Validation loss: {:?}", history.val_losses);
println!("Best epoch: {}", history.best_epoch);
```

### Advanced Training with Learning Rate Scheduling

```rust
use chronos::ml::TrainingConfig;

let training_config = TrainingConfig {
    learning_rate: 0.0001,
    batch_size: 32,
    epochs: 200,

    // Learning rate warmup
    warmup_steps: Some(4000),

    // Early stopping
    early_stopping: Some(EarlyStoppingConfig {
        patience: 15,
        min_delta: 0.0001,
    }),

    // Validation
    validation_split: 0.2,

    ..Default::default()
};

let forecaster = TransformerForecaster::with_training_config(
    &config,
    &training_config,
    Device::Auto
)?;
```

## Making Predictions

### Single-Step Prediction

```rust
// Predict next value
let input_sequence = vec![1.0, 2.0, 3.0, /* ... */, 50.0];
let prediction = forecaster.forecast(&input_sequence)?;
```

### Multi-Step Prediction

```rust
// Predict 10 steps ahead
let forecast = forecast_with_transformer(&forecaster, &input_sequence, 10)?;

for (i, value) in forecast.iter().enumerate() {
    println!("Step {}: {:.4}", i + 1, value);
}
```

### Prediction with Attention Analysis

```rust
// Get predictions with attention insights
let forecast = forecaster.forecast_with_attention(&input_sequence, 10)?;

for (step, (prediction, attention)) in forecast.iter().enumerate() {
    println!("Step {}: prediction={:.4}", step + 1, prediction);

    // Show which historical steps influenced this prediction
    let top_influences: Vec<_> = attention.iter()
        .enumerate()
        .sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap())
        .take(5)
        .collect();

    println!("  Top influences: {:?}", top_influences);
}
```

## Interpretability

### Feature Importance

```rust
// Analyze which features are most important
let analysis = forecaster.analyze_attention(sequence_length)?;

// Get feature importance scores
for (feature_idx, importance) in analysis.feature_importance.iter().enumerate() {
    println!("Feature {}: importance {:.4}", feature_idx, importance);
}
```

### Temporal Importance

```rust
// Identify critical time steps
let important_steps = analysis.important_steps;

// Steps are sorted by importance
for (rank, (step, score)) in important_steps.iter().enumerate().take(10) {
    println!("Rank {}: Step {} (score: {:.4})", rank + 1, step, score);
}
```

### Attention Entropy

```rust
// Measure how focused or distributed attention is
let entropy = analysis.entropy;

if entropy > 3.0 {
    println!("Attention is highly distributed across sequence");
} else if entropy < 1.0 {
    println!("Attention is focused on specific time steps");
} else {
    println!("Attention is moderately distributed");
}
```

## Best Practices

### 1. Sequence Length Selection

```rust
// For daily data with weekly patterns
config.sequence_length = 7 * 4;  // 4 weeks

// For hourly data with daily patterns
config.sequence_length = 24 * 3;  // 3 days

// For high-frequency data
config.sequence_length = 100;  // Adjust based on pattern periodicity
```

### 2. Model Architecture

```rust
// Small model (fast training, less capacity)
let config = TransformerConfig {
    d_model: 64,
    n_heads: 4,
    n_encoder_layers: 3,
    d_ff: 256,
    ..Default::default()
};

// Medium model (balanced)
let config = TransformerConfig {
    d_model: 128,
    n_heads: 8,
    n_encoder_layers: 6,
    d_ff: 512,
    ..Default::default()
};

// Large model (maximum capacity, slower)
let config = TransformerConfig {
    d_model: 256,
    n_heads: 16,
    n_encoder_layers: 12,
    d_ff: 1024,
    ..Default::default()
};
```

### 3. Preventing Overfitting

```rust
config.dropout_rate = 0.2;           // Layer dropout
config.attention_dropout = 0.1;      // Attention dropout

// Use early stopping
config.early_stopping = Some(EarlyStoppingConfig {
    patience: 20,
    min_delta: 0.0001,
});

// Increase validation data
config.val_ratio = 0.2;
```

### 4. Causal Masking

```rust
// Prevent the model from seeing future values
config.use_causal_mask = true;

// This is essential for time series forecasting
// Without it, model can "cheat" by looking at future values
```

## Common Patterns

### Financial Time Series

```rust
let config = TransformerConfig {
    d_model: 256,
    n_heads: 16,
    n_encoder_layers: 8,
    sequence_length: 60,      // 60 days of history
    dropout_rate: 0.3,        // Higher dropout for noisy data
    attention_dropout: 0.1,
    use_causal_mask: true,
    learning_rate: 0.0001,
    batch_size: 64,
    epochs: 300,
    ..Default::default()
};
```

### Energy Forecasting

```rust
let config = TransformerConfig {
    d_model: 128,
    n_heads: 8,
    n_encoder_layers: 6,
    sequence_length: 168,     // 1 week of hourly data
    dropout_rate: 0.2,
    pos_encoding_type: PositionalEncodingType::Temporal,  // Use timestamps
    learning_rate: 0.0001,
    batch_size: 32,
    epochs: 200,
    ..Default::default()
};
```

### Weather Prediction

```rust
let config = TransformerConfig {
    d_model: 128,
    n_heads: 8,
    n_encoder_layers: 6,
    sequence_length: 72,      // 3 days of hourly data
    dropout_rate: 0.2,
    use_causal_mask: true,
    learning_rate: 0.0001,
    batch_size: 32,
    epochs: 150,
    ..Default::default()
};
```

## GPU Acceleration

### Device Selection

```rust
use chronos::ml::Device;

// Automatic (prefers GPU)
let forecaster = TransformerForecaster::new(&config, Device::Auto)?;

// Explicit CUDA
let forecaster = TransformerForecaster::new(&config, Device::CUDA(0))?;

// Explicit Metal (macOS)
let forecaster = TransformerForecaster::new(&config, Device::Metal)?;

// CPU only
let forecaster = TransformerForecaster::new(&config, Device::CPU)?;
```

### Performance Tips

- Transformers benefit significantly from GPU (10-20x speedup)
- Larger batch sizes fully utilize GPU parallelism
- Attention computation is memory-intensive (O(n²))
- Use gradient checkpointing for very long sequences

## Troubleshooting

### Training Loss Not Decreasing

**Solutions**:
- Reduce learning rate to 0.00001
- Add learning rate warmup (warmup_steps: 4000)
- Check data normalization
- Verify causal masking is enabled
- Ensure sufficient training data

### Attention Collapse (All Attention on One Position)

**Solutions**:
- Increase attention_dropout to 0.2
- Add more encoder layers
- Reduce learning rate
- Check positional encoding implementation

### Out of Memory

**Solutions**:
- Reduce sequence_length
- Reduce batch_size to 8 or 16
- Reduce d_model or n_heads
- Use gradient checkpointing
- Switch to CPU if GPU memory insufficient

### Poor Long-Range Dependencies

**Solutions**:
- Increase n_encoder_layers
- Increase sequence_length
- Use Temporal positional encoding
- Verify causal masking is properly applied

### Unstable Training

**Solutions**:
- Reduce learning rate to 0.00001
- Add gradient clipping
- Use learning rate warmup
- Increase dropout rates
- Check for NaN values in data

## Model Persistence

### Saving Models

```rust
use chronos::ml::{save_model, ModelFormat};

// Save transformer model
save_model(&forecaster.network, "transformer_model.bin", ModelFormat::Binary)?;

// Save with metadata
save_model(&forecaster.network, "transformer_model.json", ModelFormat::JSON)?;
```

### Loading Models

```rust
use chronos::ml::load_model;

let loaded_network = load_model("transformer_model.bin", ModelFormat::Binary)?;

let mut forecaster = TransformerForecaster::new(&config, Device::Auto)?;
forecaster.network = loaded_network;
```

## Comparison with LSTM/GRU

| Feature | Transformer | LSTM | GRU |
|---------|-------------|------|-----|
| **Long Dependencies** | Excellent | Good | Good |
| **Training Speed** | Fast (parallel) | Slow (sequential) | Medium (sequential) |
| **Memory Usage** | High (O(n²)) | Medium | Medium |
| **Interpretability** | High (attention) | Low | Low |
| **Small Datasets** | Struggles | Good | Good |
| **Sequence Length** | Excellent (50+) | Good (10-50) | Good (10-50) |

## API Reference

### Core Types

```rust
pub struct TransformerConfig { /* ... */ }
pub struct TransformerForecaster { /* ... */ }
pub struct MultiHeadAttention { /* ... */ }
pub struct PositionalEncoding { /* ... */ }
pub struct AttentionAnalysis { /* ... */ }
```

### Functions

```rust
// Training and creation
pub fn create_transformer_forecaster(config: &TransformerConfig, data: &TimeSeries) -> MLResult<TransformerForecaster>

// Forecasting
pub fn forecast_with_transformer(forecaster: &TransformerForecaster, input: &[f64], steps: usize) -> MLResult<Vec<f64>>

// Analysis
pub fn analyze_attention(forecaster: &TransformerForecaster, sequence_length: usize) -> MLResult<AttentionAnalysis>
```

## Examples

See the `examples/ml/` directory:
- `examples/ml/attention_analysis.rs` - Transformer interpretability
- `examples/ml/long_sequence_forecasting.rs` - Long-range dependencies
- `examples/ml/multivariate_forecasting.rs` - Multiple time series

## Further Reading

- [LSTM/GRU Guide](lstm_gru_guide.md) - Recurrent network alternatives
- [GPU Acceleration Guide](gpu_acceleration.md) - Performance optimization
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
