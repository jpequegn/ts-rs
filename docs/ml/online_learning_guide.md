# Online Learning and Real-Time Model Adaptation Guide

## Overview

The `online_learning` module provides capabilities for real-time model adaptation and incremental learning, enabling ML models to continuously learn from streaming data and adapt to changing patterns without full retraining.

## Key Features

- **Incremental Learning**: Update models with new data points as they arrive
- **Concept Drift Detection**: Identify distribution changes in streaming data using multiple algorithms
- **Model Adaptation**: Various strategies to adapt models when drift is detected
- **Performance Monitoring**: Real-time tracking of model performance with trend analysis
- **Memory Management**: Efficient handling of streaming data buffers

## Quick Start

### Basic Online Forecaster

```rust
use chronos::ml::{
    OnlineLearningConfig, DriftDetectionConfig, DriftDetectionMethod,
    OnlineForecaster, OnlineLearningModel,
};

// Create configuration
let config = OnlineLearningConfig {
    drift_detection: DriftDetectionConfig {
        method: DriftDetectionMethod::ADWIN { delta: 0.002 },
        window_size: 100,
        sensitivity: 0.8,
        grace_period: 30,
        confirmation_threshold: 3,
    },
    ..Default::default()
};

// Create online forecaster (requires a base model implementing OnlineLearningModel)
// let base_model: Box<dyn OnlineLearningModel> = ...;
// let mut forecaster = OnlineForecaster::new(base_model, config)?;

// Make predictions and update
// let prediction = forecaster.forecast_and_update(&data, horizon, Some(&ground_truth))?;
```

## Drift Detection Algorithms

### ADWIN (Adaptive Windowing)

ADWIN maintains a sliding window that automatically adjusts its size when concept drift is detected. It's particularly effective for detecting abrupt changes in data distribution.

```rust
use chronos::ml::{ADWINDetector, DriftDetector};

let mut detector = ADWINDetector::new(0.002); // delta parameter

for value in data_stream {
    let drift_detected = detector.add_element(value)?;
    if drift_detected {
        println!("Drift detected! Adapting model...");
        // Trigger adaptation
    }
}
```

**Parameters:**
- `delta`: Confidence level (typical: 0.002 to 0.01)
  - Lower values = more sensitive to drift
  - Higher values = fewer false positives

**Best for:**
- Abrupt concept drift
- Real-time streaming data
- When false positives must be minimized

### DDM (Drift Detection Method)

DDM monitors the error rate and its standard deviation. It triggers warnings when performance degrades and detects drift when degradation exceeds critical thresholds.

```rust
use chronos::ml::{DDMDetector, DriftDetector};

let mut detector = DDMDetector::new(2.0, 3.0); // warning_level, out_control_level

for prediction_error in error_stream {
    // Convert to binary: 1.0 for error, 0.0 for correct
    let is_error = if prediction_error > threshold { 1.0 } else { 0.0 };
    let drift_detected = detector.add_element(is_error)?;

    if drift_detected {
        println!("Concept drift detected!");
    }
}
```

**Parameters:**
- `warning_level`: Threshold for warning state (typical: 2.0)
- `out_control_level`: Threshold for drift detection (typical: 3.0)

**Best for:**
- Classification problems
- Gradual concept drift
- When early warnings are valuable

### EDDM (Early Drift Detection Method)

EDDM focuses on the distance between errors rather than error rate directly. It's more sensitive to gradual changes.

```rust
use chronos::ml::{EDDMDetector, DriftDetector};

let mut detector = EDDMDetector::new(0.95, 0.90); // alpha, beta

for prediction_error in error_stream {
    let is_error = if prediction_error > threshold { 1.0 } else { 0.0 };
    let drift_detected = detector.add_element(is_error)?;
}
```

**Parameters:**
- `alpha`: Warning level multiplier (typical: 0.95)
- `beta`: Drift detection multiplier (typical: 0.90)

**Best for:**
- Early detection of gradual drift
- When drift develops slowly over time

## Adaptation Strategies

### Continuous Update

Updates the model at regular intervals regardless of drift detection.

```rust
use chronos::ml::AdaptationStrategy;

let strategy = AdaptationStrategy::ContinuousUpdate {
    update_frequency: 100, // Update every 100 samples
};
```

**Best for:**
- High-velocity data streams
- When all data is valuable
- Environments with frequent but minor changes

### Drift-Triggered Adaptation

Updates only when drift is detected, with sensitivity-based learning rate adjustment.

```rust
let strategy = AdaptationStrategy::DriftTriggered {
    sensitivity: 0.8, // How aggressively to adapt (0.0 to 1.0)
};
```

**Best for:**
- Resource-constrained environments
- When drift is infrequent
- Stable environments with occasional shifts

### Performance-Triggered Adaptation

Adapts when performance drops below a threshold.

```rust
let strategy = AdaptationStrategy::PerformanceTriggered {
    threshold: 0.9, // Minimum acceptable performance (0.0 to 1.0)
};
```

**Best for:**
- Production systems with SLA requirements
- When performance is the primary metric
- Critical applications

### Ensemble Adaptation

Maintains a pool of models with different specializations.

```rust
use chronos::ml::ModelSelection;

let strategy = AdaptationStrategy::Ensemble {
    model_pool_size: 5,
    selection_method: ModelSelection::TopK { k: 3 },
};
```

**Model Selection Methods:**
- `TopK { k }`: Select top K performing models
- `DiversityBased { diversity_threshold }`: Select diverse models
- `RecencyWeighted { decay_factor }`: Favor recent models
- `PerformanceThreshold { threshold }`: Select models above threshold

**Best for:**
- Complex environments with multiple regimes
- When robustness is critical
- Sufficient computational resources

## Performance Monitoring

### Real-Time Monitoring

```rust
use chronos::ml::{OnlinePerformanceMonitor, PerformanceMonitoringConfig};

let config = PerformanceMonitoringConfig {
    monitoring_window: 100,
    enable_trend_analysis: true,
    ..Default::default()
};

let mut monitor = OnlinePerformanceMonitor::new(config);

// Add observations
monitor.add_observation(error);

// Get current performance
let current_perf = monitor.get_current_performance();

// Analyze trends
match monitor.analyze_trend() {
    TrendAnalysis::Trend { direction, magnitude } => {
        println!("Trend: {:?}, Magnitude: {}", direction, magnitude);
    }
    TrendAnalysis::InsufficientData => {
        println!("Not enough data for trend analysis");
    }
}
```

### Alert Thresholds

```rust
use chronos::ml::AlertThresholds;

let thresholds = AlertThresholds {
    performance_degradation: 0.1, // 10% degradation triggers alert
    drift_likelihood: 0.8,        // 80% drift confidence triggers alert
    prediction_latency_ms: 100,   // 100ms latency triggers alert
    memory_usage_mb: 1000,        // 1GB memory usage triggers alert
};
```

## Memory Management

### Retention Strategies

```rust
use chronos::ml::{MemoryManagementConfig, RetentionStrategy};

let config = MemoryManagementConfig {
    buffer_size: 10000,
    retention_strategy: RetentionStrategy::ImportanceBased { threshold: 0.5 },
    ..Default::default()
};
```

**Available Strategies:**
- `FIFO`: First In, First Out (simple, predictable)
- `LRU`: Least Recently Used (favors recent data)
- `ImportanceBased { threshold }`: Keep important samples (based on importance scoring)
- `Reservoir { sample_size }`: Reservoir sampling (uniform random sample)
- `Hierarchical { levels }`: Multi-level retention (different retention for different time scales)

### Compression Methods

```rust
use chronos::ml::CompressionMethod;

let compression = CompressionMethod::Clustering { n_clusters: 100 };
```

**Available Methods:**
- `None`: No compression (full data retention)
- `Clustering { n_clusters }`: Cluster old data into representatives
- `Summarization { summary_size }`: Create statistical summaries
- `Sketching { sketch_size }`: Use data sketches for approximation

## Learning Rate Adaptation

### Learning Rate Schedules

```rust
use chronos::ml::{LearningRateConfig, LearningRateSchedule};

let config = LearningRateConfig {
    initial_lr: 0.001,
    min_lr: 0.00001,
    max_lr: 0.1,
    schedule: LearningRateSchedule::ExponentialDecay {
        decay_rate: 0.96,
        decay_steps: 1000,
    },
};
```

**Available Schedules:**
- `Constant`: Fixed learning rate
- `ExponentialDecay { decay_rate, decay_steps }`: Exponential decay
- `StepDecay { drop_rate, epochs_drop }`: Step-wise decay
- `CosineAnnealing { t_max }`: Cosine annealing
- `OneCycle { max_lr, pct_start }`: One cycle policy
- `AdaptiveOnPerformance { patience, factor }`: Adapt based on performance

## Complete Example

```rust
use chronos::ml::{
    OnlineLearningConfig, DriftDetectionConfig, DriftDetectionMethod,
    AdaptationStrategy, PerformanceMonitoringConfig, PerformanceMetric,
    OnlineForecaster, OnlineLearningModel,
};

// Configure drift detection
let drift_config = DriftDetectionConfig {
    method: DriftDetectionMethod::ADWIN { delta: 0.002 },
    window_size: 100,
    sensitivity: 0.8,
    grace_period: 30,
    confirmation_threshold: 3,
};

// Configure performance monitoring
let perf_config = PerformanceMonitoringConfig {
    metrics: vec![PerformanceMetric::MAE, PerformanceMetric::RMSE],
    monitoring_window: 100,
    enable_trend_analysis: true,
    ..Default::default()
};

// Create complete configuration
let config = OnlineLearningConfig {
    adaptation_strategy: AdaptationStrategy::DriftTriggered { sensitivity: 0.8 },
    drift_detection: drift_config,
    performance_monitoring: perf_config,
    ..Default::default()
};

// Create forecaster (requires base model)
// let base_model: Box<dyn OnlineLearningModel> = create_your_model();
// let mut forecaster = OnlineForecaster::new(base_model, config)?;

// Use in streaming context
// for (data, ground_truth) in data_stream {
//     let prediction = forecaster.forecast_and_update(&data, horizon, Some(&ground_truth))?;
//
//     // Check adaptation state
//     match forecaster.state {
//         AdaptationState::Stable => println!("Model is stable"),
//         AdaptationState::DriftDetected => println!("Drift detected, adapting..."),
//         AdaptationState::Adapting => println!("Model is adapting"),
//         AdaptationState::Recovered => println!("Model has recovered"),
//     }
// }
```

## Best Practices

### 1. Choose the Right Drift Detector

- **Fast changes**: Use ADWIN with low delta (0.001-0.002)
- **Gradual drift**: Use EDDM or DDM
- **Classification**: Use DDM or EDDM (work with binary errors)
- **Regression**: Use ADWIN (works with continuous values)

### 2. Set Appropriate Thresholds

- Start conservative (less sensitive) and adjust based on false positive rate
- Monitor both drift detection rate and false alarm rate
- Consider business cost of false positives vs. false negatives

### 3. Balance Adaptation Speed

- Higher sensitivity = faster adaptation but more false positives
- Lower sensitivity = slower adaptation but more stable
- Use ensemble methods for robustness

### 4. Monitor Performance Trends

- Enable trend analysis to catch gradual degradation
- Set alerts for unexpected performance drops
- Track adaptation frequency to tune sensitivity

### 5. Manage Memory Efficiently

- Use importance-based retention for variable-rate streams
- Apply compression for long-running systems
- Monitor memory usage and adjust buffer sizes

### 6. Learning Rate Strategy

- Start with constant rate for initial stability
- Switch to adaptive schedules after warm-up period
- Use performance-based adaptation in production

## Performance Considerations

### Computational Overhead

- ADWIN: O(log n) per sample (most efficient)
- DDM/EDDM: O(1) per sample (very efficient)
- Ensemble: O(k) where k is number of models

### Memory Requirements

- Base buffer: Configured buffer size
- ADWIN: Additional O(log n) for window management
- DDM/EDDM: O(1) additional memory
- Ensemble: O(k × model_size)

### Real-Time Performance

- Target: <100ms update latency
- ADWIN achieves ~0.1-1ms per sample
- DDM/EDDM achieve ~0.01-0.1ms per sample
- Ensemble overhead depends on model count

## Troubleshooting

### Too Many False Positives

- Increase delta (ADWIN) or thresholds (DDM/EDDM)
- Increase grace period
- Increase confirmation threshold
- Use ensemble with diverse detection

### Missing Drift

- Decrease delta or thresholds
- Reduce window size for faster response
- Use multiple detectors with voting

### Poor Adaptation Performance

- Check learning rate schedule
- Verify sufficient training data in buffer
- Consider ensemble approach
- Monitor model capacity vs. problem complexity

## References

- Bifet, A., & Gavaldà, R. (2007). Learning from Time-Changing Data with Adaptive Windowing. SIAM International Conference on Data Mining.
- Gama, J., et al. (2004). Learning with Drift Detection. Brazilian Symposium on Artificial Intelligence.
- Baena-Garcia, M., et al. (2006). Early Drift Detection Method. Fourth International Workshop on Knowledge Discovery from Data Streams.
