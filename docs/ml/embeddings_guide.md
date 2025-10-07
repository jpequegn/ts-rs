# Time Series Embeddings and Similarity Detection Guide

This guide covers the time series embeddings and similarity detection functionality in Chronos.

## Overview

The `embeddings` module provides:

- **Similarity Metrics**: Multiple methods for comparing time series (Euclidean, DTW, Cross-Correlation, etc.)
- **Embedding Types**: Framework for neural time series embeddings (Autoencoder, VAE, Transformer, LSTM)
- **Pattern Detection**: Tools for finding recurring patterns and motifs (planned)
- **Clustering**: Grouping similar time series together (planned)
- **Similarity Search**: Efficient k-nearest neighbor search

## Similarity Metrics

### Distance-Based Metrics

#### Euclidean Distance
Standard L2 distance between two time series of equal length.

```rust
use chronos::ml::embeddings::euclidean_distance;
use chronos::TimeSeries;

let ts1 = TimeSeries::new("series1".to_string(), timestamps1, vec![1.0, 2.0, 3.0])?;
let ts2 = TimeSeries::new("series2".to_string(), timestamps2, vec![4.0, 5.0, 6.0])?;

let distance = euclidean_distance(&ts1, &ts2)?;
println!("Euclidean distance: {}", distance); // ~5.196
```

#### Manhattan Distance
L1 distance, sum of absolute differences.

```rust
use chronos::ml::embeddings::manhattan_distance;

let distance = manhattan_distance(&ts1, &ts2)?;
println!("Manhattan distance: {}", distance); // 9.0
```

#### Cosine Similarity
Measures the angle between two vectors, insensitive to magnitude.

```rust
use chronos::ml::embeddings::cosine_similarity;

let similarity = cosine_similarity(&ts1, &ts2)?;
// Returns value in [-1, 1], where 1 = identical direction
println!("Cosine similarity: {}", similarity);
```

### Time Series Specific Metrics

#### Dynamic Time Warping (DTW)
Finds optimal alignment between two time series, allowing for time shifts and warping.

```rust
use chronos::ml::embeddings::dtw_distance;

// Without window constraint
let dtw_dist = dtw_distance(&ts1, &ts2, None)?;

// With Sakoe-Chiba band constraint (window size = 5)
let dtw_windowed = dtw_distance(&ts1, &ts2, Some(5))?;
```

**Use Cases:**
- Comparing time series with different speeds (e.g., speech recognition)
- Aligning shifted patterns
- Robust to temporal distortions

**Complexity:** O(n*m) without window, O(n*w) with window of size w

#### Soft-DTW
Differentiable version of DTW, useful for gradient-based optimization.

```rust
use chronos::ml::embeddings::soft_dtw_distance;

// gamma controls smoothness (smaller = more like hard DTW)
let soft_dtw = soft_dtw_distance(&ts1, &ts2, 1.0)?;
```

### Statistical Metrics

#### Cross-Correlation
Measures similarity as a function of time lag.

```rust
use chronos::ml::embeddings::cross_correlation;

// Compute cross-correlation for lags 0 to 10
let correlations = cross_correlation(&ts1, &ts2, 10)?;

// Get maximum correlation value
use chronos::ml::embeddings::max_cross_correlation;
let max_corr = max_cross_correlation(&ts1, &ts2, 10)?;
```

**Use Cases:**
- Finding optimal time alignment
- Signal processing
- Detecting delayed relationships

## Unified Similarity API

Use `compute_similarity` for a unified interface:

```rust
use chronos::ml::embeddings::{compute_similarity, SimilarityMethod};

// Euclidean
let dist = compute_similarity(&ts1, &ts2, &SimilarityMethod::Euclidean)?;

// DTW with window
let dist = compute_similarity(
    &ts1,
    &ts2,
    &SimilarityMethod::DynamicTimeWarping { window: Some(5) }
)?;

// Soft-DTW
let dist = compute_similarity(
    &ts1,
    &ts2,
    &SimilarityMethod::SoftDTW { gamma: 1.0 }
)?;

// Cross-correlation
let dist = compute_similarity(
    &ts1,
    &ts2,
    &SimilarityMethod::CrossCorrelation { max_lag: 10 }
)?;

// Cosine (returns distance = 1 - similarity)
let dist = compute_similarity(&ts1, &ts2, &SimilarityMethod::Cosine)?;
```

## Similarity Search

Find k most similar time series from a database:

```rust
use chronos::ml::embeddings::{find_similar_time_series, SimilarityMethod};

let query = TimeSeries::new("query".to_string(), timestamps, values)?;
let database: Vec<TimeSeries> = vec![ts1, ts2, ts3, ts4, ts5];

// Find 3 most similar using DTW
let results = find_similar_time_series(
    &query,
    &database,
    &SimilarityMethod::DynamicTimeWarping { window: None },
    3
)?;

for result in results {
    println!("Series: {}", result.time_series.name);
    println!("  Similarity score: {:.3}", result.similarity_score); // 0-1, higher is better
    println!("  Distance: {:.3}", result.distance); // Lower is better
}
```

## Embedding Types (Framework)

The module defines several embedding architectures for future implementation:

### Autoencoder
```rust
use chronos::ml::embeddings::{EmbeddingType, EmbeddingConfig};

let config = EmbeddingConfig {
    embedding_type: EmbeddingType::Autoencoder {
        hidden_layers: vec![64, 32],
        latent_dim: 16,
    },
    dimension: 16,
    ..Default::default()
};
```

### Variational Autoencoder (VAE)
```rust
let config = EmbeddingConfig {
    embedding_type: EmbeddingType::VariationalAutoencoder {
        latent_dim: 16,
        beta: 1.0, // β-VAE for disentanglement
    },
    ..Default::default()
};
```

### Transformer-Based
```rust
let config = EmbeddingConfig {
    embedding_type: EmbeddingType::Transformer {
        model_dim: 128,
        num_heads: 8,
        num_layers: 4,
    },
    ..Default::default()
};
```

### LSTM-Based
```rust
let config = EmbeddingConfig {
    embedding_type: EmbeddingType::LSTM {
        hidden_size: 64,
        num_layers: 2,
    },
    ..Default::default()
};
```

## Utility Functions

### Distance to Similarity Conversion
Convert distance measures to similarity scores (0-1 scale):

```rust
use chronos::ml::embeddings::distance_to_similarity;

let distance = 3.5;
let max_distance = 10.0;

let similarity = distance_to_similarity(distance, max_distance);
// Returns 0.65 (65% similarity)
```

## Best Practices

### Choosing a Similarity Metric

| Metric | Best For | Characteristics |
|--------|----------|----------------|
| Euclidean | Aligned, same-length series | Fast, intuitive |
| Manhattan | Robust to outliers | Less sensitive than Euclidean |
| Cosine | Direction more important than magnitude | Scale-invariant |
| DTW | Time-shifted patterns | Handles warping, slower |
| Soft-DTW | ML training with gradients | Differentiable |
| Cross-Correlation | Finding time lags | Good for periodic signals |

### Performance Considerations

1. **Euclidean/Manhattan**: O(n) - Very fast
2. **Cosine**: O(n) - Fast
3. **DTW without window**: O(n*m) - Slow for long series
4. **DTW with window**: O(n*w) - Much faster, w << n
5. **Cross-Correlation**: O(n*L) where L is max_lag

### Normalization

Always normalize time series before computing similarities to ensure fair comparisons:

```rust
// Z-score normalization (recommended)
let normalized = ts.normalize_zscore()?;

// Min-max normalization
let normalized = ts.normalize_minmax()?;
```

## Examples

### Example 1: Find Similar Stock Price Patterns
```rust
use chronos::ml::embeddings::{find_similar_time_series, SimilarityMethod};

// Load historical stock data
let query_pattern = load_recent_price_movement();
let historical_database = load_historical_patterns();

// Find 5 most similar historical patterns using DTW
let similar_patterns = find_similar_time_series(
    &query_pattern,
    &historical_database,
    &SimilarityMethod::DynamicTimeWarping { window: Some(10) },
    5
)?;

for pattern in similar_patterns {
    println!("Historical match from {}", pattern.time_series.name);
    println!("Similarity: {:.1}%", pattern.similarity_score * 100.0);
}
```

### Example 2: Detect Repeated Patterns
```rust
// Create sliding windows of a time series
let window_size = 50;
let windows = create_windows(&time_series, window_size);

// Compare each window to find repeating patterns
for (i, window1) in windows.iter().enumerate() {
    for (j, window2) in windows.iter().skip(i + 1).enumerate() {
        let similarity = cosine_similarity(window1, window2)?;
        if similarity > 0.95 {
            println!("Pattern at {} similar to pattern at {}", i, j);
        }
    }
}
```

### Example 3: Time-Lag Analysis
```rust
use chronos::ml::embeddings::cross_correlation;

let sensor1_data = load_sensor_data("sensor1");
let sensor2_data = load_sensor_data("sensor2");

// Find if sensor2 follows sensor1 with a delay
let correlations = cross_correlation(&sensor1_data, &sensor2_data, 20)?;

let (max_lag, max_corr) = correlations.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();

println!("Maximum correlation {} at lag {} seconds", max_corr, max_lag);
```

## Future Enhancements

The following features are planned for future releases:

1. **Neural Embeddings**: Training autoencoder/VAE models for embeddings
2. **Clustering**: K-means, DBSCAN for grouping similar series
3. **Pattern Detection**: Motif discovery and anomalous pattern identification
4. **Indexing**: Efficient similarity search using LSH, HNSW
5. **Additional Metrics**: Mahalanobis, LCSS, EDR, Mutual Information
6. **Spectral Methods**: Frequency domain similarity
7. **Wavelet Similarity**: Multi-resolution time series comparison

## References

- [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [Soft-DTW Paper](https://arxiv.org/abs/1703.01541)
- [Time Series Similarity Measures Review](https://link.springer.com/article/10.1007/s10115-012-0560-5)
