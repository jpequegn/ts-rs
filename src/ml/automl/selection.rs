//! Model Selection Module
//!
//! Automated model selection and ranking based on:
//! - Dataset characteristics
//! - Performance predictions
//! - Resource constraints
//! - Meta-learning recommendations

use super::types::*;
use crate::timeseries::TimeSeries;
use crate::ml::MLResult;
use std::collections::HashMap;

// ================================================================================================
// Model Selector
// ================================================================================================

pub struct ModelSelector {
    pub candidate_models: Vec<ModelCandidate>,
    pub selection_criteria: SelectionCriteria,
    pub performance_cache: PerformanceCache,
}

/// Performance cache for model-dataset combinations
#[derive(Debug, Clone)]
pub struct PerformanceCache {
    cache: HashMap<String, CachedPerformance>,
}

#[derive(Debug, Clone)]
struct CachedPerformance {
    performance: f64,
    features: DatasetFeatures,
}

impl ModelSelector {
    pub fn new(candidate_models: Vec<ModelCandidate>, selection_criteria: SelectionCriteria) -> Self {
        Self {
            candidate_models,
            selection_criteria,
            performance_cache: PerformanceCache {
                cache: HashMap::new(),
            },
        }
    }

    /// Rank models based on dataset characteristics
    pub fn rank_models(&self, data: &TimeSeries) -> Vec<(ModelCandidate, f64)> {
        let features = extract_dataset_features(data);

        let mut scored_models: Vec<_> = self
            .candidate_models
            .iter()
            .map(|model| {
                let score = self.score_model(model, &features);
                (model.clone(), score)
            })
            .collect();

        scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_models
    }

    /// Select top K models
    pub fn select_top_k_models(&self, data: &TimeSeries, k: usize) -> Vec<ModelCandidate> {
        let ranked = self.rank_models(data);
        ranked.into_iter().take(k).map(|(model, _)| model).collect()
    }

    /// Score a model based on selection criteria
    fn score_model(&self, model: &ModelCandidate, features: &DatasetFeatures) -> f64 {
        let performance_score = self.estimate_performance(model, features);
        let time_score = self.score_training_time(&model.estimated_training_time);
        let complexity_score = self.score_complexity(model.memory_requirements);

        self.selection_criteria.performance_weight * performance_score
            + self.selection_criteria.time_weight * time_score
            + self.selection_criteria.complexity_weight * complexity_score
    }

    /// Estimate model performance on dataset
    fn estimate_performance(&self, model: &ModelCandidate, features: &DatasetFeatures) -> f64 {
        // Simple heuristic based on model type and data characteristics
        match model.model_type {
            ModelType::LSTM | ModelType::GRU => {
                // Good for sequences with strong temporal dependencies
                features.temporal.trend_strength * 0.5 + features.temporal.seasonality_strength * 0.5
            }
            ModelType::Transformer => {
                // Good for long sequences
                if features.temporal.n_observations > 1000 {
                    0.9
                } else {
                    0.6
                }
            }
            ModelType::Ensemble => {
                // Generally robust
                0.8
            }
            _ => 0.5,
        }
    }

    /// Score training time (lower is better, normalized)
    fn score_training_time(&self, time: &std::time::Duration) -> f64 {
        let seconds = time.as_secs_f64();
        let max_acceptable = 3600.0; // 1 hour
        (1.0 - (seconds / max_acceptable).min(1.0)).max(0.0)
    }

    /// Score complexity (lower memory is better, normalized)
    fn score_complexity(&self, memory: usize) -> f64 {
        let mb = memory as f64 / 1_000_000.0;
        let max_acceptable = 1000.0; // 1GB
        (1.0 - (mb / max_acceptable).min(1.0)).max(0.0)
    }
}

// ================================================================================================
// Feature Extraction
// ================================================================================================

/// Extract dataset features for meta-learning
pub fn extract_dataset_features(data: &TimeSeries) -> DatasetFeatures {
    let values = &data.values;

    DatasetFeatures {
        statistical: extract_statistical_features(values),
        temporal: extract_temporal_features(data),
        complexity: extract_complexity_features(values),
    }
}

/// Extract statistical features
fn extract_statistical_features(values: &[f64]) -> StatisticalFeatures {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let skewness = if std > 0.0 {
        values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };

    let kurtosis = if std > 0.0 {
        values.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n - 3.0
    } else {
        0.0
    };

    let autocorrelation = compute_autocorrelation(values, 10);

    StatisticalFeatures {
        mean,
        std,
        skewness,
        kurtosis,
        autocorrelation,
    }
}

/// Extract temporal features
fn extract_temporal_features(data: &TimeSeries) -> TemporalFeatures {
    let values = &data.values;

    // Simple trend strength using linear regression slope
    let trend_strength = estimate_trend_strength(values);

    // Simple seasonality detection (placeholder)
    let seasonality_strength = estimate_seasonality_strength(values);

    TemporalFeatures {
        trend_strength,
        seasonality_strength,
        frequency: data.frequency.clone(),
        n_observations: values.len(),
    }
}

/// Extract complexity features
fn extract_complexity_features(values: &[f64]) -> ComplexityFeatures {
    ComplexityFeatures {
        entropy: compute_entropy(values),
        lempel_ziv_complexity: compute_lempel_ziv(values),
        approximate_entropy: compute_approx_entropy(values),
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Compute autocorrelation for specified lags
fn compute_autocorrelation(values: &[f64], max_lag: usize) -> Vec<f64> {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();

    (1..=max_lag)
        .map(|lag| {
            let covariance: f64 = (0..n - lag)
                .map(|i| (values[i] - mean) * (values[i + lag] - mean))
                .sum();
            covariance / variance
        })
        .collect()
}

/// Estimate trend strength using linear regression
fn estimate_trend_strength(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator > 0.0 {
        (numerator / denominator).abs().min(1.0)
    } else {
        0.0
    }
}

/// Estimate seasonality strength (simplified)
fn estimate_seasonality_strength(values: &[f64]) -> f64 {
    if values.len() < 24 {
        return 0.0;
    }

    // Check autocorrelation at typical seasonal lags
    let seasonal_lags = [7, 12, 24, 30];
    let autocorr = compute_autocorrelation(values, 30);

    seasonal_lags
        .iter()
        .filter_map(|&lag| {
            if lag < autocorr.len() {
                Some(autocorr[lag].abs())
            } else {
                None
            }
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0)
}

/// Compute Shannon entropy (simplified)
fn compute_entropy(values: &[f64]) -> f64 {
    // Discretize values into bins
    let n_bins = 10;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max - min) / n_bins as f64;

    if bin_width == 0.0 {
        return 0.0;
    }

    let mut bins = vec![0; n_bins];
    for &value in values {
        let bin = ((value - min) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        bins[bin] += 1;
    }

    let n = values.len() as f64;
    bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Compute Lempel-Ziv complexity (simplified binary version)
fn compute_lempel_ziv(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    // Convert to binary sequence (above/below median)
    let median = {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let binary: Vec<bool> = values.iter().map(|&v| v > median).collect();

    // Count distinct subsequences
    let mut complexity = 1;
    let mut i = 0;

    while i < binary.len() {
        let mut matched = false;
        for len in (1..=i).rev() {
            if i + len <= binary.len() && binary[i - len..i] == binary[i..i + len] {
                i += len;
                matched = true;
                break;
            }
        }
        if !matched {
            complexity += 1;
            i += 1;
        }
    }

    complexity as f64 / binary.len() as f64
}

/// Compute approximate entropy
fn compute_approx_entropy(values: &[f64]) -> f64 {
    if values.len() < 10 {
        return 0.0;
    }

    let m = 2; // Pattern length
    let r = 0.2 * values.iter().map(|&v| v.abs()).sum::<f64>() / values.len() as f64; // Tolerance

    let phi_m = compute_phi(values, m, r);
    let phi_m1 = compute_phi(values, m + 1, r);

    phi_m - phi_m1
}

/// Helper for approximate entropy
fn compute_phi(values: &[f64], m: usize, r: f64) -> f64 {
    let n = values.len();
    if n <= m {
        return 0.0;
    }

    let mut counts = Vec::new();

    for i in 0..=(n - m) {
        let pattern = &values[i..i + m];
        let mut count = 0;

        for j in 0..=(n - m) {
            let other = &values[j..j + m];
            let max_diff = pattern
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            if max_diff <= r {
                count += 1;
            }
        }

        counts.push(count);
    }

    let n_patterns = counts.len() as f64;
    counts
        .iter()
        .map(|&c| {
            let p = c as f64 / n_patterns;
            if p > 0.0 {
                p.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n_patterns
}

// ================================================================================================
// Public API
// ================================================================================================

/// Create default model candidates
pub fn create_default_candidates() -> Vec<ModelCandidate> {
    use std::time::Duration;

    vec![
        ModelCandidate {
            model_type: ModelType::LSTM,
            default_hyperparameters: HashMap::new(),
            search_space: SearchSpace {
                categorical_params: HashMap::new(),
                integer_params: vec![
                    ("hidden_size".to_string(), (32, 256)),
                    ("num_layers".to_string(), (1, 4)),
                ]
                .into_iter()
                .collect(),
                float_params: vec![("dropout_rate".to_string(), (0.0, 0.5))]
                    .into_iter()
                    .collect(),
                conditional_params: vec![],
            },
            estimated_training_time: Duration::from_secs(600),
            memory_requirements: 500_000_000,
        },
        ModelCandidate {
            model_type: ModelType::GRU,
            default_hyperparameters: HashMap::new(),
            search_space: SearchSpace {
                categorical_params: HashMap::new(),
                integer_params: vec![
                    ("hidden_size".to_string(), (32, 256)),
                    ("num_layers".to_string(), (1, 4)),
                ]
                .into_iter()
                .collect(),
                float_params: vec![("dropout_rate".to_string(), (0.0, 0.5))]
                    .into_iter()
                    .collect(),
                conditional_params: vec![],
            },
            estimated_training_time: Duration::from_secs(500),
            memory_requirements: 450_000_000,
        },
        ModelCandidate {
            model_type: ModelType::Transformer,
            default_hyperparameters: HashMap::new(),
            search_space: SearchSpace {
                categorical_params: HashMap::new(),
                integer_params: vec![
                    ("d_model".to_string(), (64, 512)),
                    ("num_heads".to_string(), (2, 16)),
                    ("num_layers".to_string(), (1, 6)),
                ]
                .into_iter()
                .collect(),
                float_params: vec![("dropout_rate".to_string(), (0.0, 0.5))]
                    .into_iter()
                    .collect(),
                conditional_params: vec![],
            },
            estimated_training_time: Duration::from_secs(1200),
            memory_requirements: 800_000_000,
        },
    ]
}
