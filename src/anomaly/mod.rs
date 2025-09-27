//! # Anomaly Detection Engine
//!
//! Comprehensive anomaly detection capabilities for time series data using
//! statistical, machine learning, and contextual approaches.

pub mod statistical;
pub mod timeseries;
pub mod advanced;
pub mod contextual;
pub mod streaming;
pub mod scoring;
pub mod utils;

use crate::analysis::{AnomalyDetection, Anomaly};
use crate::TimeSeries;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Detection methods to apply
    pub methods: Vec<AnomalyMethod>,

    /// Threshold configuration
    pub thresholds: ThresholdConfig,

    /// Contextual detection settings
    pub contextual: ContextualConfig,

    /// Scoring and ranking configuration
    pub scoring: ScoringConfig,

    /// Real-time detection settings
    pub streaming: StreamingConfig,
}

/// Available anomaly detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyMethod {
    // Statistical methods
    ZScore { threshold: f64 },
    ModifiedZScore { threshold: f64 },
    IQR { factor: f64 },
    Grubbs { alpha: f64 },

    // Time series specific
    SeasonalDecomposition { period: usize },
    TrendDeviation { window_size: usize },
    LevelShift { threshold: f64 },
    VolatilityAnomaly { window_size: usize },

    // Advanced ML-based
    IsolationForest { contamination: f64, n_trees: usize },
    LocalOutlierFactor { n_neighbors: usize, contamination: f64 },
    DBSCANClustering { eps: f64, min_samples: usize },

    // Contextual
    DayOfWeekAdjusted { baseline_periods: usize },
    SeasonalContext { seasonal_periods: Vec<usize> },
    MultiVariate { variables: Vec<String> },
}

/// Threshold configuration for different methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Statistical thresholds
    pub statistical: HashMap<String, f64>,

    /// Time series specific thresholds
    pub timeseries: HashMap<String, f64>,

    /// Advanced algorithm parameters
    pub advanced: HashMap<String, f64>,

    /// Severity classification thresholds
    pub severity_thresholds: SeverityThresholds,
}

/// Severity classification thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityThresholds {
    /// Low severity threshold
    pub low: f64,

    /// Medium severity threshold
    pub medium: f64,

    /// High severity threshold
    pub high: f64,

    /// Critical severity threshold
    pub critical: f64,
}

/// Contextual anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualConfig {
    /// Enable day-of-week adjustment
    pub day_of_week_adjustment: bool,

    /// Enable seasonal context
    pub seasonal_context: bool,

    /// Seasonal periods to consider
    pub seasonal_periods: Vec<usize>,

    /// Baseline periods for context establishment
    pub baseline_periods: usize,
}

/// Scoring and ranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Scoring method
    pub method: ScoringMethod,

    /// Weight combinations for multi-method detection
    pub method_weights: HashMap<String, f64>,

    /// Enable anomaly ranking
    pub enable_ranking: bool,

    /// Maximum number of top anomalies to report
    pub max_top_anomalies: usize,
}

/// Anomaly scoring methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringMethod {
    /// Use maximum score across methods
    Maximum,

    /// Use weighted average of scores
    WeightedAverage,

    /// Use ensemble voting
    EnsembleVoting,

    /// Use custom scoring function
    Custom(String),
}

/// Streaming anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable real-time detection
    pub enable_streaming: bool,

    /// Window size for streaming detection
    pub window_size: usize,

    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,

    /// Adaptation rate for thresholds
    pub adaptation_rate: f64,

    /// Enable online learning
    pub online_learning: bool,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                AnomalyMethod::ZScore { threshold: 3.0 },
                AnomalyMethod::IQR { factor: 1.5 },
            ],
            thresholds: ThresholdConfig::default(),
            contextual: ContextualConfig::default(),
            scoring: ScoringConfig::default(),
            streaming: StreamingConfig::default(),
        }
    }
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        let mut statistical = HashMap::new();
        statistical.insert("zscore".to_string(), 3.0);
        statistical.insert("modified_zscore".to_string(), 3.5);
        statistical.insert("iqr_factor".to_string(), 1.5);
        statistical.insert("grubbs_alpha".to_string(), 0.05);

        let mut timeseries = HashMap::new();
        timeseries.insert("trend_deviation".to_string(), 2.0);
        timeseries.insert("level_shift".to_string(), 3.0);
        timeseries.insert("volatility_threshold".to_string(), 2.5);

        let mut advanced = HashMap::new();
        advanced.insert("isolation_forest_contamination".to_string(), 0.1);
        advanced.insert("lof_contamination".to_string(), 0.1);
        advanced.insert("dbscan_eps".to_string(), 0.5);

        Self {
            statistical,
            timeseries,
            advanced,
            severity_thresholds: SeverityThresholds {
                low: 1.0,
                medium: 2.0,
                high: 3.0,
                critical: 4.0,
            },
        }
    }
}

impl Default for ContextualConfig {
    fn default() -> Self {
        Self {
            day_of_week_adjustment: false,
            seasonal_context: false,
            seasonal_periods: vec![7, 30, 365],
            baseline_periods: 100,
        }
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            method: ScoringMethod::Maximum,
            method_weights: HashMap::new(),
            enable_ranking: true,
            max_top_anomalies: 10,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: false,
            window_size: 100,
            adaptive_thresholds: false,
            adaptation_rate: 0.01,
            online_learning: false,
        }
    }
}

/// Main anomaly detection function
pub fn detect_anomalies(
    timeseries: &TimeSeries,
    config: &AnomalyDetectionConfig,
) -> Result<AnomalyDetection, Box<dyn std::error::Error>> {
    let mut all_anomalies = Vec::new();
    let mut method_results = HashMap::new();

    // Apply each detection method
    for method in &config.methods {
        let anomalies = match method {
            // Statistical methods
            AnomalyMethod::ZScore { threshold } => {
                statistical::detect_zscore_anomalies(timeseries, *threshold)?
            }
            AnomalyMethod::ModifiedZScore { threshold } => {
                statistical::detect_modified_zscore_anomalies(timeseries, *threshold)?
            }
            AnomalyMethod::IQR { factor } => {
                statistical::detect_iqr_anomalies(timeseries, *factor)?
            }
            AnomalyMethod::Grubbs { alpha } => {
                statistical::detect_grubbs_anomalies(timeseries, *alpha)?
            }

            // Time series specific methods
            AnomalyMethod::SeasonalDecomposition { period } => {
                timeseries::detect_seasonal_anomalies(timeseries, *period)?
            }
            AnomalyMethod::TrendDeviation { window_size } => {
                timeseries::detect_trend_deviation_anomalies(timeseries, *window_size)?
            }
            AnomalyMethod::LevelShift { threshold } => {
                timeseries::detect_level_shift_anomalies(timeseries, *threshold)?
            }
            AnomalyMethod::VolatilityAnomaly { window_size } => {
                timeseries::detect_volatility_anomalies(timeseries, *window_size)?
            }

            // Advanced ML-based methods
            AnomalyMethod::IsolationForest { contamination, n_trees } => {
                advanced::detect_isolation_forest_anomalies(timeseries, *contamination, *n_trees)?
            }
            AnomalyMethod::LocalOutlierFactor { n_neighbors, contamination } => {
                advanced::detect_lof_anomalies(timeseries, *n_neighbors, *contamination)?
            }
            AnomalyMethod::DBSCANClustering { eps, min_samples } => {
                advanced::detect_dbscan_anomalies(timeseries, *eps, *min_samples)?
            }

            // Contextual methods
            AnomalyMethod::DayOfWeekAdjusted { baseline_periods } => {
                contextual::detect_day_of_week_anomalies(timeseries, *baseline_periods)?
            }
            AnomalyMethod::SeasonalContext { seasonal_periods } => {
                contextual::detect_seasonal_context_anomalies(timeseries, seasonal_periods)?
            }
            AnomalyMethod::MultiVariate { variables: _ } => {
                // TODO: Implement multivariate detection
                Vec::new()
            }
        };

        let method_name = format!("{:?}", method);
        method_results.insert(method_name.clone(), anomalies.clone());
        all_anomalies.extend(anomalies);
    }

    // Combine and score anomalies
    let combined_anomalies = scoring::combine_and_score_anomalies(
        all_anomalies,
        &method_results,
        &config.scoring,
    )?;

    // Apply severity classification
    let classified_anomalies = scoring::classify_anomaly_severity(
        combined_anomalies,
        &config.thresholds.severity_thresholds,
    )?;

    // Create result
    let mut detection = AnomalyDetection::new("combined".to_string(), 0.0);
    detection.anomalies = classified_anomalies;
    detection.update_statistics(timeseries.values.len());

    Ok(detection)
}

/// Detect anomalies using a single method
pub fn detect_anomalies_single_method(
    timeseries: &TimeSeries,
    method: &AnomalyMethod,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    match method {
        AnomalyMethod::ZScore { threshold } => {
            statistical::detect_zscore_anomalies(timeseries, *threshold)
        }
        AnomalyMethod::ModifiedZScore { threshold } => {
            statistical::detect_modified_zscore_anomalies(timeseries, *threshold)
        }
        AnomalyMethod::IQR { factor } => {
            statistical::detect_iqr_anomalies(timeseries, *factor)
        }
        AnomalyMethod::Grubbs { alpha } => {
            statistical::detect_grubbs_anomalies(timeseries, *alpha)
        }
        _ => {
            // For other methods, use the full detection pipeline
            let config = AnomalyDetectionConfig {
                methods: vec![method.clone()],
                ..Default::default()
            };
            let result = detect_anomalies(timeseries, &config)?;
            Ok(result.anomalies)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_default_config() {
        let config = AnomalyDetectionConfig::default();
        assert_eq!(config.methods.len(), 2);
        assert!(config.scoring.enable_ranking);
        assert_eq!(config.scoring.max_top_anomalies, 10);
    }

    #[test]
    fn test_anomaly_method_serialization() {
        let method = AnomalyMethod::ZScore { threshold: 3.0 };
        let serialized = serde_json::to_string(&method).unwrap();
        let deserialized: AnomalyMethod = serde_json::from_str(&serialized).unwrap();
        assert_eq!(method, deserialized);
    }
}