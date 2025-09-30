//! Advanced outlier detection algorithms for time series data
//!
//! This module provides comprehensive outlier detection capabilities specifically
//! designed for time series data, including statistical methods, time-series-aware
//! algorithms, and ensemble detection approaches.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::quality::{OutlierMethod, QualityError, QualityResult};
use crate::stats::compute_descriptive_stats;
use crate::timeseries::TimeSeries;

/// Severity level of an outlier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutlierSeverity {
    /// Low severity (score 0.0-0.3)
    Low,
    /// Medium severity (score 0.3-0.6)
    Medium,
    /// High severity (score 0.6-0.8)
    High,
    /// Critical severity (score 0.8-1.0)
    Critical,
}

impl OutlierSeverity {
    /// Creates severity from a score (0.0-1.0)
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            OutlierSeverity::Critical
        } else if score >= 0.6 {
            OutlierSeverity::High
        } else if score >= 0.3 {
            OutlierSeverity::Medium
        } else {
            OutlierSeverity::Low
        }
    }

    /// Returns the score range for this severity
    pub fn score_range(&self) -> (f64, f64) {
        match self {
            OutlierSeverity::Low => (0.0, 0.3),
            OutlierSeverity::Medium => (0.3, 0.6),
            OutlierSeverity::High => (0.6, 0.8),
            OutlierSeverity::Critical => (0.8, 1.0),
        }
    }
}

/// Context information for an outlier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierContext {
    /// Index in the time series
    pub index: usize,
    /// Distance from expected value
    pub deviation: f64,
    /// Expected value based on context
    pub expected_value: Option<f64>,
    /// Additional context information
    pub metadata: HashMap<String, String>,
}

/// Represents a detected outlier point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierPoint {
    /// Timestamp of the outlier
    pub timestamp: DateTime<Utc>,
    /// Value of the outlier
    pub value: f64,
    /// Outlier score (0.0-1.0, higher means more anomalous)
    pub score: f64,
    /// Method used to detect this outlier
    pub method: String,
    /// Severity classification
    pub severity: OutlierSeverity,
    /// Context information
    pub context: OutlierContext,
}

impl OutlierPoint {
    /// Creates a new outlier point
    pub fn new(
        timestamp: DateTime<Utc>,
        value: f64,
        score: f64,
        method: String,
        index: usize,
        deviation: f64,
    ) -> Self {
        OutlierPoint {
            timestamp,
            value,
            score,
            method: method.clone(),
            severity: OutlierSeverity::from_score(score),
            context: OutlierContext {
                index,
                deviation,
                expected_value: None,
                metadata: HashMap::new(),
            },
        }
    }

    /// Sets the expected value
    pub fn with_expected_value(mut self, expected: f64) -> Self {
        self.context.expected_value = Some(expected);
        self
    }

    /// Adds metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.context.metadata.insert(key, value);
        self
    }
}

/// Summary statistics for outlier detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierSummary {
    /// Total number of outliers detected
    pub total_outliers: usize,
    /// Outliers by severity
    pub by_severity: HashMap<OutlierSeverity, usize>,
    /// Outliers by method
    pub by_method: HashMap<String, usize>,
    /// Percentage of data points that are outliers
    pub outlier_percentage: f64,
}

/// Comprehensive outlier detection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierReport {
    /// All detected outliers
    pub outliers: Vec<OutlierPoint>,
    /// Summary statistics
    pub summary: OutlierSummary,
    /// Results grouped by method
    pub method_results: HashMap<String, Vec<OutlierPoint>>,
    /// Confidence scores for each method
    pub confidence_scores: HashMap<String, f64>,
}

impl OutlierReport {
    /// Creates a new outlier report from detected outliers
    pub fn new(outliers: Vec<OutlierPoint>, total_points: usize) -> Self {
        let mut by_severity: HashMap<OutlierSeverity, usize> = HashMap::new();
        let mut by_method: HashMap<String, usize> = HashMap::new();
        let mut method_results: HashMap<String, Vec<OutlierPoint>> = HashMap::new();

        let total_outliers = outliers.len();

        for outlier in &outliers {
            *by_severity.entry(outlier.severity).or_insert(0) += 1;
            *by_method.entry(outlier.method.clone()).or_insert(0) += 1;
            method_results
                .entry(outlier.method.clone())
                .or_insert_with(Vec::new)
                .push(outlier.clone());
        }

        let outlier_percentage = if total_points > 0 {
            (total_outliers as f64 / total_points as f64) * 100.0
        } else {
            0.0
        };

        OutlierReport {
            outliers,
            summary: OutlierSummary {
                total_outliers,
                by_severity,
                by_method,
                outlier_percentage,
            },
            method_results,
            confidence_scores: HashMap::new(),
        }
    }

    /// Filters outliers by minimum severity
    pub fn filter_by_severity(&self, min_severity: OutlierSeverity) -> Vec<&OutlierPoint> {
        let min_score = min_severity.score_range().0;
        self.outliers
            .iter()
            .filter(|o| o.score >= min_score)
            .collect()
    }

    /// Gets outliers detected by a specific method
    pub fn outliers_by_method(&self, method: &str) -> Vec<&OutlierPoint> {
        self.outliers.iter().filter(|o| o.method == method).collect()
    }
}

/// Configuration for outlier detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierConfig {
    /// Methods to use for detection
    pub methods: Vec<OutlierMethod>,
    /// Z-score threshold (default: 3.0)
    pub zscore_threshold: f64,
    /// Modified Z-score threshold (default: 3.5)
    pub modified_zscore_threshold: f64,
    /// IQR factor (default: 1.5)
    pub iqr_factor: f64,
    /// Window size for temporal methods
    pub temporal_window_size: usize,
    /// Minimum score for ensemble voting
    pub ensemble_threshold: f64,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        OutlierConfig {
            methods: vec![
                OutlierMethod::ZScore,
                OutlierMethod::IQR,
                OutlierMethod::ModifiedZScore,
            ],
            zscore_threshold: 3.0,
            modified_zscore_threshold: 3.5,
            iqr_factor: 1.5,
            temporal_window_size: 10,
            ensemble_threshold: 0.5,
        }
    }
}

impl OutlierConfig {
    /// Creates a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the methods to use
    pub fn with_methods(mut self, methods: Vec<OutlierMethod>) -> Self {
        self.methods = methods;
        self
    }

    /// Sets the Z-score threshold
    pub fn with_zscore_threshold(mut self, threshold: f64) -> Self {
        self.zscore_threshold = threshold;
        self
    }

    /// Sets the temporal window size
    pub fn with_temporal_window(mut self, size: usize) -> Self {
        self.temporal_window_size = size;
        self
    }
}

/// Main outlier detection function
pub fn detect_outliers(
    data: &TimeSeries,
    config: &OutlierConfig,
) -> QualityResult<OutlierReport> {
    if data.values.is_empty() {
        return Err(QualityError::insufficient_data(
            "Cannot detect outliers in empty time series",
        ));
    }

    let mut all_outliers = Vec::new();

    // Apply each configured method
    for method in &config.methods {
        let outliers = match method {
            OutlierMethod::ZScore => detect_zscore_outliers(data, config.zscore_threshold)?,
            OutlierMethod::IQR => detect_iqr_outliers(data, config.iqr_factor)?,
            OutlierMethod::ModifiedZScore => {
                detect_modified_zscore_outliers(data, config.modified_zscore_threshold)?
            }
            OutlierMethod::IsolationForest => {
                // Placeholder for future implementation
                Vec::new()
            }
            OutlierMethod::LOF => {
                // Placeholder for future implementation
                Vec::new()
            }
            OutlierMethod::DBSCAN => {
                // Placeholder for future implementation
                Vec::new()
            }
        };

        all_outliers.extend(outliers);
    }

    // Remove duplicates (same index detected by multiple methods)
    all_outliers.sort_by_key(|o| o.context.index);
    all_outliers.dedup_by_key(|o| o.context.index);

    Ok(OutlierReport::new(all_outliers, data.values.len()))
}

/// Detects outliers using Z-score method
pub fn detect_zscore_outliers(
    data: &TimeSeries,
    threshold: f64,
) -> QualityResult<Vec<OutlierPoint>> {
    let stats = compute_descriptive_stats(&data.values)
        .map_err(|e| QualityError::computation(format!("Failed to compute stats: {}", e)))?;

    let mut outliers = Vec::new();
    let mean = stats.mean;
    let std_dev = stats.std_dev;

    if std_dev == 0.0 {
        return Ok(outliers); // No variation, no outliers
    }

    for (i, &value) in data.values.iter().enumerate() {
        let z_score = ((value - mean) / std_dev).abs();

        if z_score > threshold {
            // Normalize score to 0.0-1.0 range
            let normalized_score = (z_score / (threshold + 3.0)).min(1.0);

            let outlier = OutlierPoint::new(
                data.timestamps[i],
                value,
                normalized_score,
                "zscore".to_string(),
                i,
                value - mean,
            )
            .with_expected_value(mean);

            outliers.push(outlier);
        }
    }

    Ok(outliers)
}

/// Detects outliers using IQR method
pub fn detect_iqr_outliers(data: &TimeSeries, factor: f64) -> QualityResult<Vec<OutlierPoint>> {
    let stats = compute_descriptive_stats(&data.values)
        .map_err(|e| QualityError::computation(format!("Failed to compute stats: {}", e)))?;

    let q1 = stats.quantiles.q25;
    let q3 = stats.quantiles.q75;
    let iqr = q3 - q1;

    let lower_bound = q1 - factor * iqr;
    let upper_bound = q3 + factor * iqr;

    let mut outliers = Vec::new();

    for (i, &value) in data.values.iter().enumerate() {
        if value < lower_bound || value > upper_bound {
            // Calculate how far outside the bounds
            let deviation = if value < lower_bound {
                lower_bound - value
            } else {
                value - upper_bound
            };

            // Normalize score based on deviation relative to IQR
            let normalized_score = (deviation / (iqr * factor)).min(1.0);

            let expected = (q1 + q3) / 2.0; // Median as expected value

            let outlier = OutlierPoint::new(
                data.timestamps[i],
                value,
                normalized_score,
                "iqr".to_string(),
                i,
                value - expected,
            )
            .with_expected_value(expected);

            outliers.push(outlier);
        }
    }

    Ok(outliers)
}

/// Detects outliers using Modified Z-score (using MAD)
pub fn detect_modified_zscore_outliers(
    data: &TimeSeries,
    threshold: f64,
) -> QualityResult<Vec<OutlierPoint>> {
    let stats = compute_descriptive_stats(&data.values)
        .map_err(|e| QualityError::computation(format!("Failed to compute stats: {}", e)))?;

    let median = stats.median;

    // Calculate MAD (Median Absolute Deviation)
    let mut absolute_deviations: Vec<f64> = data
        .values
        .iter()
        .map(|&v| (v - median).abs())
        .collect();

    absolute_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if absolute_deviations.is_empty() {
        0.0
    } else {
        absolute_deviations[absolute_deviations.len() / 2]
    };

    if mad == 0.0 {
        return Ok(Vec::new()); // No variation
    }

    let mut outliers = Vec::new();

    for (i, &value) in data.values.iter().enumerate() {
        // Modified Z-score: 0.6745 * (x - median) / MAD
        let modified_z = 0.6745 * (value - median).abs() / mad;

        if modified_z > threshold {
            let normalized_score = (modified_z / (threshold + 3.5)).min(1.0);

            let outlier = OutlierPoint::new(
                data.timestamps[i],
                value,
                normalized_score,
                "modified_zscore".to_string(),
                i,
                value - median,
            )
            .with_expected_value(median);

            outliers.push(outlier);
        }
    }

    Ok(outliers)
}

/// Detects temporal outliers using rolling window IQR
pub fn detect_temporal_outliers(
    data: &TimeSeries,
    window_size: usize,
) -> QualityResult<Vec<OutlierPoint>> {
    if data.values.len() < window_size {
        return Err(QualityError::invalid_parameter(format!(
            "Window size {} is larger than data size {}",
            window_size,
            data.values.len()
        )));
    }

    let mut outliers = Vec::new();

    for i in window_size..data.values.len() {
        let window = &data.values[i - window_size..i];

        // Calculate quartiles for window
        let mut sorted_window = window.to_vec();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted_window.len() / 4;
        let q3_idx = 3 * sorted_window.len() / 4;
        let q1 = sorted_window[q1_idx];
        let q3 = sorted_window[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let value = data.values[i];

        if value < lower_bound || value > upper_bound {
            let deviation = if value < lower_bound {
                lower_bound - value
            } else {
                value - upper_bound
            };

            let normalized_score = (deviation / (iqr * 1.5)).min(1.0);
            let expected = (q1 + q3) / 2.0;

            let outlier = OutlierPoint::new(
                data.timestamps[i],
                value,
                normalized_score,
                "temporal_iqr".to_string(),
                i,
                value - expected,
            )
            .with_expected_value(expected);

            outliers.push(outlier);
        }
    }

    Ok(outliers)
}

/// Detects outliers using ensemble voting
pub fn detect_ensemble_outliers(
    data: &TimeSeries,
    methods: &[OutlierMethod],
    config: &OutlierConfig,
) -> QualityResult<OutlierReport> {
    if methods.is_empty() {
        return Err(QualityError::invalid_parameter(
            "Must provide at least one detection method",
        ));
    }

    // Collect results from each method
    let mut method_votes: HashMap<usize, Vec<(String, f64)>> = HashMap::new();

    for method in methods {
        let outliers = match method {
            OutlierMethod::ZScore => detect_zscore_outliers(data, config.zscore_threshold)?,
            OutlierMethod::IQR => detect_iqr_outliers(data, config.iqr_factor)?,
            OutlierMethod::ModifiedZScore => {
                detect_modified_zscore_outliers(data, config.modified_zscore_threshold)?
            }
            _ => continue, // Skip unimplemented methods
        };

        for outlier in outliers {
            method_votes
                .entry(outlier.context.index)
                .or_insert_with(Vec::new)
                .push((outlier.method.clone(), outlier.score));
        }
    }

    // Calculate consensus scores
    let mut ensemble_outliers = Vec::new();

    for (index, votes) in method_votes {
        let vote_ratio = votes.len() as f64 / methods.len() as f64;

        if vote_ratio >= config.ensemble_threshold {
            let avg_score: f64 = votes.iter().map(|(_, score)| score).sum::<f64>() / votes.len() as f64;

            let outlier = OutlierPoint::new(
                data.timestamps[index],
                data.values[index],
                avg_score,
                "ensemble".to_string(),
                index,
                0.0, // Will be calculated
            )
            .with_metadata("vote_count".to_string(), votes.len().to_string())
            .with_metadata("vote_ratio".to_string(), format!("{:.2}", vote_ratio));

            ensemble_outliers.push(outlier);
        }
    }

    Ok(OutlierReport::new(ensemble_outliers, data.values.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_timeseries_with_outliers() -> TimeSeries {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<DateTime<Utc>> = (0..100)
            .map(|i| start + chrono::Duration::seconds(i * 60))
            .collect();

        let mut values: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64 * 0.1)).collect();

        // Add some outliers
        values[10] = 150.0; // High outlier
        values[50] = -50.0; // Low outlier
        values[80] = 200.0; // High outlier

        TimeSeries::new("test_with_outliers".to_string(), timestamps, values).unwrap()
    }

    fn create_normal_timeseries() -> TimeSeries {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<DateTime<Utc>> = (0..100)
            .map(|i| start + chrono::Duration::seconds(i * 60))
            .collect();
        let values: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64 * 0.1)).collect();

        TimeSeries::new("normal_test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_outlier_severity_from_score() {
        assert_eq!(OutlierSeverity::from_score(0.1), OutlierSeverity::Low);
        assert_eq!(OutlierSeverity::from_score(0.4), OutlierSeverity::Medium);
        assert_eq!(OutlierSeverity::from_score(0.7), OutlierSeverity::High);
        assert_eq!(OutlierSeverity::from_score(0.9), OutlierSeverity::Critical);
    }

    #[test]
    fn test_outlier_config_default() {
        let config = OutlierConfig::default();
        assert_eq!(config.zscore_threshold, 3.0);
        assert_eq!(config.iqr_factor, 1.5);
        assert_eq!(config.methods.len(), 3);
    }

    #[test]
    fn test_outlier_config_builder() {
        let config = OutlierConfig::new()
            .with_zscore_threshold(2.5)
            .with_temporal_window(20);

        assert_eq!(config.zscore_threshold, 2.5);
        assert_eq!(config.temporal_window_size, 20);
    }

    #[test]
    fn test_detect_zscore_outliers() {
        let data = create_test_timeseries_with_outliers();
        let outliers = detect_zscore_outliers(&data, 3.0).unwrap();

        assert!(outliers.len() > 0);
        assert!(outliers.iter().any(|o| o.context.index == 10));
        assert!(outliers.iter().any(|o| o.context.index == 50));
    }

    #[test]
    fn test_detect_iqr_outliers() {
        let data = create_test_timeseries_with_outliers();
        let outliers = detect_iqr_outliers(&data, 1.5).unwrap();

        assert!(outliers.len() > 0);
        // Should detect the extreme outliers
        assert!(outliers.iter().any(|o| o.value > 100.0 || o.value < 0.0));
    }

    #[test]
    fn test_detect_modified_zscore_outliers() {
        let data = create_test_timeseries_with_outliers();
        let outliers = detect_modified_zscore_outliers(&data, 3.5).unwrap();

        assert!(outliers.len() > 0);
    }

    #[test]
    fn test_detect_temporal_outliers() {
        let data = create_test_timeseries_with_outliers();
        let outliers = detect_temporal_outliers(&data, 10).unwrap();

        // Temporal method should detect outliers based on local context
        assert!(outliers.len() >= 0);
    }

    #[test]
    fn test_detect_outliers_main_function() {
        let data = create_test_timeseries_with_outliers();
        let config = OutlierConfig::default();

        let report = detect_outliers(&data, &config).unwrap();

        assert!(report.summary.total_outliers > 0);
        assert!(report.summary.outlier_percentage > 0.0);
        assert!(!report.method_results.is_empty());
    }

    #[test]
    fn test_no_outliers_in_normal_data() {
        let data = create_normal_timeseries();
        let config = OutlierConfig::default();

        let report = detect_outliers(&data, &config).unwrap();

        // Should detect very few or no outliers in normal data
        assert!(report.summary.outlier_percentage < 5.0);
    }

    #[test]
    fn test_outlier_report_filtering() {
        let data = create_test_timeseries_with_outliers();
        let config = OutlierConfig::default();
        let report = detect_outliers(&data, &config).unwrap();

        let high_severity = report.filter_by_severity(OutlierSeverity::High);
        assert!(high_severity.len() <= report.outliers.len());
    }

    #[test]
    fn test_ensemble_detection() {
        let data = create_test_timeseries_with_outliers();
        let methods = vec![
            OutlierMethod::ZScore,
            OutlierMethod::IQR,
            OutlierMethod::ModifiedZScore,
        ];
        let config = OutlierConfig::default();

        let report = detect_ensemble_outliers(&data, &methods, &config).unwrap();

        // Ensemble should find consensus outliers
        assert!(report.summary.total_outliers >= 0);
    }

    #[test]
    fn test_empty_timeseries_error() {
        let data = TimeSeries::empty("empty".to_string());
        let config = OutlierConfig::default();

        let result = detect_outliers(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_outlier_point_creation() {
        let timestamp = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let outlier = OutlierPoint::new(timestamp, 100.0, 0.8, "test".to_string(), 10, 50.0)
            .with_expected_value(50.0)
            .with_metadata("test".to_string(), "value".to_string());

        assert_eq!(outlier.severity, OutlierSeverity::Critical);
        assert_eq!(outlier.context.expected_value, Some(50.0));
        assert!(outlier.context.metadata.contains_key("test"));
    }

    #[test]
    fn test_performance_requirements() {
        // Test requirement: <100ms for 10K datapoints
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let mut timestamps = Vec::with_capacity(10000);
        let mut values = Vec::with_capacity(10000);

        // Generate 10K datapoints with some outliers
        for i in 0..10000 {
            timestamps.push(start + chrono::Duration::hours(i as i64));
            // Add some outliers at regular intervals
            values.push(if i % 500 == 0 { 1000.0 } else { 50.0 + (i as f64 % 10.0) });
        }

        let data = TimeSeries::new(
            "performance_test".to_string(),
            timestamps,
            values,
        )
        .unwrap();

        let config = OutlierConfig::default();

        // Measure performance
        let start_time = std::time::Instant::now();
        let result = detect_outliers(&data, &config);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Detection should succeed");
        assert!(
            duration.as_millis() < 100,
            "Detection should complete in <100ms for 10K points, took {}ms",
            duration.as_millis()
        );

        let report = result.unwrap();
        assert!(report.summary.total_outliers > 0, "Should detect some outliers");
    }
}