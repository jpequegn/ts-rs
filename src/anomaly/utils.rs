//! # Anomaly Detection Utilities
//!
//! Common utility functions and helpers for anomaly detection algorithms.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::TimeSeries;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Common statistical functions and utilities
pub struct StatUtils;

impl StatUtils {
    /// Calculate mean of a slice of values
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate variance of a slice of values
    pub fn variance(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = Self::mean(values);
        values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64
    }

    /// Calculate standard deviation of a slice of values
    pub fn std_dev(values: &[f64]) -> f64 {
        Self::variance(values).sqrt()
    }

    /// Calculate median of a slice of values
    pub fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }

    /// Calculate percentile of a slice of values
    pub fn percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() == 1 {
            return sorted[0];
        }

        let index = (percentile / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    /// Calculate quartiles (Q1, Q2, Q3)
    pub fn quartiles(values: &[f64]) -> (f64, f64, f64) {
        let q1 = Self::percentile(values, 25.0);
        let q2 = Self::percentile(values, 50.0);
        let q3 = Self::percentile(values, 75.0);
        (q1, q2, q3)
    }

    /// Calculate Interquartile Range (IQR)
    pub fn iqr(values: &[f64]) -> f64 {
        let (q1, _, q3) = Self::quartiles(values);
        q3 - q1
    }

    /// Calculate Median Absolute Deviation (MAD)
    pub fn mad(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let median = Self::median(values);
        let deviations: Vec<f64> = values
            .iter()
            .map(|v| (v - median).abs())
            .collect();

        Self::median(&deviations)
    }

    /// Calculate skewness of a distribution
    pub fn skewness(values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }

        let mean = Self::mean(values);
        let std_dev = Self::std_dev(values);

        if std_dev == 0.0 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_cubed_deviations: f64 = values
            .iter()
            .map(|v| ((v - mean) / std_dev).powi(3))
            .sum();

        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed_deviations
    }

    /// Calculate kurtosis of a distribution
    pub fn kurtosis(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }

        let mean = Self::mean(values);
        let std_dev = Self::std_dev(values);

        if std_dev == 0.0 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_fourth_deviations: f64 = values
            .iter()
            .map(|v| ((v - mean) / std_dev).powi(4))
            .sum();

        let factor1 = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let factor2 = (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));

        factor1 * sum_fourth_deviations - factor2
    }
}

/// Data preprocessing utilities
pub struct DataUtils;

impl DataUtils {
    /// Extract valid (non-NaN) data points with indices and timestamps
    pub fn extract_valid_data(timeseries: &TimeSeries) -> Vec<(usize, f64, DateTime<Utc>)> {
        timeseries
            .values
            .iter()
            .enumerate()
            .zip(timeseries.timestamps.iter())
            .filter_map(|((i, &val), &timestamp)| {
                if !val.is_nan() {
                    Some((i, val, timestamp))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Extract only values (no indices or timestamps)
    pub fn extract_valid_values(timeseries: &TimeSeries) -> Vec<f64> {
        timeseries
            .values
            .iter()
            .filter(|&&val| !val.is_nan())
            .copied()
            .collect()
    }

    /// Check if time series has enough data for analysis
    pub fn has_sufficient_data(timeseries: &TimeSeries, min_points: usize) -> bool {
        let valid_points = Self::extract_valid_values(timeseries);
        valid_points.len() >= min_points
    }

    /// Normalize values to z-scores
    pub fn normalize_zscore(values: &[f64]) -> Vec<f64> {
        let mean = StatUtils::mean(values);
        let std_dev = StatUtils::std_dev(values);

        if std_dev == 0.0 {
            return vec![0.0; values.len()];
        }

        values
            .iter()
            .map(|v| (v - mean) / std_dev)
            .collect()
    }

    /// Normalize values to 0-1 range (min-max scaling)
    pub fn normalize_minmax(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f64::EPSILON {
            return vec![0.5; values.len()];
        }

        values
            .iter()
            .map(|v| (v - min_val) / (max_val - min_val))
            .collect()
    }

    /// Remove outliers using IQR method
    pub fn remove_outliers_iqr(values: &[f64], factor: f64) -> Vec<f64> {
        let (q1, _, q3) = StatUtils::quartiles(values);
        let iqr = q3 - q1;
        let lower_bound = q1 - factor * iqr;
        let upper_bound = q3 + factor * iqr;

        values
            .iter()
            .filter(|&&v| v >= lower_bound && v <= upper_bound)
            .copied()
            .collect()
    }

    /// Apply smoothing using simple moving average
    pub fn smooth_moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
        if window_size == 0 || window_size > values.len() {
            return values.to_vec();
        }

        let mut smoothed = Vec::new();

        for i in 0..values.len() {
            let start = if i >= window_size - 1 { i - window_size + 1 } else { 0 };
            let end = i + 1;
            let window_mean = StatUtils::mean(&values[start..end]);
            smoothed.push(window_mean);
        }

        smoothed
    }

    /// Calculate rolling statistics
    pub fn rolling_statistics(values: &[f64], window_size: usize) -> RollingStats {
        let mut means = Vec::new();
        let mut std_devs = Vec::new();
        let mut medians = Vec::new();

        for i in 0..values.len() {
            let start = if i >= window_size - 1 { i - window_size + 1 } else { 0 };
            let end = i + 1;
            let window = &values[start..end];

            means.push(StatUtils::mean(window));
            std_devs.push(StatUtils::std_dev(window));
            medians.push(StatUtils::median(window));
        }

        RollingStats {
            means,
            std_devs,
            medians,
        }
    }
}

/// Rolling statistics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingStats {
    pub means: Vec<f64>,
    pub std_devs: Vec<f64>,
    pub medians: Vec<f64>,
}

/// Threshold utilities
pub struct ThresholdUtils;

impl ThresholdUtils {
    /// Calculate adaptive threshold based on historical data
    pub fn adaptive_threshold(
        values: &[f64],
        base_threshold: f64,
        adaptation_factor: f64,
    ) -> f64 {
        if values.is_empty() {
            return base_threshold;
        }

        let std_dev = StatUtils::std_dev(values);
        let variability_factor = 1.0 + (std_dev * adaptation_factor);

        base_threshold * variability_factor
    }

    /// Calculate robust threshold using MAD
    pub fn robust_threshold(values: &[f64], mad_factor: f64) -> f64 {
        let mad = StatUtils::mad(values);
        mad_factor * mad
    }

    /// Calculate percentile-based threshold
    pub fn percentile_threshold(values: &[f64], percentile: f64) -> f64 {
        StatUtils::percentile(values, percentile)
    }

    /// Calculate dynamic threshold based on recent trend
    pub fn dynamic_threshold(
        values: &[f64],
        window_size: usize,
        base_threshold: f64,
    ) -> f64 {
        if values.len() < window_size {
            return base_threshold;
        }

        let recent_values = &values[values.len() - window_size..];
        let recent_std = StatUtils::std_dev(recent_values);
        let overall_std = StatUtils::std_dev(values);

        if overall_std == 0.0 {
            return base_threshold;
        }

        let volatility_ratio = recent_std / overall_std;
        base_threshold * (1.0 + volatility_ratio)
    }
}

/// Anomaly utilities
pub struct AnomalyUtils;

impl AnomalyUtils {
    /// Merge overlapping anomalies
    pub fn merge_overlapping_anomalies(
        mut anomalies: Vec<Anomaly>,
        max_gap: usize,
    ) -> Vec<Anomaly> {
        if anomalies.is_empty() {
            return anomalies;
        }

        // Sort by index
        anomalies.sort_by_key(|a| a.index);

        let mut merged = Vec::new();
        let mut current = anomalies[0].clone();

        for anomaly in anomalies.into_iter().skip(1) {
            if anomaly.index <= current.index + max_gap {
                // Merge anomalies - keep the one with higher score
                if anomaly.score > current.score {
                    current = anomaly;
                }
            } else {
                merged.push(current);
                current = anomaly;
            }
        }

        merged.push(current);
        merged
    }

    /// Filter anomalies by severity
    pub fn filter_by_severity(
        anomalies: Vec<Anomaly>,
        min_severity: AnomalySeverity,
    ) -> Vec<Anomaly> {
        let severity_value = |s: &AnomalySeverity| match s {
            AnomalySeverity::Low => 1,
            AnomalySeverity::Medium => 2,
            AnomalySeverity::High => 3,
            AnomalySeverity::Critical => 4,
        };

        let min_value = severity_value(&min_severity);

        anomalies
            .into_iter()
            .filter(|a| severity_value(&a.severity) >= min_value)
            .collect()
    }

    /// Calculate anomaly density (anomalies per unit time)
    pub fn calculate_anomaly_density(
        anomalies: &[Anomaly],
        total_duration_hours: f64,
    ) -> f64 {
        if total_duration_hours <= 0.0 {
            return 0.0;
        }

        anomalies.len() as f64 / total_duration_hours
    }

    /// Group anomalies by time windows
    pub fn group_by_time_windows(
        anomalies: Vec<Anomaly>,
        window_duration_minutes: i64,
    ) -> HashMap<i64, Vec<Anomaly>> {
        let mut groups: HashMap<i64, Vec<Anomaly>> = HashMap::new();

        for anomaly in anomalies {
            let window_id = anomaly.timestamp.timestamp() / (window_duration_minutes * 60);
            groups.entry(window_id).or_insert_with(Vec::new).push(anomaly);
        }

        groups
    }

    /// Calculate anomaly statistics
    pub fn calculate_statistics(anomalies: &[Anomaly]) -> AnomalyStatistics {
        if anomalies.is_empty() {
            return AnomalyStatistics::default();
        }

        let scores: Vec<f64> = anomalies.iter().map(|a| a.score).collect();
        let mut severity_counts = HashMap::new();

        for anomaly in anomalies {
            *severity_counts.entry(anomaly.severity.clone()).or_insert(0) += 1;
        }

        AnomalyStatistics {
            total_count: anomalies.len(),
            mean_score: StatUtils::mean(&scores),
            max_score: scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            min_score: scores.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            std_score: StatUtils::std_dev(&scores),
            severity_counts,
        }
    }
}

/// Anomaly statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyStatistics {
    pub total_count: usize,
    pub mean_score: f64,
    pub max_score: f64,
    pub min_score: f64,
    pub std_score: f64,
    pub severity_counts: HashMap<AnomalySeverity, usize>,
}

impl Default for AnomalyStatistics {
    fn default() -> Self {
        Self {
            total_count: 0,
            mean_score: 0.0,
            max_score: 0.0,
            min_score: 0.0,
            std_score: 0.0,
            severity_counts: HashMap::new(),
        }
    }
}

/// Distance and similarity utilities
pub struct DistanceUtils;

impl DistanceUtils {
    /// Calculate Euclidean distance between two points
    pub fn euclidean_distance(point1: &[f64], point2: &[f64]) -> f64 {
        if point1.len() != point2.len() {
            return f64::INFINITY;
        }

        point1
            .iter()
            .zip(point2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate Manhattan distance between two points
    pub fn manhattan_distance(point1: &[f64], point2: &[f64]) -> f64 {
        if point1.len() != point2.len() {
            return f64::INFINITY;
        }

        point1
            .iter()
            .zip(point2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
    }

    /// Calculate Mahalanobis distance (simplified version)
    pub fn mahalanobis_distance_simple(
        point: &[f64],
        mean: &[f64],
        std_devs: &[f64],
    ) -> f64 {
        if point.len() != mean.len() || point.len() != std_devs.len() {
            return f64::INFINITY;
        }

        point
            .iter()
            .zip(mean.iter().zip(std_devs.iter()))
            .map(|(p, (m, s))| {
                if *s > 0.0 {
                    ((p - m) / s).powi(2)
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() || vec1.is_empty() {
            return 0.0;
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|a| a * a).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|b| b * b).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm1 * norm2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_statistical_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(StatUtils::mean(&values), 3.0);
        assert_eq!(StatUtils::median(&values), 3.0);
        assert!((StatUtils::std_dev(&values) - 1.58113883008).abs() < 0.001);

        let (q1, q2, q3) = StatUtils::quartiles(&values);
        assert_eq!(q1, 2.0);
        assert_eq!(q2, 3.0);
        assert_eq!(q3, 4.0);
        assert_eq!(StatUtils::iqr(&values), 2.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(StatUtils::percentile(&values, 0.0), 1.0);
        assert_eq!(StatUtils::percentile(&values, 50.0), 3.0);
        assert_eq!(StatUtils::percentile(&values, 100.0), 5.0);
    }

    #[test]
    fn test_data_extraction() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 1, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 2, 0).unwrap(),
        ];
        let values = vec![1.0, f64::NAN, 3.0];

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        let valid_data = DataUtils::extract_valid_data(&ts);

        assert_eq!(valid_data.len(), 2);
        assert_eq!(valid_data[0].1, 1.0);
        assert_eq!(valid_data[1].1, 3.0);

        let valid_values = DataUtils::extract_valid_values(&ts);
        assert_eq!(valid_values, vec![1.0, 3.0]);
    }

    #[test]
    fn test_normalization() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let normalized_zscore = DataUtils::normalize_zscore(&values);
        let mean_normalized = StatUtils::mean(&normalized_zscore);
        assert!((mean_normalized).abs() < 0.001);

        let normalized_minmax = DataUtils::normalize_minmax(&values);
        assert_eq!(normalized_minmax[0], 0.0);
        assert_eq!(normalized_minmax[4], 1.0);
    }

    #[test]
    fn test_threshold_calculations() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let adaptive = ThresholdUtils::adaptive_threshold(&values, 2.0, 0.1);
        assert!(adaptive >= 2.0);

        let percentile_thresh = ThresholdUtils::percentile_threshold(&values, 90.0);
        assert!(percentile_thresh > 4.0);
    }

    #[test]
    fn test_distance_calculations() {
        let point1 = vec![1.0, 2.0, 3.0];
        let point2 = vec![4.0, 5.0, 6.0];

        let euclidean = DistanceUtils::euclidean_distance(&point1, &point2);
        let expected_euclidean = ((3.0_f64).powi(2) * 3.0).sqrt();
        assert!((euclidean - expected_euclidean).abs() < 0.001);

        let manhattan = DistanceUtils::manhattan_distance(&point1, &point2);
        assert_eq!(manhattan, 9.0);

        let cosine = DistanceUtils::cosine_similarity(&point1, &point1);
        assert!((cosine - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_anomaly_utilities() {
        let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();

        let anomalies = vec![
            Anomaly {
                index: 0,
                timestamp,
                value: 10.0,
                score: 3.5,
                severity: AnomalySeverity::High,
                expected_value: Some(5.0),
            },
            Anomaly {
                index: 1,
                timestamp,
                value: 15.0,
                score: 2.1,
                severity: AnomalySeverity::Medium,
                expected_value: Some(5.0),
            },
        ];

        let high_severity = AnomalyUtils::filter_by_severity(anomalies.clone(), AnomalySeverity::High);
        assert_eq!(high_severity.len(), 1);

        let stats = AnomalyUtils::calculate_statistics(&anomalies);
        assert_eq!(stats.total_count, 2);
        assert!((stats.mean_score - 2.8).abs() < 0.01);
    }

    #[test]
    fn test_rolling_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = DataUtils::rolling_statistics(&values, 3);

        assert_eq!(rolling.means.len(), 5);
        assert!((rolling.means[2] - 2.0).abs() < 0.001); // Mean of [1,2,3]
        assert!((rolling.means[4] - 4.0).abs() < 0.001); // Mean of [3,4,5]
    }
}