//! Descriptive statistics module for time series analysis
//!
//! Provides comprehensive descriptive statistics including measures of central tendency,
//! spread, shape, and position.

use serde::{Serialize, Deserialize};

/// Comprehensive descriptive statistics for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Number of observations
    pub count: usize,

    /// Number of missing/invalid values
    pub missing_count: usize,

    /// Arithmetic mean
    pub mean: f64,

    /// Median (50th percentile)
    pub median: f64,

    /// Mode (most frequent value, if applicable)
    pub mode: Option<f64>,

    /// Standard deviation
    pub std_dev: f64,

    /// Variance
    pub variance: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Range (max - min)
    pub range: f64,

    /// Quantiles
    pub quantiles: Quantiles,

    /// Skewness (measure of asymmetry)
    pub skewness: f64,

    /// Kurtosis (measure of tail heaviness)
    pub kurtosis: f64,

    /// Coefficient of variation (std_dev / mean)
    pub coefficient_of_variation: f64,
}

/// Quantile statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantiles {
    /// 25th percentile (Q1)
    pub q25: f64,

    /// 50th percentile (Q2, median)
    pub q50: f64,

    /// 75th percentile (Q3)
    pub q75: f64,

    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,

    /// 1st percentile
    pub p01: f64,

    /// 5th percentile
    pub p05: f64,

    /// 10th percentile
    pub p10: f64,

    /// 90th percentile
    pub p90: f64,

    /// 95th percentile
    pub p95: f64,

    /// 99th percentile
    pub p99: f64,
}

impl DescriptiveStats {
    /// Create a summary string representation of the statistics
    pub fn summary(&self) -> String {
        format!(
            "Descriptive Statistics Summary:
Count: {}
Missing: {}
Mean: {:.6}
Median: {:.6}
Std Dev: {:.6}
Min: {:.6}
Max: {:.6}
Range: {:.6}
Q1: {:.6}
Q3: {:.6}
IQR: {:.6}
Skewness: {:.6}
Kurtosis: {:.6}",
            self.count,
            self.missing_count,
            self.mean,
            self.median,
            self.std_dev,
            self.min,
            self.max,
            self.range,
            self.quantiles.q25,
            self.quantiles.q75,
            self.quantiles.iqr,
            self.skewness,
            self.kurtosis
        )
    }
}

/// Compute comprehensive descriptive statistics for a dataset
///
/// # Arguments
/// * `data` - A slice of f64 values (should contain only finite values)
///
/// # Returns
/// * `Result<DescriptiveStats, Box<dyn std::error::Error>>` - The computed statistics
pub fn compute_descriptive_stats(data: &[f64]) -> Result<DescriptiveStats, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err("Cannot compute statistics for empty dataset".into());
    }

    // Verify all values are finite
    if !data.iter().all(|&x| x.is_finite()) {
        return Err("Data contains non-finite values".into());
    }

    let count = data.len();

    // Sort data for quantile calculations
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Basic statistics
    let mean = data.iter().sum::<f64>() / count as f64;
    let median = compute_median(&sorted_data);
    let mode = compute_mode(data);
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();
    let min = sorted_data[0];
    let max = sorted_data[count - 1];
    let range = max - min;

    // Quantiles
    let quantiles = compute_quantiles(&sorted_data);

    // Higher moments
    let skewness = compute_skewness(data, mean, std_dev);
    let kurtosis = compute_kurtosis(data, mean, std_dev);

    // Coefficient of variation
    let coefficient_of_variation = if mean != 0.0 {
        std_dev / mean.abs()
    } else {
        f64::INFINITY
    };

    Ok(DescriptiveStats {
        count,
        missing_count: 0, // Assuming input is already cleaned
        mean,
        median,
        mode,
        std_dev,
        variance,
        min,
        max,
        range,
        quantiles,
        skewness,
        kurtosis,
        coefficient_of_variation,
    })
}

/// Compute median from sorted data
fn compute_median(sorted_data: &[f64]) -> f64 {
    let n = sorted_data.len();
    if n % 2 == 0 {
        (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
    } else {
        sorted_data[n / 2]
    }
}

/// Compute mode (most frequent value) if applicable
/// Returns None if all values are unique or if the dataset is continuous
fn compute_mode(data: &[f64]) -> Option<f64> {
    use std::collections::HashMap;

    // For continuous data, mode might not be meaningful
    // Only return mode if there are actual repeated values
    let mut counts = HashMap::new();
    for &value in data {
        *counts.entry(value as i64).or_insert(0) += 1;
    }

    let max_count = counts.values().max().copied().unwrap_or(0);
    if max_count > 1 {
        counts.iter()
            .find(|(_, &count)| count == max_count)
            .map(|(&value, _)| value as f64)
    } else {
        None
    }
}

/// Compute various quantiles from sorted data
fn compute_quantiles(sorted_data: &[f64]) -> Quantiles {
    let n = sorted_data.len() as f64;

    // Calculate percentiles using linear interpolation
    let percentile = |p: f64| -> f64 {
        let index = p * (n - 1.0);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper || upper >= sorted_data.len() {
            sorted_data[lower.min(sorted_data.len() - 1)]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    };

    let p01 = percentile(0.01);
    let p05 = percentile(0.05);
    let p10 = percentile(0.10);
    let q25 = percentile(0.25);
    let q50 = percentile(0.50);
    let q75 = percentile(0.75);
    let p90 = percentile(0.90);
    let p95 = percentile(0.95);
    let p99 = percentile(0.99);

    let iqr = q75 - q25;

    Quantiles {
        q25,
        q50,
        q75,
        iqr,
        p01,
        p05,
        p10,
        p90,
        p95,
        p99,
    }
}

/// Compute sample skewness using the method of moments
fn compute_skewness(data: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }

    let n = data.len() as f64;
    let m3 = data.iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n;

    // Apply bias correction factor
    let correction = (n * (n - 1.0)).sqrt() / (n - 2.0);
    m3 * correction
}

/// Compute sample kurtosis (excess kurtosis, where normal distribution has kurtosis = 0)
fn compute_kurtosis(data: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }

    let n = data.len() as f64;
    let m4 = data.iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / n;

    // Excess kurtosis (subtract 3 to make normal distribution have kurtosis = 0)
    let excess_kurtosis = m4 - 3.0;

    // Apply bias correction
    let correction1 = (n - 1.0) / ((n - 2.0) * (n - 3.0));
    let correction2 = ((n + 1.0) * excess_kurtosis - 3.0 * (n - 1.0)) * correction1;
    correction2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_stats_normal_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_descriptive_stats(&data).unwrap();

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert!((stats.std_dev - (2.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_descriptive_stats_single_value() {
        let data = vec![42.0];
        let stats = compute_descriptive_stats(&data).unwrap();

        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.median, 42.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
    }

    #[test]
    fn test_descriptive_stats_empty_data() {
        let data: Vec<f64> = vec![];
        let result = compute_descriptive_stats(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantiles_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = compute_descriptive_stats(&data).unwrap();

        // For this dataset, Q1 should be around 3.25, median 5.5, Q3 7.75
        assert!((stats.quantiles.q25 - 3.25).abs() < 0.1);
        assert!((stats.quantiles.q50 - 5.5).abs() < 0.1);
        assert!((stats.quantiles.q75 - 7.75).abs() < 0.1);
        assert!((stats.quantiles.iqr - 4.5).abs() < 0.1);
    }
}