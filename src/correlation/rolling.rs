//! Rolling correlation analysis module
//!
//! Implements rolling correlation calculations to analyze time-varying
//! relationships between time series.

use serde::{Serialize, Deserialize};
use crate::correlation::basic::{CorrelationType, compute_pearson_correlation};

/// Rolling correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingCorrelation {
    /// Time indices for rolling correlation values
    pub indices: Vec<usize>,

    /// Rolling correlation values
    pub correlations: Vec<f64>,

    /// Window size used for calculation
    pub window_size: usize,

    /// Correlation type used
    pub correlation_type: CorrelationType,

    /// Variable names
    pub variable1: String,
    pub variable2: String,

    /// Statistical summary
    pub summary: RollingCorrelationSummary,
}

/// Statistical summary of rolling correlations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingCorrelationSummary {
    /// Mean correlation
    pub mean_correlation: f64,

    /// Standard deviation of correlations
    pub std_correlation: f64,

    /// Minimum correlation
    pub min_correlation: f64,

    /// Maximum correlation
    pub max_correlation: f64,

    /// Median correlation
    pub median_correlation: f64,

    /// Number of rolling windows
    pub n_windows: usize,

    /// Correlation stability (1 - coefficient of variation)
    pub stability: f64,

    /// Time periods with strongest positive correlation
    pub strongest_positive: Vec<(usize, f64)>,

    /// Time periods with strongest negative correlation
    pub strongest_negative: Vec<(usize, f64)>,
}

impl RollingCorrelation {
    /// Get correlation at specific time index
    pub fn get_correlation_at(&self, index: usize) -> Option<f64> {
        self.indices.iter()
            .position(|&i| i == index)
            .and_then(|pos| self.correlations.get(pos))
            .copied()
    }

    /// Get correlations within a time range
    pub fn get_correlations_in_range(&self, start: usize, end: usize) -> Vec<(usize, f64)> {
        self.indices.iter()
            .zip(self.correlations.iter())
            .filter(|(&idx, _)| idx >= start && idx <= end)
            .map(|(&idx, &corr)| (idx, corr))
            .collect()
    }

    /// Detect significant changes in correlation
    pub fn detect_correlation_changes(&self, threshold: f64) -> Vec<CorrelationChangePoint> {
        let mut changes = Vec::new();

        if self.correlations.len() < 3 {
            return changes;
        }

        for i in 1..self.correlations.len() {
            let prev_corr = self.correlations[i - 1];
            let curr_corr = self.correlations[i];
            let change = (curr_corr - prev_corr).abs();

            if change >= threshold {
                changes.push(CorrelationChangePoint {
                    index: self.indices[i],
                    previous_correlation: prev_corr,
                    current_correlation: curr_corr,
                    change_magnitude: change,
                    change_type: if curr_corr > prev_corr {
                        ChangeType::Increase
                    } else {
                        ChangeType::Decrease
                    },
                });
            }
        }

        changes
    }
}

/// Correlation change point detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationChangePoint {
    /// Time index of the change
    pub index: usize,

    /// Correlation before change
    pub previous_correlation: f64,

    /// Correlation after change
    pub current_correlation: f64,

    /// Magnitude of change
    pub change_magnitude: f64,

    /// Type of change
    pub change_type: ChangeType,
}

/// Type of correlation change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Increase,
    Decrease,
}

/// Compute rolling correlation between two time series
pub fn compute_rolling_correlation(
    series1: &[f64],
    series2: &[f64],
    window_size: usize,
) -> Result<RollingCorrelation, Box<dyn std::error::Error>> {
    compute_rolling_correlation_with_type(series1, series2, window_size, CorrelationType::Pearson, "series1", "series2")
}

/// Compute rolling correlation with specific type and variable names
pub fn compute_rolling_correlation_with_type(
    series1: &[f64],
    series2: &[f64],
    window_size: usize,
    correlation_type: CorrelationType,
    var1_name: &str,
    var2_name: &str,
) -> Result<RollingCorrelation, Box<dyn std::error::Error>> {
    if series1.len() != series2.len() {
        return Err("Series must have the same length".into());
    }

    if window_size < 2 {
        return Err("Window size must be at least 2".into());
    }

    if series1.len() < window_size {
        return Err("Series length must be at least equal to window size".into());
    }

    let mut indices = Vec::new();
    let mut correlations = Vec::new();

    // Compute rolling correlations
    for i in window_size..=series1.len() {
        let start_idx = i - window_size;
        let window1 = &series1[start_idx..i];
        let window2 = &series2[start_idx..i];

        // Filter out missing values within the window
        let pairs: Vec<(f64, f64)> = window1.iter()
            .zip(window2.iter())
            .filter(|(&x, &y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| (x, y))
            .collect();

        if pairs.len() >= 2 {  // Need at least 2 valid pairs for correlation
            let correlation = match correlation_type {
                CorrelationType::Pearson => compute_pearson_correlation(&pairs)?,
                CorrelationType::Spearman => compute_spearman_rolling(&pairs)?,
                CorrelationType::Kendall => compute_kendall_rolling(&pairs)?,
            };

            indices.push(i - 1);  // End index of the window
            correlations.push(correlation);
        }
    }

    if correlations.is_empty() {
        return Err("No valid correlations could be computed".into());
    }

    // Compute summary statistics
    let summary = compute_rolling_summary(&correlations, &indices);

    Ok(RollingCorrelation {
        indices,
        correlations,
        window_size,
        correlation_type,
        variable1: var1_name.to_string(),
        variable2: var2_name.to_string(),
        summary,
    })
}

/// Compute Spearman correlation for rolling window
fn compute_spearman_rolling(pairs: &[(f64, f64)]) -> Result<f64, Box<dyn std::error::Error>> {
    let x_values: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let y_values: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();

    let x_ranks = compute_ranks_simple(&x_values);
    let y_ranks = compute_ranks_simple(&y_values);

    let rank_pairs: Vec<(f64, f64)> = x_ranks.into_iter().zip(y_ranks).collect();
    compute_pearson_correlation(&rank_pairs)
}

/// Compute Kendall's tau for rolling window
fn compute_kendall_rolling(pairs: &[(f64, f64)]) -> Result<f64, Box<dyn std::error::Error>> {
    let n = pairs.len();
    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let (x1, y1) = pairs[i];
            let (x2, y2) = pairs[j];

            let x_diff = x2 - x1;
            let y_diff = y2 - y1;

            if (x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0) {
                concordant += 1;
            } else if (x_diff > 0.0 && y_diff < 0.0) || (x_diff < 0.0 && y_diff > 0.0) {
                discordant += 1;
            }
        }
    }

    let total_pairs = n * (n - 1) / 2;
    if total_pairs == 0 {
        return Ok(0.0);
    }

    Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
}

/// Simple rank computation for rolling windows
fn compute_ranks_simple(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed_values: Vec<(usize, f64)> = values.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    for (rank, (original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[*original_idx] = (rank + 1) as f64;
    }

    ranks
}

/// Compute summary statistics for rolling correlations
fn compute_rolling_summary(correlations: &[f64], indices: &[usize]) -> RollingCorrelationSummary {
    let n = correlations.len();

    if n == 0 {
        return RollingCorrelationSummary {
            mean_correlation: 0.0,
            std_correlation: 0.0,
            min_correlation: 0.0,
            max_correlation: 0.0,
            median_correlation: 0.0,
            n_windows: 0,
            stability: 0.0,
            strongest_positive: Vec::new(),
            strongest_negative: Vec::new(),
        };
    }

    // Basic statistics
    let mean_correlation = correlations.iter().sum::<f64>() / n as f64;

    let variance = correlations.iter()
        .map(|&x| (x - mean_correlation).powi(2))
        .sum::<f64>() / n as f64;
    let std_correlation = variance.sqrt();

    let min_correlation = correlations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_correlation = correlations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Median
    let mut sorted_correlations = correlations.to_vec();
    sorted_correlations.sort_by(|a, b| {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,  // NaN goes to end
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => a.partial_cmp(b).unwrap(),
        }
    });
    let median_correlation = if n % 2 == 0 {
        (sorted_correlations[n / 2 - 1] + sorted_correlations[n / 2]) / 2.0
    } else {
        sorted_correlations[n / 2]
    };

    // Stability (1 - coefficient of variation)
    let stability = if mean_correlation.abs() > 1e-10 {
        1.0 - (std_correlation / mean_correlation.abs())
    } else {
        0.0
    };

    // Find strongest correlations
    let mut indexed_correlations: Vec<(usize, f64)> = indices.iter()
        .zip(correlations.iter())
        .map(|(&idx, &corr)| (idx, corr))
        .collect();

    // Sort by correlation strength
    indexed_correlations.sort_by(|a, b| {
        match (a.1.is_nan(), b.1.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,  // NaN goes to end
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.1.partial_cmp(&a.1).unwrap(),  // Sort in descending order
        }
    });
    let strongest_positive = indexed_correlations.iter()
        .filter(|(_, corr)| *corr > 0.0)
        .take(5)
        .cloned()
        .collect();

    indexed_correlations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let strongest_negative = indexed_correlations.iter()
        .filter(|(_, corr)| *corr < 0.0)
        .take(5)
        .cloned()
        .collect();

    RollingCorrelationSummary {
        mean_correlation,
        std_correlation,
        min_correlation,
        max_correlation,
        median_correlation,
        n_windows: n,
        stability,
        strongest_positive,
        strongest_negative,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_correlation_basic() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        assert_eq!(result.window_size, 3);
        assert!(result.correlations.len() > 0);

        // Should have perfect positive correlation
        for &corr in &result.correlations {
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rolling_correlation_summary() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let series2 = vec![1.0, 4.0, 2.0, 8.0, 3.0, 12.0];

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        assert!(result.summary.n_windows > 0);
        assert!(result.summary.mean_correlation.is_finite());
        assert!(result.summary.std_correlation >= 0.0);
        assert!(result.summary.min_correlation <= result.summary.max_correlation);
    }

    #[test]
    fn test_correlation_change_detection() {
        let correlations = vec![0.8, 0.9, 0.9, 0.2, 0.1, 0.0, 0.7, 0.8];
        let indices: Vec<usize> = (0..correlations.len()).collect();

        let rolling_corr = RollingCorrelation {
            indices,
            correlations,
            window_size: 3,
            correlation_type: CorrelationType::Pearson,
            variable1: "test1".to_string(),
            variable2: "test2".to_string(),
            summary: RollingCorrelationSummary {
                mean_correlation: 0.5,
                std_correlation: 0.3,
                min_correlation: 0.0,
                max_correlation: 0.9,
                median_correlation: 0.5,
                n_windows: 8,
                stability: 0.4,
                strongest_positive: Vec::new(),
                strongest_negative: Vec::new(),
            },
        };

        let changes = rolling_corr.detect_correlation_changes(0.5);
        assert!(changes.len() > 0);

        // Should detect the large drop from 0.9 to 0.2
        let large_change = changes.iter().find(|c| c.change_magnitude > 0.6);
        assert!(large_change.is_some());
    }

    #[test]
    fn test_rolling_correlation_window_size_validation() {
        let series1 = vec![1.0, 2.0, 3.0];
        let series2 = vec![2.0, 4.0, 6.0];

        // Window size too large
        let result = compute_rolling_correlation(&series1, &series2, 5);
        assert!(result.is_err());

        // Window size too small
        let result = compute_rolling_correlation(&series1, &series2, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_correlation_mismatched_lengths() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0];
        let series2 = vec![2.0, 4.0]; // Different length

        let result = compute_rolling_correlation(&series1, &series2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_correlation_with_varying_patterns() {
        // Create series with different correlation patterns
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 8.0, 6.0, 4.0];

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        // Should have correlations for multiple windows
        assert!(result.correlations.len() >= 5);
        assert_eq!(result.indices.len(), result.correlations.len());

        // Check that summary statistics are reasonable
        assert!(result.summary.mean_correlation >= -1.0 && result.summary.mean_correlation <= 1.0);
        assert!(result.summary.std_correlation >= 0.0);
        assert!(result.summary.min_correlation >= -1.0);
        assert!(result.summary.max_correlation <= 1.0);
    }

    #[test]
    fn test_rolling_correlation_with_negative_pattern() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let series2 = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        // Should have perfect negative correlation in all windows
        for &corr in &result.correlations {
            assert!((corr + 1.0).abs() < 1e-10);
        }

        assert!((result.summary.mean_correlation + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_correlation_with_missing_values() {
        let series1 = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        // Should handle NaN values gracefully
        assert!(result.correlations.len() > 0);

        // Windows containing NaN should be handled appropriately
        // (either excluded or correlation should be NaN)
        let has_nan_window = result.correlations.iter().any(|&corr| corr.is_nan());
        assert!(has_nan_window || result.correlations.len() < series1.len() - 2);
    }

    #[test]
    fn test_rolling_correlation_stability_metric() {
        // Test with stable correlation
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];

        let stable_result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        // Test with unstable correlation
        let series3 = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

        let unstable_result = compute_rolling_correlation(&series1, &series3, 3).unwrap();

        // Stable correlation should have higher stability metric
        assert!(stable_result.summary.stability > unstable_result.summary.stability);
    }

    #[test]
    fn test_correlation_change_detection_thresholds() {
        let correlations = vec![0.8, 0.7, 0.6, 0.1, 0.0, -0.1, 0.5, 0.9];
        let indices: Vec<usize> = (0..correlations.len()).collect();

        let rolling_corr = RollingCorrelation {
            indices,
            correlations,
            window_size: 3,
            correlation_type: CorrelationType::Pearson,
            variable1: "test1".to_string(),
            variable2: "test2".to_string(),
            summary: RollingCorrelationSummary {
                mean_correlation: 0.4,
                std_correlation: 0.3,
                min_correlation: -0.1,
                max_correlation: 0.9,
                median_correlation: 0.5,
                n_windows: 8,
                stability: 0.4,
                strongest_positive: Vec::new(),
                strongest_negative: Vec::new(),
            },
        };

        // Test different thresholds
        let changes_low = rolling_corr.detect_correlation_changes(0.1);
        let changes_high = rolling_corr.detect_correlation_changes(0.8);

        // Lower threshold should detect more changes
        assert!(changes_low.len() >= changes_high.len());
    }

    #[test]
    fn test_rolling_correlation_minimum_window_size() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0];

        // Test minimum valid window size (2)
        let result = compute_rolling_correlation(&series1, &series2, 2).unwrap();
        assert_eq!(result.window_size, 2);
        assert_eq!(result.correlations.len(), 3); // n - window_size + 1 = 4 - 2 + 1 = 3
    }

    #[test]
    fn test_rolling_correlation_edge_case_constant_values() {
        let series1 = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // Constant series
        let series2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        // Correlation with constant series should be NaN
        for &corr in &result.correlations {
            assert!(corr.is_nan());
        }
    }

    #[test]
    fn test_rolling_correlation_quality_metrics() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let result = compute_rolling_correlation(&series1, &series2, 3).unwrap();

        // Check quality metrics
        assert!(result.summary.n_windows > 0);
        assert!(result.summary.mean_correlation.is_finite());
        assert!(result.summary.std_correlation >= 0.0);
        assert!(result.summary.median_correlation.is_finite());
        assert!(result.summary.stability >= 0.0 && result.summary.stability <= 1.0);

        // Min should be <= median <= max
        assert!(result.summary.min_correlation <= result.summary.median_correlation);
        assert!(result.summary.median_correlation <= result.summary.max_correlation);
    }
}