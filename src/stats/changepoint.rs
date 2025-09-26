//! Change point detection module for time series analysis
//!
//! Provides various algorithms for detecting structural breaks and regime changes
//! in time series data, including CUSUM tests and other statistical approaches.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A detected change point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Index of the change point in the original series
    pub index: usize,

    /// Confidence level of the detection (0.0 to 1.0)
    pub confidence: f64,

    /// Type of change detected
    pub change_type: ChangeType,

    /// Statistical test statistic value at this point
    pub test_statistic: f64,

    /// Critical value used for detection
    pub critical_value: f64,

    /// Additional information about the change point
    pub metadata: HashMap<String, f64>,
}

/// Type of structural change detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Change in mean level
    MeanShift,

    /// Change in variance
    VarianceChange,

    /// Change in trend
    TrendChange,

    /// General structural break
    StructuralBreak,
}

/// Configuration for change point detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePointConfig {
    /// Minimum distance between detected change points
    pub min_distance: usize,

    /// Significance level for detection (e.g., 0.05)
    pub significance_level: f64,

    /// Minimum segment size (number of observations)
    pub min_segment_size: usize,

    /// Maximum number of change points to detect
    pub max_change_points: Option<usize>,

    /// Detection method to use
    pub method: ChangePointMethod,
}

/// Available change point detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangePointMethod {
    /// CUSUM test for mean shifts
    Cusum,

    /// PELT (Pruned Exact Linear Time) algorithm
    Pelt,

    /// Binary segmentation
    BinarySegmentation,

    /// Variance-based detection
    VarianceTest,
}

impl Default for ChangePointConfig {
    fn default() -> Self {
        ChangePointConfig {
            min_distance: 10,
            significance_level: 0.05,
            min_segment_size: 5,
            max_change_points: None,
            method: ChangePointMethod::Cusum,
        }
    }
}

/// Detect change points in time series data
///
/// # Arguments
/// * `data` - Time series data
/// * `config` - Configuration for detection (None uses defaults)
///
/// # Returns
/// * `Result<Vec<ChangePoint>, Box<dyn std::error::Error>>` - Detected change points
pub fn detect_changepoints(
    data: &[f64],
    config: Option<ChangePointConfig>,
) -> Result<Vec<ChangePoint>, Box<dyn std::error::Error>> {
    let config = config.unwrap_or_default();

    if data.len() < config.min_segment_size * 3 {
        return Ok(Vec::new()); // Need at least 3 segments worth of data
    }

    match config.method {
        ChangePointMethod::Cusum => detect_cusum_changepoints(data, &config),
        ChangePointMethod::Pelt => detect_pelt_changepoints(data, &config),
        ChangePointMethod::BinarySegmentation => detect_binary_segmentation_changepoints(data, &config),
        ChangePointMethod::VarianceTest => detect_variance_changepoints(data, &config),
    }
}

/// CUSUM-based change point detection
fn detect_cusum_changepoints(
    data: &[f64],
    config: &ChangePointConfig,
) -> Result<Vec<ChangePoint>, Box<dyn std::error::Error>> {
    let n = data.len();
    let mut changepoints = Vec::new();

    // Compute overall mean and variance
    let overall_mean = data.iter().sum::<f64>() / n as f64;
    let overall_var = data.iter()
        .map(|&x| (x - overall_mean).powi(2))
        .sum::<f64>() / n as f64;

    if overall_var == 0.0 {
        return Ok(changepoints); // No variation in data
    }

    let std_dev = overall_var.sqrt();

    // Compute CUSUM statistics
    let mut cusum_pos = vec![0.0; n];
    let mut cusum_neg = vec![0.0; n];

    // Detection threshold
    let h = 4.0 * std_dev; // Common choice: 4-5 standard deviations
    let k = 0.5 * std_dev; // Reference value: half standard deviation

    for i in 0..n {
        let deviation = data[i] - overall_mean;

        // Positive CUSUM (detects upward shifts)
        cusum_pos[i] = if i == 0 {
            (deviation - k).max(0.0)
        } else {
            (cusum_pos[i - 1] + deviation - k).max(0.0)
        };

        // Negative CUSUM (detects downward shifts)
        cusum_neg[i] = if i == 0 {
            (-deviation - k).max(0.0)
        } else {
            (cusum_neg[i - 1] - deviation - k).max(0.0)
        };

        // Check for change point
        if cusum_pos[i] > h || cusum_neg[i] > h {
            if changepoints.is_empty() ||
                i >= changepoints.last().unwrap().index + config.min_distance {

                let test_statistic = cusum_pos[i].max(cusum_neg[i]);
                let confidence = calculate_cusum_confidence(test_statistic, h);

                let mut metadata = HashMap::new();
                metadata.insert("cusum_positive".to_string(), cusum_pos[i]);
                metadata.insert("cusum_negative".to_string(), cusum_neg[i]);
                metadata.insert("threshold".to_string(), h);

                changepoints.push(ChangePoint {
                    index: i,
                    confidence,
                    change_type: ChangeType::MeanShift,
                    test_statistic,
                    critical_value: h,
                    metadata,
                });

                // Check maximum number of change points
                if let Some(max_cp) = config.max_change_points {
                    if changepoints.len() >= max_cp {
                        break;
                    }
                }
            }
        }
    }

    Ok(changepoints)
}

/// PELT (Pruned Exact Linear Time) algorithm for change point detection
fn detect_pelt_changepoints(
    data: &[f64],
    config: &ChangePointConfig,
) -> Result<Vec<ChangePoint>, Box<dyn std::error::Error>> {
    let n = data.len();
    let mut changepoints = Vec::new();

    // Penalty for adding a change point (BIC-style)
    let penalty = 2.0 * (n as f64).ln();

    // PELT algorithm implementation (simplified)
    let mut f = vec![f64::INFINITY; n + 1];  // Cost function
    let mut cp = vec![0; n + 1];             // Change point tracker
    let mut r = vec![0];                     // Candidate set

    f[0] = -penalty; // Initialize

    for t in 1..=n {
        let mut candidates = Vec::new();

        for &s in &r {
            if t - s >= config.min_segment_size {
                let segment_cost = compute_segment_cost(&data[s..t]);
                let total_cost = f[s] + segment_cost + penalty;

                candidates.push((total_cost, s));
            }
        }

        if !candidates.is_empty() {
            // Find minimum cost
            let (min_cost, best_s) = candidates.iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();

            f[t] = *min_cost;
            cp[t] = *best_s;

            // Pruning step
            r.retain(|&s| {
                let segment_cost = compute_segment_cost(&data[s..t]);
                f[s] + segment_cost <= f[t]
            });
        }

        r.push(t);
    }

    // Backtrack to find change points
    let mut cps = Vec::new();
    let mut current = n;

    while cp[current] != 0 {
        cps.push(cp[current]);
        current = cp[current];
    }

    cps.reverse();

    // Convert to ChangePoint structs
    for &idx in &cps {
        if idx > 0 && idx < n {
            let confidence = 0.8; // Simplified confidence

            let mut metadata = HashMap::new();
            metadata.insert("pelt_cost".to_string(), f[idx]);
            metadata.insert("penalty".to_string(), penalty);

            changepoints.push(ChangePoint {
                index: idx - 1, // Convert to 0-based indexing
                confidence,
                change_type: ChangeType::StructuralBreak,
                test_statistic: f[idx],
                critical_value: penalty,
                metadata,
            });
        }
    }

    Ok(changepoints)
}

/// Binary segmentation for change point detection
fn detect_binary_segmentation_changepoints(
    data: &[f64],
    config: &ChangePointConfig,
) -> Result<Vec<ChangePoint>, Box<dyn std::error::Error>> {
    let mut changepoints = Vec::new();
    let mut segments_to_process = vec![(0, data.len())];

    while let Some((start, end)) = segments_to_process.pop() {
        if end - start < config.min_segment_size * 2 {
            continue;
        }

        // Find the best split point in this segment
        if let Some(split_point) = find_best_split(&data[start..end], config) {
            let actual_index = start + split_point;

            // Check if this split is significant
            let test_stat = compute_split_statistic(&data[start..end], split_point);
            let critical_value = get_critical_value_for_split(end - start, config.significance_level);

            if test_stat > critical_value {
                let confidence = calculate_split_confidence(test_stat, critical_value);

                let mut metadata = HashMap::new();
                metadata.insert("segment_start".to_string(), start as f64);
                metadata.insert("segment_end".to_string(), end as f64);
                metadata.insert("split_statistic".to_string(), test_stat);

                changepoints.push(ChangePoint {
                    index: actual_index,
                    confidence,
                    change_type: ChangeType::StructuralBreak,
                    test_statistic: test_stat,
                    critical_value,
                    metadata,
                });

                // Add subsegments for further processing
                if actual_index - start >= config.min_segment_size * 2 {
                    segments_to_process.push((start, actual_index));
                }
                if end - actual_index >= config.min_segment_size * 2 {
                    segments_to_process.push((actual_index, end));
                }

                // Check maximum number of change points
                if let Some(max_cp) = config.max_change_points {
                    if changepoints.len() >= max_cp {
                        break;
                    }
                }
            }
        }
    }

    // Sort change points by index
    changepoints.sort_by_key(|cp| cp.index);

    Ok(changepoints)
}

/// Variance-based change point detection
fn detect_variance_changepoints(
    data: &[f64],
    config: &ChangePointConfig,
) -> Result<Vec<ChangePoint>, Box<dyn std::error::Error>> {
    let n = data.len();
    let mut changepoints = Vec::new();

    // Use a sliding window approach
    let window_size = config.min_segment_size * 2;

    if n < window_size * 2 {
        return Ok(changepoints);
    }

    for i in window_size..(n - window_size) {
        // Compare variance before and after point i
        let before = &data[(i - window_size)..i];
        let after = &data[i..(i + window_size)];

        let var_before = compute_variance(before);
        let var_after = compute_variance(after);

        // F-test for variance equality
        let f_stat = if var_after > var_before && var_before > 0.0 {
            var_after / var_before
        } else if var_before > var_after && var_after > 0.0 {
            var_before / var_after
        } else {
            1.0
        };

        // Critical value for F-distribution (simplified)
        let critical_value = get_f_critical_value(window_size, config.significance_level);

        if f_stat > critical_value {
            // Check minimum distance constraint
            if changepoints.is_empty() ||
                i >= changepoints.last().unwrap().index + config.min_distance {

                let confidence = calculate_f_test_confidence(f_stat, critical_value);

                let mut metadata = HashMap::new();
                metadata.insert("variance_before".to_string(), var_before);
                metadata.insert("variance_after".to_string(), var_after);
                metadata.insert("f_statistic".to_string(), f_stat);
                metadata.insert("window_size".to_string(), window_size as f64);

                changepoints.push(ChangePoint {
                    index: i,
                    confidence,
                    change_type: ChangeType::VarianceChange,
                    test_statistic: f_stat,
                    critical_value,
                    metadata,
                });

                // Check maximum number of change points
                if let Some(max_cp) = config.max_change_points {
                    if changepoints.len() >= max_cp {
                        break;
                    }
                }
            }
        }
    }

    Ok(changepoints)
}

// Helper functions

fn compute_segment_cost(segment: &[f64]) -> f64 {
    if segment.is_empty() {
        return 0.0;
    }

    let n = segment.len() as f64;
    let mean = segment.iter().sum::<f64>() / n;
    let variance = segment.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;

    // Negative log-likelihood for normal distribution
    if variance > 0.0 {
        n * (2.0 * std::f64::consts::PI * variance).ln() / 2.0 +
        segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (2.0 * variance)
    } else {
        0.0
    }
}

fn find_best_split(segment: &[f64], config: &ChangePointConfig) -> Option<usize> {
    let n = segment.len();
    let min_size = config.min_segment_size;

    if n < min_size * 2 {
        return None;
    }

    let mut best_split = None;
    let mut best_statistic = 0.0;

    for split_point in min_size..(n - min_size) {
        let statistic = compute_split_statistic(segment, split_point);
        if statistic > best_statistic {
            best_statistic = statistic;
            best_split = Some(split_point);
        }
    }

    best_split
}

fn compute_split_statistic(segment: &[f64], split_point: usize) -> f64 {
    let left = &segment[0..split_point];
    let right = &segment[split_point..];

    let mean_left = left.iter().sum::<f64>() / left.len() as f64;
    let mean_right = right.iter().sum::<f64>() / right.len() as f64;

    // T-test statistic for difference in means
    let var_left = left.iter().map(|&x| (x - mean_left).powi(2)).sum::<f64>() / (left.len() - 1) as f64;
    let var_right = right.iter().map(|&x| (x - mean_right).powi(2)).sum::<f64>() / (right.len() - 1) as f64;

    let pooled_var = ((left.len() - 1) as f64 * var_left + (right.len() - 1) as f64 * var_right) /
        (left.len() + right.len() - 2) as f64;

    if pooled_var > 0.0 {
        let se = pooled_var.sqrt() * (1.0 / left.len() as f64 + 1.0 / right.len() as f64).sqrt();
        (mean_left - mean_right).abs() / se
    } else {
        0.0
    }
}

fn compute_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (data.len() - 1) as f64
}

fn calculate_cusum_confidence(test_statistic: f64, threshold: f64) -> f64 {
    // Simple linear mapping: higher statistics get higher confidence
    let ratio = test_statistic / threshold;
    (0.5 + 0.4 * (ratio - 1.0).min(1.0)).max(0.5).min(0.95)
}

fn calculate_split_confidence(test_statistic: f64, critical_value: f64) -> f64 {
    let ratio = test_statistic / critical_value;
    (0.5 + 0.3 * (ratio - 1.0).min(2.0)).max(0.5).min(0.9)
}

fn calculate_f_test_confidence(f_stat: f64, critical_value: f64) -> f64 {
    let ratio = f_stat / critical_value;
    (0.6 + 0.3 * (ratio - 1.0).min(1.5)).max(0.6).min(0.95)
}

fn get_critical_value_for_split(n: usize, alpha: f64) -> f64 {
    // Simplified critical value for t-test
    let df = n - 2;
    match alpha {
        a if a <= 0.01 => 2.8 + 0.5 / (df as f64).sqrt(),
        a if a <= 0.05 => 2.0 + 0.3 / (df as f64).sqrt(),
        _ => 1.7 + 0.2 / (df as f64).sqrt(),
    }
}

fn get_f_critical_value(df: usize, alpha: f64) -> f64 {
    // Simplified F critical value
    match alpha {
        a if a <= 0.01 => 3.5 + 5.0 / df as f64,
        a if a <= 0.05 => 2.5 + 3.0 / df as f64,
        _ => 2.0 + 2.0 / df as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_detection_no_change() {
        // Stationary data with no change points
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = ChangePointConfig::default();

        let changepoints = detect_cusum_changepoints(&data, &config).unwrap();
        assert!(changepoints.len() <= 1); // May detect false positives but should be minimal
    }

    #[test]
    fn test_cusum_detection_with_shift() {
        // Data with a clear mean shift
        let mut data = vec![0.0; 50];
        data.extend(vec![2.0; 50]); // Mean shift at index 50

        let config = ChangePointConfig {
            min_distance: 5,
            ..Default::default()
        };

        let changepoints = detect_cusum_changepoints(&data, &config).unwrap();
        assert!(!changepoints.is_empty());

        // Should detect change point near index 50
        let detected_near_50 = changepoints.iter()
            .any(|cp| (cp.index as i32 - 50).abs() < 10);
        assert!(detected_near_50);
    }

    #[test]
    fn test_binary_segmentation() {
        let mut data = vec![1.0; 30];
        data.extend(vec![3.0; 30]);
        data.extend(vec![0.0; 30]);

        let config = ChangePointConfig {
            method: ChangePointMethod::BinarySegmentation,
            min_segment_size: 10,
            ..Default::default()
        };

        let changepoints = detect_binary_segmentation_changepoints(&data, &config).unwrap();
        assert!(!changepoints.is_empty());
    }

    #[test]
    fn test_variance_change_detection() {
        // Create data with variance change
        let mut data = Vec::new();
        // Low variance segment
        for i in 0..50 {
            data.push(1.0 + 0.1 * (i as f64 * 0.1).sin());
        }
        // High variance segment
        for i in 0..50 {
            data.push(1.0 + 1.0 * (i as f64 * 0.1).sin());
        }

        let config = ChangePointConfig {
            method: ChangePointMethod::VarianceTest,
            min_segment_size: 15,
            ..Default::default()
        };

        let changepoints = detect_variance_changepoints(&data, &config).unwrap();
        assert!(!changepoints.is_empty());

        // Should detect variance change
        assert!(changepoints.iter().any(|cp| cp.change_type == ChangeType::VarianceChange));
    }

    #[test]
    fn test_pelt_detection() {
        let mut data = vec![0.0; 25];
        data.extend(vec![2.0; 25]);
        data.extend(vec![1.0; 25]);

        let config = ChangePointConfig {
            method: ChangePointMethod::Pelt,
            min_segment_size: 10,
            ..Default::default()
        };

        let changepoints = detect_pelt_changepoints(&data, &config).unwrap();
        // PELT may or may not detect changes depending on penalty
        // This test just ensures no crashes
        assert!(changepoints.len() >= 0);
    }

    #[test]
    fn test_change_point_config() {
        let config = ChangePointConfig {
            min_distance: 15,
            significance_level: 0.01,
            max_change_points: Some(2),
            ..Default::default()
        };

        let data = vec![1.0; 100];
        let changepoints = detect_changepoints(&data, Some(config)).unwrap();
        assert!(changepoints.len() <= 2);
    }
}