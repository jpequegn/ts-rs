//! # Statistical Anomaly Detection
//!
//! Statistical methods for detecting anomalies including Z-score,
//! Modified Z-score, IQR, and Grubbs' test.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::TimeSeries;

/// Detect anomalies using Z-score method
pub fn detect_zscore_anomalies(
    timeseries: &TimeSeries,
    threshold: f64,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
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
        .collect();

    if valid_data.len() < 3 {
        return Ok(Vec::new());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, val, _)| *val).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|val| (val - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Ok(Vec::new());
    }

    let mut anomalies = Vec::new();

    for (index, value, timestamp) in valid_data {
        let z_score = (value - mean).abs() / std_dev;

        if z_score > threshold {
            let severity = classify_zscore_severity(z_score, threshold);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score: z_score,
                severity,
                expected_value: Some(mean),
            });
        }
    }

    Ok(anomalies)
}

/// Detect anomalies using Modified Z-score method (based on median absolute deviation)
pub fn detect_modified_zscore_anomalies(
    timeseries: &TimeSeries,
    threshold: f64,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
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
        .collect();

    if valid_data.len() < 3 {
        return Ok(Vec::new());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, val, _)| *val).collect();
    let median = calculate_median(&values);

    // Calculate median absolute deviation (MAD)
    let mut absolute_deviations: Vec<f64> = values
        .iter()
        .map(|val| (val - median).abs())
        .collect();
    let mad = calculate_median(&mut absolute_deviations);

    if mad == 0.0 {
        return Ok(Vec::new());
    }

    let mut anomalies = Vec::new();

    for (index, value, timestamp) in valid_data {
        // Modified Z-score formula: 0.6745 * (x - median) / MAD
        let modified_zscore = 0.6745 * (value - median).abs() / mad;

        if modified_zscore > threshold {
            let severity = classify_zscore_severity(modified_zscore, threshold);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score: modified_zscore,
                severity,
                expected_value: Some(median),
            });
        }
    }

    Ok(anomalies)
}

/// Detect anomalies using Interquartile Range (IQR) method
pub fn detect_iqr_anomalies(
    timeseries: &TimeSeries,
    factor: f64,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
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
        .collect();

    if valid_data.len() < 4 {
        return Ok(Vec::new());
    }

    let mut values: Vec<f64> = valid_data.iter().map(|(_, val, _)| *val).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = calculate_quartile(&values, 0.25);
    let q3 = calculate_quartile(&values, 0.75);
    let iqr = q3 - q1;

    let lower_bound = q1 - factor * iqr;
    let upper_bound = q3 + factor * iqr;

    let mut anomalies = Vec::new();

    for (index, value, timestamp) in valid_data {
        if value < lower_bound || value > upper_bound {
            // Calculate score as distance from nearest bound relative to IQR
            let score = if value < lower_bound {
                (lower_bound - value) / iqr
            } else {
                (value - upper_bound) / iqr
            };

            let severity = classify_iqr_severity(score, factor);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score,
                severity,
                expected_value: Some((q1 + q3) / 2.0), // Median as expected
            });
        }
    }

    Ok(anomalies)
}

/// Detect anomalies using Grubbs' test for outliers
pub fn detect_grubbs_anomalies(
    timeseries: &TimeSeries,
    alpha: f64,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
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
        .collect();

    if valid_data.len() < 3 {
        return Ok(Vec::new());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, val, _)| *val).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|val| (val - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Ok(Vec::new());
    }

    // Calculate critical value for Grubbs' test
    let t_critical = calculate_t_critical(alpha / (2.0 * n), n - 2.0);
    let grubbs_critical = ((n - 1.0) / (n).sqrt()) *
        ((t_critical * t_critical) / (n - 2.0 + t_critical * t_critical)).sqrt();

    let mut anomalies = Vec::new();

    for (index, value, timestamp) in valid_data {
        let grubbs_statistic = (value - mean).abs() / std_dev;

        if grubbs_statistic > grubbs_critical {
            let severity = classify_grubbs_severity(grubbs_statistic, grubbs_critical);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score: grubbs_statistic,
                severity,
                expected_value: Some(mean),
            });
        }
    }

    Ok(anomalies)
}

/// Calculate median of a slice of values
fn calculate_median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = sorted.len();
    if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Calculate quartile value
fn calculate_quartile(sorted_values: &[f64], quantile: f64) -> f64 {
    let n = sorted_values.len() as f64;
    let index = quantile * (n - 1.0);

    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        sorted_values[lower_index]
    } else {
        let weight = index - lower_index as f64;
        sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
    }
}

/// Approximate t-distribution critical value
fn calculate_t_critical(alpha: f64, df: f64) -> f64 {
    // Simplified approximation for t-critical value
    // For more accuracy, would need a proper t-distribution implementation
    if df >= 30.0 {
        // Use normal approximation for large df
        if alpha <= 0.001 { 3.291 }
        else if alpha <= 0.01 { 2.576 }
        else if alpha <= 0.025 { 1.96 }
        else if alpha <= 0.05 { 1.645 }
        else { 1.282 }
    } else {
        // Simple approximation for small df
        let base = if alpha <= 0.001 { 4.0 }
        else if alpha <= 0.01 { 3.0 }
        else if alpha <= 0.025 { 2.5 }
        else if alpha <= 0.05 { 2.0 }
        else { 1.5 };

        base * (1.0 + 2.0 / df).sqrt()
    }
}

/// Classify severity based on Z-score
fn classify_zscore_severity(score: f64, threshold: f64) -> AnomalySeverity {
    let ratio = score / threshold;

    if ratio >= 2.0 {
        AnomalySeverity::Critical
    } else if ratio >= 1.5 {
        AnomalySeverity::High
    } else if ratio >= 1.2 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity based on IQR factor
fn classify_iqr_severity(score: f64, factor: f64) -> AnomalySeverity {
    let ratio = score / factor;

    if ratio >= 3.0 {
        AnomalySeverity::Critical
    } else if ratio >= 2.0 {
        AnomalySeverity::High
    } else if ratio >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity based on Grubbs' statistic
fn classify_grubbs_severity(statistic: f64, critical: f64) -> AnomalySeverity {
    let ratio = statistic / critical;

    if ratio >= 2.0 {
        AnomalySeverity::Critical
    } else if ratio >= 1.5 {
        AnomalySeverity::High
    } else if ratio >= 1.2 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_timeseries() -> TimeSeries {
        let timestamps = (0..10)
            .map(|i| Utc::now() + chrono::Duration::hours(i))
            .collect();
        let values = vec![1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 100.0 is anomaly
        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_zscore_detection() {
        let ts = create_test_timeseries();
        let anomalies = detect_zscore_anomalies(&ts, 2.0).unwrap();

        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.index == 3)); // Should detect index 3 (value 100.0)
    }

    #[test]
    fn test_modified_zscore_detection() {
        let ts = create_test_timeseries();
        let anomalies = detect_modified_zscore_anomalies(&ts, 2.0).unwrap();

        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.index == 3));
    }

    #[test]
    fn test_iqr_detection() {
        let ts = create_test_timeseries();
        let anomalies = detect_iqr_anomalies(&ts, 1.5).unwrap();

        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.index == 3));
    }

    #[test]
    fn test_grubbs_detection() {
        let ts = create_test_timeseries();
        let anomalies = detect_grubbs_anomalies(&ts, 0.05).unwrap();

        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.index == 3));
    }

    #[test]
    fn test_median_calculation() {
        assert_eq!(calculate_median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(calculate_median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_quartile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_quartile(&values, 0.25), 2.0);
        assert_eq!(calculate_quartile(&values, 0.75), 4.0);
    }
}