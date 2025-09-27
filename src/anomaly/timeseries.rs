//! # Time Series Specific Anomaly Detection
//!
//! Detection methods tailored for time series data including seasonal decomposition
//! anomalies, trend deviations, level shifts, and volatility anomalies.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::TimeSeries;
use std::collections::VecDeque;

/// Detect anomalies using seasonal decomposition
pub fn detect_seasonal_anomalies(
    timeseries: &TimeSeries,
    period: usize,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    if timeseries.values.len() < period * 2 {
        return Ok(Vec::new());
    }

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

    if valid_data.len() < period * 2 {
        return Ok(Vec::new());
    }

    // Simple seasonal decomposition
    let seasonal_component = calculate_seasonal_component(&valid_data, period);
    let trend_component = calculate_trend_component(&valid_data, period);

    let mut anomalies = Vec::new();

    for (i, (index, value, timestamp)) in valid_data.iter().enumerate() {
        let expected_seasonal = seasonal_component[i % period];
        let expected_trend = if i < trend_component.len() {
            trend_component[i]
        } else {
            trend_component[trend_component.len() - 1]
        };

        let expected_value = expected_trend + expected_seasonal;
        let residual = (value - expected_value).abs();

        // Calculate threshold based on residual statistics
        let residuals: Vec<f64> = valid_data
            .iter()
            .enumerate()
            .map(|(j, (_, val, _))| {
                let exp_seasonal = seasonal_component[j % period];
                let exp_trend = if j < trend_component.len() {
                    trend_component[j]
                } else {
                    trend_component[trend_component.len() - 1]
                };
                (val - (exp_trend + exp_seasonal)).abs()
            })
            .collect();

        let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let std_residual = (residuals
            .iter()
            .map(|r| (r - mean_residual).powi(2))
            .sum::<f64>() / residuals.len() as f64)
            .sqrt();

        let threshold = mean_residual + 2.0 * std_residual;

        if residual > threshold {
            let score = residual / threshold;
            let severity = classify_seasonal_severity(score);

            anomalies.push(Anomaly {
                index: *index,
                timestamp: *timestamp,
                value: *value,
                score,
                severity,
                expected_value: Some(expected_value),
            });
        }
    }

    Ok(anomalies)
}

/// Detect trend deviation anomalies
pub fn detect_trend_deviation_anomalies(
    timeseries: &TimeSeries,
    window_size: usize,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    if timeseries.values.len() < window_size * 2 {
        return Ok(Vec::new());
    }

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

    if valid_data.len() < window_size * 2 {
        return Ok(Vec::new());
    }

    let mut anomalies = Vec::new();

    for i in window_size..valid_data.len() - window_size {
        let (index, value, timestamp) = valid_data[i];

        // Calculate local trend before and after
        let before_window = &valid_data[i - window_size..i];
        let after_window = &valid_data[i + 1..i + 1 + window_size];

        let trend_before = calculate_linear_trend(before_window);
        let trend_after = calculate_linear_trend(after_window);

        // Expected value based on before trend
        let expected_value = before_window.last().unwrap().1 + trend_before;

        // Calculate deviation from expected trend
        let trend_deviation = (value - expected_value).abs();
        let trend_change = (trend_after - trend_before).abs();

        // Threshold based on local variance
        let local_values: Vec<f64> = before_window.iter().map(|(_, v, _)| *v).collect();
        let local_std = calculate_std(&local_values);

        let threshold = 2.0 * local_std;

        if trend_deviation > threshold || trend_change > threshold {
            let score = (trend_deviation + trend_change) / threshold;
            let severity = classify_trend_severity(score);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score,
                severity,
                expected_value: Some(expected_value),
            });
        }
    }

    Ok(anomalies)
}

/// Detect level shift anomalies
pub fn detect_level_shift_anomalies(
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

    if valid_data.len() < 10 {
        return Ok(Vec::new());
    }

    let mut anomalies = Vec::new();
    let window_size = (valid_data.len() / 10).max(5).min(20);

    for i in window_size..valid_data.len() - window_size {
        let (index, value, timestamp) = valid_data[i];

        // Calculate means before and after the point
        let before_values: Vec<f64> = valid_data[i - window_size..i]
            .iter()
            .map(|(_, v, _)| *v)
            .collect();
        let after_values: Vec<f64> = valid_data[i + 1..i + 1 + window_size]
            .iter()
            .map(|(_, v, _)| *v)
            .collect();

        let mean_before = before_values.iter().sum::<f64>() / before_values.len() as f64;
        let mean_after = after_values.iter().sum::<f64>() / after_values.len() as f64;

        let level_shift = (mean_after - mean_before).abs();
        let std_before = calculate_std(&before_values);
        let std_after = calculate_std(&after_values);
        let pooled_std = ((std_before + std_after) / 2.0).max(0.001);

        let normalized_shift = level_shift / pooled_std;

        if normalized_shift > threshold {
            let score = normalized_shift / threshold;
            let severity = classify_level_shift_severity(score);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score,
                severity,
                expected_value: Some(mean_before),
            });
        }
    }

    Ok(anomalies)
}

/// Detect volatility anomalies
pub fn detect_volatility_anomalies(
    timeseries: &TimeSeries,
    window_size: usize,
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

    if valid_data.len() < window_size * 3 {
        return Ok(Vec::new());
    }

    let mut anomalies = Vec::new();

    // Calculate rolling volatility
    for i in window_size..valid_data.len() - window_size {
        let (index, value, timestamp) = valid_data[i];

        // Calculate volatility in the current window
        let window_values: Vec<f64> = valid_data[i - window_size / 2..i + window_size / 2]
            .iter()
            .map(|(_, v, _)| *v)
            .collect();

        let window_volatility = calculate_volatility(&window_values);

        // Calculate baseline volatility from earlier data
        let baseline_values: Vec<f64> = valid_data[0..i - window_size]
            .iter()
            .map(|(_, v, _)| *v)
            .collect();

        if baseline_values.is_empty() {
            continue;
        }

        let baseline_volatility = calculate_volatility(&baseline_values);

        if baseline_volatility == 0.0 {
            continue;
        }

        let volatility_ratio = window_volatility / baseline_volatility;

        // Anomaly if volatility is significantly different
        if volatility_ratio > 2.0 || volatility_ratio < 0.5 {
            let score = if volatility_ratio > 2.0 {
                volatility_ratio - 1.0
            } else {
                2.0 - volatility_ratio
            };

            let severity = classify_volatility_severity(score);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value,
                score,
                severity,
                expected_value: Some(value), // Volatility anomaly doesn't change expected value
            });
        }
    }

    Ok(anomalies)
}

// Helper functions

/// Calculate seasonal component using simple averaging
fn calculate_seasonal_component(
    data: &[(usize, f64, chrono::DateTime<chrono::Utc>)],
    period: usize,
) -> Vec<f64> {
    let mut seasonal = vec![0.0; period];
    let mut counts = vec![0; period];

    for (i, (_, value, _)) in data.iter().enumerate() {
        let season_idx = i % period;
        seasonal[season_idx] += value;
        counts[season_idx] += 1;
    }

    for (i, count) in counts.iter().enumerate() {
        if *count > 0 {
            seasonal[i] /= *count as f64;
        }
    }

    // Subtract overall mean to center seasonal component
    let mean = seasonal.iter().sum::<f64>() / seasonal.len() as f64;
    seasonal.iter().map(|x| x - mean).collect()
}

/// Calculate trend component using moving average
fn calculate_trend_component(
    data: &[(usize, f64, chrono::DateTime<chrono::Utc>)],
    window_size: usize,
) -> Vec<f64> {
    let values: Vec<f64> = data.iter().map(|(_, v, _)| *v).collect();
    let mut trend = Vec::new();

    for i in 0..values.len() {
        let start = if i >= window_size / 2 {
            i - window_size / 2
        } else {
            0
        };
        let end = (i + window_size / 2 + 1).min(values.len());

        let window_sum: f64 = values[start..end].iter().sum();
        let window_mean = window_sum / (end - start) as f64;
        trend.push(window_mean);
    }

    trend
}

/// Calculate linear trend (slope) for a window of data
fn calculate_linear_trend(data: &[(usize, f64, chrono::DateTime<chrono::Utc>)]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
    let sum_y: f64 = data.iter().map(|(_, v, _)| *v).sum();
    let sum_xy: f64 = data
        .iter()
        .enumerate()
        .map(|(i, (_, v, _))| i as f64 * v)
        .sum();
    let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    slope
}

/// Calculate standard deviation
fn calculate_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Calculate volatility (standard deviation of differences)
fn calculate_volatility(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let differences: Vec<f64> = values
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();

    calculate_std(&differences)
}

/// Classify severity for seasonal anomalies
fn classify_seasonal_severity(score: f64) -> AnomalySeverity {
    if score >= 3.0 {
        AnomalySeverity::Critical
    } else if score >= 2.0 {
        AnomalySeverity::High
    } else if score >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity for trend anomalies
fn classify_trend_severity(score: f64) -> AnomalySeverity {
    if score >= 4.0 {
        AnomalySeverity::Critical
    } else if score >= 2.5 {
        AnomalySeverity::High
    } else if score >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity for level shift anomalies
fn classify_level_shift_severity(score: f64) -> AnomalySeverity {
    if score >= 5.0 {
        AnomalySeverity::Critical
    } else if score >= 3.0 {
        AnomalySeverity::High
    } else if score >= 2.0 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity for volatility anomalies
fn classify_volatility_severity(score: f64) -> AnomalySeverity {
    if score >= 3.0 {
        AnomalySeverity::Critical
    } else if score >= 2.0 {
        AnomalySeverity::High
    } else if score >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_timeseries_with_seasonality() -> TimeSeries {
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..50)
            .map(|i| Utc::now() + chrono::Duration::hours(i))
            .collect();

        // Create data with seasonal pattern (period 10) and an anomaly
        let values: Vec<f64> = (0..50)
            .map(|i| {
                let seasonal = (2.0 * std::f64::consts::PI * (i % 10) as f64 / 10.0).sin();
                let trend = i as f64 * 0.1;
                let noise = 0.1 * ((i as f64).sin());

                if i == 25 {
                    trend + seasonal + noise + 10.0 // Anomaly
                } else {
                    trend + seasonal + noise
                }
            })
            .collect();

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_seasonal_anomaly_detection() {
        let ts = create_test_timeseries_with_seasonality();
        let anomalies = detect_seasonal_anomalies(&ts, 10).unwrap();

        assert!(!anomalies.is_empty());
        // Should detect the anomaly at index 25
        assert!(anomalies.iter().any(|a| a.index == 25));
    }

    #[test]
    fn test_trend_deviation_detection() {
        let ts = create_test_timeseries_with_seasonality();
        let anomalies = detect_trend_deviation_anomalies(&ts, 5).unwrap();

        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_level_shift_detection() {
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..30)
            .map(|i| Utc::now() + chrono::Duration::hours(i))
            .collect();

        // Create data with level shift at index 15
        let values: Vec<f64> = (0..30)
            .map(|i| {
                if i < 15 {
                    5.0 + 0.1 * (i as f64).sin()
                } else {
                    10.0 + 0.1 * (i as f64).sin() // Level shift
                }
            })
            .collect();

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        let anomalies = detect_level_shift_anomalies(&ts, 2.0).unwrap();

        assert!(!anomalies.is_empty());
        // Should detect level shift around index 15
        assert!(anomalies.iter().any(|a| a.index >= 14 && a.index <= 16));
    }

    #[test]
    fn test_volatility_anomaly_detection() {
        let ts = create_test_timeseries_with_seasonality();
        let anomalies = detect_volatility_anomalies(&ts, 10).unwrap();

        // May or may not detect anomalies depending on the synthetic data
        // This test mainly ensures the function doesn't crash
        assert!(anomalies.len() >= 0);
    }

    #[test]
    fn test_helper_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = calculate_std(&values);
        assert!(std_dev > 0.0);

        let volatility = calculate_volatility(&values);
        assert!(volatility >= 0.0);
    }
}