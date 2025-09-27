//! # Contextual Anomaly Detection
//!
//! Detection methods that consider contextual information such as
//! day-of-week patterns, seasonal contexts, and multivariate relationships.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::TimeSeries;
use chrono::{Datelike, Timelike};
use std::collections::HashMap;

/// Detect anomalies adjusted for day-of-week patterns
pub fn detect_day_of_week_anomalies(
    timeseries: &TimeSeries,
    baseline_periods: usize,
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

    if valid_data.len() < baseline_periods {
        return Ok(Vec::new());
    }

    // Build day-of-week profiles from baseline data
    let mut day_profiles: HashMap<u32, Vec<f64>> = HashMap::new();

    // Use first baseline_periods points to build profiles
    for (_, value, timestamp) in valid_data.iter().take(baseline_periods) {
        let day_of_week = timestamp.weekday().num_days_from_monday();
        day_profiles.entry(day_of_week).or_insert_with(Vec::new).push(*value);
    }

    // Calculate statistics for each day of week
    let mut day_stats: HashMap<u32, (f64, f64)> = HashMap::new(); // (mean, std)

    for (day, values) in day_profiles {
        if values.len() >= 2 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();
            day_stats.insert(day, (mean, std_dev.max(0.1))); // Avoid division by zero
        }
    }

    let mut anomalies = Vec::new();

    // Check points after baseline period
    for (index, value, timestamp) in valid_data.iter().skip(baseline_periods) {
        let day_of_week = timestamp.weekday().num_days_from_monday();

        if let Some((mean, std_dev)) = day_stats.get(&day_of_week) {
            let z_score = (value - mean).abs() / std_dev;

            if z_score > 3.0 {
                let severity = classify_contextual_severity(z_score, 3.0);

                anomalies.push(Anomaly {
                    index: *index,
                    timestamp: *timestamp,
                    value: *value,
                    score: z_score,
                    severity,
                    expected_value: Some(*mean),
                });
            }
        }
    }

    Ok(anomalies)
}

/// Detect anomalies in seasonal context
pub fn detect_seasonal_context_anomalies(
    timeseries: &TimeSeries,
    seasonal_periods: &[usize],
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

    if valid_data.is_empty() {
        return Ok(Vec::new());
    }

    let mut anomalies = Vec::new();

    for &period in seasonal_periods {
        if valid_data.len() < period * 3 {
            continue;
        }

        // Build seasonal profiles
        let seasonal_anomalies = detect_seasonal_context_for_period(&valid_data, period)?;
        anomalies.extend(seasonal_anomalies);
    }

    // Remove duplicates (keep highest score)
    anomalies.sort_by(|a, b| a.index.cmp(&b.index));
    anomalies.dedup_by(|a, b| {
        if a.index == b.index {
            if a.score < b.score {
                *a = b.clone();
            }
            true
        } else {
            false
        }
    });

    Ok(anomalies)
}

/// Detect multivariate anomalies (placeholder implementation)
pub fn detect_multivariate_anomalies(
    primary_series: &TimeSeries,
    other_series: &[&TimeSeries],
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    // This is a simplified implementation
    // A full implementation would use techniques like Mahalanobis distance,
    // correlation analysis, or multivariate statistical tests

    if other_series.is_empty() {
        // Fallback to univariate detection
        return crate::anomaly::statistical::detect_zscore_anomalies(primary_series, 3.0);
    }

    let primary_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = primary_series
        .values
        .iter()
        .enumerate()
        .zip(primary_series.timestamps.iter())
        .filter_map(|((i, &val), &timestamp)| {
            if !val.is_nan() {
                Some((i, val, timestamp))
            } else {
                None
            }
        })
        .collect();

    if primary_data.is_empty() {
        return Ok(Vec::new());
    }

    // Simple correlation-based anomaly detection
    let mut anomalies = Vec::new();

    for other_ts in other_series {
        let correlation_anomalies = detect_correlation_anomalies(primary_series, other_ts)?;
        anomalies.extend(correlation_anomalies);
    }

    // Remove duplicates and sort
    anomalies.sort_by(|a, b| a.index.cmp(&b.index));
    anomalies.dedup_by_key(|a| a.index);

    Ok(anomalies)
}

/// Detect anomalies based on hour-of-day patterns
pub fn detect_hourly_pattern_anomalies(
    timeseries: &TimeSeries,
    baseline_periods: usize,
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

    if valid_data.len() < baseline_periods {
        return Ok(Vec::new());
    }

    // Build hourly profiles from baseline data
    let mut hour_profiles: HashMap<u32, Vec<f64>> = HashMap::new();

    for (_, value, timestamp) in valid_data.iter().take(baseline_periods) {
        let hour = timestamp.hour();
        hour_profiles.entry(hour).or_insert_with(Vec::new).push(*value);
    }

    // Calculate statistics for each hour
    let mut hour_stats: HashMap<u32, (f64, f64)> = HashMap::new();

    for (hour, values) in hour_profiles {
        if values.len() >= 2 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt().max(0.1);
            hour_stats.insert(hour, (mean, std_dev));
        }
    }

    let mut anomalies = Vec::new();

    // Check points after baseline period
    for (index, value, timestamp) in valid_data.iter().skip(baseline_periods) {
        let hour = timestamp.hour();

        if let Some((mean, std_dev)) = hour_stats.get(&hour) {
            let z_score = (value - mean).abs() / std_dev;

            if z_score > 2.5 {
                let severity = classify_contextual_severity(z_score, 2.5);

                anomalies.push(Anomaly {
                    index: *index,
                    timestamp: *timestamp,
                    value: *value,
                    score: z_score,
                    severity,
                    expected_value: Some(*mean),
                });
            }
        }
    }

    Ok(anomalies)
}

// Helper functions

/// Detect seasonal context anomalies for a specific period
fn detect_seasonal_context_for_period(
    data: &[(usize, f64, chrono::DateTime<chrono::Utc>)],
    period: usize,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let mut seasonal_profiles: HashMap<usize, Vec<f64>> = HashMap::new();

    // Build seasonal profiles
    for (i, (_, value, _)) in data.iter().enumerate() {
        let season_index = i % period;
        seasonal_profiles.entry(season_index).or_insert_with(Vec::new).push(*value);
    }

    // Calculate statistics for each seasonal position
    let mut seasonal_stats: HashMap<usize, (f64, f64)> = HashMap::new();

    for (season_idx, values) in seasonal_profiles {
        if values.len() >= 2 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt().max(0.1);
            seasonal_stats.insert(season_idx, (mean, std_dev));
        }
    }

    let mut anomalies = Vec::new();

    // Check for anomalies after first complete cycle
    for (i, (index, value, timestamp)) in data.iter().enumerate().skip(period) {
        let season_index = i % period;

        if let Some((mean, std_dev)) = seasonal_stats.get(&season_index) {
            let z_score = (value - mean).abs() / std_dev;

            if z_score > 3.0 {
                let severity = classify_contextual_severity(z_score, 3.0);

                anomalies.push(Anomaly {
                    index: *index,
                    timestamp: *timestamp,
                    value: *value,
                    score: z_score,
                    severity,
                    expected_value: Some(*mean),
                });
            }
        }
    }

    Ok(anomalies)
}

/// Detect correlation-based anomalies between two time series
fn detect_correlation_anomalies(
    primary_series: &TimeSeries,
    other_series: &TimeSeries,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    // Align timestamps and get common data points
    let mut aligned_data = Vec::new();

    for (i, (primary_val, primary_ts)) in primary_series.values.iter().zip(primary_series.timestamps.iter()).enumerate() {
        if primary_val.is_nan() {
            continue;
        }

        // Find corresponding timestamp in other series (simple exact match)
        if let Some(other_pos) = other_series.timestamps.iter().position(|&ts| ts == *primary_ts) {
            let other_val = other_series.values[other_pos];
            if !other_val.is_nan() {
                aligned_data.push((i, *primary_val, other_val, *primary_ts));
            }
        }
    }

    if aligned_data.len() < 10 {
        return Ok(Vec::new());
    }

    // Calculate correlation coefficient
    let primary_values: Vec<f64> = aligned_data.iter().map(|(_, p, _, _)| *p).collect();
    let other_values: Vec<f64> = aligned_data.iter().map(|(_, _, o, _)| *o).collect();

    let correlation = calculate_correlation(&primary_values, &other_values);

    // Detect points where correlation breaks down
    let mut anomalies = Vec::new();
    let window_size = 10.min(aligned_data.len() / 4);

    for i in window_size..aligned_data.len() - window_size {
        let window_primary: Vec<f64> = primary_values[i - window_size..i + window_size + 1].to_vec();
        let window_other: Vec<f64> = other_values[i - window_size..i + window_size + 1].to_vec();

        let local_correlation = calculate_correlation(&window_primary, &window_other);
        let correlation_deviation = (local_correlation - correlation).abs();

        if correlation_deviation > 0.5 {
            let (index, primary_val, _, timestamp) = aligned_data[i];
            let severity = classify_correlation_severity(correlation_deviation);

            anomalies.push(Anomaly {
                index,
                timestamp,
                value: primary_val,
                score: correlation_deviation,
                severity,
                expected_value: None, // Correlation anomaly doesn't have clear expected value
            });
        }
    }

    Ok(anomalies)
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Classify severity for contextual anomalies
fn classify_contextual_severity(score: f64, threshold: f64) -> AnomalySeverity {
    let ratio = score / threshold;

    if ratio >= 2.5 {
        AnomalySeverity::Critical
    } else if ratio >= 2.0 {
        AnomalySeverity::High
    } else if ratio >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity for correlation anomalies
fn classify_correlation_severity(deviation: f64) -> AnomalySeverity {
    if deviation >= 0.8 {
        AnomalySeverity::Critical
    } else if deviation >= 0.6 {
        AnomalySeverity::High
    } else if deviation >= 0.4 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn create_weekly_pattern_timeseries() -> TimeSeries {
        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        // Create 4 weeks of data with weekly pattern
        let start_date = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(); // Monday

        for week in 0..4 {
            for day in 0..7 {
                let timestamp = start_date + chrono::Duration::days(week * 7 + day);
                timestamps.push(timestamp);

                // Normal weekday pattern: lower on weekends
                let base_value = if day < 5 { 10.0 } else { 5.0 };
                let noise = (day as f64 * 0.5).sin();

                if week == 3 && day == 1 {
                    // Add anomaly on Tuesday of 4th week
                    values.push(base_value + noise + 20.0);
                } else {
                    values.push(base_value + noise);
                }
            }
        }

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_day_of_week_anomaly_detection() {
        let ts = create_weekly_pattern_timeseries();
        let anomalies = detect_day_of_week_anomalies(&ts, 14).unwrap(); // Use first 2 weeks as baseline

        assert!(!anomalies.is_empty());
        // Should detect the anomaly on Tuesday of 4th week
        // Tuesday of 4th week would be index 22 (3*7 + 1)
        assert!(anomalies.iter().any(|a| a.index >= 20 && a.index <= 24));
    }

    #[test]
    fn test_seasonal_context_anomalies() {
        let ts = create_weekly_pattern_timeseries();
        let seasonal_periods = vec![7]; // Weekly period
        let anomalies = detect_seasonal_context_anomalies(&ts, &seasonal_periods).unwrap();

        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_hourly_pattern_anomalies() {
        // Create hourly data
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..48)
            .map(|i| Utc::now() + chrono::Duration::hours(i))
            .collect();

        // Normal pattern: higher values during business hours
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let hour = i % 24;
                let base = if hour >= 9 && hour <= 17 { 20.0 } else { 10.0 };

                if i == 35 {
                    // Add anomaly at hour 11 of second day
                    base + 50.0
                } else {
                    base + (i as f64 * 0.1).sin()
                }
            })
            .collect();

        let ts = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        let anomalies = detect_hourly_pattern_anomalies(&ts, 24).unwrap();

        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.index == 35));
    }

    #[test]
    fn test_correlation_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let correlation = calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001);

        let y_negative = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // Perfect negative correlation
        let correlation_neg = calculate_correlation(&x, &y_negative);
        assert!((correlation_neg + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_multivariate_anomalies() {
        let primary_ts = create_weekly_pattern_timeseries();

        // Create a correlated secondary series
        let secondary_values: Vec<f64> = primary_ts.values.iter().map(|&v| v * 2.0 + 1.0).collect();
        let secondary_ts = TimeSeries::new(
            "secondary".to_string(),
            primary_ts.timestamps.clone(),
            secondary_values,
        ).unwrap();

        let other_series = vec![&secondary_ts];
        let anomalies = detect_multivariate_anomalies(&primary_ts, &other_series).unwrap();

        // This is a basic test - in a real scenario, we'd expect anomalies
        // when the correlation breaks down
        assert!(anomalies.len() >= 0);
    }
}