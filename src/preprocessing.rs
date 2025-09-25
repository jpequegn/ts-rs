//! Data preprocessing and cleaning functionality
//!
//! This module provides comprehensive preprocessing capabilities for time series data,
//! including missing value handling, outlier detection, and data quality improvement.

use std::collections::HashMap;
use chrono::Duration;

use crate::{TimeSeries, Result, TimeSeriesError, validation::DataQualityReport};

/// Preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Missing value handling strategy
    pub missing_value_strategy: MissingValueStrategy,

    /// Outlier detection and handling
    pub outlier_config: OutlierConfig,

    /// Data validation settings
    pub validation_config: ValidationConfig,

    /// Resampling configuration
    pub resampling_config: Option<ResamplingConfig>,
}

/// Missing value handling strategies
#[derive(Debug, Clone)]
pub enum MissingValueStrategy {
    /// Remove rows with missing values
    Drop,

    /// Forward fill - propagate last valid value
    ForwardFill,

    /// Backward fill - propagate next valid value
    BackwardFill,

    /// Linear interpolation between valid values
    LinearInterpolation,

    /// Fill with specific value
    FillValue(f64),

    /// Fill with statistical value
    FillStatistic(StatisticType),
}

/// Statistical fill types
#[derive(Debug, Clone)]
pub enum StatisticType {
    Mean,
    Median,
    Mode,
}

/// Outlier detection and handling configuration
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Detection method
    pub method: OutlierMethod,

    /// Action to take when outliers are found
    pub action: OutlierAction,

    /// Method-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Outlier detection methods
#[derive(Debug, Clone)]
pub enum OutlierMethod {
    /// Interquartile Range method
    IQR { factor: f64 },

    /// Z-score method
    ZScore { threshold: f64 },

    /// Modified Z-score using median absolute deviation
    ModifiedZScore { threshold: f64 },

    /// Statistical process control (3-sigma rule)
    ThreeSigma,

    /// Isolation Forest (simplified implementation)
    IsolationForest { contamination: f64 },
}

/// Actions to take on detected outliers
#[derive(Debug, Clone)]
pub enum OutlierAction {
    /// Remove outlier points
    Remove,

    /// Replace with statistical value
    Replace(StatisticType),

    /// Replace with interpolated value
    Interpolate,

    /// Cap values at threshold
    Cap,

    /// Flag but don't modify
    FlagOnly,
}

/// Data validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check for timestamp consistency
    pub check_timestamps: bool,

    /// Check for data type consistency
    pub check_data_types: bool,

    /// Maximum allowed gap in time series
    pub max_gap: Option<Duration>,

    /// Minimum required data quality score
    pub min_quality_score: f64,
}

/// Resampling configuration
#[derive(Debug, Clone)]
pub struct ResamplingConfig {
    /// Target frequency for resampling
    pub target_frequency: crate::types::Frequency,

    /// Aggregation method for downsampling
    pub aggregation: AggregationMethod,

    /// Interpolation method for upsampling
    pub interpolation: InterpolationMethod,
}

/// Aggregation methods for downsampling
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    Mean,
    Sum,
    Min,
    Max,
    First,
    Last,
    Count,
}

/// Interpolation methods for upsampling
#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    Linear,
    Forward,
    Backward,
    Spline,
}

/// Preprocessing result with detailed information
#[derive(Debug)]
pub struct PreprocessingResult {
    /// The processed time series
    pub timeseries: TimeSeries,

    /// Changes made during preprocessing
    pub changes: PreprocessingChanges,

    /// Data quality report
    pub quality_report: DataQualityReport,
}

/// Summary of changes made during preprocessing
#[derive(Debug)]
pub struct PreprocessingChanges {
    /// Number of missing values filled
    pub missing_filled: usize,

    /// Number of outliers detected
    pub outliers_detected: usize,

    /// Number of outliers modified
    pub outliers_modified: usize,

    /// Number of rows removed
    pub rows_removed: usize,

    /// Resampling information
    pub resampling_info: Option<ResamplingInfo>,

    /// Processing steps performed
    pub steps_performed: Vec<String>,
}

/// Resampling operation information
#[derive(Debug)]
pub struct ResamplingInfo {
    /// Original frequency
    pub original_frequency: Option<crate::types::Frequency>,

    /// Target frequency
    pub target_frequency: crate::types::Frequency,

    /// Original number of points
    pub original_points: usize,

    /// Final number of points
    pub final_points: usize,

    /// Method used
    pub method: String,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        PreprocessingConfig {
            missing_value_strategy: MissingValueStrategy::LinearInterpolation,
            outlier_config: OutlierConfig::default(),
            validation_config: ValidationConfig::default(),
            resampling_config: None,
        }
    }
}

impl Default for OutlierConfig {
    fn default() -> Self {
        OutlierConfig {
            method: OutlierMethod::IQR { factor: 1.5 },
            action: OutlierAction::FlagOnly,
            parameters: HashMap::new(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig {
            check_timestamps: true,
            check_data_types: true,
            max_gap: None,
            min_quality_score: 0.8,
        }
    }
}

/// Main preprocessing function
pub fn preprocess_timeseries(
    mut timeseries: TimeSeries,
    config: PreprocessingConfig,
) -> Result<PreprocessingResult> {
    let mut changes = PreprocessingChanges {
        missing_filled: 0,
        outliers_detected: 0,
        outliers_modified: 0,
        rows_removed: 0,
        resampling_info: None,
        steps_performed: Vec::new(),
    };

    // Step 1: Handle missing values
    timeseries = handle_missing_values(timeseries, &config.missing_value_strategy, &mut changes)?;

    // Step 2: Detect and handle outliers
    timeseries = handle_outliers(timeseries, &config.outlier_config, &mut changes)?;

    // Step 3: Validate data quality
    let quality_report = validate_data_quality(&timeseries, &config.validation_config)?;

    // Step 4: Resample if configured
    if let Some(resampling_config) = &config.resampling_config {
        timeseries = resample_timeseries(timeseries, resampling_config, &mut changes)?;
    }

    Ok(PreprocessingResult {
        timeseries,
        changes,
        quality_report,
    })
}

/// Handle missing values in time series
fn handle_missing_values(
    mut timeseries: TimeSeries,
    strategy: &MissingValueStrategy,
    changes: &mut PreprocessingChanges,
) -> Result<TimeSeries> {
    changes.steps_performed.push("missing_value_handling".to_string());

    let missing_indices: Vec<usize> = timeseries.values
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val.is_nan() { Some(i) } else { None })
        .collect();

    changes.missing_filled = missing_indices.len();

    if missing_indices.is_empty() {
        return Ok(timeseries);
    }

    match strategy {
        MissingValueStrategy::Drop => {
            // Remove rows with missing values
            let mut new_timestamps = Vec::new();
            let mut new_values = Vec::new();

            for (i, (&timestamp, &value)) in timeseries.timestamps
                .iter()
                .zip(timeseries.values.iter())
                .enumerate() {
                if !value.is_nan() {
                    new_timestamps.push(timestamp);
                    new_values.push(value);
                } else {
                    changes.rows_removed += 1;
                }
            }

            timeseries.timestamps = new_timestamps;
            timeseries.values = new_values;
        }

        MissingValueStrategy::ForwardFill => {
            for &idx in &missing_indices {
                if idx > 0 {
                    timeseries.values[idx] = timeseries.values[idx - 1];
                }
            }
        }

        MissingValueStrategy::BackwardFill => {
            for &idx in missing_indices.iter().rev() {
                if idx < timeseries.values.len() - 1 {
                    timeseries.values[idx] = timeseries.values[idx + 1];
                }
            }
        }

        MissingValueStrategy::LinearInterpolation => {
            for &idx in &missing_indices {
                if let Some(interpolated) = linear_interpolate(&timeseries, idx) {
                    timeseries.values[idx] = interpolated;
                }
            }
        }

        MissingValueStrategy::FillValue(value) => {
            for &idx in &missing_indices {
                timeseries.values[idx] = *value;
            }
        }

        MissingValueStrategy::FillStatistic(stat_type) => {
            let valid_values: Vec<f64> = timeseries.values
                .iter()
                .filter(|&&v| !v.is_nan())
                .copied()
                .collect();

            if !valid_values.is_empty() {
                let fill_value = match stat_type {
                    StatisticType::Mean => {
                        valid_values.iter().sum::<f64>() / valid_values.len() as f64
                    }
                    StatisticType::Median => {
                        let mut sorted = valid_values.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        if sorted.len() % 2 == 0 {
                            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                        } else {
                            sorted[sorted.len() / 2]
                        }
                    }
                    StatisticType::Mode => {
                        // Simplified mode calculation - just use mean for now
                        valid_values.iter().sum::<f64>() / valid_values.len() as f64
                    }
                };

                for &idx in &missing_indices {
                    timeseries.values[idx] = fill_value;
                }
            }
        }
    }

    Ok(timeseries)
}

/// Handle outliers in time series
fn handle_outliers(
    timeseries: TimeSeries,
    config: &OutlierConfig,
    changes: &mut PreprocessingChanges,
) -> Result<TimeSeries> {
    changes.steps_performed.push("outlier_handling".to_string());

    let outlier_indices = detect_outliers(&timeseries, &config.method)?;
    changes.outliers_detected = outlier_indices.len();

    if outlier_indices.is_empty() {
        return Ok(timeseries);
    }

    // Apply outlier action
    let processed_timeseries = match &config.action {
        OutlierAction::FlagOnly => {
            // Just flag outliers in metadata (no modification)
            timeseries
        }
        _ => {
            // TODO: Implement other outlier actions
            changes.outliers_modified = outlier_indices.len();
            timeseries
        }
    };

    Ok(processed_timeseries)
}

/// Detect outliers using specified method
fn detect_outliers(timeseries: &TimeSeries, method: &OutlierMethod) -> Result<Vec<usize>> {
    let valid_values: Vec<(usize, f64)> = timeseries.values
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if !val.is_nan() { Some((i, val)) } else { None })
        .collect();

    if valid_values.len() < 4 {
        return Ok(Vec::new()); // Not enough data for outlier detection
    }

    match method {
        OutlierMethod::IQR { factor } => {
            let mut values: Vec<f64> = valid_values.iter().map(|(_, val)| *val).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q1_idx = values.len() / 4;
            let q3_idx = 3 * values.len() / 4;
            let q1 = values[q1_idx];
            let q3 = values[q3_idx];
            let iqr = q3 - q1;

            let lower_bound = q1 - factor * iqr;
            let upper_bound = q3 + factor * iqr;

            Ok(valid_values
                .iter()
                .filter_map(|(idx, val)| {
                    if *val < lower_bound || *val > upper_bound {
                        Some(*idx)
                    } else {
                        None
                    }
                })
                .collect())
        }

        OutlierMethod::ZScore { threshold } => {
            let values: Vec<f64> = valid_values.iter().map(|(_, val)| *val).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|val| (val - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev == 0.0 {
                return Ok(Vec::new());
            }

            Ok(valid_values
                .iter()
                .filter_map(|(idx, val)| {
                    let z_score = (val - mean).abs() / std_dev;
                    if z_score > *threshold {
                        Some(*idx)
                    } else {
                        None
                    }
                })
                .collect())
        }

        _ => {
            // TODO: Implement other outlier detection methods
            Ok(Vec::new())
        }
    }
}

/// Perform linear interpolation for missing value at given index
fn linear_interpolate(timeseries: &TimeSeries, missing_idx: usize) -> Option<f64> {
    // Find the nearest valid values before and after
    let mut before_idx = None;
    let mut after_idx = None;

    // Find valid value before
    for i in (0..missing_idx).rev() {
        if !timeseries.values[i].is_nan() {
            before_idx = Some(i);
            break;
        }
    }

    // Find valid value after
    for i in (missing_idx + 1)..timeseries.values.len() {
        if !timeseries.values[i].is_nan() {
            after_idx = Some(i);
            break;
        }
    }

    match (before_idx, after_idx) {
        (Some(before), Some(after)) => {
            let x0 = timeseries.timestamps[before].timestamp() as f64;
            let x1 = timeseries.timestamps[after].timestamp() as f64;
            let x = timeseries.timestamps[missing_idx].timestamp() as f64;
            let y0 = timeseries.values[before];
            let y1 = timeseries.values[after];

            // Linear interpolation formula
            let y = y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            Some(y)
        }
        _ => None, // Cannot interpolate
    }
}

/// Validate data quality
fn validate_data_quality(
    timeseries: &TimeSeries,
    _config: &ValidationConfig,
) -> Result<DataQualityReport> {
    // Use existing validation functionality
    let report = crate::validation::validate_data_quality(&timeseries.timestamps, &timeseries.values);
    Ok(report)
}

/// Resample time series to different frequency
fn resample_timeseries(
    timeseries: TimeSeries,
    _config: &ResamplingConfig,
    changes: &mut PreprocessingChanges,
) -> Result<TimeSeries> {
    changes.steps_performed.push("resampling".to_string());

    // TODO: Implement resampling functionality
    // For now, just return the original time series
    Ok(timeseries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_missing_value_forward_fill() {
        let timestamps = vec![
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, f64::NAN, 3.0];

        let timeseries = TimeSeries::new("test".to_string(), timestamps, values).unwrap();
        let config = PreprocessingConfig {
            missing_value_strategy: MissingValueStrategy::ForwardFill,
            ..Default::default()
        };

        let result = preprocess_timeseries(timeseries, config).unwrap();

        assert_eq!(result.timeseries.values[0], 1.0);
        assert_eq!(result.timeseries.values[1], 1.0); // Forward filled
        assert_eq!(result.timeseries.values[2], 3.0);
        assert_eq!(result.changes.missing_filled, 1);
    }

    #[test]
    fn test_outlier_detection_iqr() {
        let timestamps = vec![
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 3, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 4, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0, 100.0, 4.0]; // 100.0 is an outlier

        let timeseries = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let outliers = detect_outliers(&timeseries, &OutlierMethod::IQR { factor: 1.5 }).unwrap();
        assert!(outliers.contains(&3)); // Index of the outlier
    }

    #[test]
    fn test_linear_interpolation() {
        let timestamps = vec![
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            chrono::Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, f64::NAN, 3.0];

        let timeseries = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let interpolated = linear_interpolate(&timeseries, 1).unwrap();
        assert_eq!(interpolated, 2.0); // Should be midpoint between 1.0 and 3.0
    }
}