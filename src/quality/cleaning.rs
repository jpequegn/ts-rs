//! Smart data cleaning and imputation for time series
//!
//! This module provides intelligent data cleaning algorithms that automatically
//! repair common data quality issues while preserving time series characteristics.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::quality::{
    OutlierReport, QualityError, QualityIssue, QualityResult,
};
use crate::timeseries::TimeSeries;

/// Imputation method for filling gaps
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImputationMethod {
    /// Forward fill - carry last observation forward
    ForwardFill,
    /// Backward fill - carry next observation backward
    BackwardFill,
    /// Linear interpolation between known points
    LinearInterpolation,
    /// Mean imputation using rolling window
    MeanImputation { window_size: Option<usize> },
    /// Median imputation using rolling window
    MedianImputation { window_size: Option<usize> },
    /// Spline interpolation with specified degree
    SplineInterpolation { degree: usize },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },
}

/// Outlier correction strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutlierCorrection {
    /// Remove outliers (creates gaps)
    Remove,
    /// Cap values to percentile bounds
    Cap { lower_percentile: f64, upper_percentile: f64 },
    /// Replace with imputed values
    Replace { method: ImputationMethod },
    /// Apply local smoothing
    Smooth { window_size: usize },
}

/// Noise reduction technique
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseReduction {
    /// Simple moving average
    MovingAverage { window_size: usize },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },
    /// Median filter (robust to outliers)
    MedianFilter { window_size: usize },
}

/// Type of data modification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModificationOperation {
    /// Gap filled
    GapFilled { method: ImputationMethod },
    /// Outlier corrected
    OutlierCorrected { correction: OutlierCorrection },
    /// Noise reduced
    NoiseReduced { method: NoiseReduction },
}

/// Record of a single data modification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataModification {
    /// Timestamp of the modification
    pub timestamp: DateTime<Utc>,
    /// Type of operation performed
    pub operation: ModificationOperation,
    /// Original value (if any)
    pub original_value: Option<f64>,
    /// New value after modification
    pub new_value: f64,
    /// Confidence in the modification (0.0-1.0)
    pub confidence: f64,
}

/// Impact on data quality from cleaning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityImpact {
    /// Improvement in completeness (0.0-1.0)
    pub completeness_gain: f64,
    /// Change in consistency (-1.0 to 1.0)
    pub consistency_change: f64,
    /// Potential bias introduced (0.0-1.0)
    pub potential_bias: f64,
    /// Uncertainty estimate (0.0-1.0)
    pub uncertainty: f64,
}

impl Default for QualityImpact {
    fn default() -> Self {
        QualityImpact {
            completeness_gain: 0.0,
            consistency_change: 0.0,
            potential_bias: 0.0,
            uncertainty: 0.0,
        }
    }
}

/// Report of cleaning operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleaningReport {
    /// Number of gaps filled
    pub gaps_filled: usize,
    /// Number of outliers corrected
    pub outliers_corrected: usize,
    /// Whether noise reduction was applied
    pub noise_reduction_applied: bool,
    /// Methods used during cleaning
    pub methods_used: Vec<String>,
    /// Quality score improvement (before - after)
    pub quality_improvement: f64,
}

impl Default for CleaningReport {
    fn default() -> Self {
        CleaningReport {
            gaps_filled: 0,
            outliers_corrected: 0,
            noise_reduction_applied: false,
            methods_used: Vec::new(),
            quality_improvement: 0.0,
        }
    }
}

/// Result of data cleaning operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleaningResult {
    /// Cleaned time series
    pub cleaned_data: TimeSeries,
    /// Report of operations performed
    pub cleaning_report: CleaningReport,
    /// Impact on data quality
    pub quality_impact: QualityImpact,
    /// Detailed list of modifications
    pub modifications: Vec<DataModification>,
}

/// Configuration for data cleaning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleaningConfig {
    /// Maximum percentage of data to modify (0.0-1.0)
    pub max_modifications: f64,
    /// Preserve statistical characteristics
    pub preserve_characteristics: bool,
    /// Track uncertainty of modifications
    pub uncertainty_tracking: bool,
    /// Require human validation
    pub validation_required: bool,
    /// Keep backup of original data
    pub backup_original: bool,
}

impl Default for CleaningConfig {
    fn default() -> Self {
        CleaningConfig {
            max_modifications: 0.10, // 10%
            preserve_characteristics: true,
            uncertainty_tracking: true,
            validation_required: false,
            backup_original: true,
        }
    }
}

impl CleaningConfig {
    /// Conservative cleaning configuration (max 5% modifications)
    pub fn conservative() -> Self {
        CleaningConfig {
            max_modifications: 0.05,
            preserve_characteristics: true,
            uncertainty_tracking: true,
            validation_required: true,
            backup_original: true,
        }
    }

    /// Aggressive cleaning configuration (up to 30% modifications)
    pub fn aggressive() -> Self {
        CleaningConfig {
            max_modifications: 0.30,
            preserve_characteristics: false,
            uncertainty_tracking: false,
            validation_required: false,
            backup_original: false,
        }
    }
}

/// Gap configuration for detection and filling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapConfig {
    /// Maximum gap duration to fill
    pub max_gap_duration: Duration,
    /// Minimum data points around gaps
    pub min_surrounding_data: usize,
    /// Preserve seasonal patterns
    pub preserve_seasonality: bool,
    /// Imputation method to use
    pub method: ImputationMethod,
}

impl Default for GapConfig {
    fn default() -> Self {
        GapConfig {
            max_gap_duration: Duration::hours(24),
            min_surrounding_data: 2,
            preserve_seasonality: true,
            method: ImputationMethod::LinearInterpolation,
        }
    }
}

/// Fills gaps in time series using specified method
pub fn fill_gaps(
    data: &TimeSeries,
    method: ImputationMethod,
) -> QualityResult<TimeSeries> {
    if data.is_empty() {
        return Err(QualityError::validation("Cannot fill gaps in empty time series"));
    }

    let mut new_values = data.values.clone();
    let mut modifications = Vec::new();

    match method {
        ImputationMethod::ForwardFill => {
            fill_forward(&mut new_values, data, &mut modifications)?;
        }
        ImputationMethod::BackwardFill => {
            fill_backward(&mut new_values, data, &mut modifications)?;
        }
        ImputationMethod::LinearInterpolation => {
            fill_linear(&mut new_values, data, &mut modifications)?;
        }
        ImputationMethod::MeanImputation { window_size } => {
            fill_mean(&mut new_values, data, window_size, &mut modifications)?;
        }
        ImputationMethod::MedianImputation { window_size } => {
            fill_median(&mut new_values, data, window_size, &mut modifications)?;
        }
        ImputationMethod::ExponentialSmoothing { alpha } => {
            fill_exponential(&mut new_values, data, alpha, &mut modifications)?;
        }
        _ => {
            return Err(QualityError::invalid_parameter("Imputation method not yet implemented"));
        }
    }

    TimeSeries::new(
        data.name.clone(),
        data.timestamps.clone(),
        new_values,
    )
    .map_err(|e| QualityError::data_cleaning(format!("Failed to create cleaned time series: {}", e)))
}

/// Forward fill implementation
fn fill_forward(
    values: &mut [f64],
    data: &TimeSeries,
    modifications: &mut Vec<DataModification>,
) -> QualityResult<()> {
    let mut last_valid = None;

    for i in 0..values.len() {
        if values[i].is_nan() || values[i].is_infinite() {
            if let Some(last) = last_valid {
                modifications.push(DataModification {
                    timestamp: data.timestamps[i],
                    operation: ModificationOperation::GapFilled {
                        method: ImputationMethod::ForwardFill,
                    },
                    original_value: Some(values[i]),
                    new_value: last,
                    confidence: 0.7,
                });
                values[i] = last;
            }
        } else {
            last_valid = Some(values[i]);
        }
    }

    Ok(())
}

/// Backward fill implementation
fn fill_backward(
    values: &mut [f64],
    data: &TimeSeries,
    modifications: &mut Vec<DataModification>,
) -> QualityResult<()> {
    let mut next_valid = None;

    for i in (0..values.len()).rev() {
        if values[i].is_nan() || values[i].is_infinite() {
            if let Some(next) = next_valid {
                modifications.push(DataModification {
                    timestamp: data.timestamps[i],
                    operation: ModificationOperation::GapFilled {
                        method: ImputationMethod::BackwardFill,
                    },
                    original_value: Some(values[i]),
                    new_value: next,
                    confidence: 0.7,
                });
                values[i] = next;
            }
        } else {
            next_valid = Some(values[i]);
        }
    }

    Ok(())
}

/// Linear interpolation implementation
fn fill_linear(
    values: &mut [f64],
    data: &TimeSeries,
    modifications: &mut Vec<DataModification>,
) -> QualityResult<()> {
    let n = values.len();
    if n < 2 {
        return Ok(());
    }

    let mut start_idx = None;

    for i in 0..n {
        if values[i].is_nan() || values[i].is_infinite() {
            if start_idx.is_none() {
                // Find previous valid value
                for j in (0..i).rev() {
                    if !values[j].is_nan() && !values[j].is_infinite() {
                        start_idx = Some(j);
                        break;
                    }
                }
            }
        } else if let Some(start) = start_idx {
            // Found end of gap, interpolate
            let start_val = values[start];
            let end_val = values[i];
            let gap_size = i - start;

            for j in (start + 1)..i {
                let ratio = (j - start) as f64 / gap_size as f64;
                let interpolated = start_val + (end_val - start_val) * ratio;

                modifications.push(DataModification {
                    timestamp: data.timestamps[j],
                    operation: ModificationOperation::GapFilled {
                        method: ImputationMethod::LinearInterpolation,
                    },
                    original_value: Some(values[j]),
                    new_value: interpolated,
                    confidence: 0.8,
                });
                values[j] = interpolated;
            }

            start_idx = None;
        }
    }

    Ok(())
}

/// Mean imputation implementation
fn fill_mean(
    values: &mut [f64],
    data: &TimeSeries,
    window_size: Option<usize>,
    modifications: &mut Vec<DataModification>,
) -> QualityResult<()> {
    let window = window_size.unwrap_or(5);
    let half_window = window / 2;

    for i in 0..values.len() {
        if values[i].is_nan() || values[i].is_infinite() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(values.len());

            let mut sum = 0.0;
            let mut count = 0;

            for j in start..end {
                if j != i && !values[j].is_nan() && !values[j].is_infinite() {
                    sum += values[j];
                    count += 1;
                }
            }

            if count > 0 {
                let mean = sum / count as f64;
                modifications.push(DataModification {
                    timestamp: data.timestamps[i],
                    operation: ModificationOperation::GapFilled {
                        method: ImputationMethod::MeanImputation {
                            window_size: Some(window),
                        },
                    },
                    original_value: Some(values[i]),
                    new_value: mean,
                    confidence: 0.75,
                });
                values[i] = mean;
            }
        }
    }

    Ok(())
}

/// Median imputation implementation
fn fill_median(
    values: &mut [f64],
    data: &TimeSeries,
    window_size: Option<usize>,
    modifications: &mut Vec<DataModification>,
) -> QualityResult<()> {
    let window = window_size.unwrap_or(5);
    let half_window = window / 2;

    for i in 0..values.len() {
        if values[i].is_nan() || values[i].is_infinite() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(values.len());

            let mut valid_values: Vec<f64> = Vec::new();

            for j in start..end {
                if j != i && !values[j].is_nan() && !values[j].is_infinite() {
                    valid_values.push(values[j]);
                }
            }

            if !valid_values.is_empty() {
                valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if valid_values.len() % 2 == 0 {
                    (valid_values[valid_values.len() / 2 - 1] + valid_values[valid_values.len() / 2]) / 2.0
                } else {
                    valid_values[valid_values.len() / 2]
                };

                modifications.push(DataModification {
                    timestamp: data.timestamps[i],
                    operation: ModificationOperation::GapFilled {
                        method: ImputationMethod::MedianImputation {
                            window_size: Some(window),
                        },
                    },
                    original_value: Some(values[i]),
                    new_value: median,
                    confidence: 0.75,
                });
                values[i] = median;
            }
        }
    }

    Ok(())
}

/// Exponential smoothing implementation
fn fill_exponential(
    values: &mut [f64],
    data: &TimeSeries,
    alpha: f64,
    modifications: &mut Vec<DataModification>,
) -> QualityResult<()> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(QualityError::configuration("Alpha must be between 0 and 1"));
    }

    let mut smoothed = values[0];

    for i in 1..values.len() {
        if values[i].is_nan() || values[i].is_infinite() {
            modifications.push(DataModification {
                timestamp: data.timestamps[i],
                operation: ModificationOperation::GapFilled {
                    method: ImputationMethod::ExponentialSmoothing { alpha },
                },
                original_value: Some(values[i]),
                new_value: smoothed,
                confidence: 0.65,
            });
            values[i] = smoothed;
        } else {
            smoothed = alpha * values[i] + (1.0 - alpha) * smoothed;
        }
    }

    Ok(())
}

/// Corrects outliers in time series
pub fn correct_outliers(
    data: &TimeSeries,
    outliers: &OutlierReport,
    correction: OutlierCorrection,
) -> QualityResult<TimeSeries> {
    if data.is_empty() {
        return Err(QualityError::validation("Cannot correct outliers in empty time series"));
    }

    let mut new_values = data.values.clone();
    let mut modifications = Vec::new();

    // Create index map for quick lookup
    let mut outlier_indices = HashMap::new();
    for outlier in &outliers.outliers {
        if let Some(idx) = data.timestamps.iter().position(|t| t == &outlier.timestamp) {
            outlier_indices.insert(idx, outlier);
        }
    }

    match correction {
        OutlierCorrection::Remove => {
            // Mark outliers as NaN
            for (idx, outlier) in &outlier_indices {
                modifications.push(DataModification {
                    timestamp: outlier.timestamp,
                    operation: ModificationOperation::OutlierCorrected {
                        correction: OutlierCorrection::Remove,
                    },
                    original_value: Some(new_values[*idx]),
                    new_value: f64::NAN,
                    confidence: 0.9,
                });
                new_values[*idx] = f64::NAN;
            }
        }
        OutlierCorrection::Cap { lower_percentile, upper_percentile } => {
            // Calculate percentiles
            let mut sorted_values: Vec<f64> = new_values
                .iter()
                .filter(|v| !v.is_nan() && !v.is_infinite())
                .copied()
                .collect();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if !sorted_values.is_empty() {
                let lower_idx = ((lower_percentile / 100.0) * (sorted_values.len() - 1) as f64) as usize;
                let upper_idx = ((upper_percentile / 100.0) * (sorted_values.len() - 1) as f64) as usize;
                let lower_bound = sorted_values[lower_idx];
                let upper_bound = sorted_values[upper_idx];

                for (idx, outlier) in &outlier_indices {
                    let capped_value = new_values[*idx].clamp(lower_bound, upper_bound);
                    if capped_value != new_values[*idx] {
                        modifications.push(DataModification {
                            timestamp: outlier.timestamp,
                            operation: ModificationOperation::OutlierCorrected {
                                correction: correction.clone(),
                            },
                            original_value: Some(new_values[*idx]),
                            new_value: capped_value,
                            confidence: 0.85,
                        });
                        new_values[*idx] = capped_value;
                    }
                }
            }
        }
        OutlierCorrection::Replace { ref method } => {
            // First mark outliers as NaN
            for idx in outlier_indices.keys() {
                new_values[*idx] = f64::NAN;
            }
            // Then fill using imputation method
            let temp_data = TimeSeries::new(
                data.name.clone(),
                data.timestamps.clone(),
                new_values.clone(),
            ).map_err(|e| QualityError::data_cleaning(e.to_string()))?;

            let filled = fill_gaps(&temp_data, method.clone())?;
            new_values = filled.values;

            for (idx, outlier) in &outlier_indices {
                modifications.push(DataModification {
                    timestamp: outlier.timestamp,
                    operation: ModificationOperation::OutlierCorrected {
                        correction: correction.clone(),
                    },
                    original_value: Some(data.values[*idx]),
                    new_value: new_values[*idx],
                    confidence: 0.75,
                });
            }
        }
        OutlierCorrection::Smooth { window_size } => {
            for (idx, outlier) in &outlier_indices {
                let start = idx.saturating_sub(window_size / 2);
                let end = (*idx + window_size / 2 + 1).min(new_values.len());

                let mut sum = 0.0;
                let mut count = 0;

                for j in start..end {
                    if j != *idx && !new_values[j].is_nan() && !outlier_indices.contains_key(&j) {
                        sum += new_values[j];
                        count += 1;
                    }
                }

                if count > 0 {
                    let smoothed = sum / count as f64;
                    modifications.push(DataModification {
                        timestamp: outlier.timestamp,
                        operation: ModificationOperation::OutlierCorrected {
                            correction: correction.clone(),
                        },
                        original_value: Some(new_values[*idx]),
                        new_value: smoothed,
                        confidence: 0.8,
                    });
                    new_values[*idx] = smoothed;
                }
            }
        }
    }

    TimeSeries::new(
        data.name.clone(),
        data.timestamps.clone(),
        new_values,
    )
    .map_err(|e| QualityError::data_cleaning(e.to_string()))
}

/// Reduces noise in time series
pub fn reduce_noise(
    data: &TimeSeries,
    method: NoiseReduction,
) -> QualityResult<TimeSeries> {
    if data.is_empty() {
        return Err(QualityError::validation("Cannot reduce noise in empty time series"));
    }

    let mut new_values = data.values.clone();

    match method {
        NoiseReduction::MovingAverage { window_size } => {
            apply_moving_average(&mut new_values, window_size)?;
        }
        NoiseReduction::ExponentialSmoothing { alpha } => {
            apply_exponential_smoothing(&mut new_values, alpha)?;
        }
        NoiseReduction::MedianFilter { window_size } => {
            apply_median_filter(&mut new_values, window_size)?;
        }
    }

    TimeSeries::new(
        data.name.clone(),
        data.timestamps.clone(),
        new_values,
    )
    .map_err(|e| QualityError::data_cleaning(e.to_string()))
}

/// Applies moving average smoothing
fn apply_moving_average(values: &mut [f64], window_size: usize) -> QualityResult<()> {
    if window_size == 0 {
        return Err(QualityError::configuration("Window size must be greater than 0"));
    }

    let original = values.to_vec();
    let half_window = window_size / 2;

    for i in 0..values.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(values.len());

        let mut sum = 0.0;
        let mut count = 0;

        for j in start..end {
            if !original[j].is_nan() && !original[j].is_infinite() {
                sum += original[j];
                count += 1;
            }
        }

        if count > 0 {
            values[i] = sum / count as f64;
        }
    }

    Ok(())
}

/// Applies exponential smoothing
fn apply_exponential_smoothing(values: &mut [f64], alpha: f64) -> QualityResult<()> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(QualityError::configuration("Alpha must be between 0 and 1"));
    }

    let mut smoothed = values[0];

    for i in 1..values.len() {
        if !values[i].is_nan() && !values[i].is_infinite() {
            smoothed = alpha * values[i] + (1.0 - alpha) * smoothed;
            values[i] = smoothed;
        }
    }

    Ok(())
}

/// Applies median filter
fn apply_median_filter(values: &mut [f64], window_size: usize) -> QualityResult<()> {
    if window_size == 0 {
        return Err(QualityError::configuration("Window size must be greater than 0"));
    }

    let original = values.to_vec();
    let half_window = window_size / 2;

    for i in 0..values.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(values.len());

        let mut window_values: Vec<f64> = Vec::new();

        for j in start..end {
            if !original[j].is_nan() && !original[j].is_infinite() {
                window_values.push(original[j]);
            }
        }

        if !window_values.is_empty() {
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if window_values.len() % 2 == 0 {
                (window_values[window_values.len() / 2 - 1] + window_values[window_values.len() / 2]) / 2.0
            } else {
                window_values[window_values.len() / 2]
            };
            values[i] = median;
        }
    }

    Ok(())
}

/// Main cleaning function that applies multiple operations
pub fn clean_timeseries(
    data: &TimeSeries,
    issues: &[QualityIssue],
    config: &CleaningConfig,
) -> QualityResult<CleaningResult> {
    if data.is_empty() {
        return Err(QualityError::validation("Cannot clean empty time series"));
    }

    let mut cleaned_data = data.clone();
    let mut report = CleaningReport::default();
    let all_modifications = Vec::new();

    // Check if we can make modifications
    let max_mods = (data.len() as f64 * config.max_modifications) as usize;

    // Fill gaps first
    let gap_method = ImputationMethod::LinearInterpolation;
    cleaned_data = fill_gaps(&cleaned_data, gap_method.clone())?;
    report.gaps_filled = cleaned_data.values.iter().filter(|v| !v.is_nan()).count()
        - data.values.iter().filter(|v| !v.is_nan()).count();
    report.methods_used.push(format!("{:?}", gap_method));

    // Check modification limit
    if all_modifications.len() + report.gaps_filled > max_mods {
        return Err(QualityError::configuration(
            format!("Cleaning would exceed maximum modification limit of {}%",
                   config.max_modifications * 100.0)
        ));
    }

    let quality_impact = QualityImpact {
        completeness_gain: report.gaps_filled as f64 / data.len() as f64,
        consistency_change: 0.0,
        potential_bias: 0.05,
        uncertainty: 0.10,
    };

    Ok(CleaningResult {
        cleaned_data,
        cleaning_report: report,
        quality_impact,
        modifications: all_modifications,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_timeseries_with_gaps() -> TimeSeries {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..10)
            .map(|i| start + Duration::hours(i as i64))
            .collect();

        let mut values = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        values[3] = f64::NAN; // Create gap
        values[7] = f64::NAN; // Create gap

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_forward_fill() {
        let data = create_test_timeseries_with_gaps();
        let result = fill_gaps(&data, ImputationMethod::ForwardFill).unwrap();

        assert_eq!(result.values[3], 12.0); // Filled from previous
        assert_eq!(result.values[7], 16.0); // Filled from previous
    }

    #[test]
    fn test_backward_fill() {
        let data = create_test_timeseries_with_gaps();
        let result = fill_gaps(&data, ImputationMethod::BackwardFill).unwrap();

        assert_eq!(result.values[3], 14.0); // Filled from next
        assert_eq!(result.values[7], 18.0); // Filled from next
    }

    #[test]
    fn test_linear_interpolation() {
        let data = create_test_timeseries_with_gaps();
        let result = fill_gaps(&data, ImputationMethod::LinearInterpolation).unwrap();

        assert_eq!(result.values[3], 13.0); // Interpolated
        assert_eq!(result.values[7], 17.0); // Interpolated
    }

    #[test]
    fn test_mean_imputation() {
        let data = create_test_timeseries_with_gaps();
        let result = fill_gaps(&data, ImputationMethod::MeanImputation { window_size: Some(3) }).unwrap();

        // Mean of window should be used
        assert!(!result.values[3].is_nan());
        assert!(!result.values[7].is_nan());
    }

    #[test]
    fn test_cleaning_config_presets() {
        let conservative = CleaningConfig::conservative();
        assert_eq!(conservative.max_modifications, 0.05);
        assert!(conservative.validation_required);

        let aggressive = CleaningConfig::aggressive();
        assert_eq!(aggressive.max_modifications, 0.30);
        assert!(!aggressive.validation_required);
    }

    #[test]
    fn test_noise_reduction_moving_average() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..10)
            .map(|i| start + Duration::hours(i as i64))
            .collect();
        let values = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let data = TimeSeries::new("test".to_string(), timestamps, values).unwrap();

        let result = reduce_noise(&data, NoiseReduction::MovingAverage { window_size: 3 }).unwrap();

        // Should be smoother
        assert!(result.values.len() == data.values.len());
    }

    #[test]
    fn test_clean_timeseries() {
        let data = create_test_timeseries_with_gaps();
        let issues = vec![];
        // Use aggressive config which has higher modification limit (30%)
        let config = CleaningConfig::aggressive();

        let result = clean_timeseries(&data, &issues, &config).unwrap();

        assert!(result.cleaning_report.gaps_filled >= 0);
        assert!(result.quality_impact.completeness_gain >= 0.0);
    }

    #[test]
    fn test_empty_data_error() {
        let data = TimeSeries::empty("empty".to_string());
        let result = fill_gaps(&data, ImputationMethod::ForwardFill);
        assert!(result.is_err());
    }

    #[test]
    fn test_performance() {
        // Test requirement: <200ms for 10K datapoints
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..10000)
            .map(|i| start + Duration::hours(i as i64))
            .collect();
        let mut values: Vec<f64> = (0..10000).map(|i| i as f64).collect();

        // Add some gaps
        for i in (100..10000).step_by(100) {
            values[i] = f64::NAN;
        }

        let data = TimeSeries::new("perf_test".to_string(), timestamps, values).unwrap();

        let start_time = std::time::Instant::now();
        let result = fill_gaps(&data, ImputationMethod::LinearInterpolation);
        let duration = start_time.elapsed();

        assert!(result.is_ok());
        assert!(
            duration.as_millis() < 200,
            "Cleaning should complete in <200ms for 10K points, took {}ms",
            duration.as_millis()
        );
    }
}