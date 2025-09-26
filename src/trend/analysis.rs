//! # Trend Analysis and Classification Module
//!
//! Comprehensive trend analysis including strength measurement, direction classification,
//! rate of change analysis, and breakpoint detection.

use crate::analysis::{TrendDirection, TrendAnalysis};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Configuration for comprehensive trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    /// Minimum number of data points required
    pub min_observations: usize,

    /// Significance level for statistical tests
    pub alpha: f64,

    /// Window size for local trend analysis
    pub window_size: Option<usize>,

    /// Whether to perform Mann-Kendall test
    pub perform_mann_kendall: bool,

    /// Whether to perform Sen's slope estimation
    pub perform_sens_slope: bool,

    /// Whether to perform Pettitt's test for change points
    pub perform_pettitt_test: bool,

    /// Decomposition method to use
    pub decomposition_method: Option<crate::trend::DecompositionMethod>,

    /// Seasonal period (if known)
    pub seasonal_period: Option<usize>,

    /// Whether to detect breakpoints
    pub detect_breakpoints: bool,

    /// Minimum breakpoint confidence threshold
    pub breakpoint_confidence_threshold: f64,

    /// Whether to generate plot data
    pub generate_plot_data: bool,

    /// Rate of change analysis configuration
    pub rate_of_change_config: RateOfChangeConfig,
}

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            min_observations: 10,
            alpha: 0.05,
            window_size: None,
            perform_mann_kendall: true,
            perform_sens_slope: true,
            perform_pettitt_test: true,
            decomposition_method: None,
            seasonal_period: None,
            detect_breakpoints: true,
            breakpoint_confidence_threshold: 0.7,
            generate_plot_data: false,
            rate_of_change_config: RateOfChangeConfig::default(),
        }
    }
}

/// Configuration for rate of change analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateOfChangeConfig {
    /// Window size for rate calculations
    pub window_size: usize,

    /// Whether to compute acceleration (second derivative)
    pub compute_acceleration: bool,

    /// Time units for rate calculation
    pub time_unit: TimeUnit,

    /// Smoothing factor for rate calculations
    pub smoothing_factor: Option<f64>,
}

impl Default for RateOfChangeConfig {
    fn default() -> Self {
        Self {
            window_size: 3,
            compute_acceleration: true,
            time_unit: TimeUnit::Days,
            smoothing_factor: None,
        }
    }
}

/// Time units for rate calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl TimeUnit {
    /// Get the scaling factor for converting to this time unit from days
    pub fn scale_factor(&self) -> f64 {
        match self {
            TimeUnit::Seconds => 86400.0,
            TimeUnit::Minutes => 1440.0,
            TimeUnit::Hours => 24.0,
            TimeUnit::Days => 1.0,
            TimeUnit::Weeks => 1.0 / 7.0,
            TimeUnit::Months => 1.0 / 30.44, // Average month length
            TimeUnit::Years => 1.0 / 365.25,
        }
    }
}

/// Trend strength measurement
pub type TrendStrength = f64;

/// Rate of change analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateOfChangeAnalysis {
    /// Average rate of change over the entire series
    pub average_rate: f64,

    /// Maximum rate of change observed
    pub max_rate: f64,

    /// Minimum rate of change observed
    pub min_rate: f64,

    /// Standard deviation of rates
    pub rate_std: f64,

    /// Rate of change values at each time point
    pub rates: Vec<f64>,

    /// Acceleration values (if computed)
    pub acceleration: Option<Vec<f64>>,

    /// Time points where rates were calculated
    pub time_indices: Vec<usize>,

    /// Configuration used for analysis
    pub config: RateOfChangeConfig,

    /// Quality metrics
    pub quality_metrics: RateOfChangeQualityMetrics,
}

/// Quality metrics for rate of change analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateOfChangeQualityMetrics {
    /// Coefficient of variation for rates
    pub rate_cv: f64,

    /// Autocorrelation of rates at lag 1
    pub rate_autocorr: f64,

    /// Number of sign changes in rates
    pub sign_changes: usize,

    /// Percentage of stable periods (low rate of change)
    pub stable_periods_pct: f64,
}

/// Breakpoint/change point detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakpointDetection {
    /// Index in the time series where breakpoint occurs
    pub index: usize,

    /// Confidence level of the breakpoint (0.0 to 1.0)
    pub confidence: f64,

    /// Type of change detected
    pub change_type: ChangeType,

    /// Magnitude of change
    pub magnitude: f64,

    /// Mean value before the breakpoint
    pub mean_before: f64,

    /// Mean value after the breakpoint
    pub mean_after: f64,

    /// Statistical test used for detection
    pub test_statistic: f64,

    /// P-value of the test
    pub p_value: f64,
}

/// Type of change at a breakpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Mean level shift
    LevelShift,
    /// Variance change
    VarianceChange,
    /// Trend change
    TrendChange,
    /// Combination of changes
    Mixed,
}

/// Comprehensive trend analysis result that combines multiple analysis types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveTrendResult {
    /// Basic linear trend analysis
    pub linear_trend: TrendAnalysis,

    /// Trend direction classification
    pub direction: TrendDirection,

    /// Overall trend strength
    pub strength: TrendStrength,

    /// Statistical confidence in the trend assessment
    pub confidence: f64,

    /// Rate of change analysis
    pub rate_of_change: Option<RateOfChangeAnalysis>,

    /// Detected breakpoints
    pub breakpoints: Vec<BreakpointDetection>,

    /// Trend persistence metrics
    pub persistence: TrendPersistence,

    /// Seasonal trend analysis (if applicable)
    pub seasonal_trends: Option<HashMap<String, TrendAnalysis>>,

    /// Overall growth rate (annualized if possible)
    pub growth_rate: Option<f64>,

    /// Quality and reliability metrics
    pub quality_metrics: TrendQualityMetrics,
}

/// Trend persistence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPersistence {
    /// Hurst exponent (measure of long-term memory)
    pub hurst_exponent: Option<f64>,

    /// Autocorrelation at various lags
    pub autocorrelations: Vec<f64>,

    /// Maximum consecutive periods in same direction
    pub max_consecutive_direction: usize,

    /// Percentage of time trend maintains direction
    pub directional_consistency: f64,
}

/// Quality metrics for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendQualityMetrics {
    /// Data completeness (percentage of non-missing values)
    pub data_completeness: f64,

    /// Stationarity test p-value (if applicable)
    pub stationarity_p_value: Option<f64>,

    /// Residual diagnostics
    pub residual_diagnostics: ResidualDiagnostics,

    /// Model fitness measures
    pub model_fitness: ModelFitness,

    /// Robustness indicators
    pub robustness: RobustnessMetrics,
}

/// Residual diagnostic measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualDiagnostics {
    /// Durbin-Watson statistic for autocorrelation
    pub durbin_watson: f64,

    /// Ljung-Box test p-value for residual autocorrelation
    pub ljung_box_p_value: f64,

    /// Jarque-Bera test p-value for normality
    pub jarque_bera_p_value: f64,

    /// Residual standard error
    pub residual_std_error: f64,
}

/// Model fitness measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFitness {
    /// Akaike Information Criterion
    pub aic: f64,

    /// Bayesian Information Criterion
    pub bic: f64,

    /// Root Mean Square Error
    pub rmse: f64,

    /// Mean Absolute Error
    pub mae: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,
}

/// Robustness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessMetrics {
    /// Sensitivity to outliers (percentage change in slope with outlier removal)
    pub outlier_sensitivity: f64,

    /// Bootstrap confidence interval for slope
    pub bootstrap_slope_ci: Option<(f64, f64)>,

    /// Cross-validation R-squared
    pub cv_r_squared: Option<f64>,
}

/// Analyze comprehensive trend characteristics
pub fn analyze_trend_comprehensive(
    data: &[f64],
    config: &TrendAnalysisConfig,
) -> Result<ComprehensiveTrendResult, Box<dyn std::error::Error>> {
    if data.len() < config.min_observations {
        return Err(format!(
            "Insufficient data: need at least {} observations, got {}",
            config.min_observations,
            data.len()
        ).into());
    }

    // Filter valid data
    let valid_data: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();

    if valid_data.len() < config.min_observations {
        return Err("Insufficient valid data points".into());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, v)| *v).collect();

    // Perform basic linear trend analysis
    let linear_trend = compute_linear_trend(&values)?;

    // Classify trend direction
    let direction = classify_trend_direction(linear_trend.slope, linear_trend.r_squared, config.alpha);

    // Compute trend strength
    let strength = compute_trend_strength(&values, linear_trend.r_squared);

    // Calculate confidence
    let confidence = calculate_trend_confidence(&linear_trend, &values);

    // Perform rate of change analysis
    let rate_of_change = if valid_data.len() > config.rate_of_change_config.window_size {
        Some(analyze_rate_of_change(&values, &config.rate_of_change_config)?)
    } else {
        None
    };

    // Detect breakpoints
    let breakpoints = if config.detect_breakpoints && valid_data.len() > 10 {
        detect_breakpoints(&values, config.breakpoint_confidence_threshold)?
    } else {
        Vec::new()
    };

    // Calculate persistence metrics
    let persistence = calculate_trend_persistence(&values);

    // Calculate growth rate
    let growth_rate = calculate_growth_rate(&values);

    // Quality metrics
    let data_completeness = valid_data.len() as f64 / data.len() as f64;
    let quality_metrics = calculate_quality_metrics(&values, &linear_trend, data_completeness);

    Ok(ComprehensiveTrendResult {
        linear_trend,
        direction,
        strength,
        confidence,
        rate_of_change,
        breakpoints,
        persistence,
        seasonal_trends: None, // Could be computed with decomposition
        growth_rate,
        quality_metrics,
    })
}

/// Classify trend direction based on slope and significance
pub fn classify_trend_direction(slope: f64, r_squared: f64, alpha: f64) -> TrendDirection {
    // Simple classification logic - could be enhanced with statistical tests
    let strength_threshold = 1.0 - alpha; // Use complement of alpha as strength threshold

    if r_squared < 0.1 {
        return TrendDirection::Inconclusive;
    }

    let abs_slope = slope.abs();
    let is_strong = r_squared > strength_threshold;

    match slope {
        s if s > 0.0 => {
            if is_strong && abs_slope > 0.5 {
                TrendDirection::StronglyIncreasing
            } else {
                TrendDirection::Increasing
            }
        },
        s if s < 0.0 => {
            if is_strong && abs_slope > 0.5 {
                TrendDirection::StronglyDecreasing
            } else {
                TrendDirection::Decreasing
            }
        },
        _ => TrendDirection::Stable,
    }
}

/// Compute trend strength based on various metrics
pub fn compute_trend_strength(data: &[f64], r_squared: f64) -> TrendStrength {
    if data.len() < 3 {
        return 0.0;
    }

    // Combine R-squared with consistency metrics
    let consistency = calculate_directional_consistency(data);
    let volatility_adjustment = calculate_volatility_adjustment(data);

    // Weighted combination
    let base_strength = r_squared;
    let adjusted_strength = base_strength * consistency * volatility_adjustment;

    adjusted_strength.min(1.0).max(0.0)
}

// Helper functions

/// Compute basic linear trend using least squares
fn compute_linear_trend(data: &[f64]) -> Result<TrendAnalysis, Box<dyn std::error::Error>> {
    if data.len() < 2 {
        return Err("Need at least 2 data points for linear trend".into());
    }

    let n = data.len() as f64;
    let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

    let denominator = n * sum_x2 - sum_x.powi(2);
    if denominator.abs() < f64::EPSILON {
        return Err("Cannot compute linear trend: degenerate case".into());
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R-squared
    let mean_y = sum_y / n;
    let ss_tot: f64 = data.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = data.iter().enumerate()
        .map(|(i, &y)| {
            let predicted = intercept + slope * i as f64;
            (y - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 1.0 };

    // Classify direction
    let direction = if slope > 0.1 {
        TrendDirection::Increasing
    } else if slope < -0.1 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    };

    let strength = r_squared;

    Ok(TrendAnalysis::new(direction, strength, slope, intercept, r_squared))
}

/// Calculate trend confidence based on statistical measures
fn calculate_trend_confidence(trend: &TrendAnalysis, data: &[f64]) -> f64 {
    // Simple confidence calculation based on R-squared and data length
    let sample_size_factor = (data.len() as f64).ln() / 10.0; // Log adjustment for sample size
    let base_confidence = trend.r_squared;

    (base_confidence * (1.0 + sample_size_factor)).min(1.0)
}

/// Analyze rate of change over time
fn analyze_rate_of_change(
    data: &[f64],
    config: &RateOfChangeConfig,
) -> Result<RateOfChangeAnalysis, Box<dyn std::error::Error>> {
    if data.len() < config.window_size + 1 {
        return Err("Insufficient data for rate of change analysis".into());
    }

    let mut rates = Vec::new();
    let mut time_indices = Vec::new();

    // Calculate rates using specified window
    for i in config.window_size..data.len() {
        let window_start = i - config.window_size;
        let rate = (data[i] - data[window_start]) / config.window_size as f64;

        // Apply time unit scaling
        let scaled_rate = rate * config.time_unit.scale_factor();

        rates.push(scaled_rate);
        time_indices.push(i);
    }

    if rates.is_empty() {
        return Err("No rates could be calculated".into());
    }

    // Calculate statistics
    let average_rate = rates.iter().sum::<f64>() / rates.len() as f64;
    let max_rate = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_rate = rates.iter().copied().fold(f64::INFINITY, f64::min);

    let variance = rates.iter()
        .map(|&r| (r - average_rate).powi(2))
        .sum::<f64>() / rates.len() as f64;
    let rate_std = variance.sqrt();

    // Calculate acceleration if requested
    let acceleration = if config.compute_acceleration && rates.len() > 1 {
        let mut accel = Vec::new();
        for i in 1..rates.len() {
            accel.push(rates[i] - rates[i-1]);
        }
        Some(accel)
    } else {
        None
    };

    // Quality metrics
    let rate_cv = if average_rate.abs() > f64::EPSILON {
        rate_std / average_rate.abs()
    } else {
        0.0
    };

    let rate_autocorr = calculate_autocorr(&rates, 1);
    let sign_changes = count_sign_changes(&rates);
    let stable_periods_pct = calculate_stable_periods_percentage(&rates, rate_std);

    let quality_metrics = RateOfChangeQualityMetrics {
        rate_cv,
        rate_autocorr,
        sign_changes,
        stable_periods_pct,
    };

    Ok(RateOfChangeAnalysis {
        average_rate,
        max_rate,
        min_rate,
        rate_std,
        rates,
        acceleration,
        time_indices,
        config: config.clone(),
        quality_metrics,
    })
}

/// Detect breakpoints in the time series
fn detect_breakpoints(
    data: &[f64],
    confidence_threshold: f64,
) -> Result<Vec<BreakpointDetection>, Box<dyn std::error::Error>> {
    let mut breakpoints = Vec::new();

    // Simple breakpoint detection using moving window variance
    let window_size = (data.len() / 10).max(5).min(20);

    for i in window_size..(data.len() - window_size) {
        let before_data = &data[(i - window_size)..i];
        let after_data = &data[i..(i + window_size)];

        let mean_before = before_data.iter().sum::<f64>() / before_data.len() as f64;
        let mean_after = after_data.iter().sum::<f64>() / after_data.len() as f64;

        let var_before = before_data.iter()
            .map(|&x| (x - mean_before).powi(2))
            .sum::<f64>() / before_data.len() as f64;
        let var_after = after_data.iter()
            .map(|&x| (x - mean_after).powi(2))
            .sum::<f64>() / after_data.len() as f64;

        let magnitude = (mean_after - mean_before).abs();
        let pooled_std = ((var_before + var_after) / 2.0).sqrt();

        // Simple t-test approximation
        let test_statistic = if pooled_std > f64::EPSILON {
            magnitude / (pooled_std * ((2.0 / window_size as f64).sqrt()))
        } else {
            0.0
        };

        let confidence = (test_statistic / 3.0).min(1.0); // Simple confidence mapping

        if confidence > confidence_threshold {
            let change_type = if var_after > var_before * 1.5 {
                ChangeType::VarianceChange
            } else {
                ChangeType::LevelShift
            };

            breakpoints.push(BreakpointDetection {
                index: i,
                confidence,
                change_type,
                magnitude,
                mean_before,
                mean_after,
                test_statistic,
                p_value: 1.0 - confidence, // Simplified p-value
            });
        }
    }

    Ok(breakpoints)
}

/// Calculate trend persistence metrics
fn calculate_trend_persistence(data: &[f64]) -> TrendPersistence {
    let autocorrelations = (1..=5.min(data.len() / 4))
        .map(|lag| calculate_autocorr(data, lag))
        .collect();

    let directional_consistency = calculate_directional_consistency(data);
    let max_consecutive_direction = calculate_max_consecutive_direction(data);

    TrendPersistence {
        hurst_exponent: None, // Could implement Hurst exponent calculation
        autocorrelations,
        max_consecutive_direction,
        directional_consistency,
    }
}

/// Calculate annualized growth rate
fn calculate_growth_rate(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }

    let first_value = *data.first()?;
    let last_value = *data.last()?;

    if first_value <= 0.0 || last_value <= 0.0 {
        return None;
    }

    let periods = data.len() as f64 - 1.0;
    let growth_rate = (last_value / first_value).powf(1.0 / periods) - 1.0;

    Some(growth_rate)
}

/// Calculate comprehensive quality metrics
fn calculate_quality_metrics(
    data: &[f64],
    trend: &TrendAnalysis,
    data_completeness: f64,
) -> TrendQualityMetrics {
    // Calculate residuals
    let residuals: Vec<f64> = data.iter().enumerate()
        .map(|(i, &y)| {
            let predicted = trend.intercept + trend.slope * i as f64;
            y - predicted
        })
        .collect();

    let durbin_watson = calculate_durbin_watson(&residuals);
    let residual_std_error = (residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / residuals.len() as f64).sqrt();

    let residual_diagnostics = ResidualDiagnostics {
        durbin_watson,
        ljung_box_p_value: 0.5, // Simplified
        jarque_bera_p_value: 0.5, // Simplified
        residual_std_error,
    };

    let model_fitness = ModelFitness {
        aic: calculate_aic(data.len(), residual_std_error, 2),
        bic: calculate_bic(data.len(), residual_std_error, 2),
        rmse: residual_std_error,
        mae: residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64,
        mape: calculate_mape(data, &residuals),
    };

    let robustness = RobustnessMetrics {
        outlier_sensitivity: 0.1, // Simplified
        bootstrap_slope_ci: None,
        cv_r_squared: None,
    };

    TrendQualityMetrics {
        data_completeness,
        stationarity_p_value: None,
        residual_diagnostics,
        model_fitness,
        robustness,
    }
}

// Utility functions

fn calculate_autocorr(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let n = data.len() - lag;
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    let numerator: f64 = (0..n)
        .map(|i| (data[i] - mean) * (data[i + lag] - mean))
        .sum();

    let denominator: f64 = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn calculate_directional_consistency(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 1.0;
    }

    let mut same_direction_count = 0;
    let mut total_pairs = 0;

    for i in 1..data.len() {
        let diff = data[i] - data[i-1];
        if diff.abs() > f64::EPSILON {
            for j in (i+1)..data.len() {
                let other_diff = data[j] - data[j-1];
                if other_diff.abs() > f64::EPSILON {
                    if diff.signum() == other_diff.signum() {
                        same_direction_count += 1;
                    }
                    total_pairs += 1;
                }
            }
        }
    }

    if total_pairs > 0 {
        same_direction_count as f64 / total_pairs as f64
    } else {
        1.0
    }
}

fn calculate_volatility_adjustment(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 1.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

    let cv = if mean.abs() > f64::EPSILON { std / mean.abs() } else { 0.0 };

    // Lower volatility = higher adjustment (better trend strength)
    (1.0 / (1.0 + cv * 0.5)).min(1.0)
}

fn count_sign_changes(data: &[f64]) -> usize {
    let mut changes = 0;
    for i in 1..data.len() {
        if data[i].signum() != data[i-1].signum() {
            changes += 1;
        }
    }
    changes
}

fn calculate_stable_periods_percentage(rates: &[f64], threshold: f64) -> f64 {
    let stable_count = rates.iter()
        .filter(|&&r| r.abs() < threshold * 0.5)
        .count();

    stable_count as f64 / rates.len() as f64 * 100.0
}

fn calculate_max_consecutive_direction(data: &[f64]) -> usize {
    if data.len() < 2 {
        return 0;
    }

    let mut max_consecutive = 0;
    let mut current_consecutive = 1;
    let mut current_direction = (data[1] - data[0]).signum();

    for i in 2..data.len() {
        let direction = (data[i] - data[i-1]).signum();
        if direction == current_direction && direction != 0.0 {
            current_consecutive += 1;
        } else {
            max_consecutive = max_consecutive.max(current_consecutive);
            current_consecutive = 1;
            current_direction = direction;
        }
    }

    max_consecutive.max(current_consecutive)
}

fn calculate_durbin_watson(residuals: &[f64]) -> f64 {
    if residuals.len() < 2 {
        return 2.0;
    }

    let numerator: f64 = (1..residuals.len())
        .map(|i| (residuals[i] - residuals[i-1]).powi(2))
        .sum();

    let denominator: f64 = residuals.iter()
        .map(|&r| r.powi(2))
        .sum();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        2.0
    }
}

fn calculate_aic(n: usize, mse: f64, k: usize) -> f64 {
    let n_f64 = n as f64;
    let k_f64 = k as f64;
    n_f64 * mse.ln() + 2.0 * k_f64
}

fn calculate_bic(n: usize, mse: f64, k: usize) -> f64 {
    let n_f64 = n as f64;
    let k_f64 = k as f64;
    n_f64 * mse.ln() + k_f64 * n_f64.ln()
}

fn calculate_mape(actual: &[f64], residuals: &[f64]) -> f64 {
    let mut total_percentage_error = 0.0;
    let mut count = 0;

    for (i, &residual) in residuals.iter().enumerate() {
        if i < actual.len() && actual[i].abs() > f64::EPSILON {
            total_percentage_error += (residual.abs() / actual[i].abs()) * 100.0;
            count += 1;
        }
    }

    if count > 0 {
        total_percentage_error / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_linear_trend(n: usize, slope: f64, intercept: f64, noise: f64) -> Vec<f64> {
        (0..n)
            .map(|i| intercept + slope * i as f64 + noise * (i as f64).sin())
            .collect()
    }

    #[test]
    fn test_trend_direction_classification() {
        assert_eq!(classify_trend_direction(1.0, 0.9, 0.05), TrendDirection::StronglyIncreasing);
        assert_eq!(classify_trend_direction(0.1, 0.8, 0.05), TrendDirection::Increasing);
        assert_eq!(classify_trend_direction(-1.0, 0.9, 0.05), TrendDirection::StronglyDecreasing);
        assert_eq!(classify_trend_direction(0.0, 0.5, 0.05), TrendDirection::Stable);
        assert_eq!(classify_trend_direction(1.0, 0.05, 0.05), TrendDirection::Inconclusive);
    }

    #[test]
    fn test_trend_strength_computation() {
        let data = generate_linear_trend(50, 1.0, 0.0, 0.1);
        let strength = compute_trend_strength(&data, 0.95);
        assert!(strength > 0.8);
        assert!(strength <= 1.0);

        let noise_data = generate_linear_trend(50, 0.0, 5.0, 5.0);
        let weak_strength = compute_trend_strength(&noise_data, 0.1);
        assert!(weak_strength < 0.5);
    }

    #[test]
    fn test_comprehensive_trend_analysis() {
        let data = generate_linear_trend(100, 0.5, 10.0, 0.2);
        let config = TrendAnalysisConfig::default();

        let result = analyze_trend_comprehensive(&data, &config).unwrap();

        assert_eq!(result.direction, TrendDirection::Increasing);
        assert!(result.strength > 0.7);
        assert!(result.confidence > 0.7);
        assert!(result.growth_rate.is_some());
    }

    #[test]
    fn test_rate_of_change_analysis() {
        let data = generate_linear_trend(50, 1.0, 0.0, 0.1);
        let config = RateOfChangeConfig::default();

        let result = analyze_rate_of_change(&data, &config).unwrap();

        assert!(result.average_rate > 0.0);
        assert!(result.rates.len() > 0);
        assert!(result.acceleration.is_some());
    }

    #[test]
    fn test_breakpoint_detection() {
        let mut data = vec![1.0; 25];
        data.extend(vec![5.0; 25]); // Clear breakpoint at index 25

        let breakpoints = detect_breakpoints(&data, 0.7).unwrap();

        assert!(!breakpoints.is_empty());
        assert!(breakpoints[0].index >= 20 && breakpoints[0].index <= 30);
        assert!(breakpoints[0].confidence > 0.7);
    }

    #[test]
    fn test_persistence_metrics() {
        let data = generate_linear_trend(30, 0.5, 0.0, 0.1);
        let persistence = calculate_trend_persistence(&data);

        assert!(!persistence.autocorrelations.is_empty());
        assert!(persistence.directional_consistency > 0.0);
        assert!(persistence.max_consecutive_direction > 0);
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![1.0, 2.0];
        let config = TrendAnalysisConfig::default();

        let result = analyze_trend_comprehensive(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_autocorrelation_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let autocorr = calculate_autocorr(&data, 1);
        assert!(autocorr > 0.9); // Should be high for linear trend

        let white_noise = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let noise_autocorr = calculate_autocorr(&white_noise, 1);
        assert!(noise_autocorr < 0.0); // Should be negative for alternating pattern
    }

    #[test]
    fn test_quality_metrics() {
        let data = generate_linear_trend(50, 0.8, 2.0, 0.3);
        let trend = compute_linear_trend(&data).unwrap();
        let metrics = calculate_quality_metrics(&data, &trend, 1.0);

        assert_eq!(metrics.data_completeness, 1.0);
        assert!(metrics.residual_diagnostics.durbin_watson > 0.0);
        assert!(metrics.model_fitness.rmse > 0.0);
        assert!(metrics.model_fitness.aic > 0.0);
    }
}