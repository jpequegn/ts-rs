//! # Detrending Methods Module
//!
//! Methods for removing trends from time series data including linear detrending,
//! differencing, moving average detrending, and HP filter.

use serde::{Serialize, Deserialize};

/// Available detrending methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DetrendingMethod {
    /// Remove linear trend using least squares regression
    Linear,
    /// First-order differencing
    FirstDifference,
    /// Second-order differencing
    SecondDifference,
    /// Custom-order differencing
    Difference(usize),
    /// Moving average detrending
    MovingAverage(usize),
    /// Hodrick-Prescott filter
    HPFilter(f64),
}

impl Default for DetrendingMethod {
    fn default() -> Self {
        DetrendingMethod::Linear
    }
}

/// Result of detrending operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetrendingResult {
    /// Method used for detrending
    pub method: DetrendingMethod,

    /// Original time series data
    pub original: Vec<f64>,

    /// Detrended time series
    pub detrended: Vec<f64>,

    /// Extracted trend component (if available)
    pub trend: Option<Vec<f64>>,

    /// Quality metrics of the detrending
    pub quality_metrics: DetrendingQualityMetrics,

    /// Parameters used in detrending
    pub parameters: DetrendingParameters,
}

/// Quality metrics for detrending assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetrendingQualityMetrics {
    /// Variance reduction ratio
    pub variance_reduction: f64,

    /// Mean squared error of trend fit (if applicable)
    pub trend_mse: Option<f64>,

    /// R-squared of trend fit (if applicable)
    pub r_squared: Option<f64>,

    /// Standard deviation of residuals
    pub residual_std: f64,

    /// Autocorrelation at lag 1 of detrended series
    pub autocorr_lag1: f64,
}

/// Parameters used in detrending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetrendingParameters {
    /// Window size (for moving average methods)
    pub window_size: Option<usize>,

    /// Smoothing parameter (for HP filter)
    pub lambda: Option<f64>,

    /// Difference order (for differencing methods)
    pub difference_order: Option<usize>,

    /// Linear trend coefficients: [intercept, slope]
    pub linear_coefficients: Option<[f64; 2]>,
}

impl DetrendingResult {
    /// Create new detrending result
    pub fn new(method: DetrendingMethod, original: Vec<f64>, detrended: Vec<f64>) -> Self {
        let original_var = calculate_variance(&original);
        let detrended_var = calculate_variance(&detrended);
        let variance_reduction = if original_var > 0.0 {
            1.0 - (detrended_var / original_var)
        } else {
            0.0
        };

        let residual_std = detrended_var.sqrt();
        let autocorr_lag1 = calculate_autocorrelation(&detrended, 1);

        Self {
            method,
            original,
            detrended,
            trend: None,
            quality_metrics: DetrendingQualityMetrics {
                variance_reduction,
                trend_mse: None,
                r_squared: None,
                residual_std,
                autocorr_lag1,
            },
            parameters: DetrendingParameters {
                window_size: None,
                lambda: None,
                difference_order: None,
                linear_coefficients: None,
            },
        }
    }

    /// Add trend component
    pub fn with_trend(mut self, trend: Vec<f64>) -> Self {
        self.trend = Some(trend);
        self
    }

    /// Add quality metrics
    pub fn with_quality_metrics(mut self, metrics: DetrendingQualityMetrics) -> Self {
        self.quality_metrics = metrics;
        self
    }

    /// Add parameters
    pub fn with_parameters(mut self, parameters: DetrendingParameters) -> Self {
        self.parameters = parameters;
        self
    }
}

/// Linear detrending using least squares regression
pub fn linear_detrend(data: &[f64]) -> Result<DetrendingResult, Box<dyn std::error::Error>> {
    if data.len() < 3 {
        return Err("Linear detrending requires at least 3 data points".into());
    }

    // Filter out non-finite values
    let indexed_data: Vec<(usize, f64)> = data.iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();

    if indexed_data.len() < 3 {
        return Err("Linear detrending requires at least 3 valid data points".into());
    }

    // Calculate linear trend using least squares
    let n = indexed_data.len() as f64;
    let sum_x: f64 = indexed_data.iter().map(|(i, _)| *i as f64).sum();
    let sum_y: f64 = indexed_data.iter().map(|(_, y)| *y).sum();
    let sum_xy: f64 = indexed_data.iter().map(|(i, y)| (*i as f64) * y).sum();
    let sum_x2: f64 = indexed_data.iter().map(|(i, _)| (*i as f64).powi(2)).sum();

    let denominator = n * sum_x2 - sum_x.powi(2);
    if denominator.abs() < f64::EPSILON {
        return Err("Cannot fit linear trend: degenerate case".into());
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;

    // Generate trend line for all points (including non-finite)
    let trend: Vec<f64> = (0..data.len())
        .map(|i| intercept + slope * (i as f64))
        .collect();

    // Calculate detrended series
    let detrended: Vec<f64> = data.iter()
        .zip(trend.iter())
        .map(|(&original, &trend_val)| {
            if original.is_finite() {
                original - trend_val
            } else {
                f64::NAN
            }
        })
        .collect();

    // Calculate quality metrics
    let valid_residuals: Vec<f64> = detrended.iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    let trend_mse = if valid_residuals.is_empty() {
        None
    } else {
        Some(valid_residuals.iter().map(|&x| x.powi(2)).sum::<f64>() / valid_residuals.len() as f64)
    };

    // Calculate R-squared
    let valid_original: Vec<f64> = data.iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    let r_squared = if valid_original.len() > 1 {
        let mean_y = valid_original.iter().sum::<f64>() / valid_original.len() as f64;
        let ss_tot: f64 = valid_original.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = valid_residuals.iter().map(|&r| r.powi(2)).sum();

        if ss_tot > 0.0 {
            Some(1.0 - (ss_res / ss_tot))
        } else {
            Some(1.0)
        }
    } else {
        None
    };

    let mut result = DetrendingResult::new(DetrendingMethod::Linear, data.to_vec(), detrended)
        .with_trend(trend);

    result.quality_metrics.trend_mse = trend_mse;
    result.quality_metrics.r_squared = r_squared;
    result.parameters.linear_coefficients = Some([intercept, slope]);

    Ok(result)
}

/// Differencing detrending
pub fn difference_detrend(data: &[f64], order: usize) -> Result<DetrendingResult, Box<dyn std::error::Error>> {
    if order == 0 {
        return Err("Difference order must be greater than 0".into());
    }

    if data.len() <= order {
        return Err(format!("Data length must be greater than difference order {}", order).into());
    }

    let mut current_data = data.to_vec();

    // Apply differencing iteratively
    for _ in 0..order {
        let mut diff_data = Vec::with_capacity(current_data.len().saturating_sub(1));

        for i in 1..current_data.len() {
            if current_data[i].is_finite() && current_data[i-1].is_finite() {
                diff_data.push(current_data[i] - current_data[i-1]);
            } else {
                diff_data.push(f64::NAN);
            }
        }

        current_data = diff_data;
    }

    let method = match order {
        1 => DetrendingMethod::FirstDifference,
        2 => DetrendingMethod::SecondDifference,
        n => DetrendingMethod::Difference(n),
    };

    let mut result = DetrendingResult::new(method, data.to_vec(), current_data);
    result.parameters.difference_order = Some(order);

    Ok(result)
}

/// Moving average detrending
pub fn moving_average_detrend(data: &[f64], window: usize) -> Result<DetrendingResult, Box<dyn std::error::Error>> {
    if window == 0 {
        return Err("Window size must be greater than 0".into());
    }

    if data.len() < window {
        return Err("Data length must be at least equal to window size".into());
    }

    // Calculate centered moving average as trend
    let trend = calculate_centered_moving_average(data, window);

    // Calculate detrended series
    let detrended: Vec<f64> = data.iter()
        .zip(trend.iter())
        .map(|(&original, &trend_val)| {
            if original.is_finite() && trend_val.is_finite() {
                original - trend_val
            } else {
                f64::NAN
            }
        })
        .collect();

    let mut result = DetrendingResult::new(DetrendingMethod::MovingAverage(window), data.to_vec(), detrended)
        .with_trend(trend);

    result.parameters.window_size = Some(window);

    Ok(result)
}

/// Hodrick-Prescott filter detrending
pub fn hp_filter_detrend(data: &[f64], lambda: f64) -> Result<DetrendingResult, Box<dyn std::error::Error>> {
    if lambda <= 0.0 {
        return Err("Lambda parameter must be positive".into());
    }

    if data.len() < 4 {
        return Err("HP filter requires at least 4 data points".into());
    }

    // Filter out non-finite values and their indices
    let indexed_data: Vec<(usize, f64)> = data.iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();

    if indexed_data.len() < 4 {
        return Err("HP filter requires at least 4 valid data points".into());
    }

    let n = indexed_data.len();
    let y: Vec<f64> = indexed_data.iter().map(|(_, y)| *y).collect();

    // Solve the HP filter equations using simplified approach
    // For full implementation, would use matrix methods
    let trend = solve_hp_filter(&y, lambda)?;

    // Map trend back to original data indices
    let mut full_trend = vec![f64::NAN; data.len()];
    for (i, &(original_idx, _)) in indexed_data.iter().enumerate() {
        if i < trend.len() {
            full_trend[original_idx] = trend[i];
        }
    }

    // Calculate detrended series
    let detrended: Vec<f64> = data.iter()
        .zip(full_trend.iter())
        .map(|(&original, &trend_val)| {
            if original.is_finite() && trend_val.is_finite() {
                original - trend_val
            } else {
                f64::NAN
            }
        })
        .collect();

    let mut result = DetrendingResult::new(DetrendingMethod::HPFilter(lambda), data.to_vec(), detrended)
        .with_trend(full_trend);

    result.parameters.lambda = Some(lambda);

    Ok(result)
}

/// Perform detrending using specified method
pub fn perform_detrending(data: &[f64], method: DetrendingMethod) -> Result<DetrendingResult, Box<dyn std::error::Error>> {
    match method {
        DetrendingMethod::Linear => linear_detrend(data),
        DetrendingMethod::FirstDifference => difference_detrend(data, 1),
        DetrendingMethod::SecondDifference => difference_detrend(data, 2),
        DetrendingMethod::Difference(order) => difference_detrend(data, order),
        DetrendingMethod::MovingAverage(window) => moving_average_detrend(data, window),
        DetrendingMethod::HPFilter(lambda) => hp_filter_detrend(data, lambda),
    }
}

// Helper functions

/// Calculate variance of a dataset
fn calculate_variance(data: &[f64]) -> f64 {
    let valid_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if valid_data.len() <= 1 {
        return 0.0;
    }

    let mean = valid_data.iter().sum::<f64>() / valid_data.len() as f64;
    valid_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / valid_data.len() as f64
}

/// Calculate autocorrelation at specified lag
fn calculate_autocorrelation(data: &[f64], lag: usize) -> f64 {
    let valid_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if valid_data.len() <= lag + 1 {
        return 0.0;
    }

    let n = valid_data.len();
    let mean = valid_data.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..(n - lag) {
        let x_i = valid_data[i] - mean;
        let x_i_lag = valid_data[i + lag] - mean;
        numerator += x_i * x_i_lag;
        denominator += x_i * x_i;
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate centered moving average
fn calculate_centered_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    let mut trend = vec![f64::NAN; data.len()];
    let half_window = window / 2;

    for i in half_window..(data.len().saturating_sub(half_window)) {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(data.len());

        let window_data: Vec<f64> = data[start..end]
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        if !window_data.is_empty() {
            trend[i] = window_data.iter().sum::<f64>() / window_data.len() as f64;
        }
    }

    trend
}

/// Simplified HP filter solver
fn solve_hp_filter(data: &[f64], lambda: f64) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = data.len();
    if n < 4 {
        return Err("HP filter requires at least 4 data points".into());
    }

    // Simplified HP filter using iterative approach
    let mut trend = data.to_vec();
    let max_iterations = 100;
    let tolerance = 1e-6;

    for iteration in 0..max_iterations {
        let mut new_trend = trend.clone();
        let mut max_change: f64 = 0.0;

        // Update interior points
        for t in 2..(n - 2) {
            let penalty_term = lambda * (trend[t-2] - 4.0 * trend[t-1] + 6.0 * trend[t] - 4.0 * trend[t+1] + trend[t+2]);
            let data_term = data[t] - trend[t];
            let update = (data_term - penalty_term) / (1.0 + 6.0 * lambda);

            new_trend[t] = trend[t] + update;
            max_change = max_change.max(update.abs());
        }

        // Update boundary points with special handling
        new_trend[0] = (data[0] + lambda * (2.0 * new_trend[1] - new_trend[2])) / (1.0 + lambda);
        new_trend[1] = (data[1] + lambda * (new_trend[0] + new_trend[2] - 2.0 * new_trend[1])) / (1.0 + lambda);
        new_trend[n-2] = (data[n-2] + lambda * (new_trend[n-3] + new_trend[n-1] - 2.0 * new_trend[n-2])) / (1.0 + lambda);
        new_trend[n-1] = (data[n-1] + lambda * (2.0 * new_trend[n-2] - new_trend[n-3])) / (1.0 + lambda);

        trend = new_trend;

        if max_change < tolerance {
            break;
        }

        if iteration == max_iterations - 1 {
            eprintln!("Warning: HP filter did not converge after {} iterations", max_iterations);
        }
    }

    Ok(trend)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_trend_data(n: usize, intercept: f64, slope: f64, noise: f64) -> Vec<f64> {
        (0..n)
            .map(|i| intercept + slope * i as f64 + noise * (i as f64 * 0.1).sin())
            .collect()
    }

    #[test]
    fn test_linear_detrend() {
        let data = generate_trend_data(50, 10.0, 0.5, 1.0);
        let result = linear_detrend(&data).unwrap();

        assert_eq!(result.method, DetrendingMethod::Linear);
        assert_eq!(result.original.len(), 50);
        assert_eq!(result.detrended.len(), 50);
        assert!(result.trend.is_some());
        assert!(result.quality_metrics.r_squared.is_some());
        assert!(result.quality_metrics.r_squared.unwrap() > 0.7);
        assert!(result.parameters.linear_coefficients.is_some());

        let coeffs = result.parameters.linear_coefficients.unwrap();
        assert!((coeffs[0] - 10.0).abs() < 2.0); // intercept
        assert!((coeffs[1] - 0.5).abs() < 0.1);  // slope
    }

    #[test]
    fn test_first_difference() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = difference_detrend(&data, 1).unwrap();

        assert_eq!(result.method, DetrendingMethod::FirstDifference);
        assert_eq!(result.detrended, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.parameters.difference_order, Some(1));
    }

    #[test]
    fn test_second_difference() {
        let data = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]; // x^2
        let result = difference_detrend(&data, 2).unwrap();

        assert_eq!(result.method, DetrendingMethod::SecondDifference);
        // Second difference of x^2 should be constant (2)
        let expected_len = data.len() - 2;
        assert_eq!(result.detrended.len(), expected_len);
        assert_eq!(result.parameters.difference_order, Some(2));
    }

    #[test]
    fn test_moving_average_detrend() {
        let data = generate_trend_data(30, 5.0, 0.2, 0.5);
        let result = moving_average_detrend(&data, 5).unwrap();

        assert_eq!(result.method, DetrendingMethod::MovingAverage(5));
        assert_eq!(result.original.len(), 30);
        assert_eq!(result.detrended.len(), 30);
        assert!(result.trend.is_some());
        assert_eq!(result.parameters.window_size, Some(5));
    }

    #[test]
    fn test_hp_filter() {
        let data = generate_trend_data(20, 0.0, 1.0, 0.5);
        let result = hp_filter_detrend(&data, 1600.0).unwrap();

        assert_eq!(result.method, DetrendingMethod::HPFilter(1600.0));
        assert_eq!(result.original.len(), 20);
        assert_eq!(result.detrended.len(), 20);
        assert!(result.trend.is_some());
        assert_eq!(result.parameters.lambda, Some(1600.0));
    }

    #[test]
    fn test_perform_detrending_all_methods() {
        let data = generate_trend_data(25, 2.0, 0.3, 0.2);

        let linear_result = perform_detrending(&data, DetrendingMethod::Linear).unwrap();
        assert_eq!(linear_result.method, DetrendingMethod::Linear);

        let diff_result = perform_detrending(&data, DetrendingMethod::FirstDifference).unwrap();
        assert_eq!(diff_result.method, DetrendingMethod::FirstDifference);

        let ma_result = perform_detrending(&data, DetrendingMethod::MovingAverage(7)).unwrap();
        assert_eq!(ma_result.method, DetrendingMethod::MovingAverage(7));

        let hp_result = perform_detrending(&data, DetrendingMethod::HPFilter(100.0)).unwrap();
        assert_eq!(hp_result.method, DetrendingMethod::HPFilter(100.0));
    }

    #[test]
    fn test_detrending_with_missing_values() {
        let mut data = generate_trend_data(20, 1.0, 0.1, 0.1);
        data[5] = f64::NAN;
        data[15] = f64::INFINITY;

        let result = linear_detrend(&data).unwrap();
        assert_eq!(result.original.len(), 20);
        assert_eq!(result.detrended.len(), 20);
        assert!(result.detrended[5].is_nan());
        assert!(result.detrended[15].is_nan());
    }

    #[test]
    fn test_insufficient_data_errors() {
        let short_data = vec![1.0, 2.0];

        let linear_result = linear_detrend(&short_data);
        assert!(linear_result.is_err());

        let diff_result = difference_detrend(&short_data, 3);
        assert!(diff_result.is_err());

        let hp_result = hp_filter_detrend(&short_data, 100.0);
        assert!(hp_result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let data = generate_trend_data(10, 0.0, 0.0, 0.1);

        let zero_diff_result = difference_detrend(&data, 0);
        assert!(zero_diff_result.is_err());

        let zero_window_result = moving_average_detrend(&data, 0);
        assert!(zero_window_result.is_err());

        let negative_lambda_result = hp_filter_detrend(&data, -1.0);
        assert!(negative_lambda_result.is_err());
    }

    #[test]
    fn test_variance_reduction() {
        let data = generate_trend_data(40, 0.0, 1.0, 0.1);
        let result = linear_detrend(&data).unwrap();

        assert!(result.quality_metrics.variance_reduction > 0.0);
        assert!(result.quality_metrics.variance_reduction <= 1.0);
    }

    #[test]
    fn test_quality_metrics() {
        let data = generate_trend_data(30, 5.0, 0.5, 0.2);
        let result = linear_detrend(&data).unwrap();

        assert!(result.quality_metrics.residual_std >= 0.0);
        assert!(result.quality_metrics.autocorr_lag1.abs() <= 1.0);
        assert!(result.quality_metrics.trend_mse.is_some());
        assert!(result.quality_metrics.r_squared.is_some());
    }
}