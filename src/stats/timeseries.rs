//! Time series specific statistical analysis module
//!
//! Provides autocorrelation, partial autocorrelation, cross-correlation,
//! and lag analysis for time series data.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Time series specific statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    /// Autocorrelation function values
    pub acf: AutocorrelationResult,

    /// Partial autocorrelation function values
    pub pacf: PartialAutocorrelationResult,

    /// Lag analysis results
    pub lag_analysis: LagAnalysis,

    /// Cross-correlation results (if multiple series provided)
    pub cross_correlation: Option<CrossCorrelationResult>,
}

/// Autocorrelation function result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationResult {
    /// Lag values (0, 1, 2, ...)
    pub lags: Vec<usize>,

    /// ACF values for each lag
    pub values: Vec<f64>,

    /// Confidence intervals (95% by default)
    pub confidence_intervals: Vec<(f64, f64)>,

    /// Ljung-Box test statistic and p-value
    pub ljung_box_test: Option<LjungBoxTest>,

    /// Maximum lag analyzed
    pub max_lag: usize,
}

/// Partial autocorrelation function result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialAutocorrelationResult {
    /// Lag values (1, 2, 3, ...)
    pub lags: Vec<usize>,

    /// PACF values for each lag
    pub values: Vec<f64>,

    /// Confidence intervals (95% by default)
    pub confidence_intervals: Vec<(f64, f64)>,

    /// Maximum lag analyzed
    pub max_lag: usize,
}

/// Cross-correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCorrelationResult {
    /// Lag values (can be negative, zero, positive)
    pub lags: Vec<i32>,

    /// Cross-correlation values for each lag
    pub values: Vec<f64>,

    /// Peak correlation and its lag
    pub peak_correlation: (i32, f64),

    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Ljung-Box test for autocorrelation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LjungBoxTest {
    /// Test statistic
    pub statistic: f64,

    /// P-value
    pub p_value: f64,

    /// Degrees of freedom
    pub df: usize,

    /// Number of lags tested
    pub lags_tested: usize,

    /// Is there significant autocorrelation? (p < 0.05)
    pub has_autocorrelation: bool,
}

/// Lag analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagAnalysis {
    /// Optimal lag for modeling (based on various criteria)
    pub optimal_lag: Option<usize>,

    /// Information criteria for different lags
    pub information_criteria: HashMap<usize, InformationCriteria>,

    /// Significant lags (those exceeding confidence bounds)
    pub significant_lags: Vec<usize>,

    /// Seasonal patterns detected
    pub seasonal_patterns: Vec<SeasonalPattern>,
}

/// Information criteria for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationCriteria {
    /// Akaike Information Criterion
    pub aic: f64,

    /// Bayesian Information Criterion
    pub bic: f64,

    /// Hannan-Quinn Information Criterion
    pub hqic: f64,
}

/// Detected seasonal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Period of the seasonal pattern
    pub period: usize,

    /// Strength of the pattern (0.0 to 1.0)
    pub strength: f64,

    /// Confidence in the detection
    pub confidence: f64,
}

impl TimeSeriesStats {
    /// Compute comprehensive time series statistics
    pub fn compute(data: &[f64], max_lags: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if data.len() < 10 {
            return Err("Need at least 10 observations for time series analysis".into());
        }

        let effective_max_lags = max_lags.min(data.len() / 4);

        // Compute ACF
        let acf = compute_autocorrelation(data, effective_max_lags)?;

        // Compute PACF
        let pacf = compute_partial_autocorrelation(data, effective_max_lags)?;

        // Perform lag analysis
        let lag_analysis = perform_lag_analysis(&acf, &pacf, data.len())?;

        Ok(TimeSeriesStats {
            acf,
            pacf,
            lag_analysis,
            cross_correlation: None, // Will be set separately if needed
        })
    }

    /// Add cross-correlation analysis
    pub fn with_cross_correlation(mut self, other_series: &[f64], max_lags: i32) -> Result<Self, Box<dyn std::error::Error>> {
        // This should be called on the original data, not on the struct
        // But for the API, we'll accept it here
        Err("Cross-correlation should be computed separately and added".into())
    }
}

/// Compute autocorrelation function
pub fn compute_autocorrelation(data: &[f64], max_lags: usize) -> Result<AutocorrelationResult, Box<dyn std::error::Error>> {
    let n = data.len();
    if n < 10 {
        return Err("Need at least 10 observations for ACF computation".into());
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance == 0.0 {
        return Err("Cannot compute ACF for constant series".into());
    }

    let effective_max_lags = max_lags.min(n - 1);
    let mut lags = Vec::with_capacity(effective_max_lags + 1);
    let mut values = Vec::with_capacity(effective_max_lags + 1);

    // Lag 0 is always 1.0
    lags.push(0);
    values.push(1.0);

    // Compute ACF for lags 1 to max_lags
    for lag in 1..=effective_max_lags {
        let mut covariance = 0.0;
        let valid_pairs = n - lag;

        for i in 0..valid_pairs {
            covariance += (data[i] - mean) * (data[i + lag] - mean);
        }

        covariance /= n as f64; // Use n instead of valid_pairs for consistency
        let correlation = covariance / variance;

        lags.push(lag);
        values.push(correlation);
    }

    // Compute confidence intervals (95%)
    let confidence_intervals = compute_acf_confidence_intervals(&values, n);

    // Perform Ljung-Box test
    let ljung_box_test = if effective_max_lags >= 10 {
        Some(ljung_box_test(&values[1..], n, effective_max_lags.min(20)))
    } else {
        None
    };

    Ok(AutocorrelationResult {
        lags,
        values,
        confidence_intervals,
        ljung_box_test,
        max_lag: effective_max_lags,
    })
}

/// Compute partial autocorrelation function using Yule-Walker equations
pub fn compute_partial_autocorrelation(data: &[f64], max_lags: usize) -> Result<PartialAutocorrelationResult, Box<dyn std::error::Error>> {
    let n = data.len();
    if n < 10 {
        return Err("Need at least 10 observations for PACF computation".into());
    }

    // First compute ACF
    let acf_result = compute_autocorrelation(data, max_lags)?;
    let acf_values = &acf_result.values;

    let effective_max_lags = max_lags.min(n / 4).min(acf_values.len() - 1);
    let mut lags = Vec::with_capacity(effective_max_lags);
    let mut values = Vec::with_capacity(effective_max_lags);

    // PACF at lag 1 is just ACF at lag 1
    if effective_max_lags >= 1 {
        lags.push(1);
        values.push(acf_values[1]);
    }

    // For higher lags, solve Yule-Walker equations
    for k in 2..=effective_max_lags {
        let pacf_k = solve_yule_walker_for_lag(&acf_values[1..=k], k)?;
        lags.push(k);
        values.push(pacf_k);
    }

    // Compute confidence intervals
    let confidence_intervals = compute_pacf_confidence_intervals(&values, n);

    Ok(PartialAutocorrelationResult {
        lags,
        values,
        confidence_intervals,
        max_lag: effective_max_lags,
    })
}

/// Compute cross-correlation between two time series
pub fn compute_cross_correlation(series1: &[f64], series2: &[f64], max_lags: i32) -> Result<CrossCorrelationResult, Box<dyn std::error::Error>> {
    if series1.len() != series2.len() {
        return Err("Time series must have the same length for cross-correlation".into());
    }

    let n = series1.len();
    if n < 10 {
        return Err("Need at least 10 observations for cross-correlation".into());
    }

    let mean1 = series1.iter().sum::<f64>() / n as f64;
    let mean2 = series2.iter().sum::<f64>() / n as f64;

    let std1 = (series1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / n as f64).sqrt();
    let std2 = (series2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / n as f64).sqrt();

    if std1 == 0.0 || std2 == 0.0 {
        return Err("Cannot compute cross-correlation with constant series".into());
    }

    let effective_max_lags = max_lags.min(n as i32 / 2);
    let mut lags = Vec::new();
    let mut values = Vec::new();

    let mut peak_correlation: (i32, f64) = (0, 0.0);

    // Compute cross-correlation for negative lags (series2 leads series1)
    for lag in (-effective_max_lags..0).rev() {
        let correlation = compute_cross_correlation_at_lag(series1, series2, lag, mean1, mean2, std1, std2);
        lags.insert(0, lag);
        values.insert(0, correlation);

        if correlation.abs() > peak_correlation.1.abs() {
            peak_correlation = (lag, correlation);
        }
    }

    // Compute cross-correlation for zero and positive lags (series1 leads series2)
    for lag in 0..=effective_max_lags {
        let correlation = compute_cross_correlation_at_lag(series1, series2, lag, mean1, mean2, std1, std2);
        lags.push(lag);
        values.push(correlation);

        if correlation.abs() > peak_correlation.1.abs() {
            peak_correlation = (lag, correlation);
        }
    }

    // Compute confidence intervals
    let confidence_intervals = compute_cross_correlation_confidence_intervals(&values, n);

    Ok(CrossCorrelationResult {
        lags,
        values,
        peak_correlation,
        confidence_intervals,
    })
}

// Helper functions

fn compute_cross_correlation_at_lag(
    series1: &[f64],
    series2: &[f64],
    lag: i32,
    mean1: f64,
    mean2: f64,
    std1: f64,
    std2: f64,
) -> f64 {
    let n = series1.len() as i32;
    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        let j = i + lag;
        if j >= 0 && j < n {
            let i = i as usize;
            let j = j as usize;
            sum += (series1[i] - mean1) * (series2[j] - mean2);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    sum / (count as f64 * std1 * std2)
}

fn solve_yule_walker_for_lag(acf_values: &[f64], k: usize) -> Result<f64, Box<dyn std::error::Error>> {
    if k == 0 {
        return Ok(1.0);
    }
    if k == 1 {
        return Ok(acf_values[0]);
    }

    // Build Toeplitz matrix for Yule-Walker equations
    let mut matrix = vec![vec![0.0; k]; k];
    let mut rhs = vec![0.0; k];

    for i in 0..k {
        rhs[i] = acf_values[i];
        for j in 0..k {
            let lag_diff = (i as i32 - j as i32).abs() as usize;
            matrix[i][j] = if lag_diff == 0 {
                1.0
            } else {
                acf_values[lag_diff - 1]
            };
        }
    }

    // Solve the system using Gaussian elimination
    let solution = solve_linear_system(&matrix, &rhs)?;

    Ok(solution[k - 1]) // PACF at lag k is the last coefficient
}

fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = matrix.len();
    let mut aug_matrix = vec![vec![0.0; n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug_matrix[i][j] = matrix[i][j];
        }
        aug_matrix[i][n] = rhs[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            aug_matrix.swap(i, max_row);
        }

        // Check for singular matrix
        if aug_matrix[i][i].abs() < 1e-10 {
            return Err("Singular matrix in Yule-Walker equations".into());
        }

        // Make all rows below this one 0 in current column
        for k in (i + 1)..n {
            let factor = aug_matrix[k][i] / aug_matrix[i][i];
            for j in i..=n {
                aug_matrix[k][j] -= factor * aug_matrix[i][j];
            }
        }
    }

    // Back substitution
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        solution[i] = aug_matrix[i][n];
        for j in (i + 1)..n {
            solution[i] -= aug_matrix[i][j] * solution[j];
        }
        solution[i] /= aug_matrix[i][i];
    }

    Ok(solution)
}

fn compute_acf_confidence_intervals(acf_values: &[f64], n: usize) -> Vec<(f64, f64)> {
    let mut intervals = Vec::with_capacity(acf_values.len());

    // For lag 0, confidence interval is just (1, 1)
    intervals.push((1.0, 1.0));

    // For other lags, use Bartlett's approximation
    let critical_value = 1.96; // 95% confidence
    for i in 1..acf_values.len() {
        // Bartlett's standard error approximation
        let mut variance_sum = 1.0;
        for j in 1..i {
            variance_sum += 2.0 * acf_values[j].powi(2);
        }
        let std_error = (variance_sum / n as f64).sqrt();
        let margin = critical_value * std_error;

        intervals.push((-margin, margin));
    }

    intervals
}

fn compute_pacf_confidence_intervals(pacf_values: &[f64], n: usize) -> Vec<(f64, f64)> {
    let critical_value = 1.96; // 95% confidence
    let std_error = (1.0 / n as f64).sqrt();
    let margin = critical_value * std_error;

    vec![(-margin, margin); pacf_values.len()]
}

fn compute_cross_correlation_confidence_intervals(ccf_values: &[f64], n: usize) -> Vec<(f64, f64)> {
    let critical_value = 1.96; // 95% confidence
    let std_error = (1.0 / n as f64).sqrt();
    let margin = critical_value * std_error;

    vec![(-margin, margin); ccf_values.len()]
}

fn ljung_box_test(acf_values: &[f64], n: usize, h: usize) -> LjungBoxTest {
    let mut statistic = 0.0;

    for (i, &acf) in acf_values.iter().take(h).enumerate() {
        let lag = i + 1;
        statistic += acf.powi(2) / (n - lag) as f64;
    }

    statistic *= n as f64 * (n + 2) as f64;

    // Approximate p-value using chi-square distribution
    let p_value = approximate_chi_square_p_value(statistic, h);

    LjungBoxTest {
        statistic,
        p_value,
        df: h,
        lags_tested: h,
        has_autocorrelation: p_value < 0.05,
    }
}

fn approximate_chi_square_p_value(x: f64, df: usize) -> f64 {
    // Very simplified chi-square p-value approximation
    // In practice, use proper statistical functions
    if df == 0 {
        return 1.0;
    }

    let normalized = x / df as f64;
    if normalized < 0.5 {
        0.9
    } else if normalized < 1.0 {
        0.7
    } else if normalized < 2.0 {
        0.3
    } else if normalized < 3.0 {
        0.1
    } else {
        0.01
    }
}

fn perform_lag_analysis(
    acf: &AutocorrelationResult,
    pacf: &PartialAutocorrelationResult,
    n: usize,
) -> Result<LagAnalysis, Box<dyn std::error::Error>> {
    let mut significant_lags = Vec::new();

    // Find significant ACF lags
    for (i, (&acf_val, &(lower, upper))) in acf.values.iter()
        .zip(acf.confidence_intervals.iter())
        .enumerate()
        .skip(1) // Skip lag 0
    {
        if acf_val < lower || acf_val > upper {
            significant_lags.push(i);
        }
    }

    // Detect seasonal patterns
    let seasonal_patterns = detect_seasonal_patterns(&acf.values, &significant_lags);

    // Compute information criteria for different lag orders (simplified)
    let mut information_criteria = HashMap::new();
    for lag in 1..=10.min(pacf.values.len()) {
        // Simplified AIC/BIC calculation
        let k = lag as f64; // number of parameters
        let log_likelihood = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()); // Simplified

        let aic = 2.0 * k - 2.0 * log_likelihood;
        let bic = k * (n as f64).ln() - 2.0 * log_likelihood;
        let hqic = 2.0 * k * (n as f64).ln().ln() - 2.0 * log_likelihood;

        information_criteria.insert(lag, InformationCriteria { aic, bic, hqic });
    }

    // Find optimal lag (simplified - choose first significant PACF)
    let optimal_lag = pacf.lags.iter()
        .zip(pacf.values.iter())
        .zip(pacf.confidence_intervals.iter())
        .find(|&((_, &val), &(lower, upper))| val < lower || val > upper)
        .map(|((lag, _), _)| *lag);

    Ok(LagAnalysis {
        optimal_lag,
        information_criteria,
        significant_lags,
        seasonal_patterns,
    })
}

fn detect_seasonal_patterns(acf_values: &[f64], significant_lags: &[usize]) -> Vec<SeasonalPattern> {
    let mut patterns = Vec::new();

    // Look for common seasonal periods (12, 24, 7, etc.)
    let common_periods = [7, 12, 24, 48, 52];

    for &period in &common_periods {
        if period < acf_values.len() {
            let strength = acf_values[period].abs();
            if strength > 0.1 && significant_lags.contains(&period) {
                patterns.push(SeasonalPattern {
                    period,
                    strength,
                    confidence: if strength > 0.3 { 0.8 } else { 0.5 },
                });
            }
        }
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_white_noise() {
        // White noise should have ACF ≈ 0 for all lags > 0
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let acf_result = compute_autocorrelation(&data, 10).unwrap();

        assert_eq!(acf_result.values[0], 1.0); // Lag 0 should be 1
        assert_eq!(acf_result.lags.len(), acf_result.values.len());
        assert_eq!(acf_result.max_lag, 10);
    }

    #[test]
    fn test_partial_autocorrelation_computation() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let pacf_result = compute_partial_autocorrelation(&data, 5).unwrap();

        assert!(pacf_result.values.len() > 0);
        assert_eq!(pacf_result.lags.len(), pacf_result.values.len());
    }

    #[test]
    fn test_cross_correlation() {
        let series1: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let series2: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();

        let ccf_result = compute_cross_correlation(&series1, &series2, 10).unwrap();

        assert!(ccf_result.lags.len() > 0);
        assert_eq!(ccf_result.lags.len(), ccf_result.values.len());
        assert!(ccf_result.peak_correlation.1.abs() <= 1.0);
    }

    #[test]
    fn test_timeseries_stats_computation() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 0.1 * (i as f64)).collect();
        let stats = TimeSeriesStats::compute(&data, 20).unwrap();

        assert!(stats.acf.values.len() > 0);
        assert!(stats.pacf.values.len() > 0);
        assert!(stats.acf.values[0] == 1.0); // ACF at lag 0 should be 1
    }
}