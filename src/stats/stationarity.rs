//! Stationarity testing module for time series analysis
//!
//! Provides various statistical tests for stationarity including
//! Augmented Dickey-Fuller (ADF), KPSS, and Phillips-Perron tests.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Result of a stationarity test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTest {
    /// Name of the test
    pub test_name: String,

    /// Test statistic
    pub statistic: f64,

    /// P-value of the test
    pub p_value: f64,

    /// Critical values at different significance levels
    pub critical_values: HashMap<String, f64>,

    /// Is the series stationary according to this test?
    pub is_stationary: bool,

    /// Null hypothesis of the test
    pub null_hypothesis: String,

    /// Alternative hypothesis of the test
    pub alternative_hypothesis: String,

    /// Number of lags used in the test
    pub lags_used: Option<usize>,

    /// Additional test-specific information
    pub additional_info: HashMap<String, f64>,
}

/// Augmented Dickey-Fuller test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdfTest {
    /// Base test result
    pub base: StationarityTest,

    /// Type of ADF test (no constant, constant, constant+trend)
    pub test_type: AdfTestType,

    /// Regression coefficients
    pub coefficients: Vec<f64>,

    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,

    /// Information criteria for lag selection
    pub ic_values: HashMap<usize, f64>,
}

/// Type of ADF test regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdfTestType {
    /// No constant term: Δy_t = φy_{t-1} + ... + ε_t
    NoConstant,
    /// Constant term: Δy_t = α + φy_{t-1} + ... + ε_t
    Constant,
    /// Constant and trend: Δy_t = α + βt + φy_{t-1} + ... + ε_t
    ConstantTrend,
}

/// KPSS test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpssTest {
    /// Base test result
    pub base: StationarityTest,

    /// Type of KPSS test (level or trend stationarity)
    pub test_type: KpssTestType,

    /// Bandwidth parameter used
    pub bandwidth: usize,

    /// Long-run variance estimate
    pub long_run_variance: f64,
}

/// Type of KPSS test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KpssTestType {
    /// Test for level stationarity (constant mean)
    Level,
    /// Test for trend stationarity (constant trend)
    Trend,
}

/// Phillips-Perron test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhillipsPerronTest {
    /// Base test result
    pub base: StationarityTest,

    /// Type of PP test
    pub test_type: PpTestType,

    /// Bandwidth parameter
    pub bandwidth: usize,

    /// Long-run variance
    pub long_run_variance: f64,

    /// Short-run variance
    pub short_run_variance: f64,
}

/// Type of Phillips-Perron test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PpTestType {
    /// No constant
    NoConstant,
    /// With constant
    Constant,
    /// With constant and trend
    ConstantTrend,
}

/// Perform stationarity test on time series data
///
/// # Arguments
/// * `data` - Time series data
/// * `test_type` - Type of test ("adf", "kpss", "pp")
///
/// # Returns
/// * `Result<StationarityTest, Box<dyn std::error::Error>>` - Test result
pub fn test_stationarity(
    data: &[f64],
    test_type: &str,
) -> Result<StationarityTest, Box<dyn std::error::Error>> {
    match test_type.to_lowercase().as_str() {
        "adf" => adf_test(data, None),
        "kpss" => kpss_test(data, KpssTestType::Level),
        "pp" => phillips_perron_test(data, PpTestType::Constant),
        _ => Err(format!("Unknown test type: {}", test_type).into()),
    }
}

/// Augmented Dickey-Fuller test for unit root
pub fn adf_test(
    data: &[f64],
    max_lags: Option<usize>,
) -> Result<StationarityTest, Box<dyn std::error::Error>> {
    if data.len() < 20 {
        return Err("ADF test requires at least 20 observations".into());
    }

    let n = data.len();

    // Determine optimal number of lags using AIC
    let optimal_lags = if let Some(lags) = max_lags {
        lags
    } else {
        select_adf_lags(data)?
    };

    // Run ADF regression: Δy_t = α + φy_{t-1} + Σγ_i Δy_{t-i} + ε_t
    let adf_result = run_adf_regression(data, optimal_lags)?;

    // Get critical values
    let mut critical_values = HashMap::new();
    critical_values.insert("1%".to_string(), get_adf_critical_value(n, 0.01));
    critical_values.insert("5%".to_string(), get_adf_critical_value(n, 0.05));
    critical_values.insert("10%".to_string(), get_adf_critical_value(n, 0.10));

    // Calculate approximate p-value
    let p_value = approximate_adf_p_value(adf_result.t_statistic, n);

    // Test decision (reject H0 if statistic < critical value)
    let is_stationary = adf_result.t_statistic < critical_values["5%"];

    let mut additional_info = HashMap::new();
    additional_info.insert("t_statistic".to_string(), adf_result.t_statistic);
    additional_info.insert("n_observations".to_string(), n as f64);

    Ok(StationarityTest {
        test_name: "Augmented Dickey-Fuller".to_string(),
        statistic: adf_result.t_statistic,
        p_value,
        critical_values,
        is_stationary,
        null_hypothesis: "Series has a unit root (non-stationary)".to_string(),
        alternative_hypothesis: "Series is stationary".to_string(),
        lags_used: Some(optimal_lags),
        additional_info,
    })
}

/// KPSS test for stationarity
pub fn kpss_test(
    data: &[f64],
    test_type: KpssTestType,
) -> Result<StationarityTest, Box<dyn std::error::Error>> {
    if data.len() < 20 {
        return Err("KPSS test requires at least 20 observations".into());
    }

    let n = data.len();

    // Compute residuals from regression
    let residuals = match test_type {
        KpssTestType::Level => {
            // Regression on constant only
            let mean = data.iter().sum::<f64>() / n as f64;
            data.iter().map(|&x| x - mean).collect::<Vec<f64>>()
        }
        KpssTestType::Trend => {
            // Regression on constant and trend
            compute_trend_residuals(data)?
        }
    };

    // Compute partial sums
    let mut partial_sums = vec![0.0; n];
    let mut cumsum = 0.0;
    for (i, &residual) in residuals.iter().enumerate() {
        cumsum += residual;
        partial_sums[i] = cumsum;
    }

    // Select bandwidth using Newey-West method
    let bandwidth = select_bandwidth_newey_west(n);

    // Compute long-run variance
    let long_run_variance = compute_long_run_variance(&residuals, bandwidth);

    // Compute KPSS statistic
    let sum_squares = partial_sums.iter().map(|&x| x.powi(2)).sum::<f64>();
    let kpss_statistic = sum_squares / (n.pow(2) as f64 * long_run_variance);

    // Get critical values
    let mut critical_values = HashMap::new();
    match test_type {
        KpssTestType::Level => {
            critical_values.insert("1%".to_string(), 0.739);
            critical_values.insert("5%".to_string(), 0.463);
            critical_values.insert("10%".to_string(), 0.347);
        }
        KpssTestType::Trend => {
            critical_values.insert("1%".to_string(), 0.216);
            critical_values.insert("5%".to_string(), 0.146);
            critical_values.insert("10%".to_string(), 0.119);
        }
    }

    // Approximate p-value
    let p_value = approximate_kpss_p_value(kpss_statistic, &test_type);

    // Test decision (reject H0 if statistic > critical value)
    let is_stationary = kpss_statistic <= critical_values["5%"];

    let mut additional_info = HashMap::new();
    additional_info.insert("bandwidth".to_string(), bandwidth as f64);
    additional_info.insert("long_run_variance".to_string(), long_run_variance);

    Ok(StationarityTest {
        test_name: "KPSS".to_string(),
        statistic: kpss_statistic,
        p_value,
        critical_values,
        is_stationary,
        null_hypothesis: "Series is stationary".to_string(),
        alternative_hypothesis: "Series has a unit root (non-stationary)".to_string(),
        lags_used: Some(bandwidth),
        additional_info,
    })
}

/// Phillips-Perron test for unit root
pub fn phillips_perron_test(
    data: &[f64],
    test_type: PpTestType,
) -> Result<StationarityTest, Box<dyn std::error::Error>> {
    if data.len() < 20 {
        return Err("Phillips-Perron test requires at least 20 observations".into());
    }

    let n = data.len();

    // Run simple regression (no lags): Δy_t = α + φy_{t-1} + ε_t
    let regression_result = run_pp_regression(data, &test_type)?;

    // Select bandwidth
    let bandwidth = select_bandwidth_newey_west(n);

    // Compute variance estimates
    let long_run_variance = compute_long_run_variance(&regression_result.residuals, bandwidth);
    let short_run_variance = regression_result.residuals.iter()
        .map(|&x| x.powi(2))
        .sum::<f64>() / (n - regression_result.n_params) as f64;

    // Compute PP statistic (Phillips-Perron correction)
    let correction_factor = (long_run_variance / short_run_variance).sqrt();
    let pp_statistic = regression_result.t_statistic * correction_factor -
        0.5 * (long_run_variance - short_run_variance) / short_run_variance *
        (n as f64 * regression_result.phi_se).sqrt();

    // Get critical values (same as ADF)
    let mut critical_values = HashMap::new();
    critical_values.insert("1%".to_string(), get_adf_critical_value(n, 0.01));
    critical_values.insert("5%".to_string(), get_adf_critical_value(n, 0.05));
    critical_values.insert("10%".to_string(), get_adf_critical_value(n, 0.10));

    // Approximate p-value
    let p_value = approximate_adf_p_value(pp_statistic, n);

    // Test decision
    let is_stationary = pp_statistic < critical_values["5%"];

    let mut additional_info = HashMap::new();
    additional_info.insert("bandwidth".to_string(), bandwidth as f64);
    additional_info.insert("long_run_variance".to_string(), long_run_variance);
    additional_info.insert("short_run_variance".to_string(), short_run_variance);

    Ok(StationarityTest {
        test_name: "Phillips-Perron".to_string(),
        statistic: pp_statistic,
        p_value,
        critical_values,
        is_stationary,
        null_hypothesis: "Series has a unit root (non-stationary)".to_string(),
        alternative_hypothesis: "Series is stationary".to_string(),
        lags_used: Some(bandwidth),
        additional_info,
    })
}

// Helper structures and functions

struct AdfRegressionResult {
    t_statistic: f64,
    coefficients: Vec<f64>,
    std_errors: Vec<f64>,
    residuals: Vec<f64>,
}

struct PpRegressionResult {
    t_statistic: f64,
    phi_coefficient: f64,
    phi_se: f64,
    residuals: Vec<f64>,
    n_params: usize,
}

fn select_adf_lags(data: &[f64]) -> Result<usize, Box<dyn std::error::Error>> {
    let max_lags = ((data.len() as f64).powf(1.0 / 3.0) * 12.0 / 100.0).floor() as usize;
    let max_lags = max_lags.max(1).min(data.len() / 4);

    let mut best_lags = 1;
    let mut best_aic = f64::INFINITY;

    for lags in 1..=max_lags {
        if let Ok(result) = run_adf_regression(data, lags) {
            let n_effective = data.len() - lags - 1;
            let k = lags + 2; // Number of parameters
            let rss = result.residuals.iter().map(|&x| x.powi(2)).sum::<f64>();
            let aic = n_effective as f64 * (rss / n_effective as f64).ln() + 2.0 * k as f64;

            if aic < best_aic {
                best_aic = aic;
                best_lags = lags;
            }
        }
    }

    Ok(best_lags)
}

fn run_adf_regression(data: &[f64], lags: usize) -> Result<AdfRegressionResult, Box<dyn std::error::Error>> {
    let n = data.len();
    if n < lags + 10 {
        return Err("Insufficient data for ADF regression".into());
    }

    // Prepare data: Δy_t = α + φy_{t-1} + Σγ_i Δy_{t-i} + ε_t
    let n_obs = n - lags - 1;
    let n_params = lags + 2; // constant + φ + γ_1...γ_lags

    let mut y = Vec::with_capacity(n_obs);  // Δy_t
    let mut x = vec![vec![0.0; n_params]; n_obs];  // Design matrix

    for t in (lags + 1)..n {
        // Dependent variable: Δy_t = y_t - y_{t-1}
        y.push(data[t] - data[t - 1]);

        let row_idx = t - lags - 1;

        // Constant term
        x[row_idx][0] = 1.0;

        // Lagged level: y_{t-1}
        x[row_idx][1] = data[t - 1];

        // Lagged differences: Δy_{t-i} for i = 1..lags
        for i in 1..=lags {
            if t > i {
                x[row_idx][i + 1] = data[t - i] - data[t - i - 1];
            }
        }
    }

    // Solve least squares: β = (X'X)^(-1)X'y
    let coefficients = solve_ols(&x, &y)?;

    // Calculate residuals and standard errors
    let mut residuals = Vec::with_capacity(n_obs);
    for (i, &y_val) in y.iter().enumerate() {
        let y_hat = x[i].iter().zip(coefficients.iter())
            .map(|(&x_val, &coef)| x_val * coef)
            .sum::<f64>();
        residuals.push(y_val - y_hat);
    }

    let rss = residuals.iter().map(|&x| x.powi(2)).sum::<f64>();
    let mse = rss / (n_obs - n_params) as f64;

    // Compute (X'X)^(-1)
    let xtx_inv = compute_matrix_inverse(&compute_xtx(&x))?;
    let std_errors: Vec<f64> = xtx_inv.iter().map(|row| (mse * row[1]).sqrt()).collect(); // We want SE of φ

    // t-statistic for φ (coefficient of y_{t-1})
    let t_statistic = coefficients[1] / std_errors[1];

    Ok(AdfRegressionResult {
        t_statistic,
        coefficients,
        std_errors,
        residuals,
    })
}

fn run_pp_regression(data: &[f64], test_type: &PpTestType) -> Result<PpRegressionResult, Box<dyn std::error::Error>> {
    let n = data.len();
    let n_obs = n - 1;

    let mut y = Vec::with_capacity(n_obs);  // Δy_t
    let n_params = match test_type {
        PpTestType::NoConstant => 1,
        PpTestType::Constant => 2,
        PpTestType::ConstantTrend => 3,
    };

    let mut x = vec![vec![0.0; n_params]; n_obs];

    for t in 1..n {
        y.push(data[t] - data[t - 1]);
        let row_idx = t - 1;

        let mut col = 0;

        // Constant term
        if matches!(test_type, PpTestType::Constant | PpTestType::ConstantTrend) {
            x[row_idx][col] = 1.0;
            col += 1;
        }

        // Trend term
        if matches!(test_type, PpTestType::ConstantTrend) {
            x[row_idx][col] = t as f64;
            col += 1;
        }

        // Lagged level: y_{t-1}
        x[row_idx][col] = data[t - 1];
    }

    let coefficients = solve_ols(&x, &y)?;

    // Calculate residuals
    let mut residuals = Vec::with_capacity(n_obs);
    for (i, &y_val) in y.iter().enumerate() {
        let y_hat = x[i].iter().zip(coefficients.iter())
            .map(|(&x_val, &coef)| x_val * coef)
            .sum::<f64>();
        residuals.push(y_val - y_hat);
    }

    let rss = residuals.iter().map(|&x| x.powi(2)).sum::<f64>();
    let mse = rss / (n_obs - n_params) as f64;

    // Standard error of φ (last coefficient)
    let xtx_inv = compute_matrix_inverse(&compute_xtx(&x))?;
    let phi_se = (mse * xtx_inv[n_params - 1][n_params - 1]).sqrt();

    let phi_coefficient = coefficients[n_params - 1];
    let t_statistic = phi_coefficient / phi_se;

    Ok(PpRegressionResult {
        t_statistic,
        phi_coefficient,
        phi_se,
        residuals,
        n_params,
    })
}

fn compute_trend_residuals(data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = data.len();

    // Regression: y_t = α + βt + ε_t
    let mut x = vec![vec![0.0; 2]; n];
    for i in 0..n {
        x[i][0] = 1.0; // constant
        x[i][1] = (i + 1) as f64; // trend
    }

    let coefficients = solve_ols(&x, data)?;

    let mut residuals = Vec::with_capacity(n);
    for (i, &y_val) in data.iter().enumerate() {
        let y_hat = coefficients[0] + coefficients[1] * (i + 1) as f64;
        residuals.push(y_val - y_hat);
    }

    Ok(residuals)
}

fn select_bandwidth_newey_west(n: usize) -> usize {
    // Newey-West bandwidth selection: floor(4(T/100)^(2/9))
    let bandwidth = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
    bandwidth.max(1)
}

fn compute_long_run_variance(residuals: &[f64], bandwidth: usize) -> f64 {
    let n = residuals.len();
    let mean = residuals.iter().sum::<f64>() / n as f64;

    let mut variance = 0.0;

    // γ₀ (variance)
    let gamma_0 = residuals.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n as f64;

    variance += gamma_0;

    // Weighted autocovariances
    for j in 1..=bandwidth {
        let mut gamma_j = 0.0;
        let valid_pairs = n - j;

        for i in 0..valid_pairs {
            gamma_j += (residuals[i] - mean) * (residuals[i + j] - mean);
        }

        gamma_j /= n as f64;

        // Bartlett kernel
        let weight = 1.0 - j as f64 / (bandwidth + 1) as f64;
        variance += 2.0 * weight * gamma_j;
    }

    variance.max(1e-8) // Ensure positive variance
}

fn solve_ols(x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n_obs = x.len();
    let n_params = x[0].len();

    if n_obs != y.len() {
        return Err("Dimension mismatch in OLS".into());
    }

    // Compute X'X
    let xtx = compute_xtx(x);

    // Compute X'y
    let mut xty = vec![0.0; n_params];
    for j in 0..n_params {
        for i in 0..n_obs {
            xty[j] += x[i][j] * y[i];
        }
    }

    // Solve (X'X)β = X'y
    let xtx_inv = compute_matrix_inverse(&xtx)?;
    let mut coefficients = vec![0.0; n_params];

    for i in 0..n_params {
        for j in 0..n_params {
            coefficients[i] += xtx_inv[i][j] * xty[j];
        }
    }

    Ok(coefficients)
}

fn compute_xtx(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_params = x[0].len();
    let mut xtx = vec![vec![0.0; n_params]; n_params];

    for i in 0..n_params {
        for j in 0..n_params {
            for row in x {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    xtx
}

fn compute_matrix_inverse(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let n = matrix.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Create augmented matrix [A|I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
            aug[i][j + n] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            aug.swap(i, max_row);
        }

        // Check for singularity
        if aug[i][i].abs() < 1e-10 {
            return Err("Matrix is singular".into());
        }

        // Scale pivot row
        let pivot = aug[i][i];
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..(2 * n) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][j + n];
        }
    }

    Ok(inverse)
}

fn get_adf_critical_value(n: usize, alpha: f64) -> f64 {
    // Approximation of ADF critical values (MacKinnon, 1996)
    // For constant term case
    let c_alpha = if (alpha - 0.01).abs() < f64::EPSILON {
        (-3.43, -6.5, -16.786)
    } else if (alpha - 0.05).abs() < f64::EPSILON {
        (-2.86, -2.885, -4.234)
    } else if (alpha - 0.10).abs() < f64::EPSILON {
        (-2.57, -1.95, -2.809)
    } else {
        (-2.86, -2.885, -4.234) // Default to 5%
    };

    let inv_n = 1.0 / n as f64;
    let inv_n2 = inv_n * inv_n;

    c_alpha.0 + c_alpha.1 * inv_n + c_alpha.2 * inv_n2
}

fn approximate_adf_p_value(statistic: f64, n: usize) -> f64 {
    // Very simplified p-value approximation
    // In practice, use MacKinnon (1996) response surface or lookup tables

    let critical_1pct = get_adf_critical_value(n, 0.01);
    let critical_5pct = get_adf_critical_value(n, 0.05);
    let critical_10pct = get_adf_critical_value(n, 0.10);

    if statistic <= critical_1pct {
        0.01
    } else if statistic <= critical_5pct {
        0.025
    } else if statistic <= critical_10pct {
        0.075
    } else {
        0.15
    }
}

fn approximate_kpss_p_value(statistic: f64, test_type: &KpssTestType) -> f64 {
    // Simplified KPSS p-value approximation
    let (cv_1, cv_5, cv_10) = match test_type {
        KpssTestType::Level => (0.739, 0.463, 0.347),
        KpssTestType::Trend => (0.216, 0.146, 0.119),
    };

    if statistic >= cv_1 {
        0.01
    } else if statistic >= cv_5 {
        0.025
    } else if statistic >= cv_10 {
        0.075
    } else {
        0.15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_test_random_walk() {
        // Generate a random walk (non-stationary)
        let mut data = vec![0.0];
        for i in 1..100 {
            data.push(data[i - 1] + 0.1 * (i as f64).sin());
        }

        let result = adf_test(&data, None).unwrap();
        assert_eq!(result.test_name, "Augmented Dickey-Fuller");
        assert!(result.critical_values.contains_key("5%"));
    }

    #[test]
    fn test_kpss_test_stationary() {
        // Generate stationary data
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();

        let result = kpss_test(&data, KpssTestType::Level).unwrap();
        assert_eq!(result.test_name, "KPSS");
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_phillips_perron_test() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.1).sin()).collect();

        let result = phillips_perron_test(&data, PpTestType::Constant).unwrap();
        assert_eq!(result.test_name, "Phillips-Perron");
        assert!(result.additional_info.contains_key("bandwidth"));
    }

    #[test]
    fn test_stationarity_test_dispatcher() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).cos()).collect();

        let adf_result = test_stationarity(&data, "adf").unwrap();
        assert_eq!(adf_result.test_name, "Augmented Dickey-Fuller");

        let kpss_result = test_stationarity(&data, "kpss").unwrap();
        assert_eq!(kpss_result.test_name, "KPSS");

        let pp_result = test_stationarity(&data, "pp").unwrap();
        assert_eq!(pp_result.test_name, "Phillips-Perron");
    }
}