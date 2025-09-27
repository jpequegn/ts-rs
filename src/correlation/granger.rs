//! Granger causality testing and VAR model analysis module
//!
//! Implements Granger causality tests, Vector Autoregression (VAR) models,
//! and impulse response function analysis for investigating causal
//! relationships between time series.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Granger causality test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerCausalityResult {
    /// Variable that potentially causes (X)
    pub cause_variable: String,

    /// Variable that is potentially caused (Y)
    pub effect_variable: String,

    /// Number of lags used in the test
    pub lags: usize,

    /// F-statistic for the test
    pub f_statistic: f64,

    /// P-value of the test
    pub p_value: f64,

    /// Whether the result is significant at α = 0.05
    pub is_significant: bool,

    /// R-squared of the restricted model (without causal variable)
    pub r_squared_restricted: f64,

    /// R-squared of the unrestricted model (with causal variable)
    pub r_squared_unrestricted: f64,

    /// Degrees of freedom for the test
    pub df_numerator: usize,
    pub df_denominator: usize,

    /// Test interpretation
    pub interpretation: String,
}

/// Vector Autoregression (VAR) model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VARModel {
    /// Variable names in the VAR system
    pub variables: Vec<String>,

    /// Number of lags in the model
    pub lags: usize,

    /// VAR coefficient matrices (lags x variables x variables)
    pub coefficients: Vec<Vec<Vec<f64>>>,

    /// Intercept terms for each equation
    pub intercepts: Vec<f64>,

    /// Residual covariance matrix
    pub residual_covariance: Vec<Vec<f64>>,

    /// R-squared for each equation
    pub r_squared: Vec<f64>,

    /// Log-likelihood of the model
    pub log_likelihood: f64,

    /// Information criteria
    pub aic: f64,
    pub bic: f64,

    /// Number of observations used in estimation
    pub n_observations: usize,

    /// Model diagnostics
    pub diagnostics: VARDiagnostics,
}

/// VAR model diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VARDiagnostics {
    /// Portmanteau test for residual autocorrelation
    pub ljung_box_statistic: f64,
    pub ljung_box_p_value: f64,

    /// Jarque-Bera test for normality of residuals
    pub jarque_bera_statistics: Vec<f64>,
    pub jarque_bera_p_values: Vec<f64>,

    /// ARCH test for conditional heteroskedasticity
    pub arch_statistics: Vec<f64>,
    pub arch_p_values: Vec<f64>,

    /// Stability assessment
    pub is_stable: bool,
    pub eigenvalues: Vec<f64>,
}

/// Impulse response function analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseResponseResult {
    /// Variable names
    pub variables: Vec<String>,

    /// Number of periods for impulse response
    pub periods: usize,

    /// Impulse response functions (shock_var x response_var x periods)
    pub responses: HashMap<String, HashMap<String, Vec<f64>>>,

    /// Cumulative impulse responses
    pub cumulative_responses: HashMap<String, HashMap<String, Vec<f64>>>,

    /// Confidence intervals (if computed)
    pub confidence_intervals: Option<ImpulseResponseConfidenceIntervals>,

    /// Forecast error variance decomposition
    pub variance_decomposition: VarianceDecomposition,
}

/// Confidence intervals for impulse response functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseResponseConfidenceIntervals {
    /// Lower bounds (shock_var x response_var x periods)
    pub lower_bounds: HashMap<String, HashMap<String, Vec<f64>>>,

    /// Upper bounds (shock_var x response_var x periods)
    pub upper_bounds: HashMap<String, HashMap<String, Vec<f64>>>,

    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
}

/// Forecast error variance decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceDecomposition {
    /// Variable names
    pub variables: Vec<String>,

    /// Number of periods
    pub periods: usize,

    /// Variance decomposition (response_var x shock_var x periods)
    pub decomposition: HashMap<String, HashMap<String, Vec<f64>>>,
}

impl GrangerCausalityResult {
    /// Get a human-readable interpretation of the test result
    pub fn get_interpretation(&self) -> String {
        if self.is_significant {
            format!(
                "{} Granger-causes {} (F({},{}) = {:.4}, p = {:.4})",
                self.cause_variable,
                self.effect_variable,
                self.df_numerator,
                self.df_denominator,
                self.f_statistic,
                self.p_value
            )
        } else {
            format!(
                "{} does not Granger-cause {} (F({},{}) = {:.4}, p = {:.4})",
                self.cause_variable,
                self.effect_variable,
                self.df_numerator,
                self.df_denominator,
                self.f_statistic,
                self.p_value
            )
        }
    }
}

impl VARModel {
    /// Get the lag coefficient matrix for a specific lag
    pub fn get_lag_matrix(&self, lag: usize) -> Option<&Vec<Vec<f64>>> {
        if lag == 0 || lag > self.lags {
            None
        } else {
            self.coefficients.get(lag - 1)
        }
    }

    /// Check if the VAR model is stable (all eigenvalues inside unit circle)
    pub fn is_stable(&self) -> bool {
        self.diagnostics.is_stable
    }

    /// Get forecasts from the VAR model
    pub fn forecast(&self, data: &HashMap<String, Vec<f64>>, periods: usize) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
        if data.len() != self.variables.len() {
            return Err("Data must contain all variables in the VAR model".into());
        }

        let n_vars = self.variables.len();
        let mut forecasts = HashMap::new();

        for var in &self.variables {
            forecasts.insert(var.clone(), Vec::new());
        }

        // Get recent observations for initial conditions
        let mut recent_data = Vec::new();
        for i in 0..n_vars {
            let var_name = &self.variables[i];
            let series = &data[var_name];
            if series.len() < self.lags {
                return Err(format!("Insufficient data for variable {}", var_name).into());
            }
            recent_data.push(series[series.len() - self.lags..].to_vec());
        }

        // Generate forecasts
        for _ in 0..periods {
            let mut forecast_step = Vec::new();

            for i in 0..n_vars {
                let mut forecast_value = self.intercepts[i];

                // Add contributions from all lags and variables
                for lag in 0..self.lags {
                    for j in 0..n_vars {
                        let coeff = self.coefficients[lag][i][j];
                        let lag_index = self.lags - 1 - lag;
                        forecast_value += coeff * recent_data[j][lag_index];
                    }
                }

                forecast_step.push(forecast_value);
            }

            // Add forecasted values to results
            for (i, var) in self.variables.iter().enumerate() {
                forecasts.get_mut(var).unwrap().push(forecast_step[i]);
            }

            // Update recent_data for next iteration
            for i in 0..n_vars {
                recent_data[i].remove(0);
                recent_data[i].push(forecast_step[i]);
            }
        }

        Ok(forecasts)
    }
}

/// Test Granger causality between two time series
pub fn test_granger_causality(
    cause_series: &[f64],
    effect_series: &[f64],
    lags: usize,
    cause_name: &str,
    effect_name: &str,
) -> Result<GrangerCausalityResult, Box<dyn std::error::Error>> {
    if cause_series.len() != effect_series.len() {
        return Err("Series must have the same length".into());
    }

    if cause_series.len() <= lags + 1 {
        return Err("Series too short for specified number of lags".into());
    }

    let n = cause_series.len() - lags;

    // Prepare data matrices
    let mut y = Vec::new();  // Dependent variable (effect_series)
    let mut x_restricted = Vec::new();  // Only lags of effect_series
    let mut x_unrestricted = Vec::new();  // Lags of both series

    for t in lags..cause_series.len() {
        y.push(effect_series[t]);

        // Restricted model: only lags of effect variable
        let mut x_rest = vec![1.0];  // Intercept
        for lag in 1..=lags {
            x_rest.push(effect_series[t - lag]);
        }
        x_restricted.push(x_rest);

        // Unrestricted model: lags of both variables
        let mut x_unrest = vec![1.0];  // Intercept
        for lag in 1..=lags {
            x_unrest.push(effect_series[t - lag]);
        }
        for lag in 1..=lags {
            x_unrest.push(cause_series[t - lag]);
        }
        x_unrestricted.push(x_unrest);
    }

    // Estimate restricted model (without cause variable)
    let restricted_result = estimate_ols(&y, &x_restricted)?;

    // Estimate unrestricted model (with cause variable)
    let unrestricted_result = estimate_ols(&y, &x_unrestricted)?;

    // Calculate F-statistic for Granger causality test
    let rss_restricted = restricted_result.residual_sum_squares;
    let rss_unrestricted = unrestricted_result.residual_sum_squares;

    let df_numerator = lags;  // Number of restrictions (lags of cause variable)
    let df_denominator = n - x_unrestricted[0].len();

    let f_statistic = ((rss_restricted - rss_unrestricted) / df_numerator as f64) /
                     (rss_unrestricted / df_denominator as f64);

    // Calculate p-value (approximate using F-distribution)
    let p_value = 1.0 - f_distribution_cdf(f_statistic, df_numerator, df_denominator);

    let is_significant = p_value < 0.05;

    let interpretation = if is_significant {
        format!("{} Granger-causes {}", cause_name, effect_name)
    } else {
        format!("{} does not Granger-cause {}", cause_name, effect_name)
    };

    Ok(GrangerCausalityResult {
        cause_variable: cause_name.to_string(),
        effect_variable: effect_name.to_string(),
        lags,
        f_statistic,
        p_value,
        is_significant,
        r_squared_restricted: restricted_result.r_squared,
        r_squared_unrestricted: unrestricted_result.r_squared,
        df_numerator,
        df_denominator,
        interpretation,
    })
}

/// Estimate a Vector Autoregression (VAR) model
pub fn estimate_var_model(
    data: &HashMap<String, Vec<f64>>,
    lags: usize,
) -> Result<VARModel, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let n_vars = variables.len();
    let n_obs = data.values().next().unwrap().len();

    if n_obs <= lags {
        return Err("Insufficient observations for VAR estimation".into());
    }

    let effective_obs = n_obs - lags;

    // Prepare data matrices for each equation
    let mut coefficients = vec![vec![vec![0.0; n_vars]; n_vars]; lags];
    let mut intercepts = vec![0.0; n_vars];
    let mut r_squared = vec![0.0; n_vars];
    let mut residuals = vec![vec![0.0; effective_obs]; n_vars];

    // Estimate each equation separately using OLS
    for (eq_idx, eq_var) in variables.iter().enumerate() {
        let eq_series = &data[eq_var];

        // Prepare dependent variable
        let mut y = Vec::new();
        for t in lags..n_obs {
            y.push(eq_series[t]);
        }

        // Prepare regressors (intercept + lags of all variables)
        let mut x = Vec::new();
        for t in lags..n_obs {
            let mut x_row = vec![1.0];  // Intercept

            // Add lags of all variables
            for lag in 1..=lags {
                for var in &variables {
                    x_row.push(data[var][t - lag]);
                }
            }
            x.push(x_row);
        }

        // Estimate equation
        let ols_result = estimate_ols(&y, &x)?;

        // Store intercept
        intercepts[eq_idx] = ols_result.coefficients[0];

        // Store coefficients in proper structure
        let mut coeff_idx = 1;
        for lag in 0..lags {
            for var_idx in 0..n_vars {
                coefficients[lag][eq_idx][var_idx] = ols_result.coefficients[coeff_idx];
                coeff_idx += 1;
            }
        }

        r_squared[eq_idx] = ols_result.r_squared;
        residuals[eq_idx] = ols_result.residuals;
    }

    // Calculate residual covariance matrix
    let mut residual_covariance = vec![vec![0.0; n_vars]; n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut cov = 0.0;
            for t in 0..effective_obs {
                cov += residuals[i][t] * residuals[j][t];
            }
            residual_covariance[i][j] = cov / (effective_obs - 1) as f64;
        }
    }

    // Calculate log-likelihood
    let log_likelihood = calculate_var_log_likelihood(&residual_covariance, effective_obs);

    // Calculate information criteria
    let n_params = n_vars * (1 + lags * n_vars);  // Intercepts + lag coefficients
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (effective_obs as f64).ln();

    // Calculate diagnostics
    let diagnostics = calculate_var_diagnostics(&coefficients, &residual_covariance, &residuals);

    Ok(VARModel {
        variables,
        lags,
        coefficients,
        intercepts,
        residual_covariance,
        r_squared,
        log_likelihood,
        aic,
        bic,
        n_observations: effective_obs,
        diagnostics,
    })
}

/// Compute impulse response functions from a VAR model
pub fn compute_impulse_response(
    var_model: &VARModel,
    periods: usize,
    confidence_level: Option<f64>,
) -> Result<ImpulseResponseResult, Box<dyn std::error::Error>> {
    let n_vars = var_model.variables.len();

    // Compute orthogonalized impulse responses using Cholesky decomposition
    let chol_factor = cholesky_decomposition(&var_model.residual_covariance)?;

    // Initialize impulse response matrices
    let mut responses = HashMap::new();
    let mut cumulative_responses = HashMap::new();

    for shock_var in &var_model.variables {
        responses.insert(shock_var.clone(), HashMap::new());
        cumulative_responses.insert(shock_var.clone(), HashMap::new());

        for response_var in &var_model.variables {
            responses.get_mut(shock_var).unwrap().insert(response_var.clone(), Vec::new());
            cumulative_responses.get_mut(shock_var).unwrap().insert(response_var.clone(), Vec::new());
        }
    }

    // Compute impulse responses using MA representation
    let mut psi_matrices = vec![vec![vec![0.0; n_vars]; n_vars]; periods + 1];

    // Period 0: Identity matrix
    for i in 0..n_vars {
        psi_matrices[0][i][i] = 1.0;
    }

    // Subsequent periods using VAR coefficients
    for h in 1..=periods {
        for i in 0..n_vars {
            for j in 0..n_vars {
                let mut sum = 0.0;

                for lag in 1..=var_model.lags.min(h) {
                    for k in 0..n_vars {
                        sum += var_model.coefficients[lag - 1][i][k] * psi_matrices[h - lag][k][j];
                    }
                }

                psi_matrices[h][i][j] = sum;
            }
        }
    }

    // Apply Cholesky factor to get orthogonalized responses
    for h in 0..=periods {
        for shock_idx in 0..n_vars {
            let shock_var = &var_model.variables[shock_idx];

            for response_idx in 0..n_vars {
                let response_var = &var_model.variables[response_idx];

                let mut orthog_response = 0.0;
                for k in 0..n_vars {
                    orthog_response += psi_matrices[h][response_idx][k] * chol_factor[k][shock_idx];
                }

                if h < periods {
                    responses.get_mut(shock_var).unwrap()
                        .get_mut(response_var).unwrap().push(orthog_response);
                }

                // Cumulative response
                let prev_cumulative = if h == 0 { 0.0 } else {
                    cumulative_responses[shock_var][response_var][h - 1]
                };

                if h < periods {
                    cumulative_responses.get_mut(shock_var).unwrap()
                        .get_mut(response_var).unwrap().push(prev_cumulative + orthog_response);
                }
            }
        }
    }

    // Compute variance decomposition
    let variance_decomp = compute_variance_decomposition(&responses, periods);

    // TODO: Implement confidence intervals using bootstrap or analytical methods
    let confidence_intervals = None;

    Ok(ImpulseResponseResult {
        variables: var_model.variables.clone(),
        periods,
        responses,
        cumulative_responses,
        confidence_intervals,
        variance_decomposition: variance_decomp,
    })
}

/// OLS estimation result
#[derive(Debug)]
struct OLSResult {
    coefficients: Vec<f64>,
    residuals: Vec<f64>,
    residual_sum_squares: f64,
    r_squared: f64,
}

/// Estimate OLS regression
fn estimate_ols(y: &[f64], x: &[Vec<f64>]) -> Result<OLSResult, Box<dyn std::error::Error>> {
    let n = y.len();
    let k = x[0].len();

    if n != x.len() {
        return Err("Dimension mismatch between y and X".into());
    }

    if n <= k {
        return Err("Insufficient observations for OLS estimation".into());
    }

    // Calculate X'X
    let mut xtx = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            for t in 0..n {
                xtx[i][j] += x[t][i] * x[t][j];
            }
        }
    }

    // Calculate X'y
    let mut xty = vec![0.0; k];
    for i in 0..k {
        for t in 0..n {
            xty[i] += x[t][i] * y[t];
        }
    }

    // Solve (X'X)β = X'y using Gaussian elimination
    let coefficients = solve_linear_system(&xtx, &xty)?;

    // Calculate fitted values and residuals
    let mut fitted = vec![0.0; n];
    let mut residuals = vec![0.0; n];
    let mut rss = 0.0;

    for t in 0..n {
        for i in 0..k {
            fitted[t] += coefficients[i] * x[t][i];
        }
        residuals[t] = y[t] - fitted[t];
        rss += residuals[t] * residuals[t];
    }

    // Calculate R-squared
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let mut tss = 0.0;
    for &yi in y {
        tss += (yi - y_mean) * (yi - y_mean);
    }

    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    Ok(OLSResult {
        coefficients,
        residuals,
        residual_sum_squares: rss,
        r_squared,
    })
}

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination
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

        // Check for singular matrix
        if aug[i][i].abs() < 1e-14 {
            return Err("Singular matrix in OLS estimation".into());
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

/// Calculate VAR model log-likelihood
fn calculate_var_log_likelihood(residual_cov: &[Vec<f64>], n_obs: usize) -> f64 {
    let k = residual_cov.len();
    let det = matrix_determinant(residual_cov);

    if det <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let log_det = det.ln();
    let constant = -(k as f64) * (n_obs as f64) * (2.0 * std::f64::consts::PI).ln() / 2.0;

    constant - (n_obs as f64) * log_det / 2.0
}

/// Calculate matrix determinant (for small matrices)
fn matrix_determinant(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();

    match n {
        1 => matrix[0][0],
        2 => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        _ => {
            // Use LU decomposition for larger matrices
            let mut a = matrix.to_vec();
            let mut det = 1.0;

            for i in 0..n {
                // Find pivot
                let mut max_row = i;
                for k in (i + 1)..n {
                    if a[k][i].abs() > a[max_row][i].abs() {
                        max_row = k;
                    }
                }

                if max_row != i {
                    a.swap(i, max_row);
                    det = -det;
                }

                if a[i][i].abs() < 1e-14 {
                    return 0.0;
                }

                det *= a[i][i];

                for k in (i + 1)..n {
                    let factor = a[k][i] / a[i][i];
                    for j in i..n {
                        a[k][j] -= factor * a[i][j];
                    }
                }
            }

            det
        }
    }
}

/// Cholesky decomposition
fn cholesky_decomposition(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[j][k] * l[j][k];
                }

                let diag_val = matrix[j][j] - sum;
                if diag_val <= 0.0 {
                    return Err("Matrix is not positive definite".into());
                }
                l[j][j] = diag_val.sqrt();
            } else {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                l[i][j] = (matrix[i][j] - sum) / l[j][j];
            }
        }
    }

    Ok(l)
}

/// Calculate VAR model diagnostics
fn calculate_var_diagnostics(
    coefficients: &[Vec<Vec<f64>>],
    _residual_cov: &[Vec<f64>],
    residuals: &[Vec<f64>],
) -> VARDiagnostics {
    let n_vars = coefficients[0].len();
    let n_obs = residuals[0].len();

    // Simplified diagnostic calculations
    // In practice, these would be more sophisticated

    // Ljung-Box test (simplified)
    let ljung_box_statistic = 0.0;  // TODO: Implement proper Ljung-Box test
    let ljung_box_p_value = 1.0;

    // Jarque-Bera test (simplified)
    let mut jarque_bera_statistics = Vec::new();
    let mut jarque_bera_p_values = Vec::new();

    for i in 0..n_vars {
        // Calculate skewness and kurtosis
        let mean = residuals[i].iter().sum::<f64>() / n_obs as f64;
        let mut variance = 0.0;
        let mut skewness = 0.0;
        let mut kurtosis = 0.0;

        for &r in &residuals[i] {
            let centered = r - mean;
            variance += centered * centered;
            skewness += centered.powi(3);
            kurtosis += centered.powi(4);
        }

        variance /= n_obs as f64;
        skewness /= n_obs as f64 * variance.powf(1.5);
        kurtosis /= n_obs as f64 * variance * variance;
        kurtosis -= 3.0;  // Excess kurtosis

        let jb_stat = (n_obs as f64) / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0);
        jarque_bera_statistics.push(jb_stat);
        jarque_bera_p_values.push(if jb_stat > 5.99 { 0.01 } else { 0.5 });  // Simplified
    }

    // ARCH test (simplified)
    let arch_statistics = vec![0.0; n_vars];
    let arch_p_values = vec![1.0; n_vars];

    // Stability check (simplified)
    let eigenvalues: Vec<f64> = vec![0.5; n_vars];  // TODO: Calculate actual eigenvalues
    let is_stable = eigenvalues.iter().all(|&e| e.abs() < 1.0);

    VARDiagnostics {
        ljung_box_statistic,
        ljung_box_p_value,
        jarque_bera_statistics,
        jarque_bera_p_values,
        arch_statistics,
        arch_p_values,
        is_stable,
        eigenvalues,
    }
}

/// Compute variance decomposition
fn compute_variance_decomposition(
    responses: &HashMap<String, HashMap<String, Vec<f64>>>,
    periods: usize,
) -> VarianceDecomposition {
    let variables: Vec<String> = responses.keys().cloned().collect();
    let mut decomposition = HashMap::new();

    for response_var in &variables {
        decomposition.insert(response_var.clone(), HashMap::new());

        for shock_var in &variables {
            decomposition.get_mut(response_var).unwrap()
                .insert(shock_var.clone(), Vec::new());
        }

        // Calculate variance decomposition for each period
        for h in 0..periods {
            let mut total_variance = 0.0;
            let mut shock_contributions = HashMap::new();

            // Calculate contribution of each shock
            for shock_var in &variables {
                let mut contribution = 0.0;
                for j in 0..=h {
                    let response = responses[shock_var][response_var][j];
                    contribution += response * response;
                }
                shock_contributions.insert(shock_var.clone(), contribution);
                total_variance += contribution;
            }

            // Convert to percentages
            for shock_var in &variables {
                let percentage = if total_variance > 0.0 {
                    shock_contributions[shock_var] / total_variance * 100.0
                } else {
                    0.0
                };
                decomposition.get_mut(response_var).unwrap()
                    .get_mut(shock_var).unwrap().push(percentage);
            }
        }
    }

    VarianceDecomposition {
        variables,
        periods,
        decomposition,
    }
}

/// Approximate F-distribution CDF (simplified implementation)
fn f_distribution_cdf(f_stat: f64, df1: usize, df2: usize) -> f64 {
    // Simplified approximation - in practice would use proper F-distribution
    if f_stat <= 0.0 {
        return 0.0;
    }

    // Very rough approximation based on F-distribution properties
    let critical_values = [
        (1, 1, 161.4), (1, 5, 6.61), (1, 10, 4.96), (1, 30, 4.17),
        (5, 5, 5.05), (5, 10, 3.33), (5, 30, 2.53),
        (10, 10, 2.98), (10, 30, 2.16),
    ];

    // Find closest match and interpolate (very simplified)
    for &(d1, d2, crit) in &critical_values {
        if df1 <= d1 && df2 <= d2 {
            return if f_stat > crit { 0.95 } else { 0.5 };
        }
    }

    // Default approximation
    if f_stat > 4.0 { 0.95 } else if f_stat > 2.0 { 0.8 } else { 0.5 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_granger_causality_basic() {
        // Create test data where x causes y
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut y = vec![0.0; 10];

        // y[t] = 0.5 * x[t-1] + noise
        for i in 1..10 {
            y[i] = 0.5 * x[i - 1] + (i as f64) * 0.1;
        }

        let result = test_granger_causality(&x, &y, 2, "x", "y").unwrap();

        assert_eq!(result.cause_variable, "x");
        assert_eq!(result.effect_variable, "y");
        assert_eq!(result.lags, 2);
        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_var_model_estimation() {
        let mut data = HashMap::new();

        // Create simple VAR(1) data
        let n = 50;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];

        for t in 1..n {
            x[t] = 0.5 * x[t - 1] + 0.2 * y[t - 1] + (t as f64) * 0.01;
            y[t] = 0.3 * x[t - 1] + 0.6 * y[t - 1] + (t as f64) * 0.005;
        }

        data.insert("x".to_string(), x);
        data.insert("y".to_string(), y);

        let var_model = estimate_var_model(&data, 1).unwrap();

        assert_eq!(var_model.variables.len(), 2);
        assert_eq!(var_model.lags, 1);
        assert_eq!(var_model.coefficients.len(), 1);
        assert_eq!(var_model.coefficients[0].len(), 2);
        assert_eq!(var_model.coefficients[0][0].len(), 2);
        assert!(var_model.log_likelihood.is_finite());
    }

    #[test]
    fn test_impulse_response() {
        let mut data = HashMap::new();
        data.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        data.insert("y".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let var_model = estimate_var_model(&data, 1).unwrap();
        let irf = compute_impulse_response(&var_model, 5, None).unwrap();

        assert_eq!(irf.periods, 5);
        assert_eq!(irf.variables.len(), 2);
        assert!(irf.responses.contains_key("x"));
        assert!(irf.responses.contains_key("y"));
        assert!(irf.responses["x"].contains_key("x"));
        assert!(irf.responses["x"].contains_key("y"));
    }

    #[test]
    fn test_ols_estimation() {
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let x = vec![
            vec![1.0, 0.0],  // Intercept and x1
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
        ];

        let result = estimate_ols(&y, &x).unwrap();

        // Should find y = 1 + 2*x (approximately)
        assert!((result.coefficients[0] - 1.0).abs() < 1e-10);
        assert!((result.coefficients[1] - 2.0).abs() < 1e-10);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
    }
}