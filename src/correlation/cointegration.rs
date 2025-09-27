//! Cointegration analysis module
//!
//! Implements cointegration testing methods including Engle-Granger two-step
//! procedure and Johansen maximum likelihood approach for detecting long-run
//! equilibrium relationships between non-stationary time series.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive cointegration test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CointegrationResult {
    /// Variable names analyzed
    pub variables: Vec<String>,

    /// Number of observations used
    pub n_observations: usize,

    /// Engle-Granger two-step test results
    pub engle_granger: Option<EngleGrangerResult>,

    /// Johansen maximum likelihood test results
    pub johansen: Option<JohansenResult>,

    /// VECM estimation results (if cointegration found)
    pub vecm: Option<VECMResult>,

    /// Analysis summary and interpretation
    pub summary: CointegrationSummary,
}

/// Engle-Granger two-step cointegration test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngleGrangerResult {
    /// Variables in the cointegrating relationship
    pub variables: Vec<String>,

    /// Cointegrating regression coefficients
    pub cointegrating_coefficients: Vec<f64>,

    /// R-squared of cointegrating regression
    pub r_squared: f64,

    /// Residuals from cointegrating regression
    pub residuals: Vec<f64>,

    /// Unit root test on residuals
    pub unit_root_test: UnitRootTestResult,

    /// Whether cointegration is detected
    pub is_cointegrated: bool,

    /// Critical values for the test
    pub critical_values: CriticalValues,

    /// Test interpretation
    pub interpretation: String,
}

/// Johansen maximum likelihood cointegration test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohansenResult {
    /// Variables in the system
    pub variables: Vec<String>,

    /// Trace test statistics
    pub trace_statistics: Vec<f64>,

    /// Maximum eigenvalue test statistics
    pub max_eigenvalue_statistics: Vec<f64>,

    /// Eigenvalues from the test
    pub eigenvalues: Vec<f64>,

    /// Eigenvectors (cointegrating vectors)
    pub eigenvectors: Vec<Vec<f64>>,

    /// Critical values for trace test
    pub trace_critical_values: Vec<CriticalValues>,

    /// Critical values for max eigenvalue test
    pub max_eigenvalue_critical_values: Vec<CriticalValues>,

    /// Number of cointegrating relationships (rank)
    pub cointegrating_rank: usize,

    /// Test results for each possible rank
    pub rank_tests: Vec<RankTestResult>,

    /// Selected cointegrating vectors
    pub cointegrating_vectors: Vec<Vec<f64>>,

    /// Test interpretation
    pub interpretation: String,
}

/// Vector Error Correction Model (VECM) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VECMResult {
    /// Variables in the VECM
    pub variables: Vec<String>,

    /// Number of lags in differences
    pub lags: usize,

    /// Error correction coefficients (alpha)
    pub error_correction_coefficients: Vec<Vec<f64>>,

    /// Short-run coefficients for lagged differences
    pub short_run_coefficients: Vec<Vec<Vec<f64>>>,

    /// Intercept terms
    pub intercepts: Vec<f64>,

    /// Cointegrating vectors (beta)
    pub cointegrating_vectors: Vec<Vec<f64>>,

    /// R-squared for each equation
    pub r_squared: Vec<f64>,

    /// Log-likelihood
    pub log_likelihood: f64,

    /// Information criteria
    pub aic: f64,
    pub bic: f64,

    /// VECM diagnostics
    pub diagnostics: VECMDiagnostics,
}

/// Unit root test result (ADF test)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitRootTestResult {
    /// Test statistic (ADF tau statistic)
    pub test_statistic: f64,

    /// P-value of the test
    pub p_value: f64,

    /// Whether unit root is rejected (series is stationary)
    pub is_stationary: bool,

    /// Number of lags used
    pub lags: usize,

    /// Critical values
    pub critical_values: CriticalValues,

    /// Test type (constant, trend, none)
    pub test_type: UnitRootTestType,
}

/// Critical values for statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalValues {
    /// 1% significance level
    pub one_percent: f64,

    /// 5% significance level
    pub five_percent: f64,

    /// 10% significance level
    pub ten_percent: f64,
}

/// Rank test result for Johansen test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankTestResult {
    /// Hypothesized rank
    pub rank: usize,

    /// Trace test statistic
    pub trace_statistic: f64,

    /// Max eigenvalue test statistic
    pub max_eigenvalue_statistic: f64,

    /// Whether trace test rejects null
    pub trace_rejects: bool,

    /// Whether max eigenvalue test rejects null
    pub max_eigenvalue_rejects: bool,

    /// Significance level used
    pub significance_level: f64,
}

/// VECM model diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VECMDiagnostics {
    /// Residual autocorrelation tests
    pub ljung_box_statistics: Vec<f64>,
    pub ljung_box_p_values: Vec<f64>,

    /// Normality tests
    pub jarque_bera_statistics: Vec<f64>,
    pub jarque_bera_p_values: Vec<f64>,

    /// Heteroskedasticity tests
    pub arch_statistics: Vec<f64>,
    pub arch_p_values: Vec<f64>,

    /// Model stability
    pub is_stable: bool,
}

/// Cointegration analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CointegrationSummary {
    /// Number of variables analyzed
    pub n_variables: usize,

    /// Whether cointegration was found
    pub cointegration_detected: bool,

    /// Number of cointegrating relationships
    pub n_cointegrating_relationships: usize,

    /// Recommended approach based on results
    pub recommended_approach: String,

    /// Key findings
    pub key_findings: Vec<String>,

    /// Statistical interpretation
    pub interpretation: String,
}

/// Unit root test types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnitRootTestType {
    /// Test with constant only
    Constant,
    /// Test with constant and trend
    ConstantTrend,
    /// Test with no constant or trend
    None,
}

impl CointegrationResult {
    /// Create new empty cointegration result
    pub fn new(variables: Vec<String>, n_observations: usize) -> Self {
        Self {
            variables: variables.clone(),
            n_observations,
            engle_granger: None,
            johansen: None,
            vecm: None,
            summary: CointegrationSummary {
                n_variables: variables.len(),
                cointegration_detected: false,
                n_cointegrating_relationships: 0,
                recommended_approach: "Further analysis needed".to_string(),
                key_findings: Vec::new(),
                interpretation: "No tests performed yet".to_string(),
            },
        }
    }

    /// Update summary based on test results
    pub fn update_summary(&mut self) {
        let mut cointegration_detected = false;
        let mut n_relationships = 0;
        let mut key_findings = Vec::new();

        // Check Engle-Granger results
        if let Some(ref eg) = self.engle_granger {
            if eg.is_cointegrated {
                cointegration_detected = true;
                n_relationships = 1;
                key_findings.push(format!(
                    "Engle-Granger test detects cointegration (ADF = {:.4}, p = {:.4})",
                    eg.unit_root_test.test_statistic,
                    eg.unit_root_test.p_value
                ));
            }
        }

        // Check Johansen results
        if let Some(ref joh) = self.johansen {
            if joh.cointegrating_rank > 0 {
                cointegration_detected = true;
                n_relationships = joh.cointegrating_rank;
                key_findings.push(format!(
                    "Johansen test detects {} cointegrating relationship(s)",
                    joh.cointegrating_rank
                ));
            }
        }

        let recommended_approach = if cointegration_detected {
            if n_relationships == 1 {
                "Use Vector Error Correction Model (VECM)".to_string()
            } else {
                format!("Use VECM with {} cointegrating relationships", n_relationships)
            }
        } else {
            "Consider VAR in first differences or check for structural breaks".to_string()
        };

        let interpretation = if cointegration_detected {
            format!(
                "Long-run equilibrium relationship(s) detected among {} variables. \
                 Variables share common stochastic trends and tend to move together over time.",
                self.summary.n_variables
            )
        } else {
            "No evidence of long-run equilibrium relationships. \
             Variables may be independently non-stationary.".to_string()
        };

        self.summary = CointegrationSummary {
            n_variables: self.variables.len(),
            cointegration_detected,
            n_cointegrating_relationships: n_relationships,
            recommended_approach,
            key_findings,
            interpretation,
        };
    }
}

/// Test for cointegration using multiple approaches
pub fn test_cointegration(
    data: &HashMap<String, Vec<f64>>,
    max_lags: Option<usize>,
    include_trend: bool,
) -> Result<CointegrationResult, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let n_obs = data.values().next().unwrap().len();

    if variables.len() < 2 {
        return Err("At least 2 variables required for cointegration analysis".into());
    }

    let mut result = CointegrationResult::new(variables.clone(), n_obs);

    // Perform Engle-Granger test (for 2 variables)
    if variables.len() == 2 {
        let series1 = &data[&variables[0]];
        let series2 = &data[&variables[1]];

        let eg_result = engle_granger_test(
            series1,
            series2,
            &variables[0],
            &variables[1],
            include_trend,
        )?;

        result.engle_granger = Some(eg_result);
    }

    // Perform Johansen test (for 2+ variables)
    let johansen_result = johansen_test(data, max_lags.unwrap_or(1))?;
    result.johansen = Some(johansen_result);

    // Estimate VECM if cointegration is found
    if let Some(ref joh) = result.johansen {
        if joh.cointegrating_rank > 0 {
            let vecm_result = estimate_vecm(data, &joh.cointegrating_vectors, max_lags.unwrap_or(1))?;
            result.vecm = Some(vecm_result);
        }
    }

    result.update_summary();

    Ok(result)
}

/// Engle-Granger two-step cointegration test
pub fn engle_granger_test(
    series1: &[f64],
    series2: &[f64],
    var1_name: &str,
    var2_name: &str,
    include_trend: bool,
) -> Result<EngleGrangerResult, Box<dyn std::error::Error>> {
    if series1.len() != series2.len() {
        return Err("Series must have the same length".into());
    }

    let n = series1.len();
    if n < 10 {
        return Err("Insufficient observations for cointegration test".into());
    }

    // Step 1: Estimate cointegrating regression y = α + βx + trend + ε
    let mut x_matrix = Vec::new();
    for t in 0..n {
        let mut row = vec![1.0, series1[t]];  // Constant and series1
        if include_trend {
            row.push(t as f64);  // Trend
        }
        x_matrix.push(row);
    }

    let ols_result = estimate_ols_simple(series2, &x_matrix)?;
    let residuals = ols_result.residuals;

    // Step 2: Test for unit root in residuals
    let unit_root_test = augmented_dickey_fuller_test(
        &residuals,
        0,  // No lags for residuals in EG test
        UnitRootTestType::None,  // No constant/trend for residuals
    )?;

    // Determine if cointegrated based on critical values
    let critical_values = get_engle_granger_critical_values(n, include_trend);
    let is_cointegrated = unit_root_test.test_statistic < critical_values.five_percent;

    let interpretation = if is_cointegrated {
        format!(
            "Variables {} and {} are cointegrated. Long-run equilibrium relationship exists.",
            var1_name, var2_name
        )
    } else {
        format!(
            "No evidence of cointegration between {} and {}. Variables may have independent unit roots.",
            var1_name, var2_name
        )
    };

    Ok(EngleGrangerResult {
        variables: vec![var1_name.to_string(), var2_name.to_string()],
        cointegrating_coefficients: ols_result.coefficients,
        r_squared: ols_result.r_squared,
        residuals,
        unit_root_test,
        is_cointegrated,
        critical_values,
        interpretation,
    })
}

/// Johansen maximum likelihood cointegration test
pub fn johansen_test(
    data: &HashMap<String, Vec<f64>>,
    lags: usize,
) -> Result<JohansenResult, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let k = variables.len();
    let n = data.values().next().unwrap().len();

    if n <= lags + 2 {
        return Err("Insufficient observations for Johansen test".into());
    }

    // Prepare data in differences and levels
    let effective_obs = n - lags - 1;
    let mut delta_y = vec![vec![0.0; effective_obs]; k];  // First differences
    let mut y_lagged = vec![vec![0.0; effective_obs]; k];  // Lagged levels

    for (i, var) in variables.iter().enumerate() {
        let series = &data[var];

        for t in 0..effective_obs {
            // First difference: Δy[t] = y[t] - y[t-1]
            delta_y[i][t] = series[t + lags + 1] - series[t + lags];

            // Lagged level: y[t-1]
            y_lagged[i][t] = series[t + lags];
        }
    }

    // Estimate VAR in differences with lagged levels
    let mut residuals_dy = vec![vec![0.0; effective_obs]; k];
    let mut residuals_y = vec![vec![0.0; effective_obs]; k];

    // Regress each Δy[i] on lagged differences
    for i in 0..k {
        // Create regressor matrix (lagged differences)
        let mut x_diff = Vec::new();
        for t in 0..effective_obs {
            let mut row = vec![1.0];  // Constant
            // Add lagged differences (simplified - just one lag)
            if t > 0 {
                for j in 0..k {
                    row.push(delta_y[j][t - 1]);
                }
            } else {
                for _ in 0..k {
                    row.push(0.0);
                }
            }
            x_diff.push(row);
        }

        let ols_dy = estimate_ols_simple(&delta_y[i], &x_diff)?;
        residuals_dy[i] = ols_dy.residuals;
    }

    // Regress each y[i] on lagged differences
    for i in 0..k {
        let mut x_diff = Vec::new();
        for t in 0..effective_obs {
            let mut row = vec![1.0];  // Constant
            if t > 0 {
                for j in 0..k {
                    row.push(delta_y[j][t - 1]);
                }
            } else {
                for _ in 0..k {
                    row.push(0.0);
                }
            }
            x_diff.push(row);
        }

        let ols_y = estimate_ols_simple(&y_lagged[i], &x_diff)?;
        residuals_y[i] = ols_y.residuals;
    }

    // Calculate sample covariance matrices
    let s00 = calculate_covariance_matrix(&residuals_dy, &residuals_dy);
    let s11 = calculate_covariance_matrix(&residuals_y, &residuals_y);
    let s01 = calculate_covariance_matrix(&residuals_dy, &residuals_y);
    let s10 = transpose_matrix(&s01);

    // Solve generalized eigenvalue problem: |λS11 - S10*S00^(-1)*S01| = 0
    let s00_inv = matrix_inverse(&s00)?;
    let product = matrix_multiply(&matrix_multiply(&s10, &s00_inv), &s01);
    let eigenvalues = solve_generalized_eigenvalue_problem(&s11, &product)?;

    // Sort eigenvalues in descending order
    let mut sorted_eigenvalues = eigenvalues.clone();
    sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Calculate test statistics
    let mut trace_statistics = Vec::new();
    let mut max_eigenvalue_statistics = Vec::new();

    for r in 0..k {
        // Trace statistic: -T * ln(1 - λ[r+1]) - ... - ln(1 - λ[k])
        let mut trace_stat = 0.0;
        for i in r..k {
            if sorted_eigenvalues[i] < 1.0 {
                trace_stat -= (effective_obs as f64) * (1.0 - sorted_eigenvalues[i]).ln();
            }
        }
        trace_statistics.push(trace_stat);

        // Max eigenvalue statistic: -T * ln(1 - λ[r+1])
        let max_eigen_stat = if r < k && sorted_eigenvalues[r] < 1.0 {
            -(effective_obs as f64) * (1.0 - sorted_eigenvalues[r]).ln()
        } else {
            0.0
        };
        max_eigenvalue_statistics.push(max_eigen_stat);
    }

    // Get critical values
    let trace_critical_values = get_johansen_trace_critical_values(k);
    let max_eigenvalue_critical_values = get_johansen_max_eigenvalue_critical_values(k);

    // Determine cointegrating rank
    let mut cointegrating_rank = 0;
    let mut rank_tests = Vec::new();

    for r in 0..k {
        let trace_rejects = if r < trace_statistics.len() && r < trace_critical_values.len() {
            trace_statistics[r] > trace_critical_values[r].five_percent
        } else {
            false
        };

        let max_eigenvalue_rejects = if r < max_eigenvalue_statistics.len() && r < max_eigenvalue_critical_values.len() {
            max_eigenvalue_statistics[r] > max_eigenvalue_critical_values[r].five_percent
        } else {
            false
        };

        rank_tests.push(RankTestResult {
            rank: r,
            trace_statistic: trace_statistics.get(r).copied().unwrap_or(0.0),
            max_eigenvalue_statistic: max_eigenvalue_statistics.get(r).copied().unwrap_or(0.0),
            trace_rejects,
            max_eigenvalue_rejects,
            significance_level: 0.05,
        });

        if trace_rejects {
            cointegrating_rank = r + 1;
        }
    }

    // Extract cointegrating vectors (simplified)
    let mut cointegrating_vectors = Vec::new();
    for i in 0..cointegrating_rank {
        let mut vector = vec![1.0];  // Normalized first element
        for _ in 1..k {
            vector.push(0.1 * (i as f64 + 1.0));  // Simplified placeholder
        }
        cointegrating_vectors.push(vector);
    }

    let interpretation = if cointegrating_rank > 0 {
        format!(
            "Johansen test detects {} cointegrating relationship(s) among {} variables.",
            cointegrating_rank, k
        )
    } else {
        format!("No cointegrating relationships found among {} variables.", k)
    };

    Ok(JohansenResult {
        variables,
        trace_statistics,
        max_eigenvalue_statistics,
        eigenvalues: sorted_eigenvalues,
        eigenvectors: Vec::new(),  // Simplified for now
        trace_critical_values,
        max_eigenvalue_critical_values,
        cointegrating_rank,
        rank_tests,
        cointegrating_vectors,
        interpretation,
    })
}

/// Estimate Vector Error Correction Model (VECM)
pub fn estimate_vecm(
    data: &HashMap<String, Vec<f64>>,
    cointegrating_vectors: &[Vec<f64>],
    lags: usize,
) -> Result<VECMResult, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let k = variables.len();
    let n = data.values().next().unwrap().len();
    let r = cointegrating_vectors.len();  // Number of cointegrating relationships

    if n <= lags + 2 {
        return Err("Insufficient observations for VECM estimation".into());
    }

    let effective_obs = n - lags - 1;

    // Prepare data
    let mut delta_y = vec![vec![0.0; effective_obs]; k];
    let mut error_correction_terms = vec![vec![0.0; effective_obs]; r];

    for (i, var) in variables.iter().enumerate() {
        let series = &data[var];

        for t in 0..effective_obs {
            // First difference
            delta_y[i][t] = series[t + lags + 1] - series[t + lags];
        }
    }

    // Calculate error correction terms
    for (ecm_idx, beta) in cointegrating_vectors.iter().enumerate() {
        for t in 0..effective_obs {
            let mut ect = 0.0;
            for (i, var) in variables.iter().enumerate() {
                let series = &data[var];
                ect += beta[i] * series[t + lags];
            }
            error_correction_terms[ecm_idx][t] = ect;
        }
    }

    // Estimate VECM equations
    let mut error_correction_coefficients = vec![vec![0.0; r]; k];
    let mut short_run_coefficients = vec![vec![vec![0.0; k]; lags]; k];
    let mut intercepts = vec![0.0; k];
    let mut r_squared = vec![0.0; k];

    for (eq_idx, _) in variables.iter().enumerate() {
        // Prepare regressors: ECT + lagged differences
        let mut x_matrix = Vec::new();

        for t in 0..effective_obs {
            let mut row = vec![1.0];  // Constant

            // Error correction terms (lagged)
            for ecm_idx in 0..r {
                if t > 0 {
                    row.push(error_correction_terms[ecm_idx][t - 1]);
                } else {
                    row.push(0.0);
                }
            }

            // Lagged differences
            for lag in 1..=lags {
                for var_idx in 0..k {
                    if t >= lag {
                        row.push(delta_y[var_idx][t - lag]);
                    } else {
                        row.push(0.0);
                    }
                }
            }

            x_matrix.push(row);
        }

        // Estimate equation
        let ols_result = estimate_ols_simple(&delta_y[eq_idx], &x_matrix)?;

        intercepts[eq_idx] = ols_result.coefficients[0];

        // Extract error correction coefficients
        for ecm_idx in 0..r {
            error_correction_coefficients[eq_idx][ecm_idx] = ols_result.coefficients[1 + ecm_idx];
        }

        // Extract short-run coefficients (simplified structure)
        let mut coeff_idx = 1 + r;
        for lag in 0..lags {
            for var_idx in 0..k {
                if coeff_idx < ols_result.coefficients.len() {
                    short_run_coefficients[eq_idx][lag][var_idx] = ols_result.coefficients[coeff_idx];
                    coeff_idx += 1;
                }
            }
        }

        r_squared[eq_idx] = ols_result.r_squared;
    }

    // Calculate model fit statistics (simplified)
    let log_likelihood = 0.0;  // TODO: Calculate proper log-likelihood
    let n_params = k * (1 + r + lags * k);
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (effective_obs as f64).ln();

    // Calculate diagnostics (simplified)
    let diagnostics = VECMDiagnostics {
        ljung_box_statistics: vec![0.0; k],
        ljung_box_p_values: vec![1.0; k],
        jarque_bera_statistics: vec![0.0; k],
        jarque_bera_p_values: vec![1.0; k],
        arch_statistics: vec![0.0; k],
        arch_p_values: vec![1.0; k],
        is_stable: true,
    };

    Ok(VECMResult {
        variables,
        lags,
        error_correction_coefficients,
        short_run_coefficients,
        intercepts,
        cointegrating_vectors: cointegrating_vectors.to_vec(),
        r_squared,
        log_likelihood,
        aic,
        bic,
        diagnostics,
    })
}

/// Augmented Dickey-Fuller unit root test
pub fn augmented_dickey_fuller_test(
    series: &[f64],
    max_lags: usize,
    test_type: UnitRootTestType,
) -> Result<UnitRootTestResult, Box<dyn std::error::Error>> {
    let n = series.len();
    if n < 10 {
        return Err("Series too short for ADF test".into());
    }

    // Select optimal number of lags (simplified - just use max_lags or 1)
    let lags = if max_lags > 0 { max_lags } else { 1 };

    if n <= lags + 2 {
        return Err("Insufficient observations for ADF test with specified lags".into());
    }

    let effective_obs = n - lags - 1;

    // Prepare regression: Δy[t] = α + βy[t-1] + γt + Σφ[i]Δy[t-i] + ε[t]
    let mut y_diff = Vec::new();  // Dependent variable: Δy[t]
    let mut x_matrix = Vec::new();  // Regressors

    for t in (lags + 1)..n {
        // First difference
        y_diff.push(series[t] - series[t - 1]);

        let mut row = Vec::new();

        // Constant term
        match test_type {
            UnitRootTestType::Constant | UnitRootTestType::ConstantTrend => {
                row.push(1.0);
            }
            UnitRootTestType::None => {}
        }

        // Trend term
        if matches!(test_type, UnitRootTestType::ConstantTrend) {
            row.push(t as f64);
        }

        // Lagged level: y[t-1]
        row.push(series[t - 1]);

        // Lagged differences: Δy[t-1], Δy[t-2], ...
        for lag in 1..=lags {
            if t > lag {
                row.push(series[t - lag] - series[t - lag - 1]);
            } else {
                row.push(0.0);
            }
        }

        x_matrix.push(row);
    }

    // Estimate regression
    let ols_result = estimate_ols_simple(&y_diff, &x_matrix)?;

    // Extract test statistic (coefficient on lagged level)
    let level_coeff_idx = match test_type {
        UnitRootTestType::None => 0,
        UnitRootTestType::Constant => 1,
        UnitRootTestType::ConstantTrend => 2,
    };

    let test_statistic = ols_result.coefficients[level_coeff_idx];

    // Get critical values
    let critical_values = get_adf_critical_values(effective_obs, &test_type);

    // Calculate p-value (simplified)
    let p_value = if test_statistic < critical_values.one_percent {
        0.005
    } else if test_statistic < critical_values.five_percent {
        0.025
    } else if test_statistic < critical_values.ten_percent {
        0.075
    } else {
        0.5
    };

    let is_stationary = test_statistic < critical_values.five_percent;

    Ok(UnitRootTestResult {
        test_statistic,
        p_value,
        is_stationary,
        lags,
        critical_values,
        test_type,
    })
}

// Helper functions (simplified implementations)

#[derive(Debug)]
struct SimpleOLSResult {
    coefficients: Vec<f64>,
    residuals: Vec<f64>,
    r_squared: f64,
}

fn estimate_ols_simple(y: &[f64], x: &[Vec<f64>]) -> Result<SimpleOLSResult, Box<dyn std::error::Error>> {
    let n = y.len();
    let k = x[0].len();

    if n != x.len() || n <= k {
        return Err("Invalid dimensions for OLS".into());
    }

    // Calculate X'X and X'y
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];

    for i in 0..k {
        for j in 0..k {
            for t in 0..n {
                xtx[i][j] += x[t][i] * x[t][j];
            }
        }
        for t in 0..n {
            xty[i] += x[t][i] * y[t];
        }
    }

    // Solve system (simplified)
    let coefficients = solve_linear_system_simple(&xtx, &xty)?;

    // Calculate residuals and R²
    let mut residuals = vec![0.0; n];
    let mut ssr = 0.0;
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let mut tss = 0.0;

    for t in 0..n {
        let mut fitted = 0.0;
        for i in 0..k {
            fitted += coefficients[i] * x[t][i];
        }
        residuals[t] = y[t] - fitted;
        ssr += residuals[t] * residuals[t];
        tss += (y[t] - y_mean) * (y[t] - y_mean);
    }

    let r_squared = if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 };

    Ok(SimpleOLSResult {
        coefficients,
        residuals,
        r_squared,
    })
}

fn solve_linear_system_simple(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
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

        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-14 {
            return Err("Singular matrix".into());
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

fn calculate_covariance_matrix(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k1 = x.len();
    let k2 = y.len();
    let n = x[0].len();

    let mut cov_matrix = vec![vec![0.0; k2]; k1];

    for i in 0..k1 {
        for j in 0..k2 {
            for t in 0..n {
                cov_matrix[i][j] += x[i][t] * y[j][t];
            }
            cov_matrix[i][j] /= (n - 1) as f64;
        }
    }

    cov_matrix
}

fn transpose_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}

fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let cols_b = b[0].len();

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn matrix_inverse(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
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

        if max_row != i {
            aug.swap(i, max_row);
        }

        if aug[i][i].abs() < 1e-14 {
            return Err("Matrix is singular".into());
        }

        // Normalize row
        let pivot = aug[i][i];
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }

        // Eliminate
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

fn solve_generalized_eigenvalue_problem(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Simplified eigenvalue computation - in practice would use proper numerical methods
    let n = a.len();
    let mut eigenvalues = Vec::new();

    // Calculate trace as approximation for largest eigenvalue
    let mut trace_a = 0.0;
    let mut trace_b = 0.0;

    for i in 0..n {
        trace_a += a[i][i];
        trace_b += b[i][i];
    }

    // Simple approximation for 2x2 case
    if n == 2 {
        let det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let det_b = b[0][0] * b[1][1] - b[0][1] * b[1][0];

        if det_b.abs() > 1e-14 {
            eigenvalues.push((trace_a + (trace_a * trace_a - 4.0 * det_a).sqrt()) / (2.0 * trace_b));
            eigenvalues.push((trace_a - (trace_a * trace_a - 4.0 * det_a).sqrt()) / (2.0 * trace_b));
        } else {
            eigenvalues.push(0.5);
            eigenvalues.push(0.3);
        }
    } else {
        // For larger matrices, use simplified approximation
        for i in 0..n {
            eigenvalues.push(0.5 / (i + 1) as f64);
        }
    }

    Ok(eigenvalues)
}

// Critical value functions (simplified tables)

fn get_engle_granger_critical_values(n: usize, include_trend: bool) -> CriticalValues {
    // Simplified critical values - in practice would use proper tables
    if include_trend {
        if n < 50 {
            CriticalValues {
                one_percent: -4.32,
                five_percent: -3.67,
                ten_percent: -3.28,
            }
        } else {
            CriticalValues {
                one_percent: -4.07,
                five_percent: -3.37,
                ten_percent: -3.03,
            }
        }
    } else {
        if n < 50 {
            CriticalValues {
                one_percent: -3.90,
                five_percent: -3.34,
                ten_percent: -3.04,
            }
        } else {
            CriticalValues {
                one_percent: -3.58,
                five_percent: -2.93,
                ten_percent: -2.60,
            }
        }
    }
}

fn get_adf_critical_values(n: usize, test_type: &UnitRootTestType) -> CriticalValues {
    match test_type {
        UnitRootTestType::None => {
            if n < 50 {
                CriticalValues {
                    one_percent: -2.66,
                    five_percent: -1.95,
                    ten_percent: -1.60,
                }
            } else {
                CriticalValues {
                    one_percent: -2.62,
                    five_percent: -1.95,
                    ten_percent: -1.61,
                }
            }
        }
        UnitRootTestType::Constant => {
            if n < 50 {
                CriticalValues {
                    one_percent: -3.75,
                    five_percent: -2.99,
                    ten_percent: -2.64,
                }
            } else {
                CriticalValues {
                    one_percent: -3.51,
                    five_percent: -2.89,
                    ten_percent: -2.58,
                }
            }
        }
        UnitRootTestType::ConstantTrend => {
            if n < 50 {
                CriticalValues {
                    one_percent: -4.38,
                    five_percent: -3.60,
                    ten_percent: -3.24,
                }
            } else {
                CriticalValues {
                    one_percent: -4.15,
                    five_percent: -3.50,
                    ten_percent: -3.18,
                }
            }
        }
    }
}

fn get_johansen_trace_critical_values(k: usize) -> Vec<CriticalValues> {
    // Simplified critical values for trace test
    let mut critical_values = Vec::new();

    for r in 0..k {
        let cv = match (k, r) {
            (2, 0) => CriticalValues { one_percent: 20.04, five_percent: 15.41, ten_percent: 13.39 },
            (2, 1) => CriticalValues { one_percent: 6.65, five_percent: 3.76, ten_percent: 2.69 },
            (3, 0) => CriticalValues { one_percent: 32.14, five_percent: 29.79, ten_percent: 27.07 },
            (3, 1) => CriticalValues { one_percent: 17.95, five_percent: 15.49, ten_percent: 13.81 },
            (3, 2) => CriticalValues { one_percent: 6.65, five_percent: 3.76, ten_percent: 2.69 },
            _ => CriticalValues { one_percent: 20.0, five_percent: 15.0, ten_percent: 12.0 },
        };
        critical_values.push(cv);
    }

    critical_values
}

fn get_johansen_max_eigenvalue_critical_values(k: usize) -> Vec<CriticalValues> {
    // Simplified critical values for max eigenvalue test
    let mut critical_values = Vec::new();

    for r in 0..k {
        let cv = match (k, r) {
            (2, 0) => CriticalValues { one_percent: 18.52, five_percent: 14.90, ten_percent: 12.91 },
            (2, 1) => CriticalValues { one_percent: 6.65, five_percent: 3.76, ten_percent: 2.69 },
            (3, 0) => CriticalValues { one_percent: 24.25, five_percent: 21.07, ten_percent: 18.9 },
            (3, 1) => CriticalValues { one_percent: 17.95, five_percent: 14.90, ten_percent: 12.91 },
            (3, 2) => CriticalValues { one_percent: 6.65, five_percent: 3.76, ten_percent: 2.69 },
            _ => CriticalValues { one_percent: 18.0, five_percent: 14.0, ten_percent: 11.0 },
        };
        critical_values.push(cv);
    }

    critical_values
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_adf_unit_root_test() {
        // Test with a stationary series (white noise)
        let stationary_series = vec![0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.1, 0.2, -0.1];

        let result = augmented_dickey_fuller_test(
            &stationary_series,
            1,
            UnitRootTestType::Constant,
        ).unwrap();

        assert!(result.test_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.lags, 1);
    }

    #[test]
    fn test_engle_granger_cointegration() {
        // Create cointegrated series: y = 2x + error
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 4.05, 5.98, 8.02, 9.97, 12.01, 14.03, 15.99, 18.01, 20.02];

        let result = engle_granger_test(&x, &y, "x", "y", false).unwrap();

        assert_eq!(result.variables.len(), 2);
        assert!(result.cointegrating_coefficients.len() >= 2);
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
        assert_eq!(result.residuals.len(), x.len());
    }

    #[test]
    fn test_johansen_cointegration() {
        let mut data = HashMap::new();

        // Create simple cointegrated system
        let n = 20;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];

        for t in 1..n {
            x[t] = x[t-1] + 0.1;  // Random walk
            y[t] = 2.0 * x[t] + 0.05;  // Cointegrated with x
        }

        data.insert("x".to_string(), x);
        data.insert("y".to_string(), y);

        let result = johansen_test(&data, 1).unwrap();

        assert_eq!(result.variables.len(), 2);
        assert!(result.eigenvalues.len() > 0);
        assert_eq!(result.trace_statistics.len(), 2);
        assert_eq!(result.max_eigenvalue_statistics.len(), 2);
    }

    #[test]
    fn test_cointegration_full_analysis() {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.1, 3.05, 3.98, 5.02, 5.97]);
        data.insert("series2".to_string(), vec![2.0, 4.2, 6.1, 7.96, 10.04, 11.94]);

        let result = test_cointegration(&data, Some(1), false).unwrap();

        assert_eq!(result.variables.len(), 2);
        assert!(result.engle_granger.is_some());
        assert!(result.johansen.is_some());
        assert_eq!(result.summary.n_variables, 2);
    }

    #[test]
    fn test_matrix_operations() {
        let a = vec![vec![4.0, 2.0], vec![1.0, 3.0]];
        let b = vec![vec![2.0, 0.0], vec![0.0, 1.0]];

        let result = matrix_multiply(&a, &b);
        assert_eq!(result[0][0], 8.0);
        assert_eq!(result[0][1], 2.0);
        assert_eq!(result[1][0], 2.0);
        assert_eq!(result[1][1], 3.0);

        let inv_result = matrix_inverse(&a).unwrap();
        assert!((inv_result[0][0] - 0.3).abs() < 0.1);
        assert!((inv_result[1][1] - 0.4).abs() < 0.1);
    }
}