//! # Correlation and Relationship Analysis Module
//!
//! Comprehensive multivariate analysis and correlation detection for time series data,
//! including correlation analysis, cross-correlation, Granger causality, cointegration,
//! Dynamic Time Warping, and Principal Component Analysis.

pub mod basic;
pub mod rolling;
pub mod cross_correlation;
pub mod granger;
pub mod cointegration;
pub mod dtw;
pub mod pca;

// Re-export commonly used types and functions
pub use basic::{CorrelationMatrix, CorrelationType, compute_correlation_matrix, compute_pairwise_correlation};
pub use rolling::{RollingCorrelation, compute_rolling_correlation};
pub use cross_correlation::{CrossCorrelationAnalysis, LeadLagResult, compute_lead_lag_analysis};
pub use granger::{GrangerCausalityResult, VARModel, ImpulseResponseResult, test_granger_causality};
pub use cointegration::{CointegrationResult, JohansenResult, EngleGrangerResult, test_cointegration};
pub use dtw::{DTWResult, DTWAlignment, compute_dtw_distance, compute_dtw_alignment};
pub use pca::{PCAResult, FactorAnalysisResult, compute_pca, extract_common_trends};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive correlation and relationship analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysisResult {
    /// Correlation matrix for all variable pairs
    pub correlation_matrix: Option<CorrelationMatrix>,

    /// Rolling correlation analysis
    pub rolling_correlations: Option<RollingCorrelation>,

    /// Cross-correlation analysis with lead-lag relationships
    pub cross_correlation: Option<CrossCorrelationAnalysis>,

    /// Granger causality test results
    pub granger_causality: HashMap<String, GrangerCausalityResult>,

    /// VAR model results
    pub var_model: Option<VARModel>,

    /// Impulse response functions
    pub impulse_response: Option<ImpulseResponseResult>,

    /// Cointegration test results
    pub cointegration: Option<CointegrationResult>,

    /// Dynamic Time Warping results
    pub dtw_results: HashMap<String, DTWResult>,

    /// Principal Component Analysis results
    pub pca_results: Option<PCAResult>,

    /// Factor analysis results
    pub factor_analysis: Option<FactorAnalysisResult>,

    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Metadata for correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Number of variables analyzed
    pub n_variables: usize,

    /// Number of observations per variable
    pub n_observations: usize,

    /// Variable names
    pub variable_names: Vec<String>,

    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Analysis configuration
    pub config: AnalysisConfig,

    /// Analysis duration in milliseconds
    pub duration_ms: u64,
}

/// Configuration for correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Correlation types to compute
    pub correlation_types: Vec<CorrelationType>,

    /// Rolling window size for rolling correlations
    pub rolling_window: Option<usize>,

    /// Maximum lag for cross-correlation analysis
    pub max_lag: usize,

    /// Significance level for statistical tests
    pub alpha: f64,

    /// Number of lags for Granger causality tests
    pub granger_lags: usize,

    /// Number of principal components to extract
    pub n_components: Option<usize>,

    /// DTW window constraint (Sakoe-Chiba band)
    pub dtw_window: Option<usize>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            correlation_types: vec![CorrelationType::Pearson, CorrelationType::Spearman],
            rolling_window: Some(30),
            max_lag: 50,
            alpha: 0.05,
            granger_lags: 5,
            n_components: None,
            dtw_window: None,
        }
    }
}

impl CorrelationAnalysisResult {
    /// Create a new empty correlation analysis result
    pub fn new(variable_names: Vec<String>, n_observations: usize) -> Self {
        Self {
            correlation_matrix: None,
            rolling_correlations: None,
            cross_correlation: None,
            granger_causality: HashMap::new(),
            var_model: None,
            impulse_response: None,
            cointegration: None,
            dtw_results: HashMap::new(),
            pca_results: None,
            factor_analysis: None,
            metadata: AnalysisMetadata {
                n_variables: variable_names.len(),
                n_observations,
                variable_names,
                timestamp: chrono::Utc::now(),
                config: AnalysisConfig::default(),
                duration_ms: 0,
            },
        }
    }

    /// Generate a comprehensive summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("Correlation and Relationship Analysis Report\n");
        report.push_str("==========================================\n\n");

        report.push_str(&format!("Variables: {} (Observations: {})\n",
            self.metadata.n_variables, self.metadata.n_observations));
        report.push_str(&format!("Analyzed: {}\n\n",
            self.metadata.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));

        // Correlation matrix summary
        if let Some(ref corr_matrix) = self.correlation_matrix {
            report.push_str("Correlation Matrix Summary:\n");
            for corr_type in &corr_matrix.correlation_types {
                report.push_str(&format!("  {:?} correlations computed\n", corr_type));
            }
            report.push_str("\n");
        }

        // Granger causality summary
        if !self.granger_causality.is_empty() {
            report.push_str("Granger Causality Results:\n");
            for (pair, result) in &self.granger_causality {
                report.push_str(&format!("  {}: p-value = {:.4} ({})\n",
                    pair,
                    result.p_value,
                    if result.is_significant { "Significant" } else { "Not significant" }
                ));
            }
            report.push_str("\n");
        }

        // PCA summary
        if let Some(ref pca) = self.pca_results {
            report.push_str("Principal Component Analysis:\n");
            report.push_str(&format!("  Components extracted: {}\n", pca.n_components));
            report.push_str(&format!("  Total variance explained: {:.2}%\n",
                pca.explained_variance_ratio.iter().sum::<f64>() * 100.0));
            report.push_str("\n");
        }

        report.push_str(&format!("Analysis completed in {} ms\n", self.metadata.duration_ms));

        report
    }
}

/// Perform comprehensive correlation and relationship analysis
pub fn analyze_correlations(
    data: &HashMap<String, Vec<f64>>,
    config: AnalysisConfig,
) -> Result<CorrelationAnalysisResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    let variable_names: Vec<String> = data.keys().cloned().collect();
    let n_observations = data.values().next().map(|v| v.len()).unwrap_or(0);

    // Validate input data
    for (name, values) in data {
        if values.len() != n_observations {
            return Err(format!("Variable '{}' has {} observations, expected {}",
                name, values.len(), n_observations).into());
        }
    }

    let mut result = CorrelationAnalysisResult::new(variable_names.clone(), n_observations);
    result.metadata.config = config.clone();

    // Compute correlation matrix
    if !config.correlation_types.is_empty() {
        let correlation_matrix = compute_correlation_matrix(data, &config.correlation_types)?;
        result.correlation_matrix = Some(correlation_matrix);
    }

    // Compute rolling correlations if requested
    if let Some(window) = config.rolling_window {
        if data.len() >= 2 {
            let pairs: Vec<_> = variable_names.iter().take(2).cloned().collect();
            if pairs.len() == 2 {
                let series1 = &data[&pairs[0]];
                let series2 = &data[&pairs[1]];
                let rolling_corr = compute_rolling_correlation(series1, series2, window)?;
                result.rolling_correlations = Some(rolling_corr);
            }
        }
    }

    result.metadata.duration_ms = start_time.elapsed().as_millis() as u64;

    Ok(result)
}