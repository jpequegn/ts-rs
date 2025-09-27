//! # Statistical Analysis Module
//!
//! Provides comprehensive statistical analysis capabilities for time series data,
//! including descriptive statistics, distribution analysis, time series specific
//! metrics, and stationarity testing.

pub mod descriptive;
pub mod distribution;
pub mod timeseries;
pub mod stationarity;
pub mod changepoint;
pub mod export;

// Re-export commonly used types and functions
pub use descriptive::{DescriptiveStats, compute_descriptive_stats};
pub use distribution::{DistributionAnalysis, NormalityTest, compute_distribution_analysis};
pub use timeseries::{TimeSeriesStats, compute_autocorrelation, compute_partial_autocorrelation, compute_cross_correlation};
pub use stationarity::{StationarityTest, AdfTest, KpssTest, PhillipsPerronTest, test_stationarity};
pub use changepoint::{ChangePoint, detect_changepoints};
pub use export::{ExportFormat, export_stats_results};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive statistical analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult {
    /// Descriptive statistics
    pub descriptive: Option<DescriptiveStats>,

    /// Distribution analysis
    pub distribution: Option<DistributionAnalysis>,

    /// Time series specific statistics
    pub timeseries_stats: Option<TimeSeriesStats>,

    /// Stationarity test results
    pub stationarity_tests: HashMap<String, StationarityTest>,

    /// Detected change points
    pub changepoints: Vec<ChangePoint>,

    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Metadata for the statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Number of data points analyzed
    pub n_samples: usize,

    /// Number of missing values
    pub n_missing: usize,

    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Column name analyzed
    pub column_name: String,

    /// Analysis duration in milliseconds
    pub duration_ms: u64,
}

impl StatisticalAnalysisResult {
    /// Create a new empty analysis result
    pub fn new(column_name: String, n_samples: usize) -> Self {
        Self {
            descriptive: None,
            distribution: None,
            timeseries_stats: None,
            stationarity_tests: HashMap::new(),
            changepoints: Vec::new(),
            metadata: AnalysisMetadata {
                n_samples,
                n_missing: 0,
                timestamp: chrono::Utc::now(),
                column_name,
                duration_ms: 0,
            },
        }
    }

    /// Add descriptive statistics to the result
    pub fn with_descriptive(mut self, stats: DescriptiveStats) -> Self {
        self.descriptive = Some(stats);
        self
    }

    /// Add distribution analysis to the result
    pub fn with_distribution(mut self, analysis: DistributionAnalysis) -> Self {
        self.distribution = Some(analysis);
        self
    }

    /// Add time series statistics to the result
    pub fn with_timeseries_stats(mut self, stats: TimeSeriesStats) -> Self {
        self.timeseries_stats = Some(stats);
        self
    }

    /// Add a stationarity test result
    pub fn add_stationarity_test(mut self, test_name: String, test: StationarityTest) -> Self {
        self.stationarity_tests.insert(test_name, test);
        self
    }

    /// Add detected change points
    pub fn with_changepoints(mut self, changepoints: Vec<ChangePoint>) -> Self {
        self.changepoints = changepoints;
        self
    }

    /// Generate a summary report of the analysis
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Statistical Analysis Report\n"));
        report.push_str(&format!("===========================\n\n"));
        report.push_str(&format!("Column: {}\n", self.metadata.column_name));
        report.push_str(&format!("Samples: {} (Missing: {})\n\n",
            self.metadata.n_samples, self.metadata.n_missing));

        // Descriptive statistics summary
        if let Some(ref desc) = self.descriptive {
            report.push_str("Descriptive Statistics:\n");
            report.push_str(&format!("  Mean: {:.4}\n", desc.mean));
            report.push_str(&format!("  Median: {:.4}\n", desc.median));
            report.push_str(&format!("  Std Dev: {:.4}\n", desc.std_dev));
            report.push_str(&format!("  Range: [{:.4}, {:.4}]\n\n", desc.min, desc.max));
        }

        // Distribution analysis summary
        if let Some(ref dist) = self.distribution {
            report.push_str("Distribution:\n");
            report.push_str(&format!("  Skewness: {:.4}\n", dist.skewness));
            report.push_str(&format!("  Kurtosis: {:.4}\n", dist.kurtosis));
            if let Some(ref norm) = dist.normality_test {
                report.push_str(&format!("  Normality (p-value): {:.4}\n\n", norm.p_value));
            }
        }

        // Stationarity tests summary
        if !self.stationarity_tests.is_empty() {
            report.push_str("Stationarity Tests:\n");
            for (name, test) in &self.stationarity_tests {
                report.push_str(&format!("  {}: {} (p={:.4})\n",
                    name,
                    if test.is_stationary { "Stationary" } else { "Non-stationary" },
                    test.p_value
                ));
            }
            report.push_str("\n");
        }

        // Change points summary
        if !self.changepoints.is_empty() {
            report.push_str(&format!("Change Points Detected: {}\n", self.changepoints.len()));
            for (i, cp) in self.changepoints.iter().enumerate().take(5) {
                report.push_str(&format!("  Point {}: Index {} (confidence: {:.2})\n",
                    i + 1, cp.index, cp.confidence));
            }
            if self.changepoints.len() > 5 {
                report.push_str(&format!("  ... and {} more\n", self.changepoints.len() - 5));
            }
        }

        report
    }
}

/// Perform comprehensive statistical analysis on time series data
pub fn analyze_timeseries(
    _timestamps: &[chrono::DateTime<chrono::Utc>],
    values: &[f64],
    column_name: &str,
) -> Result<StatisticalAnalysisResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    // Filter out non-finite values
    let valid_values: Vec<f64> = values.iter()
        .filter(|&&v| v.is_finite())
        .copied()
        .collect();

    let n_missing = values.len() - valid_values.len();

    let mut result = StatisticalAnalysisResult::new(
        column_name.to_string(),
        values.len()
    );
    result.metadata.n_missing = n_missing;

    // Compute descriptive statistics
    if !valid_values.is_empty() {
        let desc_stats = compute_descriptive_stats(&valid_values)?;
        result = result.with_descriptive(desc_stats);

        // Compute distribution analysis
        let dist_analysis = compute_distribution_analysis(&valid_values)?;
        result = result.with_distribution(dist_analysis);

        // Compute time series specific statistics
        let ts_stats = TimeSeriesStats::compute(&valid_values, 50)?;
        result = result.with_timeseries_stats(ts_stats);

        // Perform stationarity tests
        if valid_values.len() >= 20 {  // Minimum samples for meaningful tests
            // ADF test
            if let Ok(adf_test) = test_stationarity(&valid_values, "adf") {
                result = result.add_stationarity_test("ADF".to_string(), adf_test);
            }

            // KPSS test
            if let Ok(kpss_test) = test_stationarity(&valid_values, "kpss") {
                result = result.add_stationarity_test("KPSS".to_string(), kpss_test);
            }
        }

        // Detect change points
        if valid_values.len() >= 10 {
            if let Ok(changepoints) = detect_changepoints(&valid_values, None) {
                result = result.with_changepoints(changepoints);
            }
        }
    }

    result.metadata.duration_ms = start_time.elapsed().as_millis() as u64;

    Ok(result)
}