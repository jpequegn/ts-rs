//! # Trend Analysis and Decomposition Module
//!
//! Provides comprehensive trend analysis including decomposition, trend fitting,
//! change detection, and detrending methods for time series data.

pub mod decomposition;
pub mod detection;
pub mod detrending;
pub mod analysis;
pub mod plotting;

// Re-export commonly used types and functions
pub use crate::analysis::TrendDirection;
pub use decomposition::{
    DecompositionMethod, DecompositionResult, ClassicalDecomposition,
    StlDecomposition, perform_decomposition
};
pub use detection::{
    TrendTest, TrendTestResult, MannKendallTest, SensSlope, PettittTest,
    detect_trend, test_trend_significance
};
pub use detrending::{
    DetrendingMethod, DetrendingResult, linear_detrend, difference_detrend,
    moving_average_detrend, hp_filter_detrend, perform_detrending
};
pub use analysis::{
    TrendAnalysisConfig, TrendStrength,
    RateOfChangeAnalysis, BreakpointDetection, analyze_trend_comprehensive,
    classify_trend_direction, compute_trend_strength
};
pub use plotting::{
    TrendPlotData, DecompositionPlotData, generate_trend_plot_data,
    generate_decomposition_plot_data
};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Comprehensive trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveTrendAnalysis {
    /// Overall trend direction and strength
    pub trend_summary: TrendSummary,

    /// Decomposition results if performed
    pub decomposition: Option<DecompositionResult>,

    /// Trend test results
    pub trend_tests: HashMap<String, TrendTestResult>,

    /// Detrending results if performed
    pub detrended_series: Option<DetrendingResult>,

    /// Rate of change analysis
    pub rate_of_change: Option<RateOfChangeAnalysis>,

    /// Detected breakpoints/change points
    pub breakpoints: Vec<BreakpointDetection>,

    /// Plot data for visualization
    pub plot_data: Option<TrendPlotData>,

    /// Analysis metadata
    pub metadata: TrendAnalysisMetadata,
}

/// Summary of trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSummary {
    /// Overall trend direction
    pub direction: TrendDirection,

    /// Trend strength (0.0 to 1.0)
    pub strength: f64,

    /// Statistical significance (p-value)
    pub p_value: Option<f64>,

    /// Confidence level for the assessment
    pub confidence: f64,

    /// Annual growth rate (if applicable)
    pub growth_rate: Option<f64>,

    /// Seasonal strength (0.0 to 1.0)
    pub seasonal_strength: Option<f64>,

    /// Component contributions
    pub component_contributions: Option<ComponentContributions>,
}

/// Contribution of different components to the time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentContributions {
    /// Trend component contribution (percentage)
    pub trend_percentage: f64,

    /// Seasonal component contribution (percentage)
    pub seasonal_percentage: f64,

    /// Residual/noise component contribution (percentage)
    pub residual_percentage: f64,
}

/// Metadata for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisMetadata {
    /// Number of data points analyzed
    pub n_observations: usize,

    /// Time span of the data
    pub time_span: Option<chrono::Duration>,

    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,

    /// Methods used in the analysis
    pub methods_used: Vec<String>,

    /// Analysis duration in milliseconds
    pub duration_ms: u64,

    /// Data quality indicators
    pub data_quality: DataQualityIndicators,
}

/// Data quality indicators relevant to trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityIndicators {
    /// Percentage of missing values
    pub missing_percentage: f64,

    /// Number of gaps in the time series
    pub gap_count: usize,

    /// Regularity of time intervals
    pub regularity_score: f64,

    /// Presence of outliers that might affect trend
    pub outlier_count: usize,
}

impl ComprehensiveTrendAnalysis {
    /// Create a new empty trend analysis result
    pub fn new(n_observations: usize) -> Self {
        Self {
            trend_summary: TrendSummary {
                direction: TrendDirection::Stable,
                strength: 0.0,
                p_value: None,
                confidence: 0.0,
                growth_rate: None,
                seasonal_strength: None,
                component_contributions: None,
            },
            decomposition: None,
            trend_tests: HashMap::new(),
            detrended_series: None,
            rate_of_change: None,
            breakpoints: Vec::new(),
            plot_data: None,
            metadata: TrendAnalysisMetadata {
                n_observations,
                time_span: None,
                timestamp: Utc::now(),
                methods_used: Vec::new(),
                duration_ms: 0,
                data_quality: DataQualityIndicators {
                    missing_percentage: 0.0,
                    gap_count: 0,
                    regularity_score: 1.0,
                    outlier_count: 0,
                },
            },
        }
    }

    /// Generate a comprehensive summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("Trend Analysis Results\n");
        report.push_str("======================\n\n");

        // Overall trend
        let trend_symbol = match self.trend_summary.direction {
            TrendDirection::StronglyIncreasing => "↗↗",
            TrendDirection::Increasing => "↗",
            TrendDirection::Stable => "→",
            TrendDirection::Decreasing => "↘",
            TrendDirection::StronglyDecreasing => "↘↘",
            TrendDirection::Inconclusive => "?",
        };

        report.push_str(&format!(
            "Overall Trend: {} {} (p = {:.3})\n",
            trend_symbol,
            format!("{:?}", self.trend_summary.direction),
            self.trend_summary.p_value.unwrap_or(1.0)
        ));

        report.push_str(&format!(
            "Trend Strength: {:.2} ({})\n",
            self.trend_summary.strength,
            strength_description(self.trend_summary.strength)
        ));

        if let Some(growth_rate) = self.trend_summary.growth_rate {
            report.push_str(&format!("Annual Growth Rate: {:.1}%\n", growth_rate * 100.0));
        }

        if let Some(seasonal_strength) = self.trend_summary.seasonal_strength {
            report.push_str(&format!(
                "Seasonal Strength: {:.2} ({})\n",
                seasonal_strength,
                strength_description(seasonal_strength)
            ));
        }

        // Component contributions
        if let Some(ref components) = self.trend_summary.component_contributions {
            report.push_str("\nComponents:\n");
            report.push_str(&format!("- Trend: {:.0}% of variation\n", components.trend_percentage));
            report.push_str(&format!("- Seasonal: {:.0}% of variation\n", components.seasonal_percentage));
            report.push_str(&format!("- Residual: {:.0}% of variation\n", components.residual_percentage));
        }

        // Change points/breakpoints
        if !self.breakpoints.is_empty() {
            report.push_str("\nChange Points Detected:\n");
            for (i, breakpoint) in self.breakpoints.iter().enumerate().take(5) {
                report.push_str(&format!(
                    "- Point {}: Index {} (confidence: {:.0}%)\n",
                    i + 1,
                    breakpoint.index,
                    breakpoint.confidence * 100.0
                ));
            }
            if self.breakpoints.len() > 5 {
                report.push_str(&format!("... and {} more\n", self.breakpoints.len() - 5));
            }
        }

        // Data quality summary
        report.push_str(&format!(
            "\nData Quality:\n- {} observations\n- {:.1}% missing values\n- {} gaps detected\n",
            self.metadata.n_observations,
            self.metadata.data_quality.missing_percentage,
            self.metadata.data_quality.gap_count
        ));

        report.push_str(&format!(
            "\nAnalysis completed in {} ms\n",
            self.metadata.duration_ms
        ));

        report
    }
}

/// Perform comprehensive trend analysis on time series data
pub fn analyze_comprehensive(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    config: Option<TrendAnalysisConfig>,
) -> Result<ComprehensiveTrendAnalysis, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let config = config.unwrap_or_default();

    // Filter out non-finite values
    let valid_data: Vec<(DateTime<Utc>, f64)> = timestamps
        .iter()
        .zip(values.iter())
        .filter(|(_, &v)| v.is_finite())
        .map(|(&t, &v)| (t, v))
        .collect();

    if valid_data.len() < 10 {
        return Err("Need at least 10 valid observations for trend analysis".into());
    }

    let valid_timestamps: Vec<DateTime<Utc>> = valid_data.iter().map(|(t, _)| *t).collect();
    let valid_values: Vec<f64> = valid_data.iter().map(|(_, v)| *v).collect();

    let mut result = ComprehensiveTrendAnalysis::new(valid_values.len());

    // Update metadata
    result.metadata.time_span = if valid_timestamps.len() > 1 {
        Some(valid_timestamps[valid_timestamps.len() - 1] - valid_timestamps[0])
    } else {
        None
    };

    result.metadata.data_quality.missing_percentage =
        (values.len() - valid_values.len()) as f64 / values.len() as f64 * 100.0;

    // Perform comprehensive trend analysis
    let comprehensive_analysis = analyze_trend_comprehensive(&valid_values, &config)?;
    result.trend_summary.direction = comprehensive_analysis.direction;
    result.trend_summary.strength = comprehensive_analysis.strength;
    result.trend_summary.confidence = comprehensive_analysis.confidence;
    result.trend_summary.growth_rate = comprehensive_analysis.growth_rate;
    result.metadata.methods_used.push("comprehensive_analysis".to_string());

    // Perform trend tests
    if config.perform_mann_kendall {
        let mk_test = detect_trend(&valid_values, "mann_kendall")?;
        result.trend_summary.p_value = Some(mk_test.p_value);
        result.trend_tests.insert("Mann-Kendall".to_string(), mk_test);
        result.metadata.methods_used.push("mann_kendall".to_string());
    }

    // Perform decomposition if requested
    if let Some(decomp_method) = config.decomposition_method {
        let decomposition = perform_decomposition(&valid_values, decomp_method, config.seasonal_period)?;

        // Calculate component contributions
        if let Some(ref trend) = decomposition.trend {
            let trend_var = calculate_variance(trend);
            let seasonal_var = decomposition.seasonal.as_ref()
                .map(|s| calculate_variance(s)).unwrap_or(0.0);
            let residual_var = calculate_variance(&decomposition.residual);

            let total_var = trend_var + seasonal_var + residual_var;
            if total_var > 0.0 {
                result.trend_summary.component_contributions = Some(ComponentContributions {
                    trend_percentage: trend_var / total_var * 100.0,
                    seasonal_percentage: seasonal_var / total_var * 100.0,
                    residual_percentage: residual_var / total_var * 100.0,
                });

                result.trend_summary.seasonal_strength = Some(seasonal_var / total_var);
            }
        }

        result.decomposition = Some(decomposition);
        result.metadata.methods_used.push(format!("{:?}", decomp_method));
    }

    // Generate plot data if requested
    if config.generate_plot_data {
        result.plot_data = Some(generate_trend_plot_data(&valid_timestamps, &valid_values)?);
        result.metadata.methods_used.push("plot_data_generation".to_string());
    }

    result.metadata.duration_ms = start_time.elapsed().as_millis() as u64;

    Ok(result)
}

// Helper functions
fn strength_description(strength: f64) -> &'static str {
    if strength >= 0.8 {
        "Very Strong"
    } else if strength >= 0.6 {
        "Strong"
    } else if strength >= 0.4 {
        "Moderate"
    } else if strength >= 0.2 {
        "Weak"
    } else {
        "Very Weak"
    }
}

fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trend_analysis_creation() {
        let analysis = ComprehensiveTrendAnalysis::new(100);
        assert_eq!(analysis.metadata.n_observations, 100);
        assert_eq!(analysis.trend_summary.direction, TrendDirection::Stable);
    }

    #[test]
    fn test_component_contributions() {
        let contributions = ComponentContributions {
            trend_percentage: 60.0,
            seasonal_percentage: 30.0,
            residual_percentage: 10.0,
        };

        let total = contributions.trend_percentage +
                   contributions.seasonal_percentage +
                   contributions.residual_percentage;

        assert!((total - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_strength_description() {
        assert_eq!(strength_description(0.9), "Very Strong");
        assert_eq!(strength_description(0.7), "Strong");
        assert_eq!(strength_description(0.5), "Moderate");
        assert_eq!(strength_description(0.3), "Weak");
        assert_eq!(strength_description(0.1), "Very Weak");
    }
}