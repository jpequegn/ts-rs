//! Data Quality Module
//!
//! Comprehensive tools for assessing, monitoring, and improving the quality of time series data.
//!
//! # Overview
//!
//! The data quality module provides a complete suite of functionality for managing time series
//! data quality across five key dimensions:
//!
//! - **Completeness** (0-100): Presence of all expected data points
//! - **Consistency** (0-100): Data follows expected patterns and rules
//! - **Validity** (0-100): Data values are within expected ranges
//! - **Timeliness** (0-100): Data arrives at expected frequencies
//! - **Accuracy** (0-100): Data correctness and statistical soundness
//!
//! # Features
//!
//! - **Quality Assessment**: Multi-dimensional scoring with automated recommendations
//! - **Data Profiling**: Comprehensive analysis of data characteristics
//! - **Outlier Detection**: Multiple detection methods (Z-score, IQR, Modified Z-score, Temporal, Ensemble)
//! - **Data Cleaning**: Smart imputation, outlier correction, and noise reduction
//! - **Quality Monitoring**: Continuous tracking with degradation detection and alerting
//!
//! # Quick Start
//!
//! ## Basic Quality Assessment
//!
//! ```rust,no_run
//! use chronos::quality::*;
//! use chronos::TimeSeries;
//!
//! # fn main() -> Result<(), QualityError> {
//! // Load time series data
//! let data = TimeSeries::from_csv("data.csv")?;
//!
//! // Assess quality with default configuration
//! let config = QualityConfig::default();
//! let assessment = assess_quality(&data, &config)?;
//!
//! println!("Overall quality: {:.1}/100", assessment.overall_score);
//! println!("Completeness: {:.1}/100", assessment.dimension_scores.completeness);
//! println!("Issues found: {}", assessment.quality_issues.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Data Cleaning
//!
//! ```rust,no_run
//! use chronos::quality::*;
//! use chronos::TimeSeries;
//!
//! # fn main() -> Result<(), QualityError> {
//! let data = TimeSeries::from_csv("data.csv")?;
//!
//! // Clean data with conservative settings
//! let config = CleaningConfig::conservative();
//! let result = clean_timeseries(&data, &config)?;
//!
//! println!("Gaps filled: {}", result.cleaning_report.gaps_filled);
//! println!("Outliers corrected: {}", result.cleaning_report.outliers_corrected);
//! # Ok(())
//! # }
//! ```
//!
//! ## Quality Monitoring
//!
//! ```rust,no_run
//! use chronos::quality::*;
//! use chronos::TimeSeries;
//!
//! # fn main() -> Result<(), QualityError> {
//! // Set up continuous monitoring
//! let config = MonitoringConfig::default();
//! let mut tracker = QualityTracker::new(config);
//!
//! // Track quality over time
//! let data = TimeSeries::from_csv("data.csv")?;
//! let assessment = assess_quality(&data, &QualityConfig::default())?;
//! tracker.track_quality_metrics(&assessment)?;
//!
//! // Check for degradation
//! let trend = tracker.detect_quality_degradation(&QualityThresholds::default());
//! # Ok(())
//! # }
//! ```
//!
//! # Quality Dimensions
//!
//! The module assesses data quality across four key dimensions:
//!
//! ## 1. Completeness
//! - Missing data detection
//! - Gap identification and analysis
//! - Coverage metrics
//!
//! ## 2. Validity
//! - Outlier detection using multiple methods
//! - Range validation
//! - Value type checking
//!
//! ## 3. Consistency
//! - Duplicate detection
//! - Temporal consistency
//! - Value consistency checks
//!
//! ## 4. Timeliness
//! - Frequency analysis
//! - Irregular spacing detection
//! - Gap analysis
//!
//! # Configuration Presets
//!
//! The module provides several preset configurations:
//!
//! ```rust
//! use chronos::quality::QualityConfig;
//!
//! // Strict quality requirements (high thresholds)
//! let strict = QualityConfig::strict();
//!
//! // Lenient quality requirements (low thresholds)
//! let lenient = QualityConfig::lenient();
//!
//! // Default balanced configuration
//! let default = QualityConfig::default();
//! ```
//!
//! # Future Capabilities
//!
//! The following features are planned for future releases:
//!
//! - Data profiling and statistics
//! - Advanced outlier detection algorithms
//! - Automated data cleaning and imputation
//! - Quality scoring and reporting
//! - Integration with the main TimeSeries type

pub mod types;
pub mod config;
pub mod errors;
pub mod profiling;
pub mod outlier_detection;
pub mod scoring;
pub mod cleaning;
pub mod monitoring;

// Re-export commonly used types for convenience
pub use types::{
    QualityAssessment, QualityIssue, QualityMetrics, OutlierMethod, TimeRange,
};

pub use config::{
    QualityConfig, OutlierDetectionConfig, ConsistencyConfig,
    TimelinessConfig, RangeConfig,
};

pub use errors::{QualityError, QualityResult};

pub use profiling::{
    profile_timeseries, analyze_completeness, generate_statistical_profile,
    analyze_temporal_coverage, DataProfile, CompletenessReport, TemporalCoverage,
    StatisticalProfile, ProfilingConfig, DataGap, QualityIndicators,
    ValueRange, TrendIndicators,
};

pub use outlier_detection::{
    detect_outliers, detect_zscore_outliers, detect_iqr_outliers,
    detect_modified_zscore_outliers, detect_temporal_outliers,
    detect_ensemble_outliers,
    OutlierSeverity, OutlierContext, OutlierPoint, OutlierSummary,
    OutlierReport, OutlierConfig,
};

pub use scoring::{
    assess_quality, calculate_completeness_score, calculate_consistency_score,
    calculate_validity_score, calculate_timeliness_score, calculate_accuracy_score,
    calculate_overall_score, generate_recommendations,
    DimensionScores, QualityWeights, QualityBenchmarks, Priority,
    ImputationMethod, QualityRecommendation, QualityProfile,
    EnhancedQualityAssessment,
};

pub use cleaning::{
    fill_gaps, correct_outliers, reduce_noise, clean_timeseries,
    OutlierCorrection, NoiseReduction,
    ModificationOperation, DataModification, QualityImpact,
    CleaningReport, CleaningResult, CleaningConfig, GapConfig,
};

pub use monitoring::{
    monitor_quality_over_time, detect_quality_degradation, track_quality_metrics,
    QualityTracker, QualityTrend, QualityAlert, QualityDataPoint, QualityTimeSeries,
    TrendDirection, AlertType, AlertSeverity, MonitoringConfig, QualityThresholds,
    QualityThresholdConfig, QualityBaseline, NotificationChannel,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that all public types are accessible
        let _config = QualityConfig::default();
        let _metrics = QualityMetrics::default();

        // Test configuration presets
        let _strict = QualityConfig::strict();
        let _lenient = QualityConfig::lenient();
    }

    #[test]
    fn test_error_creation() {
        let err = QualityError::configuration("test");
        assert!(err.is_configuration_error());
    }

    #[test]
    fn test_outlier_methods() {
        let methods = vec![
            OutlierMethod::ZScore,
            OutlierMethod::IQR,
            OutlierMethod::ModifiedZScore,
        ];

        assert_eq!(methods.len(), 3);
        assert_eq!(OutlierMethod::ZScore.as_str(), "zscore");
    }
}