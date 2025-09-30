//! Data Quality Module
//!
//! This module provides comprehensive data quality assessment and validation
//! capabilities for time series data. It includes tools for detecting and
//! handling various data quality issues such as missing values, outliers,
//! inconsistencies, and timeliness problems.
//!
//! # Overview
//!
//! The quality module is organized into several key components:
//!
//! - **Types**: Core data structures for quality assessment results and issues
//! - **Configuration**: Flexible configuration options for quality checks
//! - **Errors**: Comprehensive error handling for quality operations
//!
//! # Usage
//!
//! ```rust,ignore
//! use chronos::quality::{QualityConfig, QualityAssessment};
//! use chronos::TimeSeries;
//!
//! // Create a quality configuration
//! let config = QualityConfig::default()
//!     .with_completeness_threshold(0.95)
//!     .with_cleaning(false);
//!
//! // Assess quality (future implementation)
//! // let assessment = assess_quality(&timeseries, &config)?;
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