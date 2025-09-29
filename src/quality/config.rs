//! Configuration types for data quality assessment

use serde::{Deserialize, Serialize};
use crate::quality::types::OutlierMethod;

/// Configuration for data quality assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Minimum acceptable completeness score (0.0-1.0)
    pub completeness_threshold: f64,
    /// Methods to use for outlier detection
    pub outlier_detection_methods: Vec<OutlierMethod>,
    /// Maximum acceptable gap ratio (0.0-1.0)
    pub acceptable_gap_ratio: f64,
    /// Enable automatic data cleaning
    pub enable_cleaning: bool,
    /// Configuration for outlier detection
    pub outlier_config: OutlierDetectionConfig,
    /// Configuration for consistency checks
    pub consistency_config: ConsistencyConfig,
    /// Configuration for timeliness checks
    pub timeliness_config: TimelinessConfig,
    /// Configuration for range validation
    pub range_config: RangeConfig,
}

impl Default for QualityConfig {
    fn default() -> Self {
        QualityConfig {
            completeness_threshold: 0.95,
            outlier_detection_methods: vec![
                OutlierMethod::ZScore,
                OutlierMethod::IQR,
                OutlierMethod::ModifiedZScore,
            ],
            acceptable_gap_ratio: 0.05,
            enable_cleaning: false,
            outlier_config: OutlierDetectionConfig::default(),
            consistency_config: ConsistencyConfig::default(),
            timeliness_config: TimelinessConfig::default(),
            range_config: RangeConfig::default(),
        }
    }
}

impl QualityConfig {
    /// Creates a new QualityConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a strict quality configuration (high thresholds)
    pub fn strict() -> Self {
        QualityConfig {
            completeness_threshold: 0.99,
            outlier_detection_methods: vec![
                OutlierMethod::ZScore,
                OutlierMethod::IQR,
                OutlierMethod::ModifiedZScore,
                OutlierMethod::IsolationForest,
            ],
            acceptable_gap_ratio: 0.01,
            enable_cleaning: false,
            outlier_config: OutlierDetectionConfig::strict(),
            consistency_config: ConsistencyConfig::strict(),
            timeliness_config: TimelinessConfig::strict(),
            range_config: RangeConfig::default(),
        }
    }

    /// Creates a lenient quality configuration (low thresholds)
    pub fn lenient() -> Self {
        QualityConfig {
            completeness_threshold: 0.80,
            outlier_detection_methods: vec![OutlierMethod::ZScore],
            acceptable_gap_ratio: 0.15,
            enable_cleaning: false,
            outlier_config: OutlierDetectionConfig::lenient(),
            consistency_config: ConsistencyConfig::lenient(),
            timeliness_config: TimelinessConfig::lenient(),
            range_config: RangeConfig::default(),
        }
    }

    /// Sets the completeness threshold
    pub fn with_completeness_threshold(mut self, threshold: f64) -> Self {
        self.completeness_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the outlier detection methods
    pub fn with_outlier_methods(mut self, methods: Vec<OutlierMethod>) -> Self {
        self.outlier_detection_methods = methods;
        self
    }

    /// Enables or disables automatic cleaning
    pub fn with_cleaning(mut self, enable: bool) -> Self {
        self.enable_cleaning = enable;
        self
    }

    /// Sets the acceptable gap ratio
    pub fn with_gap_ratio(mut self, ratio: f64) -> Self {
        self.acceptable_gap_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Validates the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.completeness_threshold < 0.0 || self.completeness_threshold > 1.0 {
            return Err("Completeness threshold must be between 0.0 and 1.0".to_string());
        }
        if self.acceptable_gap_ratio < 0.0 || self.acceptable_gap_ratio > 1.0 {
            return Err("Acceptable gap ratio must be between 0.0 and 1.0".to_string());
        }
        if self.outlier_detection_methods.is_empty() {
            return Err("At least one outlier detection method must be specified".to_string());
        }
        Ok(())
    }
}

/// Configuration for outlier detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlierDetectionConfig {
    /// Z-score threshold for outlier detection
    pub zscore_threshold: f64,
    /// IQR multiplier for outlier detection
    pub iqr_multiplier: f64,
    /// Modified Z-score threshold using MAD
    pub modified_zscore_threshold: f64,
    /// Contamination parameter for isolation forest (0.0-0.5)
    pub contamination: f64,
    /// Number of neighbors for LOF algorithm
    pub lof_neighbors: usize,
    /// Epsilon for DBSCAN clustering
    pub dbscan_epsilon: f64,
    /// Minimum samples for DBSCAN
    pub dbscan_min_samples: usize,
}

impl Default for OutlierDetectionConfig {
    fn default() -> Self {
        OutlierDetectionConfig {
            zscore_threshold: 3.0,
            iqr_multiplier: 1.5,
            modified_zscore_threshold: 3.5,
            contamination: 0.1,
            lof_neighbors: 20,
            dbscan_epsilon: 0.5,
            dbscan_min_samples: 5,
        }
    }
}

impl OutlierDetectionConfig {
    /// Creates a strict outlier detection configuration
    pub fn strict() -> Self {
        OutlierDetectionConfig {
            zscore_threshold: 2.5,
            iqr_multiplier: 1.2,
            modified_zscore_threshold: 3.0,
            contamination: 0.05,
            lof_neighbors: 30,
            dbscan_epsilon: 0.3,
            dbscan_min_samples: 10,
        }
    }

    /// Creates a lenient outlier detection configuration
    pub fn lenient() -> Self {
        OutlierDetectionConfig {
            zscore_threshold: 4.0,
            iqr_multiplier: 2.0,
            modified_zscore_threshold: 4.5,
            contamination: 0.15,
            lof_neighbors: 15,
            dbscan_epsilon: 0.8,
            dbscan_min_samples: 3,
        }
    }
}

/// Configuration for consistency checks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Check for duplicate timestamps
    pub check_duplicates: bool,
    /// Check for monotonicity in timestamps
    pub check_monotonicity: bool,
    /// Tolerance for timestamp comparison (milliseconds)
    pub timestamp_tolerance_ms: i64,
    /// Check for value consistency (e.g., running totals should be non-decreasing)
    pub check_value_consistency: bool,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        ConsistencyConfig {
            check_duplicates: true,
            check_monotonicity: true,
            timestamp_tolerance_ms: 0,
            check_value_consistency: false,
        }
    }
}

impl ConsistencyConfig {
    /// Creates a strict consistency configuration
    pub fn strict() -> Self {
        ConsistencyConfig {
            check_duplicates: true,
            check_monotonicity: true,
            timestamp_tolerance_ms: 0,
            check_value_consistency: true,
        }
    }

    /// Creates a lenient consistency configuration
    pub fn lenient() -> Self {
        ConsistencyConfig {
            check_duplicates: false,
            check_monotonicity: false,
            timestamp_tolerance_ms: 1000,
            check_value_consistency: false,
        }
    }
}

/// Configuration for timeliness checks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelinessConfig {
    /// Expected frequency in seconds (None for auto-detection)
    pub expected_frequency_seconds: Option<f64>,
    /// Tolerance for frequency deviation (0.0-1.0)
    pub frequency_tolerance: f64,
    /// Maximum acceptable gap in seconds
    pub max_gap_seconds: Option<i64>,
    /// Check for irregular spacing
    pub check_irregular_spacing: bool,
}

impl Default for TimelinessConfig {
    fn default() -> Self {
        TimelinessConfig {
            expected_frequency_seconds: None,
            frequency_tolerance: 0.1,
            max_gap_seconds: None,
            check_irregular_spacing: true,
        }
    }
}

impl TimelinessConfig {
    /// Creates a strict timeliness configuration
    pub fn strict() -> Self {
        TimelinessConfig {
            expected_frequency_seconds: None,
            frequency_tolerance: 0.05,
            max_gap_seconds: None,
            check_irregular_spacing: true,
        }
    }

    /// Creates a lenient timeliness configuration
    pub fn lenient() -> Self {
        TimelinessConfig {
            expected_frequency_seconds: None,
            frequency_tolerance: 0.25,
            max_gap_seconds: None,
            check_irregular_spacing: false,
        }
    }
}

/// Configuration for range validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RangeConfig {
    /// Minimum expected value (None for no minimum)
    pub min_value: Option<f64>,
    /// Maximum expected value (None for no maximum)
    pub max_value: Option<f64>,
    /// Check for negative values
    pub allow_negative: bool,
    /// Check for zero values
    pub allow_zero: bool,
    /// Check for infinite values
    pub allow_infinite: bool,
    /// Check for NaN values
    pub allow_nan: bool,
}

impl Default for RangeConfig {
    fn default() -> Self {
        RangeConfig {
            min_value: None,
            max_value: None,
            allow_negative: true,
            allow_zero: true,
            allow_infinite: false,
            allow_nan: false,
        }
    }
}

impl RangeConfig {
    /// Creates a configuration for non-negative values only
    pub fn non_negative() -> Self {
        RangeConfig {
            min_value: Some(0.0),
            max_value: None,
            allow_negative: false,
            allow_zero: true,
            allow_infinite: false,
            allow_nan: false,
        }
    }

    /// Creates a configuration for positive values only
    pub fn positive_only() -> Self {
        RangeConfig {
            min_value: Some(0.0),
            max_value: None,
            allow_negative: false,
            allow_zero: false,
            allow_infinite: false,
            allow_nan: false,
        }
    }

    /// Creates a configuration with a specific range
    pub fn with_range(min: f64, max: f64) -> Self {
        RangeConfig {
            min_value: Some(min),
            max_value: Some(max),
            allow_negative: min < 0.0,
            allow_zero: min <= 0.0 && max >= 0.0,
            allow_infinite: false,
            allow_nan: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_config_default() {
        let config = QualityConfig::default();
        assert_eq!(config.completeness_threshold, 0.95);
        assert_eq!(config.outlier_detection_methods.len(), 3);
        assert!(!config.enable_cleaning);
    }

    #[test]
    fn test_quality_config_strict() {
        let config = QualityConfig::strict();
        assert_eq!(config.completeness_threshold, 0.99);
        assert_eq!(config.outlier_detection_methods.len(), 4);
        assert_eq!(config.acceptable_gap_ratio, 0.01);
    }

    #[test]
    fn test_quality_config_lenient() {
        let config = QualityConfig::lenient();
        assert_eq!(config.completeness_threshold, 0.80);
        assert_eq!(config.outlier_detection_methods.len(), 1);
        assert_eq!(config.acceptable_gap_ratio, 0.15);
    }

    #[test]
    fn test_quality_config_builder() {
        let config = QualityConfig::new()
            .with_completeness_threshold(0.9)
            .with_gap_ratio(0.1)
            .with_cleaning(true);

        assert_eq!(config.completeness_threshold, 0.9);
        assert_eq!(config.acceptable_gap_ratio, 0.1);
        assert!(config.enable_cleaning);
    }

    #[test]
    fn test_quality_config_validation() {
        let mut config = QualityConfig::default();
        assert!(config.validate().is_ok());

        config.completeness_threshold = 1.5;
        assert!(config.validate().is_err());

        config.completeness_threshold = 0.95;
        config.outlier_detection_methods.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_outlier_config_variants() {
        let default = OutlierDetectionConfig::default();
        let strict = OutlierDetectionConfig::strict();
        let lenient = OutlierDetectionConfig::lenient();

        assert!(strict.zscore_threshold < default.zscore_threshold);
        assert!(lenient.zscore_threshold > default.zscore_threshold);
    }

    #[test]
    fn test_consistency_config_variants() {
        let default = ConsistencyConfig::default();
        let strict = ConsistencyConfig::strict();
        let lenient = ConsistencyConfig::lenient();

        assert!(default.check_duplicates);
        assert!(strict.check_value_consistency);
        assert!(!lenient.check_duplicates);
    }

    #[test]
    fn test_timeliness_config_variants() {
        let default = TimelinessConfig::default();
        let strict = TimelinessConfig::strict();
        let lenient = TimelinessConfig::lenient();

        assert!(strict.frequency_tolerance < default.frequency_tolerance);
        assert!(lenient.frequency_tolerance > default.frequency_tolerance);
    }

    #[test]
    fn test_range_config_variants() {
        let non_negative = RangeConfig::non_negative();
        assert_eq!(non_negative.min_value, Some(0.0));
        assert!(!non_negative.allow_negative);
        assert!(non_negative.allow_zero);

        let positive = RangeConfig::positive_only();
        assert_eq!(positive.min_value, Some(0.0));
        assert!(!positive.allow_negative);
        assert!(!positive.allow_zero);

        let range = RangeConfig::with_range(-10.0, 10.0);
        assert_eq!(range.min_value, Some(-10.0));
        assert_eq!(range.max_value, Some(10.0));
        assert!(range.allow_negative);
    }
}