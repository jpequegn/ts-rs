//! Error types for data quality assessment

use thiserror::Error;

/// Result type for quality operations
pub type QualityResult<T> = std::result::Result<T, QualityError>;

/// Errors that can occur during data quality assessment
#[derive(Debug, Error)]
pub enum QualityError {
    /// Configuration validation error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Data validation error
    #[error("Data validation error: {0}")]
    Validation(String),

    /// Insufficient data for analysis
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Invalid parameter or argument
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Outlier detection error
    #[error("Outlier detection error: {0}")]
    OutlierDetection(String),

    /// Consistency check error
    #[error("Consistency check error: {0}")]
    ConsistencyCheck(String),

    /// Data cleaning error
    #[error("Data cleaning error: {0}")]
    DataCleaning(String),

    /// Threshold violation
    #[error("Quality threshold violated: {metric} = {value:.4}, expected >= {threshold:.4}")]
    ThresholdViolation {
        metric: String,
        value: f64,
        threshold: f64,
    },

    /// Computation error
    #[error("Computation error: {0}")]
    Computation(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Generic error
    #[error("Quality assessment error: {0}")]
    Generic(String),
}

impl QualityError {
    /// Creates a configuration error
    pub fn configuration(msg: impl Into<String>) -> Self {
        QualityError::Configuration(msg.into())
    }

    /// Creates a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        QualityError::Validation(msg.into())
    }

    /// Creates an insufficient data error
    pub fn insufficient_data(msg: impl Into<String>) -> Self {
        QualityError::InsufficientData(msg.into())
    }

    /// Creates an invalid parameter error
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        QualityError::InvalidParameter(msg.into())
    }

    /// Creates an outlier detection error
    pub fn outlier_detection(msg: impl Into<String>) -> Self {
        QualityError::OutlierDetection(msg.into())
    }

    /// Creates a consistency check error
    pub fn consistency_check(msg: impl Into<String>) -> Self {
        QualityError::ConsistencyCheck(msg.into())
    }

    /// Creates a data cleaning error
    pub fn data_cleaning(msg: impl Into<String>) -> Self {
        QualityError::DataCleaning(msg.into())
    }

    /// Creates a threshold violation error
    pub fn threshold_violation(metric: impl Into<String>, value: f64, threshold: f64) -> Self {
        QualityError::ThresholdViolation {
            metric: metric.into(),
            value,
            threshold,
        }
    }

    /// Creates a computation error
    pub fn computation(msg: impl Into<String>) -> Self {
        QualityError::Computation(msg.into())
    }

    /// Creates a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        QualityError::Serialization(msg.into())
    }

    /// Creates a generic error
    pub fn generic(msg: impl Into<String>) -> Self {
        QualityError::Generic(msg.into())
    }

    /// Returns true if this is a configuration error
    pub fn is_configuration_error(&self) -> bool {
        matches!(self, QualityError::Configuration(_))
    }

    /// Returns true if this is a validation error
    pub fn is_validation_error(&self) -> bool {
        matches!(self, QualityError::Validation(_))
    }

    /// Returns true if this is a threshold violation
    pub fn is_threshold_violation(&self) -> bool {
        matches!(self, QualityError::ThresholdViolation { .. })
    }

    /// Returns the error message
    pub fn message(&self) -> String {
        self.to_string()
    }
}

/// Conversion from TimeSeriesError to QualityError
impl From<crate::TimeSeriesError> for QualityError {
    fn from(err: crate::TimeSeriesError) -> Self {
        match err {
            crate::TimeSeriesError::Validation(msg) => QualityError::Validation(msg),
            crate::TimeSeriesError::DataInconsistency(msg) => QualityError::ConsistencyCheck(msg),
            crate::TimeSeriesError::MissingData(msg) => QualityError::InsufficientData(msg),
            crate::TimeSeriesError::InvalidInput(msg) => QualityError::InvalidParameter(msg),
            other => QualityError::Generic(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = QualityError::configuration("Invalid config");
        assert!(err.is_configuration_error());
        assert!(err.message().contains("Invalid config"));

        let err = QualityError::validation("Invalid data");
        assert!(err.is_validation_error());

        let err = QualityError::threshold_violation("completeness", 0.8, 0.95);
        assert!(err.is_threshold_violation());
        assert!(err.message().contains("0.8"));
        assert!(err.message().contains("0.95"));
    }

    #[test]
    fn test_error_types() {
        let config_err = QualityError::configuration("test");
        assert!(config_err.is_configuration_error());

        let validation_err = QualityError::validation("test");
        assert!(validation_err.is_validation_error());

        let threshold_err = QualityError::threshold_violation("test", 0.5, 0.9);
        assert!(threshold_err.is_threshold_violation());
    }

    #[test]
    fn test_threshold_violation_display() {
        let err = QualityError::threshold_violation("completeness", 0.85, 0.95);
        let msg = format!("{}", err);
        assert!(msg.contains("completeness"));
        assert!(msg.contains("0.8500"));
        assert!(msg.contains("0.9500"));
    }

    #[test]
    fn test_error_conversion_from_timeseries_error() {
        let ts_err = crate::TimeSeriesError::validation("test validation");
        let quality_err: QualityError = ts_err.into();
        assert!(quality_err.is_validation_error());

        let ts_err = crate::TimeSeriesError::missing_data("test missing");
        let quality_err: QualityError = ts_err.into();
        match quality_err {
            QualityError::InsufficientData(_) => {},
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_error_messages() {
        let err = QualityError::insufficient_data("Need at least 10 points");
        assert!(err.message().contains("Need at least 10 points"));

        let err = QualityError::outlier_detection("Z-score failed");
        assert!(err.message().contains("Z-score failed"));

        let err = QualityError::data_cleaning("Cannot clean NaN values");
        assert!(err.message().contains("Cannot clean NaN values"));
    }
}