//! # Chronos Time Series Analysis Library
//!
//! Core data structures and functionality for time series analysis,
//! providing robust, efficient data handling and analysis capabilities.

pub mod types;
pub mod timeseries;
pub mod analysis;
pub mod validation;
pub mod import;
pub mod preprocessing;

// Re-export commonly used types
pub use types::{Frequency, MissingValuePolicy};
pub use timeseries::TimeSeries;
pub use analysis::{AnalysisResult, TrendAnalysis, SeasonalAnalysis, AnomalyDetection};
pub use import::{ImportConfig, ImportResult, ImportStats, import_from_file, import_csv, import_json};
pub use preprocessing::{PreprocessingConfig, PreprocessingResult, preprocess_timeseries};

/// Result type used throughout the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Error types for the time series library
#[derive(Debug, thiserror::Error)]
pub enum TimeSeriesError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Data inconsistency: {0}")]
    DataInconsistency(String),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error("Missing data: {0}")]
    MissingData(String),

    #[error("Analysis error: {0}")]
    Analysis(String),
}

impl TimeSeriesError {
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    pub fn data_inconsistency(msg: impl Into<String>) -> Self {
        Self::DataInconsistency(msg.into())
    }

    pub fn invalid_timestamp(msg: impl Into<String>) -> Self {
        Self::InvalidTimestamp(msg.into())
    }

    pub fn missing_data(msg: impl Into<String>) -> Self {
        Self::MissingData(msg.into())
    }

    pub fn analysis(msg: impl Into<String>) -> Self {
        Self::Analysis(msg.into())
    }
}