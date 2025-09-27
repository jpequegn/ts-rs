//! # Configuration Management
//!
//! Comprehensive configuration system for Chronos time series analysis.
//! Supports multiple file formats (TOML, YAML, JSON), profiles, and environment variables.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use figment::{Figment, providers::{Format, Toml, Yaml, Json, Env}};
use merge::Merge;

pub mod defaults;
pub mod validation;
pub mod loader;

/// Main configuration structure containing all settings
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct Config {
    /// Metadata about the configuration
    pub metadata: ConfigMetadata,

    /// Analysis-related configurations
    pub analysis: AnalysisConfig,

    /// Visualization and plotting configurations
    pub visualization: VisualizationConfig,

    /// Output format and export configurations
    pub output: OutputConfig,

    /// Performance and system configurations
    pub performance: PerformanceConfig,

    /// Profile management settings
    pub profiles: ProfilesConfig,
}

/// Configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ConfigMetadata {
    /// Configuration version for migration support
    pub version: String,

    /// Active profile name
    pub active_profile: String,

    /// Configuration file sources (for debugging)
    pub sources: Vec<String>,

    /// Last modified timestamp
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,

    /// User-defined description
    pub description: Option<String>,
}

/// Analysis configurations for all analysis types
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct AnalysisConfig {
    /// Statistical analysis defaults
    pub statistics: StatisticsConfig,

    /// Trend analysis defaults
    pub trend: TrendConfig,

    /// Seasonality analysis defaults
    pub seasonality: SeasonalityConfig,

    /// Anomaly detection defaults
    pub anomaly: AnomalyConfig,

    /// Forecasting defaults
    pub forecasting: ForecastingConfig,

    /// Correlation analysis defaults
    pub correlation: CorrelationConfig,
}

/// Statistical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct StatisticsConfig {
    /// Default confidence level for statistical tests
    pub confidence_level: f64,

    /// Default significance level
    pub significance_level: f64,

    /// Enable normality testing by default
    pub auto_normality_test: bool,

    /// Enable stationarity testing by default
    pub auto_stationarity_test: bool,

    /// Maximum lags for autocorrelation
    pub max_autocorrelation_lags: usize,

    /// Enable changepoint detection by default
    pub auto_changepoint_detection: bool,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct TrendConfig {
    /// Default decomposition method
    pub default_method: String,

    /// Default seasonal period
    pub default_seasonal_period: Option<usize>,

    /// Smoothing parameter for trend extraction
    pub smoothing_alpha: f64,

    /// Enable trend significance testing
    pub auto_significance_test: bool,

    /// Generate plot data by default
    pub generate_plots: bool,
}

/// Seasonality analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct SeasonalityConfig {
    /// Default detection method
    pub default_method: String,

    /// Maximum period to search for seasonality
    pub max_period: usize,

    /// Minimum period to search for seasonality
    pub min_period: usize,

    /// Enable multiple seasonality detection
    pub detect_multiple: bool,

    /// Enable calendar effects detection
    pub detect_calendar_effects: bool,

    /// Default seasonal adjustment method
    pub default_adjustment_method: String,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct AnomalyConfig {
    /// Default detection methods
    pub default_methods: Vec<String>,

    /// Default threshold for z-score method
    pub zscore_threshold: f64,

    /// Default threshold for modified z-score
    pub modified_zscore_threshold: f64,

    /// Default IQR factor
    pub iqr_factor: f64,

    /// Default contamination rate for isolation forest
    pub contamination: f64,

    /// Default scoring method
    pub scoring_method: String,

    /// Default minimum severity to report
    pub min_severity: String,

    /// Maximum anomalies to report
    pub max_anomalies: usize,
}

/// Forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ForecastingConfig {
    /// Default forecasting method
    pub default_method: String,

    /// Default forecast horizon
    pub default_horizon: usize,

    /// Default confidence level for intervals
    pub confidence_level: f64,

    /// Enable prediction intervals by default
    pub include_intervals: bool,

    /// Enable model evaluation by default
    pub auto_evaluation: bool,

    /// Cross-validation folds for evaluation
    pub cv_folds: usize,

    /// ARIMA model parameters
    pub arima: ArimaConfig,

    /// Exponential smoothing parameters
    pub exponential_smoothing: ExponentialSmoothingConfig,
}

/// ARIMA model configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ArimaConfig {
    /// Default AR order
    pub default_p: usize,

    /// Default differencing order
    pub default_d: usize,

    /// Default MA order
    pub default_q: usize,

    /// Maximum model order for auto-selection
    pub max_order: usize,

    /// Use AIC for model selection
    pub use_aic: bool,
}

/// Exponential smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ExponentialSmoothingConfig {
    /// Default alpha parameter
    pub alpha: f64,

    /// Default beta parameter
    pub beta: f64,

    /// Default gamma parameter
    pub gamma: f64,

    /// Default seasonal type
    pub seasonal_type: String,
}

/// Correlation analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct CorrelationConfig {
    /// Default correlation types to compute
    pub default_types: Vec<String>,

    /// Default rolling window size
    pub rolling_window: Option<usize>,

    /// Enable cross-correlation by default
    pub enable_cross_correlation: bool,

    /// Maximum lag for cross-correlation
    pub max_lag: usize,

    /// Default significance level
    pub significance_level: f64,

    /// Enable Granger causality testing
    pub enable_granger_causality: bool,
}

/// Visualization and plotting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct VisualizationConfig {
    /// Default plot theme
    pub default_theme: String,

    /// Default plot dimensions
    pub default_dimensions: (usize, usize),

    /// Default DPI for exports
    pub default_dpi: usize,

    /// Color palette settings
    pub colors: ColorConfig,

    /// Font settings
    pub fonts: FontConfig,

    /// Enable interactive plots by default
    pub interactive_by_default: bool,

    /// Enable grid by default
    pub show_grid: bool,

    /// Default export formats
    pub export_formats: Vec<String>,
}

/// Color configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ColorConfig {
    /// Primary color palette
    pub palette: Vec<String>,

    /// Background color
    pub background: String,

    /// Text color
    pub text: String,

    /// Grid color
    pub grid: String,

    /// Anomaly highlight color
    pub anomaly: String,

    /// Trend line color
    pub trend: String,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct FontConfig {
    /// Default font family
    pub family: String,

    /// Default font size
    pub size: usize,

    /// Title font size
    pub title_size: usize,

    /// Label font size
    pub label_size: usize,

    /// Legend font size
    pub legend_size: usize,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct OutputConfig {
    /// Default output format
    pub default_format: String,

    /// Default output directory
    pub default_directory: Option<PathBuf>,

    /// Include timestamps in filenames
    pub timestamp_filenames: bool,

    /// Compression level for exports
    pub compression_level: usize,

    /// Include metadata in exports
    pub include_metadata: bool,

    /// Export precision for numeric values
    pub numeric_precision: usize,

    /// Auto-open exported files
    pub auto_open: bool,
}

/// Performance and system configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub enable_parallel: bool,

    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,

    /// Memory limit in MB
    pub memory_limit_mb: Option<usize>,

    /// Chunk size for large datasets
    pub chunk_size: usize,

    /// Enable caching
    pub enable_caching: bool,

    /// Cache directory
    pub cache_directory: Option<PathBuf>,

    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,

    /// Progress reporting threshold
    pub progress_threshold: usize,

    /// Enable database integration
    pub enable_database: bool,
}

/// Profile management configuration
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ProfilesConfig {
    /// Available profiles
    pub available: Vec<String>,

    /// Profile-specific configurations
    pub definitions: std::collections::HashMap<String, ProfileDefinition>,

    /// Auto-switch profiles based on data characteristics
    pub auto_switch: bool,

    /// Profile detection rules
    pub detection_rules: ProfileDetectionRules,
}

/// Profile definition
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ProfileDefinition {
    /// Profile name
    pub name: String,

    /// Profile description
    pub description: String,

    /// Domain-specific overrides
    pub overrides: Config,

    /// Recommended data characteristics
    pub data_characteristics: DataCharacteristics,
}

/// Data characteristics for profile matching
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct DataCharacteristics {
    /// Typical frequency (daily, hourly, etc.)
    pub frequency: Option<String>,

    /// Expected seasonality patterns
    pub seasonality_patterns: Vec<usize>,

    /// Data value ranges
    pub value_ranges: Option<(f64, f64)>,

    /// Expected volatility level
    pub volatility_level: Option<String>,

    /// Domain keywords for auto-detection
    pub domain_keywords: Vec<String>,
}

/// Profile detection rules
#[derive(Debug, Clone, Serialize, Deserialize, Merge)]
pub struct ProfileDetectionRules {
    /// Enable automatic profile detection
    pub enabled: bool,

    /// Confidence threshold for auto-switching
    pub confidence_threshold: f64,

    /// Minimum data points required for detection
    pub min_data_points: usize,
}

/// Configuration error types
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    #[error("Invalid configuration format: {0}")]
    InvalidFormat(String),

    #[error("Configuration validation failed: {0}")]
    ValidationFailed(String),

    #[error("Profile not found: {0}")]
    ProfileNotFound(String),

    #[error("Environment variable error: {0}")]
    EnvironmentError(String),

    #[error("Migration error: {0}")]
    MigrationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

// Re-export from submodules
pub use loader::{ConfigLoader, ConfigManager, ConfigFormat};
pub use validation::{ConfigValidator, ValidationResult};