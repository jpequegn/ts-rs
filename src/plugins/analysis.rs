//! # Analysis Plugin Interface
//!
//! Defines the interface for analysis plugins that can perform custom statistical methods
//! and domain-specific algorithms on time series data.

use super::{PluginError, PluginResult, PluginContext};
use crate::TimeSeries;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Analysis plugin trait
pub trait AnalysisPlugin: Send + Sync {
    /// Get analysis methods supported by this plugin
    fn supported_methods(&self) -> Vec<AnalysisMethod>;

    /// Get analysis categories supported by this plugin
    fn supported_categories(&self) -> Vec<AnalysisCategory>;

    /// Validate analysis configuration
    fn validate_config(&self, config: &AnalysisConfig) -> PluginResult<()>;

    /// Perform analysis on time series data
    fn analyze(
        &self,
        data: &[TimeSeries],
        config: &AnalysisConfig,
        context: &PluginContext,
    ) -> PluginResult<AnalysisResult>;

    /// Get required parameters for a specific analysis method
    fn get_required_parameters(&self, method: &str) -> Vec<ParameterDefinition>;

    /// Get optional parameters for a specific analysis method
    fn get_optional_parameters(&self, method: &str) -> Vec<ParameterDefinition>;

    /// Check if the plugin supports batch processing
    fn supports_batch_processing(&self) -> bool {
        false
    }

    /// Check if the plugin supports incremental processing
    fn supports_incremental_processing(&self) -> bool {
        false
    }

    /// Perform batch analysis on multiple datasets
    fn analyze_batch(
        &self,
        _datasets: &[Vec<TimeSeries>],
        _configs: &[AnalysisConfig],
        _context: &PluginContext,
    ) -> PluginResult<Vec<AnalysisResult>> {
        Err(PluginError::ExecutionError("Batch processing not supported".to_string()))
    }

    /// Start incremental analysis
    fn start_incremental_analysis(
        &self,
        _config: &AnalysisConfig,
        _context: &PluginContext,
    ) -> PluginResult<Box<dyn IncrementalAnalyzer>> {
        Err(PluginError::ExecutionError("Incremental processing not supported".to_string()))
    }

    /// Get analysis capabilities and constraints
    fn get_capabilities(&self) -> AnalysisCapabilities;

    /// Estimate computational complexity for given data size
    fn estimate_complexity(&self, data_size: usize, method: &str) -> ComplexityEstimate;
}

/// Analysis method specification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnalysisMethod {
    /// Method identifier
    pub id: String,
    /// Method name
    pub name: String,
    /// Method description
    pub description: String,
    /// Method category
    pub category: AnalysisCategory,
    /// Method version
    pub version: String,
    /// Whether method is experimental
    pub experimental: bool,
}

/// Analysis categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisCategory {
    /// Statistical analysis methods
    Statistical,
    /// Time series decomposition
    Decomposition,
    /// Forecasting methods
    Forecasting,
    /// Anomaly detection
    AnomalyDetection,
    /// Correlation analysis
    Correlation,
    /// Spectral analysis
    Spectral,
    /// Machine learning methods
    MachineLearning,
    /// Signal processing
    SignalProcessing,
    /// Custom domain-specific methods
    DomainSpecific(String),
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Analysis method to use
    pub method: String,
    /// Method parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Input data configuration
    pub input: InputConfig,
    /// Output configuration
    pub output: OutputConfig,
    /// Performance configuration
    pub performance: Option<PerformanceConfig>,
    /// Validation configuration
    pub validation: Option<ValidationConfig>,
}

/// Input configuration for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Columns to analyze
    pub columns: Vec<String>,
    /// Data preprocessing options
    pub preprocessing: Option<PreprocessingConfig>,
    /// Data filtering options
    pub filter: Option<DataFilter>,
    /// Window configuration for windowed analysis
    pub window: Option<WindowConfig>,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Missing value handling
    pub missing_values: MissingValueHandling,
    /// Outlier handling
    pub outliers: OutlierHandling,
    /// Normalization method
    pub normalization: Option<NormalizationMethod>,
    /// Smoothing configuration
    pub smoothing: Option<SmoothingConfig>,
}

/// Missing value handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueHandling {
    /// Drop rows with missing values
    Drop,
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// Linear interpolation
    LinearInterpolation,
    /// Spline interpolation
    SplineInterpolation,
    /// Use constant value
    Constant(f64),
    /// Use mean value
    Mean,
    /// Use median value
    Median,
}

/// Outlier handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierHandling {
    /// Keep all values
    Keep,
    /// Remove outliers
    Remove,
    /// Cap outliers to percentile values
    Cap { lower: f64, upper: f64 },
    /// Transform outliers
    Transform(TransformMethod),
}

/// Transform methods for outliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformMethod {
    Log,
    Sqrt,
    BoxCox(f64),
    YeoJohnson,
}

/// Normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Z-score normalization
    ZScore,
    /// Min-max normalization
    MinMax { min: f64, max: f64 },
    /// Robust scaling
    Robust,
    /// Unit vector scaling
    UnitVector,
}

/// Smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingConfig {
    /// Smoothing method
    pub method: SmoothingMethod,
    /// Window size for smoothing
    pub window_size: usize,
    /// Smoothing parameters
    pub parameters: HashMap<String, f64>,
}

/// Smoothing methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothingMethod {
    MovingAverage,
    ExponentialSmoothing,
    Lowess,
    SavitzkyGolay,
    Kalman,
}

/// Data filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFilter {
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Value range filters
    pub value_ranges: HashMap<String, ValueRange>,
    /// Custom filter expressions
    pub expressions: Vec<String>,
}

/// Date range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}

/// Value range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueRange {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub inclusive: bool,
}

/// Window configuration for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window type
    pub window_type: WindowType,
    /// Window size
    pub size: usize,
    /// Window step size
    pub step: Option<usize>,
    /// Window overlap
    pub overlap: Option<f64>,
}

/// Window types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    /// Fixed size sliding window
    Sliding,
    /// Tumbling window (non-overlapping)
    Tumbling,
    /// Session windows
    Session { timeout: std::time::Duration },
    /// Custom window function
    Custom(String),
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,
    /// Include confidence intervals
    pub include_confidence: bool,
    /// Confidence level
    pub confidence_level: Option<f64>,
    /// Include diagnostics
    pub include_diagnostics: bool,
    /// Include intermediate results
    pub include_intermediate: bool,
    /// Precision for numerical results
    pub precision: Option<usize>,
}

/// Output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Structured result object
    Structured,
    /// Time series format
    TimeSeries,
    /// Statistical summary
    Summary,
    /// Full detailed report
    Detailed,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Maximum execution time
    pub max_duration: Option<std::time::Duration>,
    /// Number of parallel threads
    pub threads: Option<usize>,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Chunk size for batch processing
    pub chunk_size: Option<usize>,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Cross-validation method
    pub cross_validation: Option<CrossValidationMethod>,
    /// Validation metrics to compute
    pub metrics: Vec<ValidationMetric>,
    /// Validation data split ratio
    pub validation_split: Option<f64>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Cross-validation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossValidationMethod {
    /// K-fold cross-validation
    KFold { k: usize },
    /// Time series cross-validation
    TimeSeriesSplit { test_size: usize },
    /// Walk-forward validation
    WalkForward { initial_window: usize, step: usize },
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMetric {
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
    R2Score,
    AdjustedR2Score,
    AIC,
    BIC,
    Custom(String),
}

/// Parameter definition for analysis methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Valid value range or options
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    String,
    Array(Box<ParameterType>),
    Object,
    Enum(Vec<String>),
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraints {
    /// Numeric range
    Range { min: f64, max: f64 },
    /// String length constraints
    Length { min: usize, max: usize },
    /// Allowed values
    Values(Vec<serde_json::Value>),
    /// Regular expression pattern
    Pattern(String),
}

/// Analysis capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCapabilities {
    /// Maximum data size that can be processed
    pub max_data_size: Option<usize>,
    /// Supported data types
    pub supported_data_types: Vec<String>,
    /// Memory requirements
    pub memory_requirements: MemoryRequirements,
    /// Computational requirements
    pub compute_requirements: ComputeRequirements,
    /// Whether GPU acceleration is supported
    pub gpu_acceleration: bool,
    /// Whether streaming processing is supported
    pub streaming: bool,
}

/// Memory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Base memory overhead
    pub base_overhead: usize,
    /// Memory per data point
    pub per_data_point: f64,
    /// Peak memory multiplier
    pub peak_multiplier: f64,
}

/// Computational requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequirements {
    /// Computational complexity class
    pub complexity_class: ComplexityClass,
    /// CPU intensity level (1-10)
    pub cpu_intensity: u8,
    /// I/O intensity level (1-10)
    pub io_intensity: u8,
}

/// Computational complexity classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    LogLinear,
    Quadratic,
    Cubic,
    Exponential,
    Unknown,
}

/// Complexity estimate for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityEstimate {
    /// Estimated execution time
    pub estimated_duration: std::time::Duration,
    /// Estimated memory usage
    pub estimated_memory: usize,
    /// Confidence in estimate (0.0-1.0)
    pub confidence: f64,
    /// Complexity class
    pub complexity_class: ComplexityClass,
}

/// Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Analysis method used
    pub method: String,
    /// Analysis results
    pub results: HashMap<String, serde_json::Value>,
    /// Statistical measures
    pub statistics: Option<AnalysisStatistics>,
    /// Model information (if applicable)
    pub model: Option<ModelInfo>,
    /// Diagnostic information
    pub diagnostics: Option<DiagnosticInfo>,
    /// Execution metadata
    pub metadata: AnalysisMetadata,
    /// Warnings and notes
    pub warnings: Vec<String>,
}

/// Analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    /// Model fit statistics
    pub goodness_of_fit: HashMap<String, f64>,
    /// Residual statistics
    pub residuals: Option<ResidualStatistics>,
    /// Information criteria
    pub information_criteria: HashMap<String, f64>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
}

/// Residual statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualStatistics {
    pub mean: f64,
    pub std: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub autocorrelation: Vec<f64>,
    pub normality_test: Option<NormalityTest>,
}

/// Normality test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTest {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub is_normal: bool,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model type
    pub model_type: String,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model equation or formula
    pub formula: Option<String>,
    /// Model serialized state
    pub serialized_model: Option<Vec<u8>>,
}

/// Diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticInfo {
    /// Convergence information
    pub convergence: Option<ConvergenceInfo>,
    /// Validation results
    pub validation: Option<ValidationResults>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Quality indicators
    pub quality: QualityIndicators,
}

/// Convergence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: u32,
    pub final_error: f64,
    pub convergence_criteria: String,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub cross_validation_scores: Vec<f64>,
    pub validation_metrics: HashMap<String, f64>,
    pub validation_method: String,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time: std::time::Duration,
    pub memory_usage: usize,
    pub cpu_usage: Option<f64>,
    pub cache_hits: Option<u64>,
    pub cache_misses: Option<u64>,
}

/// Quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIndicators {
    pub data_quality_score: f64,
    pub model_quality_score: f64,
    pub reliability_score: f64,
    pub uncertainty_level: f64,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis ID
    pub analysis_id: String,
    /// Plugin ID
    pub plugin_id: String,
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Configuration used
    pub config_hash: String,
    /// Plugin version
    pub plugin_version: String,
}

/// Data characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub num_series: usize,
    pub num_points: usize,
    pub data_range: DateRange,
    pub sampling_frequency: Option<String>,
    pub has_missing_values: bool,
    pub has_outliers: bool,
}

/// Incremental analyzer trait for streaming analysis
pub trait IncrementalAnalyzer: Send + Sync {
    /// Process next data point or batch
    fn process_data(&mut self, data: &HashMap<String, serde_json::Value>) -> PluginResult<Option<AnalysisResult>>;

    /// Get current analysis state
    fn get_current_result(&self) -> PluginResult<AnalysisResult>;

    /// Update analysis parameters
    fn update_parameters(&mut self, parameters: &HashMap<String, serde_json::Value>) -> PluginResult<()>;

    /// Reset analysis state
    fn reset(&mut self) -> PluginResult<()>;

    /// Finalize analysis and get final result
    fn finalize(&mut self) -> PluginResult<AnalysisResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_config_serialization() {
        let config = AnalysisConfig {
            method: "linear_regression".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("alpha".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.05).unwrap()));
                params
            },
            input: InputConfig {
                columns: vec!["value".to_string()],
                preprocessing: None,
                filter: None,
                window: None,
            },
            output: OutputConfig {
                format: OutputFormat::Structured,
                include_confidence: true,
                confidence_level: Some(0.95),
                include_diagnostics: true,
                include_intermediate: false,
                precision: Some(4),
            },
            performance: None,
            validation: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AnalysisConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.method, "linear_regression");
    }

    #[test]
    fn test_analysis_categories() {
        let statistical = AnalysisCategory::Statistical;
        let domain_specific = AnalysisCategory::DomainSpecific("finance".to_string());

        assert_eq!(statistical, AnalysisCategory::Statistical);
        match domain_specific {
            AnalysisCategory::DomainSpecific(domain) => assert_eq!(domain, "finance"),
            _ => panic!("Expected domain-specific category"),
        }
    }

    #[test]
    fn test_parameter_definition() {
        let param = ParameterDefinition {
            name: "window_size".to_string(),
            description: "Size of the analysis window".to_string(),
            param_type: ParameterType::Integer,
            default: Some(serde_json::Value::Number(serde_json::Number::from(10))),
            constraints: Some(ParameterConstraints::Range { min: 1.0, max: 1000.0 }),
        };

        assert_eq!(param.name, "window_size");
        match param.param_type {
            ParameterType::Integer => {},
            _ => panic!("Expected integer parameter type"),
        }
    }

    #[test]
    fn test_complexity_estimate() {
        let estimate = ComplexityEstimate {
            estimated_duration: std::time::Duration::from_secs(30),
            estimated_memory: 1024 * 1024, // 1MB
            confidence: 0.8,
            complexity_class: ComplexityClass::Linear,
        };

        assert_eq!(estimate.estimated_duration, std::time::Duration::from_secs(30));
        assert_eq!(estimate.complexity_class, ComplexityClass::Linear);
    }
}