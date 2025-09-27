//! # Reporting Types and Configurations
//!
//! Core data structures and types for the comprehensive reporting system.

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::stats::StatisticalAnalysisResult;
use crate::trend::ComprehensiveTrendAnalysis;
use crate::seasonality::ComprehensiveSeasonalityAnalysis;
use crate::validation::DataQualityReport;

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Report template to use
    pub template: ReportTemplate,

    /// Export format for the report
    pub export_format: ReportExportFormat,

    /// Analysis sections to include
    pub sections: ReportSections,

    /// Output file path (optional)
    pub output_path: Option<String>,

    /// Report metadata
    pub metadata: ReportMetadata,

    /// Advanced configuration options
    pub advanced: AdvancedReportConfig,
}

/// Available report templates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportTemplate {
    /// Executive summary focused on key insights and business impact
    Executive,

    /// Technical analysis with detailed statistical information
    Technical,

    /// Data quality assessment and recommendations
    DataQuality,

    /// Forecasting analysis with predictions and model evaluation
    Forecasting,

    /// Custom template with user-defined sections
    Custom {
        /// Template name
        name: String,
        /// Sections to include
        sections: Vec<String>,
    },
}

/// Export formats for reports
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportExportFormat {
    /// Markdown format for documentation
    Markdown,

    /// HTML format with styling and interactivity
    HTML,

    /// PDF format for sharing and printing
    PDF,

    /// JSON format for programmatic access
    JSON,

    /// Plain text format
    Text,
}

/// Configuration for report sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSections {
    /// Include executive summary
    pub executive_summary: bool,

    /// Include data quality assessment
    pub data_quality: bool,

    /// Include descriptive statistics
    pub descriptive_stats: bool,

    /// Include trend analysis
    pub trend_analysis: bool,

    /// Include seasonality analysis
    pub seasonality_analysis: bool,

    /// Include anomaly detection
    pub anomaly_detection: bool,

    /// Include forecasting analysis
    pub forecasting: bool,

    /// Include correlation analysis
    pub correlation_analysis: bool,

    /// Include visualizations
    pub visualizations: bool,

    /// Include automated insights
    pub automated_insights: bool,

    /// Include recommendations
    pub recommendations: bool,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,

    /// Report author
    pub author: Option<String>,

    /// Data source information
    pub data_source: String,

    /// Analysis period
    pub analysis_period: Option<(DateTime<Utc>, DateTime<Utc>)>,

    /// Generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Report version
    pub version: String,

    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

/// Advanced report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedReportConfig {
    /// Include confidence intervals in analyses
    pub include_confidence_intervals: bool,

    /// Significance level for statistical tests
    pub significance_level: f64,

    /// Maximum number of insights to generate
    pub max_insights: usize,

    /// Include detailed technical appendix
    pub include_technical_appendix: bool,

    /// Custom styling options
    pub styling: Option<ReportStyling>,

    /// Language for the report
    pub language: String,
}

/// Report styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportStyling {
    /// Color scheme
    pub color_scheme: String,

    /// Font family
    pub font_family: String,

    /// Custom CSS (for HTML output)
    pub custom_css: Option<String>,

    /// Logo URL or path
    pub logo: Option<String>,
}

/// Comprehensive report result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportResult {
    /// Report content in the requested format
    pub content: ReportContent,

    /// Analysis data used in the report
    pub analysis_data: AnalysisData,

    /// Generated insights and recommendations
    pub insights: Vec<Insight>,

    /// Report metadata
    pub metadata: ReportMetadata,

    /// Export information
    pub export_info: Option<ReportExportInfo>,
}

/// Report content variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportContent {
    /// Markdown content
    Markdown(String),

    /// HTML content with styling
    HTML(String),

    /// PDF file path or binary data
    PDF(Vec<u8>),

    /// JSON structured data
    JSON(serde_json::Value),

    /// Plain text content
    Text(String),
}

/// Aggregated analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisData {
    /// Statistical analysis results
    pub statistics: Option<StatisticalAnalysisResult>,

    /// Trend analysis results
    pub trend: Option<ComprehensiveTrendAnalysis>,

    /// Seasonality analysis results
    pub seasonality: Option<ComprehensiveSeasonalityAnalysis>,

    /// Anomaly detection results (placeholder)
    pub anomalies: Option<String>,

    /// Forecasting results (placeholder)
    pub forecasting: Option<String>,

    /// Correlation analysis results (placeholder)
    pub correlation: Option<String>,

    /// Data quality assessment
    pub data_quality: Option<DataQualityReport>,

    /// Generated visualizations (placeholder)
    pub plots: Vec<String>,

    /// Raw time series data summary
    pub data_summary: DataSummary,
}

/// Summary of the analyzed time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSummary {
    /// Number of data points
    pub n_points: usize,

    /// Date range
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,

    /// Sampling frequency
    pub frequency: Option<String>,

    /// Column names analyzed
    pub columns: Vec<String>,

    /// Missing data percentage
    pub missing_percentage: f64,
}

/// Automated insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    /// Insight type/category
    pub category: InsightCategory,

    /// Insight title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Importance level
    pub importance: InsightImportance,

    /// Supporting evidence
    pub evidence: Vec<String>,

    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Insight categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InsightCategory {
    /// Data quality issues
    DataQuality,

    /// Trend patterns
    Trend,

    /// Seasonal patterns
    Seasonality,

    /// Anomalies and outliers
    Anomaly,

    /// Forecasting insights
    Forecasting,

    /// Statistical properties
    Statistical,

    /// Business intelligence
    Business,
}

/// Insight importance levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsightImportance {
    /// Low importance
    Low,

    /// Medium importance
    Medium,

    /// High importance
    High,

    /// Critical importance
    Critical,
}

/// Report export information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportExportInfo {
    /// Export format used
    pub format: ReportExportFormat,

    /// File path where report was saved
    pub file_path: Option<String>,

    /// File size in bytes
    pub file_size: Option<u64>,

    /// Export timestamp
    pub exported_at: DateTime<Utc>,

    /// Export duration in milliseconds
    pub export_duration_ms: u64,
}

/// Batch reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReportConfig {
    /// Individual report configurations
    pub reports: Vec<ReportConfig>,

    /// Common output directory
    pub output_directory: String,

    /// Batch processing options
    pub batch_options: BatchOptions,
}

/// Batch processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptions {
    /// Generate consolidated summary report
    pub generate_summary: bool,

    /// Include comparative analysis
    pub include_comparison: bool,

    /// Maximum parallel processing
    pub max_parallel: usize,

    /// Continue on individual failures
    pub continue_on_error: bool,
}

/// Default implementations
impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            template: ReportTemplate::Executive,
            export_format: ReportExportFormat::HTML,
            sections: ReportSections::default(),
            output_path: None,
            metadata: ReportMetadata::default(),
            advanced: AdvancedReportConfig::default(),
        }
    }
}

impl Default for ReportSections {
    fn default() -> Self {
        Self {
            executive_summary: true,
            data_quality: true,
            descriptive_stats: true,
            trend_analysis: true,
            seasonality_analysis: true,
            anomaly_detection: true,
            forecasting: false,
            correlation_analysis: false,
            visualizations: true,
            automated_insights: true,
            recommendations: true,
        }
    }
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self {
            title: "Time Series Analysis Report".to_string(),
            author: None,
            data_source: "Unknown".to_string(),
            analysis_period: None,
            generated_at: Utc::now(),
            version: "1.0".to_string(),
            custom_fields: HashMap::new(),
        }
    }
}

impl Default for AdvancedReportConfig {
    fn default() -> Self {
        Self {
            include_confidence_intervals: true,
            significance_level: 0.05,
            max_insights: 10,
            include_technical_appendix: false,
            styling: None,
            language: "en".to_string(),
        }
    }
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            generate_summary: true,
            include_comparison: true,
            max_parallel: 4,
            continue_on_error: true,
        }
    }
}

impl std::fmt::Display for ReportTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportTemplate::Executive => write!(f, "Executive Summary"),
            ReportTemplate::Technical => write!(f, "Technical Analysis"),
            ReportTemplate::DataQuality => write!(f, "Data Quality Assessment"),
            ReportTemplate::Forecasting => write!(f, "Forecasting Analysis"),
            ReportTemplate::Custom { name, .. } => write!(f, "Custom: {}", name),
        }
    }
}

impl std::fmt::Display for ReportExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportExportFormat::Markdown => write!(f, "Markdown"),
            ReportExportFormat::HTML => write!(f, "HTML"),
            ReportExportFormat::PDF => write!(f, "PDF"),
            ReportExportFormat::JSON => write!(f, "JSON"),
            ReportExportFormat::Text => write!(f, "Text"),
        }
    }
}