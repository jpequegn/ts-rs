//! # Plotting Type Definitions
//!
//! Core data structures and types for the plotting system.

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Main plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    /// Type of plot to create
    pub plot_type: PlotType,

    /// Primary column for single-series plots
    pub primary_column: String,

    /// Additional columns for multi-series plots
    pub additional_columns: Vec<String>,

    /// Plot title
    pub title: Option<String>,

    /// X-axis label
    pub x_label: Option<String>,

    /// Y-axis label
    pub y_label: Option<String>,

    /// Theme to apply
    pub theme: Theme,

    /// Width in pixels
    pub width: usize,

    /// Height in pixels
    pub height: usize,

    /// Export format
    pub export_format: ExportFormat,

    /// Interactive mode
    pub interactive: bool,

    /// Custom styling options
    pub custom_style: Option<HashMap<String, String>>,

    /// Show legend
    pub show_legend: bool,

    /// Show grid
    pub show_grid: bool,
}

/// Types of plots available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotType {
    /// Basic line plot
    Line,

    /// Scatter plot
    Scatter,

    /// Multiple series on same plot
    MultipleSeries,

    /// Subplot layout
    Subplots,

    /// Histogram
    Histogram,

    /// Box plot
    BoxPlot,

    /// Violin plot
    ViolinPlot,

    /// Q-Q plot for normality testing
    QQPlot,

    /// Autocorrelation function plot
    ACF,

    /// Partial autocorrelation function plot
    PACF,

    /// Density plot
    Density,

    /// Correlation heatmap
    Heatmap,

    /// Scatter matrix for correlation
    ScatterMatrix,

    /// Trend decomposition plot
    Decomposition,

    /// Seasonal pattern visualization
    SeasonalPattern,

    /// Anomaly highlighting plot
    AnomalyHighlight,

    /// Forecast plot with confidence intervals
    Forecast,
}

/// Export formats supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Interactive HTML file
    HTML,

    /// PNG image
    PNG,

    /// SVG vector graphics
    SVG,

    /// PDF document
    PDF,

    /// JSON data for external tools
    JSON,

    /// Only display, no export
    Display,
}

/// Available themes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Theme {
    /// Light theme with professional colors
    Default,

    /// Dark theme
    Dark,

    /// Publication-ready theme
    Publication,

    /// High contrast theme
    HighContrast,

    /// Custom theme with user-defined colors
    Custom(ThemeConfig),
}

/// Custom theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Background color
    pub background: String,

    /// Primary color palette
    pub colors: Vec<String>,

    /// Grid color
    pub grid_color: String,

    /// Text color
    pub text_color: String,

    /// Axis color
    pub axis_color: String,
}

/// Plot result containing rendered output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotResult {
    /// Plot content (HTML, JSON, or file path)
    pub content: PlotContent,

    /// Plot metadata
    pub metadata: PlotMetadata,

    /// Export information
    pub export_info: Option<ExportInfo>,
}

/// Plot content variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotContent {
    /// HTML content for interactive plots
    HTML(String),

    /// JSON representation for external tools
    JSON(serde_json::Value),

    /// File path for exported plots
    FilePath(String),

    /// Base64-encoded image data
    ImageData(String),
}

/// Plot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotMetadata {
    /// Plot type used
    pub plot_type: PlotType,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Number of data points
    pub data_points: usize,

    /// Number of series
    pub series_count: usize,

    /// Plot dimensions
    pub dimensions: (usize, usize),

    /// Theme used
    pub theme: Theme,
}

/// Export information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    /// Export format used
    pub format: ExportFormat,

    /// File path (if applicable)
    pub file_path: Option<String>,

    /// File size in bytes
    pub file_size: Option<u64>,

    /// Export timestamp
    pub exported_at: DateTime<Utc>,
}

/// Plot data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotData {
    /// Multiple series data
    pub series: Vec<PlotSeries>,

    /// Metadata about the data
    pub metadata: HashMap<String, String>,
}

/// Individual data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotSeries {
    /// Series name
    pub name: String,

    /// X-axis values
    pub x_values: Vec<PlotPoint>,

    /// Y-axis values
    pub y_values: Vec<f64>,

    /// Series type
    pub series_type: SeriesType,

    /// Custom color (optional)
    pub color: Option<String>,

    /// Custom style (optional)
    pub style: Option<SeriesStyle>,
}

/// Plot point that can be timestamp, numeric, or categorical
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotPoint {
    /// Timestamp with index
    Timestamp(DateTime<Utc>, usize),

    /// Numeric value
    Numeric(f64),

    /// Index value
    Index(usize),

    /// Categorical value
    Category(String),
}

/// Series rendering type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeriesType {
    /// Line plot
    Line,

    /// Scatter plot
    Scatter,

    /// Bar chart
    Bar,

    /// Area plot
    Area,

    /// Histogram bars
    Histogram,

    /// Box plot
    Box,

    /// Violin plot
    Violin,
}

/// Series styling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStyle {
    /// Line width
    pub line_width: Option<f64>,

    /// Marker size
    pub marker_size: Option<f64>,

    /// Opacity
    pub opacity: Option<f64>,

    /// Line style (solid, dashed, dotted)
    pub line_style: Option<String>,

    /// Fill color
    pub fill_color: Option<String>,
}

/// Statistical plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalPlotConfig {
    /// Number of bins for histograms
    pub bins: Option<usize>,

    /// Density estimation for distribution plots
    pub density: bool,

    /// Confidence level for confidence intervals
    pub confidence_level: f64,

    /// Bandwidth for density estimation
    pub bandwidth: Option<f64>,
}

/// Correlation plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPlotConfig {
    /// Correlation method
    pub method: CorrelationMethod,

    /// Show correlation values in heatmap
    pub show_values: bool,

    /// Color scale for heatmap
    pub color_scale: String,

    /// Cluster rows and columns
    pub cluster: bool,
}

/// Correlation methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
}

/// Decomposition plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionPlotConfig {
    /// Components to show
    pub components: Vec<DecompositionComponent>,

    /// Shared x-axis across subplots
    pub shared_x: bool,

    /// Subplot height ratios
    pub height_ratios: Option<Vec<f64>>,
}

/// Decomposition components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionComponent {
    Original,
    Trend,
    Seasonal,
    Residual,
    Reconstructed,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            plot_type: PlotType::Line,
            primary_column: "value".to_string(),
            additional_columns: Vec::new(),
            title: None,
            x_label: Some("X".to_string()),
            y_label: Some("Y".to_string()),
            theme: Theme::Default,
            width: 800,
            height: 600,
            export_format: ExportFormat::HTML,
            interactive: true,
            custom_style: None,
            show_legend: true,
            show_grid: true,
        }
    }
}

impl Default for StatisticalPlotConfig {
    fn default() -> Self {
        Self {
            bins: None,
            density: false,
            confidence_level: 0.95,
            bandwidth: None,
        }
    }
}

impl Default for CorrelationPlotConfig {
    fn default() -> Self {
        Self {
            method: CorrelationMethod::Pearson,
            show_values: true,
            color_scale: "RdBu".to_string(),
            cluster: false,
        }
    }
}

impl Default for DecompositionPlotConfig {
    fn default() -> Self {
        Self {
            components: vec![
                DecompositionComponent::Original,
                DecompositionComponent::Trend,
                DecompositionComponent::Seasonal,
                DecompositionComponent::Residual,
            ],
            shared_x: true,
            height_ratios: None,
        }
    }
}