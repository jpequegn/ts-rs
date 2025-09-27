//! # Visualization Plugin Interface
//!
//! Defines the interface for visualization plugins that can create custom plot types,
//! interactive visualizations, and various export formats.

use super::{PluginError, PluginResult, PluginContext};
use crate::TimeSeries;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Visualization plugin trait
pub trait VisualizationPlugin: Send + Sync {
    /// Get supported plot types
    fn supported_plot_types(&self) -> Vec<PlotType>;

    /// Get supported export formats
    fn supported_export_formats(&self) -> Vec<ExportFormat>;

    /// Get supported themes
    fn supported_themes(&self) -> Vec<String>;

    /// Validate visualization configuration
    fn validate_config(&self, config: &VisualizationConfig) -> PluginResult<()>;

    /// Create visualization from time series data
    fn create_visualization(
        &self,
        data: &[TimeSeries],
        config: &VisualizationConfig,
        context: &PluginContext,
    ) -> PluginResult<VisualizationResult>;

    /// Render visualization to specific format
    fn render(
        &self,
        visualization: &VisualizationResult,
        format: &ExportFormat,
        output_path: &PathBuf,
    ) -> PluginResult<RenderResult>;

    /// Get visualization capabilities
    fn get_capabilities(&self) -> VisualizationCapabilities;

    /// Support for interactive visualizations
    fn supports_interactive(&self) -> bool {
        false
    }

    /// Create interactive visualization
    fn create_interactive_visualization(
        &self,
        _data: &[TimeSeries],
        _config: &VisualizationConfig,
        _context: &PluginContext,
    ) -> PluginResult<InteractiveVisualization> {
        Err(PluginError::ExecutionError("Interactive visualizations not supported".to_string()))
    }

    /// Support for animation
    fn supports_animation(&self) -> bool {
        false
    }

    /// Create animated visualization
    fn create_animation(
        &self,
        _data: &[TimeSeries],
        _config: &AnimationConfig,
        _context: &PluginContext,
    ) -> PluginResult<AnimationResult> {
        Err(PluginError::ExecutionError("Animation not supported".to_string()))
    }

    /// Get plot recommendations for given data
    fn recommend_plots(&self, data: &[TimeSeries]) -> Vec<PlotRecommendation>;

    /// Validate data compatibility with plot type
    fn validate_data_compatibility(&self, data: &[TimeSeries], plot_type: &PlotType) -> PluginResult<CompatibilityReport>;
}

/// Plot type specification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlotType {
    /// Plot type identifier
    pub id: String,
    /// Plot type name
    pub name: String,
    /// Plot type description
    pub description: String,
    /// Plot category
    pub category: PlotCategory,
    /// Data requirements
    pub data_requirements: DataRequirements,
    /// Whether plot supports multiple series
    pub multi_series: bool,
    /// Whether plot supports 3D rendering
    pub supports_3d: bool,
}

/// Plot categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlotCategory {
    /// Time series plots
    TimeSeries,
    /// Statistical plots
    Statistical,
    /// Distribution plots
    Distribution,
    /// Correlation plots
    Correlation,
    /// Decomposition plots
    Decomposition,
    /// Forecast plots
    Forecast,
    /// Network/Graph plots
    Network,
    /// Geospatial plots
    Geospatial,
    /// Custom domain-specific plots
    Custom(String),
}

/// Data requirements for plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRequirements {
    /// Minimum number of data points
    pub min_points: usize,
    /// Maximum number of data points (for performance)
    pub max_points: Option<usize>,
    /// Required data columns
    pub required_columns: Vec<String>,
    /// Optional data columns
    pub optional_columns: Vec<String>,
    /// Supported data types
    pub supported_types: Vec<DataType>,
    /// Whether temporal data is required
    pub requires_temporal: bool,
}

/// Data types for visualization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Numeric,
    Temporal,
    Categorical,
    Boolean,
    Text,
    Geospatial,
}

/// Export formats
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    /// PNG image
    PNG,
    /// JPEG image
    JPEG,
    /// SVG vector graphics
    SVG,
    /// PDF document
    PDF,
    /// HTML file
    HTML,
    /// Interactive HTML with JavaScript
    InteractiveHTML,
    /// JSON data format
    JSON,
    /// CSV data export
    CSV,
    /// Custom format
    Custom(String),
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Plot type to create
    pub plot_type: String,
    /// Plot title
    pub title: Option<String>,
    /// Plot styling configuration
    pub style: StyleConfig,
    /// Data configuration
    pub data: DataConfig,
    /// Layout configuration
    pub layout: LayoutConfig,
    /// Interactive features configuration
    pub interactive: Option<InteractiveConfig>,
    /// Export configuration
    pub export: ExportConfig,
    /// Performance configuration
    pub performance: Option<PerformanceConfig>,
}

/// Styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConfig {
    /// Theme name
    pub theme: Option<String>,
    /// Color palette
    pub colors: Option<ColorPalette>,
    /// Font configuration
    pub fonts: Option<FontConfig>,
    /// Line styles
    pub lines: Option<LineConfig>,
    /// Marker styles
    pub markers: Option<MarkerConfig>,
    /// Background configuration
    pub background: Option<BackgroundConfig>,
    /// Custom CSS or styling
    pub custom_style: Option<String>,
}

/// Color palette specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Primary colors for data series
    pub primary: Vec<String>,
    /// Background colors
    pub background: Option<String>,
    /// Text colors
    pub text: Option<String>,
    /// Grid colors
    pub grid: Option<String>,
    /// Accent colors
    pub accent: Vec<String>,
    /// Color mapping strategy
    pub strategy: ColorStrategy,
}

/// Color mapping strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorStrategy {
    /// Sequential color mapping
    Sequential,
    /// Diverging color mapping
    Diverging,
    /// Qualitative color mapping
    Qualitative,
    /// Custom color mapping
    Custom(HashMap<String, String>),
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    /// Title font
    pub title: Option<FontSpec>,
    /// Axis label font
    pub axis_labels: Option<FontSpec>,
    /// Tick label font
    pub tick_labels: Option<FontSpec>,
    /// Legend font
    pub legend: Option<FontSpec>,
    /// Annotation font
    pub annotations: Option<FontSpec>,
}

/// Font specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSpec {
    /// Font family
    pub family: String,
    /// Font size
    pub size: f64,
    /// Font weight
    pub weight: FontWeight,
    /// Font style
    pub style: FontStyle,
    /// Font color
    pub color: Option<String>,
}

/// Font weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    Custom(u16),
}

/// Font styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

/// Line configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineConfig {
    /// Line width
    pub width: Option<f64>,
    /// Line style
    pub style: Option<LineStyle>,
    /// Line opacity
    pub opacity: Option<f64>,
    /// Anti-aliasing
    pub anti_alias: bool,
}

/// Line styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
    Custom(String),
}

/// Marker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerConfig {
    /// Marker size
    pub size: Option<f64>,
    /// Marker shape
    pub shape: Option<MarkerShape>,
    /// Marker opacity
    pub opacity: Option<f64>,
    /// Marker border
    pub border: Option<BorderConfig>,
}

/// Marker shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkerShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross,
    Plus,
    Star,
    Custom(String),
}

/// Border configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderConfig {
    pub width: f64,
    pub color: String,
    pub style: LineStyle,
}

/// Background configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundConfig {
    /// Background color
    pub color: Option<String>,
    /// Background gradient
    pub gradient: Option<GradientConfig>,
    /// Background image
    pub image: Option<String>,
    /// Background opacity
    pub opacity: Option<f64>,
}

/// Gradient configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    pub direction: GradientDirection,
    pub colors: Vec<GradientStop>,
}

/// Gradient direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientDirection {
    Horizontal,
    Vertical,
    Diagonal,
    Radial,
}

/// Gradient color stop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStop {
    pub position: f64,
    pub color: String,
}

/// Data configuration for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Series to plot
    pub series: Vec<SeriesConfig>,
    /// X-axis configuration
    pub x_axis: AxisConfig,
    /// Y-axis configuration
    pub y_axis: AxisConfig,
    /// Secondary Y-axis configuration
    pub y2_axis: Option<AxisConfig>,
    /// Data filtering
    pub filter: Option<DataFilter>,
    /// Data aggregation
    pub aggregation: Option<AggregationConfig>,
}

/// Series configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesConfig {
    /// Series name
    pub name: String,
    /// Data column
    pub column: String,
    /// Series type (line, bar, scatter, etc.)
    pub series_type: String,
    /// Series-specific styling
    pub style: Option<SeriesStyle>,
    /// Y-axis to use (primary or secondary)
    pub y_axis: AxisSelection,
    /// Visibility
    pub visible: bool,
}

/// Series-specific styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStyle {
    pub color: Option<String>,
    pub line_width: Option<f64>,
    pub line_style: Option<LineStyle>,
    pub marker_size: Option<f64>,
    pub marker_shape: Option<MarkerShape>,
    pub opacity: Option<f64>,
}

/// Axis selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AxisSelection {
    Primary,
    Secondary,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Axis title
    pub title: Option<String>,
    /// Axis scale type
    pub scale: AxisScale,
    /// Axis range
    pub range: Option<AxisRange>,
    /// Tick configuration
    pub ticks: Option<TickConfig>,
    /// Grid configuration
    pub grid: Option<GridConfig>,
    /// Axis formatting
    pub format: Option<AxisFormat>,
}

/// Axis scale types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AxisScale {
    Linear,
    Log,
    Symlog,
    Time,
    Category,
}

/// Axis range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisRange {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub auto: bool,
}

/// Tick configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickConfig {
    /// Number of ticks
    pub count: Option<usize>,
    /// Tick spacing
    pub spacing: Option<f64>,
    /// Tick format
    pub format: Option<String>,
    /// Tick rotation angle
    pub rotation: Option<f64>,
    /// Show tick labels
    pub show_labels: bool,
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Show major grid lines
    pub major: bool,
    /// Show minor grid lines
    pub minor: bool,
    /// Grid line style
    pub style: Option<LineStyle>,
    /// Grid line color
    pub color: Option<String>,
    /// Grid line opacity
    pub opacity: Option<f64>,
}

/// Axis formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisFormat {
    /// Number format
    pub number_format: Option<NumberFormat>,
    /// Date format
    pub date_format: Option<String>,
    /// Custom format function
    pub custom_format: Option<String>,
}

/// Number formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFormat {
    /// Decimal places
    pub decimal_places: Option<usize>,
    /// Use thousands separator
    pub thousands_separator: bool,
    /// Percentage format
    pub percentage: bool,
    /// Scientific notation
    pub scientific: bool,
    /// Custom format string
    pub custom: Option<String>,
}

/// Data filtering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFilter {
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Value filters
    pub value_filters: Vec<ValueFilter>,
    /// Sample size (for large datasets)
    pub sample_size: Option<usize>,
    /// Sampling strategy
    pub sampling_strategy: Option<SamplingStrategy>,
}

/// Date range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}

/// Value filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFilter {
    pub column: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// Filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    Between,
    In,
    NotIn,
}

/// Sampling strategies for large datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Random,
    Systematic,
    Stratified,
    TimeBasedDownsampling,
}

/// Data aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Aggregation function
    pub function: AggregationFunction,
    /// Aggregation period
    pub period: AggregationPeriod,
    /// Fill missing values
    pub fill_missing: bool,
}

/// Aggregation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Mean,
    Sum,
    Count,
    Min,
    Max,
    Median,
    First,
    Last,
    Std,
    Var,
}

/// Aggregation periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationPeriod {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
    Custom(String),
}

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Plot dimensions
    pub dimensions: Option<Dimensions>,
    /// Margins
    pub margins: Option<Margins>,
    /// Legend configuration
    pub legend: Option<LegendConfig>,
    /// Subplot configuration
    pub subplots: Option<SubplotConfig>,
    /// Annotation configuration
    pub annotations: Vec<Annotation>,
}

/// Plot dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimensions {
    pub width: f64,
    pub height: f64,
    pub aspect_ratio: Option<f64>,
}

/// Plot margins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Margins {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

/// Legend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendConfig {
    /// Show legend
    pub show: bool,
    /// Legend position
    pub position: LegendPosition,
    /// Legend orientation
    pub orientation: LegendOrientation,
    /// Legend styling
    pub style: Option<LegendStyle>,
}

/// Legend positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Bottom,
    Left,
    Right,
    Custom { x: f64, y: f64 },
}

/// Legend orientations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendOrientation {
    Horizontal,
    Vertical,
}

/// Legend styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendStyle {
    pub background_color: Option<String>,
    pub border_color: Option<String>,
    pub border_width: Option<f64>,
    pub font: Option<FontSpec>,
    pub padding: Option<f64>,
}

/// Subplot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubplotConfig {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Shared x-axis
    pub shared_x: bool,
    /// Shared y-axis
    pub shared_y: bool,
    /// Subplot spacing
    pub spacing: Option<SubplotSpacing>,
}

/// Subplot spacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubplotSpacing {
    pub horizontal: f64,
    pub vertical: f64,
}

/// Annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation type
    pub annotation_type: AnnotationType,
    /// Annotation text
    pub text: String,
    /// Position
    pub position: AnnotationPosition,
    /// Styling
    pub style: Option<AnnotationStyle>,
}

/// Annotation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationType {
    Text,
    Arrow,
    Rectangle,
    Circle,
    Line,
}

/// Annotation position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationPosition {
    pub x: f64,
    pub y: f64,
    pub coordinate_system: CoordinateSystem,
}

/// Coordinate systems for positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinateSystem {
    Data,
    Pixel,
    Normalized,
}

/// Annotation styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationStyle {
    pub font: Option<FontSpec>,
    pub color: Option<String>,
    pub background_color: Option<String>,
    pub border: Option<BorderConfig>,
}

/// Interactive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Enable zoom
    pub zoom: bool,
    /// Enable pan
    pub pan: bool,
    /// Enable hover tooltips
    pub hover: bool,
    /// Enable selection
    pub selection: bool,
    /// Enable brush selection
    pub brush: bool,
    /// Crossfilter support
    pub crossfilter: bool,
    /// Custom interactions
    pub custom_interactions: Vec<CustomInteraction>,
}

/// Custom interaction specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomInteraction {
    pub name: String,
    pub trigger: InteractionTrigger,
    pub action: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Interaction triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionTrigger {
    Click,
    DoubleClick,
    Hover,
    Drag,
    KeyPress,
    Custom(String),
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Default export format
    pub format: ExportFormat,
    /// Output resolution (DPI)
    pub dpi: Option<u32>,
    /// Quality setting (for lossy formats)
    pub quality: Option<u8>,
    /// Transparency support
    pub transparent: bool,
    /// Include metadata
    pub include_metadata: bool,
}

/// Performance configuration for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    /// Maximum memory usage
    pub max_memory: Option<usize>,
    /// Rendering timeout
    pub timeout: Option<std::time::Duration>,
    /// Level of detail for large datasets
    pub level_of_detail: bool,
    /// Chunk size for data processing
    pub chunk_size: Option<usize>,
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Animation type
    pub animation_type: AnimationType,
    /// Animation duration
    pub duration: std::time::Duration,
    /// Frame rate
    pub frame_rate: u32,
    /// Easing function
    pub easing: EasingFunction,
    /// Loop animation
    pub loop_animation: bool,
    /// Export format for animation
    pub export_format: AnimationExportFormat,
}

/// Animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationType {
    TimeProgression,
    DataTransition,
    ParameterSweep,
    Custom(String),
}

/// Easing functions for animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
    Custom(String),
}

/// Animation export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationExportFormat {
    GIF,
    MP4,
    WebM,
    WebP,
    ImageSequence,
}

/// Visualization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationResult {
    /// Visualization ID
    pub id: String,
    /// Plot type used
    pub plot_type: String,
    /// Visualization data
    pub data: PlotData,
    /// Layout information
    pub layout: PlotLayout,
    /// Style information
    pub style: PlotStyle,
    /// Metadata
    pub metadata: VisualizationMetadata,
    /// Generated code/markup (if applicable)
    pub code: Option<String>,
}

/// Plot data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotData {
    /// Data series
    pub series: Vec<DataSeries>,
    /// Axis data
    pub axes: HashMap<String, AxisData>,
    /// Additional plot elements
    pub elements: Vec<PlotElement>,
}

/// Data series for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSeries {
    pub name: String,
    pub x_values: Vec<serde_json::Value>,
    pub y_values: Vec<serde_json::Value>,
    pub metadata: SeriesMetadata,
}

/// Series metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesMetadata {
    pub data_type: DataType,
    pub units: Option<String>,
    pub description: Option<String>,
    pub source: Option<String>,
}

/// Axis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisData {
    pub title: Option<String>,
    pub min: f64,
    pub max: f64,
    pub scale: AxisScale,
    pub ticks: Vec<TickValue>,
}

/// Tick value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickValue {
    pub value: f64,
    pub label: String,
}

/// Plot element (annotations, shapes, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotElement {
    pub element_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Plot layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotLayout {
    pub dimensions: Dimensions,
    pub margins: Margins,
    pub title: Option<String>,
    pub legend: Option<LegendLayout>,
}

/// Legend layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendLayout {
    pub position: LegendPosition,
    pub size: Dimensions,
}

/// Plot style information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotStyle {
    pub theme: String,
    pub colors: ColorPalette,
    pub fonts: FontConfig,
    pub custom_css: Option<String>,
}

/// Visualization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub plugin_id: String,
    pub plugin_version: String,
    pub data_hash: String,
    pub config_hash: String,
    pub render_time: std::time::Duration,
}

/// Render result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderResult {
    /// Output file path
    pub output_path: PathBuf,
    /// File size in bytes
    pub file_size: u64,
    /// Render duration
    pub render_duration: std::time::Duration,
    /// Render quality metrics
    pub quality_metrics: Option<QualityMetrics>,
    /// Any warnings during rendering
    pub warnings: Vec<String>,
}

/// Quality metrics for rendered output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub resolution: (u32, u32),
    pub color_depth: u8,
    pub compression_ratio: Option<f64>,
    pub estimated_quality_score: f64,
}

/// Interactive visualization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveVisualization {
    /// HTML content
    pub html: String,
    /// JavaScript code
    pub javascript: String,
    /// CSS styles
    pub css: String,
    /// Required external libraries
    pub dependencies: Vec<Dependency>,
    /// Interaction capabilities
    pub capabilities: InteractiveCapabilities,
}

/// External dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub source: DependencySource,
}

/// Dependency sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencySource {
    CDN(String),
    Local(PathBuf),
    Inline,
}

/// Interactive capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveCapabilities {
    pub zoom: bool,
    pub pan: bool,
    pub hover_tooltips: bool,
    pub selection: bool,
    pub brushing: bool,
    pub linked_views: bool,
    pub real_time_updates: bool,
}

/// Animation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationResult {
    /// Animation frames
    pub frames: Vec<AnimationFrame>,
    /// Animation metadata
    pub metadata: AnimationMetadata,
    /// Export paths (if rendered)
    pub export_paths: Vec<PathBuf>,
}

/// Animation frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationFrame {
    pub frame_number: u32,
    pub timestamp: f64,
    pub visualization: VisualizationResult,
}

/// Animation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationMetadata {
    pub total_frames: u32,
    pub duration: std::time::Duration,
    pub frame_rate: u32,
    pub file_size: Option<u64>,
}

/// Plot recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotRecommendation {
    /// Recommended plot type
    pub plot_type: PlotType,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Reasoning for recommendation
    pub reasoning: String,
    /// Suggested configuration
    pub suggested_config: Option<VisualizationConfig>,
}

/// Data compatibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    /// Whether data is compatible
    pub compatible: bool,
    /// Compatibility score (0.0-1.0)
    pub score: f64,
    /// Issues found
    pub issues: Vec<CompatibilityIssue>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// Compatibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Issue description
    pub description: String,
    /// Severity level
    pub severity: IssueSeverity,
    /// Affected data columns/series
    pub affected_data: Vec<String>,
}

/// Issue types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    InsufficientData,
    DataTypeMismatch,
    MissingRequiredColumn,
    DataQualityIssue,
    PerformanceConcern,
    UnsupportedFeature,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Visualization capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationCapabilities {
    /// Maximum data points per series
    pub max_data_points: Option<usize>,
    /// Maximum number of series
    pub max_series: Option<usize>,
    /// Supported data types
    pub supported_data_types: Vec<DataType>,
    /// Rendering capabilities
    pub rendering: RenderingCapabilities,
    /// Interactive features
    pub interactive_features: Vec<InteractiveFeature>,
    /// Export capabilities
    pub export_capabilities: ExportCapabilities,
}

/// Rendering capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingCapabilities {
    /// GPU acceleration support
    pub gpu_acceleration: bool,
    /// Maximum resolution
    pub max_resolution: (u32, u32),
    /// 3D rendering support
    pub supports_3d: bool,
    /// Animation support
    pub supports_animation: bool,
    /// Real-time rendering
    pub real_time_rendering: bool,
}

/// Interactive features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveFeature {
    Zoom,
    Pan,
    Hover,
    Selection,
    Brushing,
    Linking,
    RealTimeUpdates,
    CustomInteractions,
}

/// Export capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportCapabilities {
    /// Supported static formats
    pub static_formats: Vec<ExportFormat>,
    /// Supported interactive formats
    pub interactive_formats: Vec<ExportFormat>,
    /// Supported animation formats
    pub animation_formats: Vec<AnimationExportFormat>,
    /// Maximum export resolution
    pub max_export_resolution: (u32, u32),
    /// Supports batch export
    pub batch_export: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_config_serialization() {
        let config = VisualizationConfig {
            plot_type: "line_plot".to_string(),
            title: Some("Test Plot".to_string()),
            style: StyleConfig {
                theme: Some("dark".to_string()),
                colors: None,
                fonts: None,
                lines: None,
                markers: None,
                background: None,
                custom_style: None,
            },
            data: DataConfig {
                series: vec![SeriesConfig {
                    name: "Series 1".to_string(),
                    column: "value".to_string(),
                    series_type: "line".to_string(),
                    style: None,
                    y_axis: AxisSelection::Primary,
                    visible: true,
                }],
                x_axis: AxisConfig {
                    title: Some("Time".to_string()),
                    scale: AxisScale::Time,
                    range: None,
                    ticks: None,
                    grid: None,
                    format: None,
                },
                y_axis: AxisConfig {
                    title: Some("Value".to_string()),
                    scale: AxisScale::Linear,
                    range: None,
                    ticks: None,
                    grid: None,
                    format: None,
                },
                y2_axis: None,
                filter: None,
                aggregation: None,
            },
            layout: LayoutConfig {
                dimensions: Some(Dimensions {
                    width: 800.0,
                    height: 600.0,
                    aspect_ratio: None,
                }),
                margins: None,
                legend: None,
                subplots: None,
                annotations: vec![],
            },
            interactive: None,
            export: ExportConfig {
                format: ExportFormat::PNG,
                dpi: Some(300),
                quality: None,
                transparent: false,
                include_metadata: true,
            },
            performance: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VisualizationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.plot_type, "line_plot");
    }

    #[test]
    fn test_plot_categories() {
        let ts_category = PlotCategory::TimeSeries;
        let custom_category = PlotCategory::Custom("finance".to_string());

        assert_eq!(ts_category, PlotCategory::TimeSeries);
        match custom_category {
            PlotCategory::Custom(domain) => assert_eq!(domain, "finance"),
            _ => panic!("Expected custom category"),
        }
    }

    #[test]
    fn test_export_formats() {
        assert_eq!(ExportFormat::PNG, ExportFormat::PNG);
        assert_ne!(ExportFormat::PNG, ExportFormat::SVG);

        let custom_format = ExportFormat::Custom("webgl".to_string());
        match custom_format {
            ExportFormat::Custom(format) => assert_eq!(format, "webgl"),
            _ => panic!("Expected custom format"),
        }
    }

    #[test]
    fn test_plot_recommendation() {
        let recommendation = PlotRecommendation {
            plot_type: PlotType {
                id: "line_plot".to_string(),
                name: "Line Plot".to_string(),
                description: "Basic line plot".to_string(),
                category: PlotCategory::TimeSeries,
                data_requirements: DataRequirements {
                    min_points: 2,
                    max_points: None,
                    required_columns: vec!["time".to_string(), "value".to_string()],
                    optional_columns: vec![],
                    supported_types: vec![DataType::Temporal, DataType::Numeric],
                    requires_temporal: true,
                },
                multi_series: true,
                supports_3d: false,
            },
            confidence: 0.95,
            reasoning: "Data has temporal component and numeric values".to_string(),
            suggested_config: None,
        };

        assert_eq!(recommendation.confidence, 0.95);
        assert_eq!(recommendation.plot_type.id, "line_plot");
    }
}