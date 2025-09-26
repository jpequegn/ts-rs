//! # Trend Analysis Plotting Module
//!
//! Data structures and functions for generating plot data for trend analysis visualization.
//! This module provides data that can be consumed by external plotting libraries.

use crate::trend::{DecompositionResult, BreakpointDetection};
use crate::trend::decomposition::{DecompositionMethod, DecompositionQuality};
use crate::trend::analysis::{ComprehensiveTrendResult, RateOfChangeAnalysis};
use crate::analysis::TrendDirection;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Plot data for trend analysis visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPlotData {
    /// Time series data points
    pub series: TimeSeries,

    /// Linear trend line data
    pub trend_line: Option<TrendLine>,

    /// Confidence bands around the trend
    pub confidence_bands: Option<ConfidenceBands>,

    /// Breakpoint markers
    pub breakpoints: Vec<BreakpointMarker>,

    /// Rate of change visualization data
    pub rate_of_change: Option<RateOfChangePlot>,

    /// Statistical annotations
    pub annotations: Vec<PlotAnnotation>,

    /// Plot styling and configuration
    pub styling: PlotStyling,

    /// Metadata for the plot
    pub metadata: PlotMetadata,
}

/// Time series data for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// X-axis values (timestamps or indices)
    pub x_values: Vec<PlotValue>,

    /// Y-axis values (data points)
    pub y_values: Vec<f64>,

    /// Optional error bars or confidence intervals
    pub error_bars: Option<Vec<f64>>,

    /// Point colors (for highlighting specific values)
    pub colors: Option<Vec<String>>,

    /// Point sizes (for emphasis)
    pub sizes: Option<Vec<f64>>,

    /// Point labels
    pub labels: Option<Vec<String>>,

    /// Series name
    pub name: String,
}

/// Trend line data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendLine {
    /// X-axis values for trend line
    pub x_values: Vec<PlotValue>,

    /// Y-axis values for trend line
    pub y_values: Vec<f64>,

    /// Trend direction for styling
    pub direction: TrendDirection,

    /// Trend strength (0.0 to 1.0) for styling intensity
    pub strength: f64,

    /// Statistical significance
    pub p_value: Option<f64>,

    /// Line styling properties
    pub style: LineStyle,
}

/// Confidence bands around trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBands {
    /// X-axis values
    pub x_values: Vec<PlotValue>,

    /// Upper confidence bound
    pub upper_bound: Vec<f64>,

    /// Lower confidence bound
    pub lower_bound: Vec<f64>,

    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,

    /// Band styling
    pub style: BandStyle,
}

/// Breakpoint marker for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakpointMarker {
    /// X position of the breakpoint
    pub x_position: PlotValue,

    /// Y position for marker placement
    pub y_position: f64,

    /// Confidence level of the breakpoint
    pub confidence: f64,

    /// Type of change at breakpoint
    pub change_type: String,

    /// Magnitude of change
    pub magnitude: f64,

    /// Marker styling
    pub style: MarkerStyle,

    /// Annotation text
    pub annotation: Option<String>,
}

/// Rate of change plot data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateOfChangePlot {
    /// X-axis values (time indices)
    pub x_values: Vec<PlotValue>,

    /// Rate of change values
    pub rates: Vec<f64>,

    /// Acceleration values (if available)
    pub acceleration: Option<Vec<f64>>,

    /// Zero line for reference
    pub zero_line: bool,

    /// Moving average of rates
    pub smoothed_rates: Option<Vec<f64>>,

    /// Styling for rate plot
    pub style: LineStyle,
}

/// Plot annotation for statistical information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotAnnotation {
    /// Text content
    pub text: String,

    /// Position on plot
    pub position: AnnotationPosition,

    /// Styling
    pub style: AnnotationStyle,

    /// Type of annotation
    pub annotation_type: AnnotationType,
}

/// Decomposition plot data for trend decomposition visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionPlotData {
    /// Original time series
    pub original: TimeSeries,

    /// Trend component
    pub trend: Option<TimeSeries>,

    /// Seasonal component
    pub seasonal: Option<TimeSeries>,

    /// Residual component
    pub residual: TimeSeries,

    /// Reconstructed series (trend + seasonal)
    pub reconstructed: Option<TimeSeries>,

    /// Component statistics
    pub component_stats: ComponentStatistics,

    /// Plot layout configuration
    pub layout: DecompositionLayout,

    /// Overall styling
    pub styling: PlotStyling,
}

/// Statistics for decomposition components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatistics {
    /// Percentage of total variance explained by trend
    pub trend_variance_pct: f64,

    /// Percentage of total variance explained by seasonal component
    pub seasonal_variance_pct: f64,

    /// Percentage of total variance in residuals
    pub residual_variance_pct: f64,

    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: f64,

    /// Quality score of decomposition (0.0 to 1.0)
    pub quality_score: f64,
}

/// Layout configuration for decomposition plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionLayout {
    /// Number of subplot rows
    pub rows: usize,

    /// Subplot titles
    pub subplot_titles: Vec<String>,

    /// Shared x-axis
    pub shared_x_axis: bool,

    /// Individual y-axis ranges
    pub y_axis_ranges: Option<Vec<(f64, f64)>>,

    /// Height ratios for subplots
    pub height_ratios: Option<Vec<f64>>,
}

/// Plot value that can be either timestamp or numeric
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PlotValue {
    /// Timestamp value
    Timestamp(DateTime<Utc>),
    /// Numeric value (index or other)
    Numeric(f64),
    /// String value (categories)
    Category(String),
}

/// Line styling properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineStyle {
    /// Line color (hex, rgb, or named color)
    pub color: String,

    /// Line width
    pub width: f64,

    /// Line pattern (solid, dashed, dotted)
    pub pattern: LinePattern,

    /// Opacity (0.0 to 1.0)
    pub opacity: f64,
}

/// Band styling properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandStyle {
    /// Fill color
    pub fill_color: String,

    /// Border color
    pub border_color: String,

    /// Opacity (0.0 to 1.0)
    pub opacity: f64,

    /// Show border
    pub show_border: bool,
}

/// Marker styling properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerStyle {
    /// Marker symbol
    pub symbol: MarkerSymbol,

    /// Marker size
    pub size: f64,

    /// Marker color
    pub color: String,

    /// Border color
    pub border_color: String,

    /// Border width
    pub border_width: f64,
}

/// Annotation styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationStyle {
    /// Text color
    pub color: String,

    /// Font size
    pub font_size: f64,

    /// Font weight
    pub font_weight: FontWeight,

    /// Background color
    pub background_color: Option<String>,

    /// Border
    pub border: Option<String>,
}

/// Line patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinePattern {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Marker symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkerSymbol {
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross,
    X,
    Star,
    Arrow,
}

/// Font weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
}

/// Annotation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationType {
    Statistical,
    Informational,
    Warning,
    Error,
}

/// Annotation position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationPosition {
    /// X coordinate
    pub x: PlotValue,

    /// Y coordinate
    pub y: f64,

    /// X anchor (left, center, right)
    pub x_anchor: String,

    /// Y anchor (top, middle, bottom)
    pub y_anchor: String,
}

/// Overall plot styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotStyling {
    /// Plot theme (light, dark, custom)
    pub theme: String,

    /// Color palette
    pub color_palette: Vec<String>,

    /// Grid settings
    pub grid: GridSettings,

    /// Axis settings
    pub axes: AxisSettings,

    /// Legend settings
    pub legend: LegendSettings,

    /// Title and labels
    pub labels: PlotLabels,
}

/// Grid settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSettings {
    /// Show grid
    pub show: bool,

    /// Grid color
    pub color: String,

    /// Grid opacity
    pub opacity: f64,

    /// Grid line width
    pub width: f64,
}

/// Axis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisSettings {
    /// X-axis title
    pub x_title: String,

    /// Y-axis title
    pub y_title: String,

    /// Show tick marks
    pub show_ticks: bool,

    /// Tick format
    pub tick_format: Option<String>,

    /// Axis ranges
    pub x_range: Option<(PlotValue, PlotValue)>,
    pub y_range: Option<(f64, f64)>,
}

/// Legend settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendSettings {
    /// Show legend
    pub show: bool,

    /// Legend position
    pub position: String,

    /// Legend orientation
    pub orientation: String,
}

/// Plot labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotLabels {
    /// Main title
    pub title: String,

    /// Subtitle
    pub subtitle: Option<String>,

    /// Caption
    pub caption: Option<String>,
}

/// Plot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Plot type identifier
    pub plot_type: String,

    /// Data source information
    pub data_source: Option<String>,

    /// Analysis parameters
    pub parameters: HashMap<String, String>,

    /// Statistical summary
    pub stats_summary: HashMap<String, f64>,
}

/// Generate trend plot data from time series and analysis results
pub fn generate_trend_plot_data(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<TrendPlotData, Box<dyn std::error::Error>> {
    if timestamps.len() != values.len() {
        return Err("Timestamps and values must have the same length".into());
    }

    if values.len() < 2 {
        return Err("Need at least 2 data points for trend plotting".into());
    }

    // Create time series
    let x_values: Vec<PlotValue> = timestamps.iter().map(|&t| PlotValue::Timestamp(t)).collect();
    let series = TimeSeries {
        x_values: x_values.clone(),
        y_values: values.to_vec(),
        error_bars: None,
        colors: None,
        sizes: None,
        labels: None,
        name: "Time Series".to_string(),
    };

    // Calculate basic linear trend
    let (slope, intercept, r_squared) = calculate_linear_trend(values)?;

    // Generate trend line
    let trend_y_values: Vec<f64> = (0..values.len())
        .map(|i| intercept + slope * i as f64)
        .collect();

    let direction = classify_direction(slope, r_squared);
    let trend_line = TrendLine {
        x_values: x_values.clone(),
        y_values: trend_y_values,
        direction: direction.clone(),
        strength: r_squared,
        p_value: None,
        style: LineStyle {
            color: trend_color_for_direction(&direction),
            width: 2.0,
            pattern: LinePattern::Solid,
            opacity: 0.8,
        },
    };

    // Default styling
    let styling = create_default_styling();

    // Metadata
    let metadata = PlotMetadata {
        created_at: Utc::now(),
        plot_type: "trend_analysis".to_string(),
        data_source: None,
        parameters: create_trend_parameters(slope, intercept, r_squared),
        stats_summary: create_trend_stats(values, slope, r_squared),
    };

    Ok(TrendPlotData {
        series,
        trend_line: Some(trend_line),
        confidence_bands: None,
        breakpoints: Vec::new(),
        rate_of_change: None,
        annotations: Vec::new(),
        styling,
        metadata,
    })
}

/// Generate comprehensive trend plot data with full analysis
pub fn generate_comprehensive_trend_plot_data(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    analysis: &ComprehensiveTrendResult,
) -> Result<TrendPlotData, Box<dyn std::error::Error>> {
    let mut plot_data = generate_trend_plot_data(timestamps, values)?;

    // Add confidence bands if available
    if let Some(ci) = analysis.linear_trend.confidence_interval {
        let confidence_bands = generate_confidence_bands(timestamps, values, &analysis.linear_trend, ci)?;
        plot_data.confidence_bands = Some(confidence_bands);
    }

    // Add breakpoint markers
    plot_data.breakpoints = analysis.breakpoints.iter()
        .map(|bp| create_breakpoint_marker(timestamps, values, bp))
        .collect();

    // Add rate of change plot if available
    if let Some(ref roc) = analysis.rate_of_change {
        plot_data.rate_of_change = Some(create_rate_of_change_plot(timestamps, roc)?);
    }

    // Add statistical annotations
    plot_data.annotations = create_trend_annotations(&analysis);

    // Update metadata with comprehensive analysis
    plot_data.metadata.parameters.extend(create_comprehensive_parameters(analysis));
    plot_data.metadata.stats_summary.extend(create_comprehensive_stats(analysis));

    Ok(plot_data)
}

/// Generate decomposition plot data
pub fn generate_decomposition_plot_data(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    decomposition: &DecompositionResult,
) -> Result<DecompositionPlotData, Box<dyn std::error::Error>> {
    if timestamps.len() != values.len() {
        return Err("Timestamps and values must have the same length".into());
    }

    let x_values: Vec<PlotValue> = timestamps.iter().map(|&t| PlotValue::Timestamp(t)).collect();

    // Original time series
    let original = TimeSeries {
        x_values: x_values.clone(),
        y_values: values.to_vec(),
        error_bars: None,
        colors: None,
        sizes: None,
        labels: None,
        name: "Original".to_string(),
    };

    // Trend component
    let trend = decomposition.trend.as_ref().map(|trend_values| TimeSeries {
        x_values: x_values.clone(),
        y_values: trend_values.clone(),
        error_bars: None,
        colors: None,
        sizes: None,
        labels: None,
        name: "Trend".to_string(),
    });

    // Seasonal component
    let seasonal = decomposition.seasonal.as_ref().map(|seasonal_values| TimeSeries {
        x_values: x_values.clone(),
        y_values: seasonal_values.clone(),
        error_bars: None,
        colors: None,
        sizes: None,
        labels: None,
        name: "Seasonal".to_string(),
    });

    // Residual component
    let residual = TimeSeries {
        x_values: x_values.clone(),
        y_values: decomposition.residual.clone(),
        error_bars: None,
        colors: None,
        sizes: None,
        labels: None,
        name: "Residual".to_string(),
    };

    // Reconstructed series if possible
    let reconstructed = if let (Some(ref trend_vals), Some(ref seasonal_vals)) =
        (&decomposition.trend, &decomposition.seasonal) {
        let is_additive = matches!(decomposition.method, DecompositionMethod::ClassicalAdditive | DecompositionMethod::Stl);
        let reconstructed_values: Vec<f64> = trend_vals.iter()
            .zip(seasonal_vals.iter())
            .map(|(&t, &s)| if is_additive { t + s } else { t * s })
            .collect();

        Some(TimeSeries {
            x_values: x_values.clone(),
            y_values: reconstructed_values,
            error_bars: None,
            colors: None,
            sizes: None,
            labels: None,
            name: "Reconstructed".to_string(),
        })
    } else {
        None
    };

    // Calculate component statistics
    let component_stats = calculate_component_statistics(&decomposition);

    // Layout configuration
    let num_components = 2 + trend.is_some() as usize + seasonal.is_some() as usize;
    let layout = DecompositionLayout {
        rows: num_components,
        subplot_titles: create_subplot_titles(&decomposition),
        shared_x_axis: true,
        y_axis_ranges: None,
        height_ratios: None,
    };

    // Styling
    let styling = create_default_styling();

    Ok(DecompositionPlotData {
        original,
        trend,
        seasonal,
        residual,
        reconstructed,
        component_stats,
        layout,
        styling,
    })
}

// Helper functions

fn calculate_linear_trend(values: &[f64]) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

    let denominator = n * sum_x2 - sum_x.powi(2);
    if denominator.abs() < f64::EPSILON {
        return Err("Cannot calculate linear trend: degenerate case".into());
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R-squared
    let mean_y = sum_y / n;
    let ss_tot: f64 = values.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = values.iter().enumerate()
        .map(|(i, &y)| {
            let predicted = intercept + slope * i as f64;
            (y - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 1.0 };

    Ok((slope, intercept, r_squared))
}

fn classify_direction(slope: f64, r_squared: f64) -> TrendDirection {
    if r_squared < 0.1 {
        return TrendDirection::Inconclusive;
    }

    match slope {
        s if s > 0.5 && r_squared > 0.8 => TrendDirection::StronglyIncreasing,
        s if s > 0.0 => TrendDirection::Increasing,
        s if s < -0.5 && r_squared > 0.8 => TrendDirection::StronglyDecreasing,
        s if s < 0.0 => TrendDirection::Decreasing,
        _ => TrendDirection::Stable,
    }
}

fn trend_color_for_direction(direction: &TrendDirection) -> String {
    match direction {
        TrendDirection::StronglyIncreasing => "#00AA00".to_string(),
        TrendDirection::Increasing => "#66CC66".to_string(),
        TrendDirection::Stable => "#CCCCCC".to_string(),
        TrendDirection::Decreasing => "#CC6666".to_string(),
        TrendDirection::StronglyDecreasing => "#AA0000".to_string(),
        TrendDirection::Inconclusive => "#FFAA00".to_string(),
    }
}

fn create_default_styling() -> PlotStyling {
    PlotStyling {
        theme: "light".to_string(),
        color_palette: vec![
            "#1f77b4".to_string(), "#ff7f0e".to_string(), "#2ca02c".to_string(),
            "#d62728".to_string(), "#9467bd".to_string(), "#8c564b".to_string(),
        ],
        grid: GridSettings {
            show: true,
            color: "#E0E0E0".to_string(),
            opacity: 0.5,
            width: 0.5,
        },
        axes: AxisSettings {
            x_title: "Time".to_string(),
            y_title: "Value".to_string(),
            show_ticks: true,
            tick_format: None,
            x_range: None,
            y_range: None,
        },
        legend: LegendSettings {
            show: true,
            position: "top-right".to_string(),
            orientation: "vertical".to_string(),
        },
        labels: PlotLabels {
            title: "Trend Analysis".to_string(),
            subtitle: None,
            caption: None,
        },
    }
}

fn generate_confidence_bands(
    timestamps: &[DateTime<Utc>],
    _values: &[f64],
    trend: &crate::analysis::TrendAnalysis,
    ci: (f64, f64),
) -> Result<ConfidenceBands, Box<dyn std::error::Error>> {
    let x_values: Vec<PlotValue> = timestamps.iter().map(|&t| PlotValue::Timestamp(t)).collect();

    let upper_bound: Vec<f64> = (0..timestamps.len())
        .map(|i| trend.intercept + (trend.slope + ci.1) * i as f64)
        .collect();

    let lower_bound: Vec<f64> = (0..timestamps.len())
        .map(|i| trend.intercept + (trend.slope + ci.0) * i as f64)
        .collect();

    Ok(ConfidenceBands {
        x_values,
        upper_bound,
        lower_bound,
        confidence_level: 0.95,
        style: BandStyle {
            fill_color: "#1f77b4".to_string(),
            border_color: "#1f77b4".to_string(),
            opacity: 0.2,
            show_border: false,
        },
    })
}

fn create_breakpoint_marker(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    breakpoint: &BreakpointDetection,
) -> BreakpointMarker {
    let x_position = if breakpoint.index < timestamps.len() {
        PlotValue::Timestamp(timestamps[breakpoint.index])
    } else {
        PlotValue::Numeric(breakpoint.index as f64)
    };

    let y_position = if breakpoint.index < values.len() {
        values[breakpoint.index]
    } else {
        breakpoint.mean_after
    };

    BreakpointMarker {
        x_position,
        y_position,
        confidence: breakpoint.confidence,
        change_type: format!("{:?}", breakpoint.change_type),
        magnitude: breakpoint.magnitude,
        style: MarkerStyle {
            symbol: MarkerSymbol::Triangle,
            size: 10.0 + (breakpoint.confidence * 10.0),
            color: "#FF4444".to_string(),
            border_color: "#AA0000".to_string(),
            border_width: 1.0,
        },
        annotation: Some(format!("Change: {:.2}", breakpoint.magnitude)),
    }
}

fn create_rate_of_change_plot(
    timestamps: &[DateTime<Utc>],
    roc: &RateOfChangeAnalysis,
) -> Result<RateOfChangePlot, Box<dyn std::error::Error>> {
    let x_values: Vec<PlotValue> = roc.time_indices.iter()
        .filter_map(|&idx| {
            if idx < timestamps.len() {
                Some(PlotValue::Timestamp(timestamps[idx]))
            } else {
                None
            }
        })
        .collect();

    Ok(RateOfChangePlot {
        x_values,
        rates: roc.rates.clone(),
        acceleration: roc.acceleration.clone(),
        zero_line: true,
        smoothed_rates: None,
        style: LineStyle {
            color: "#FF7F0E".to_string(),
            width: 1.5,
            pattern: LinePattern::Solid,
            opacity: 0.8,
        },
    })
}

fn create_trend_annotations(analysis: &ComprehensiveTrendResult) -> Vec<PlotAnnotation> {
    let mut annotations = Vec::new();

    // Add trend strength annotation
    let strength_text = format!("Trend Strength: {:.2}", analysis.strength);
    annotations.push(PlotAnnotation {
        text: strength_text,
        position: AnnotationPosition {
            x: PlotValue::Numeric(0.05),
            y: 0.95,
            x_anchor: "left".to_string(),
            y_anchor: "top".to_string(),
        },
        style: AnnotationStyle {
            color: "#333333".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            background_color: Some("#FFFFFF".to_string()),
            border: Some("#CCCCCC".to_string()),
        },
        annotation_type: AnnotationType::Statistical,
    });

    // Add R-squared annotation
    let r2_text = format!("R² = {:.3}", analysis.linear_trend.r_squared);
    annotations.push(PlotAnnotation {
        text: r2_text,
        position: AnnotationPosition {
            x: PlotValue::Numeric(0.05),
            y: 0.88,
            x_anchor: "left".to_string(),
            y_anchor: "top".to_string(),
        },
        style: AnnotationStyle {
            color: "#333333".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            background_color: Some("#FFFFFF".to_string()),
            border: Some("#CCCCCC".to_string()),
        },
        annotation_type: AnnotationType::Statistical,
    });

    annotations
}

fn calculate_component_statistics(decomposition: &DecompositionResult) -> ComponentStatistics {
    let total_var = if let Some(ref trend) = decomposition.trend {
        calculate_variance(trend)
    } else {
        0.0
    };

    let seasonal_var = if let Some(ref seasonal) = decomposition.seasonal {
        calculate_variance(seasonal)
    } else {
        0.0
    };

    let residual_var = calculate_variance(&decomposition.residual);
    let sum_var = total_var + seasonal_var + residual_var;

    let (trend_pct, seasonal_pct, residual_pct) = if sum_var > 0.0 {
        (
            total_var / sum_var * 100.0,
            seasonal_var / sum_var * 100.0,
            residual_var / sum_var * 100.0,
        )
    } else {
        (0.0, 0.0, 100.0)
    };

    let signal_to_noise = if residual_var > 0.0 {
        (total_var + seasonal_var) / residual_var
    } else {
        f64::INFINITY
    };

    let quality_score = if sum_var > 0.0 {
        (total_var + seasonal_var) / sum_var
    } else {
        0.0
    };

    ComponentStatistics {
        trend_variance_pct: trend_pct,
        seasonal_variance_pct: seasonal_pct,
        residual_variance_pct: residual_pct,
        signal_to_noise_ratio: signal_to_noise,
        quality_score,
    }
}

fn create_subplot_titles(decomposition: &DecompositionResult) -> Vec<String> {
    let mut titles = vec!["Original".to_string()];

    if decomposition.trend.is_some() {
        titles.push("Trend".to_string());
    }

    if decomposition.seasonal.is_some() {
        titles.push("Seasonal".to_string());
    }

    titles.push("Residual".to_string());

    titles
}

fn calculate_variance(data: &[f64]) -> f64 {
    let valid_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if valid_data.len() <= 1 {
        return 0.0;
    }

    let mean = valid_data.iter().sum::<f64>() / valid_data.len() as f64;
    valid_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / valid_data.len() as f64
}

fn create_trend_parameters(slope: f64, intercept: f64, r_squared: f64) -> HashMap<String, String> {
    let mut params = HashMap::new();
    params.insert("slope".to_string(), format!("{:.6}", slope));
    params.insert("intercept".to_string(), format!("{:.6}", intercept));
    params.insert("r_squared".to_string(), format!("{:.6}", r_squared));
    params
}

fn create_trend_stats(values: &[f64], slope: f64, r_squared: f64) -> HashMap<String, f64> {
    let mut stats = HashMap::new();
    stats.insert("slope".to_string(), slope);
    stats.insert("r_squared".to_string(), r_squared);
    stats.insert("n_points".to_string(), values.len() as f64);

    if !values.is_empty() {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        stats.insert("mean".to_string(), mean);
        stats.insert("std".to_string(), variance.sqrt());
    }

    stats
}

fn create_comprehensive_parameters(analysis: &ComprehensiveTrendResult) -> HashMap<String, String> {
    let mut params = HashMap::new();
    params.insert("direction".to_string(), format!("{:?}", analysis.direction));
    params.insert("strength".to_string(), format!("{:.3}", analysis.strength));
    params.insert("confidence".to_string(), format!("{:.3}", analysis.confidence));

    if let Some(growth_rate) = analysis.growth_rate {
        params.insert("growth_rate".to_string(), format!("{:.6}", growth_rate));
    }

    params.insert("n_breakpoints".to_string(), analysis.breakpoints.len().to_string());

    params
}

fn create_comprehensive_stats(analysis: &ComprehensiveTrendResult) -> HashMap<String, f64> {
    let mut stats = HashMap::new();
    stats.insert("strength".to_string(), analysis.strength);
    stats.insert("confidence".to_string(), analysis.confidence);

    if let Some(growth_rate) = analysis.growth_rate {
        stats.insert("growth_rate".to_string(), growth_rate);
    }

    stats.insert("n_breakpoints".to_string(), analysis.breakpoints.len() as f64);
    stats.insert("max_consecutive_direction".to_string(), analysis.persistence.max_consecutive_direction as f64);
    stats.insert("directional_consistency".to_string(), analysis.persistence.directional_consistency);

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Vec<DateTime<Utc>>, Vec<f64>) {
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| now + chrono::Duration::days(i))
            .collect();
        let values = vec![1.0, 2.1, 3.2, 4.1, 5.3, 6.0, 7.2, 8.1, 9.3, 10.1];
        (timestamps, values)
    }

    #[test]
    fn test_generate_trend_plot_data() {
        let (timestamps, values) = create_test_data();
        let plot_data = generate_trend_plot_data(&timestamps, &values).unwrap();

        assert_eq!(plot_data.series.x_values.len(), 10);
        assert_eq!(plot_data.series.y_values.len(), 10);
        assert!(plot_data.trend_line.is_some());

        let trend_line = plot_data.trend_line.unwrap();
        assert_eq!(trend_line.direction, TrendDirection::Increasing);
        assert!(trend_line.strength > 0.9);
    }

    #[test]
    fn test_plot_value_serialization() {
        let timestamp_val = PlotValue::Timestamp(Utc::now());
        let numeric_val = PlotValue::Numeric(42.0);
        let category_val = PlotValue::Category("test".to_string());

        // Test that serialization works (basic check)
        let _json1 = serde_json::to_string(&timestamp_val).unwrap();
        let _json2 = serde_json::to_string(&numeric_val).unwrap();
        let _json3 = serde_json::to_string(&category_val).unwrap();
    }

    #[test]
    fn test_trend_color_mapping() {
        assert_eq!(trend_color_for_direction(&TrendDirection::StronglyIncreasing), "#00AA00");
        assert_eq!(trend_color_for_direction(&TrendDirection::StronglyDecreasing), "#AA0000");
        assert_eq!(trend_color_for_direction(&TrendDirection::Stable), "#CCCCCC");
    }

    #[test]
    fn test_component_statistics() {
        // Mock decomposition result
        let decomposition = DecompositionResult {
            method: DecompositionMethod::ClassicalAdditive,
            trend: Some(vec![1.0, 2.0, 3.0, 4.0]),
            seasonal: Some(vec![0.1, -0.1, 0.1, -0.1]),
            residual: vec![0.01, -0.01, 0.01, -0.01],
            seasonal_periods: vec![],
            quality_metrics: DecompositionQuality {
                r_squared: 0.95,
                mae_residuals: 0.01,
                std_residuals: 0.01,
                seasonality_strength: 0.1,
                trend_strength: 0.9,
            },
            metadata: std::collections::HashMap::new(),
        };

        let stats = calculate_component_statistics(&decomposition);
        assert!(stats.trend_variance_pct > 80.0);
        assert!(stats.quality_score > 0.8);
        assert!(stats.signal_to_noise_ratio > 1.0);
    }

    #[test]
    fn test_invalid_input_handling() {
        let timestamps = vec![Utc::now()];
        let values = vec![1.0, 2.0]; // Mismatched lengths

        let result = generate_trend_plot_data(&timestamps, &values);
        assert!(result.is_err());

        let empty_values = vec![];
        let result2 = generate_trend_plot_data(&[], &empty_values);
        assert!(result2.is_err());
    }

    #[test]
    fn test_styling_defaults() {
        let styling = create_default_styling();
        assert_eq!(styling.theme, "light");
        assert!(!styling.color_palette.is_empty());
        assert!(styling.grid.show);
        assert!(styling.legend.show);
    }
}