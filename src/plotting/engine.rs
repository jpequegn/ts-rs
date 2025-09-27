//! # Plotting Engine
//!
//! Core plotting engine using plotly for rendering interactive and static plots.

use crate::plotting::types::*;
use plotly::{Plot, Scatter, Histogram, HeatMap, Layout, color::NamedColor};
use plotly::common::{Mode, Title, Visible, TickMode, Font};
use plotly::layout::{Axis, GridPattern, RowOrder, LayoutGrid, Annotation, TicksDirection};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Main plotting engine
pub struct PlotEngine {
    config: PlotConfig,
    plot: Plot,
}

impl PlotEngine {
    /// Create a new plotting engine with configuration
    pub fn new(config: PlotConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut plot = Plot::new();

        // Apply theme and basic layout
        let layout = create_layout(&config)?;
        plot.set_layout(layout);

        Ok(Self { config, plot })
    }

    /// Create a line plot
    pub fn create_line_plot(&self, series: &[PlotSeries]) -> Result<PlotResult, Box<dyn std::error::Error>> {
        let mut plot = self.plot.clone();

        for (idx, series_data) in series.iter().enumerate() {
            let x_values = extract_x_values(&series_data.x_values)?;
            let trace = Scatter::new(x_values, series_data.y_values.clone())
                .name(&series_data.name)
                .mode(Mode::Lines);

            plot.add_trace(trace);
        }

        self.finalize_plot(plot, PlotType::Line, series)
    }

    /// Create a scatter plot
    pub fn create_scatter_plot(&self, series: &[PlotSeries]) -> Result<PlotResult, Box<dyn std::error::Error>> {
        let mut plot = self.plot.clone();

        for (idx, series_data) in series.iter().enumerate() {
            let x_values = extract_x_values(&series_data.x_values)?;
            let trace = Scatter::new(x_values, series_data.y_values.clone())
                .name(&series_data.name)
                .mode(Mode::Markers);

            plot.add_trace(trace);
        }

        self.finalize_plot(plot, PlotType::Scatter, series)
    }

    /// Create a histogram
    pub fn create_histogram(&self, values: &[f64]) -> Result<PlotResult, Box<dyn std::error::Error>> {
        let mut plot = self.plot.clone();

        let trace = Histogram::new(values.to_vec())
            .name("Distribution");

        plot.add_trace(trace);

        let series = vec![PlotSeries {
            name: "Histogram".to_string(),
            x_values: Vec::new(),
            y_values: values.to_vec(),
            series_type: SeriesType::Histogram,
            color: None,
            style: None,
        }];

        self.finalize_plot(plot, PlotType::Histogram, &series)
    }

    /// Create a box plot (placeholder implementation)
    pub fn create_box_plot(&self, data: &HashMap<String, Vec<f64>>) -> Result<PlotResult, Box<dyn std::error::Error>> {
        // For now, create a scatter plot as placeholder for box plot
        // In a full implementation, this would use the correct Box plot type
        let mut plot_data = Vec::new();
        for (name, values) in data {
            for (i, &value) in values.iter().enumerate() {
                plot_data.push((i, value, name.clone()));
            }
        }

        let series: Vec<PlotSeries> = data.iter().map(|(name, values)| PlotSeries {
            name: name.clone(),
            x_values: (0..values.len()).map(PlotPoint::Index).collect(),
            y_values: values.clone(),
            series_type: SeriesType::Box,
            color: None,
            style: None,
        }).collect();

        self.finalize_plot(self.plot.clone(), PlotType::BoxPlot, &series)
    }

    /// Create a correlation heatmap
    pub fn create_correlation_heatmap(&self, data: &HashMap<String, Vec<f64>>) -> Result<PlotResult, Box<dyn std::error::Error>> {
        // Calculate correlation matrix
        let correlation_matrix = calculate_correlation_matrix(data)?;
        let variables: Vec<String> = data.keys().cloned().collect();

        let mut plot = self.plot.clone();

        let trace = HeatMap::new(
            variables.clone(),
            variables.clone(),
            correlation_matrix,
        );

        plot.add_trace(trace);

        let series = vec![PlotSeries {
            name: "Correlation Matrix".to_string(),
            x_values: Vec::new(),
            y_values: Vec::new(),
            series_type: SeriesType::Box, // Using Box as placeholder
            color: None,
            style: None,
        }];

        self.finalize_plot(plot, PlotType::Heatmap, &series)
    }

    /// Create decomposition plot with subplots
    pub fn create_decomposition_plot(&self, series: &[PlotSeries]) -> Result<PlotResult, Box<dyn std::error::Error>> {
        if series.is_empty() {
            return Err("No series data provided for decomposition plot".into());
        }

        // For now, create a simple multi-series plot
        // In a full implementation, this would create subplots for trend, seasonal, residual components
        self.create_line_plot(series)
    }

    /// Finalize plot and create result
    fn finalize_plot(&self, plot: Plot, plot_type: PlotType, series: &[PlotSeries]) -> Result<PlotResult, Box<dyn std::error::Error>> {
        let data_points = series.iter().map(|s| s.y_values.len()).sum();

        let metadata = PlotMetadata {
            plot_type,
            created_at: Utc::now(),
            data_points,
            series_count: series.len(),
            dimensions: (self.config.width, self.config.height),
            theme: self.config.theme.clone(),
        };

        let content = match &self.config.export_format {
            ExportFormat::HTML => PlotContent::HTML(plot.to_html()),
            ExportFormat::JSON => PlotContent::JSON(serde_json::to_value(&plot)?),
            _ => PlotContent::HTML(plot.to_html()), // Default to HTML for now
        };

        Ok(PlotResult {
            content,
            metadata,
            export_info: None,
        })
    }
}

/// Create plot layout based on configuration
fn create_layout(config: &PlotConfig) -> Result<Layout, Box<dyn std::error::Error>> {
    let mut layout = Layout::new()
        .width(config.width)
        .height(config.height)
        .show_legend(config.show_legend);

    // Set title if provided
    if let Some(ref title) = config.title {
        layout = layout.title(Title::new(title));
    }

    // Set axis labels
    if let Some(ref x_label) = config.x_label {
        layout = layout.x_axis(Axis::new().title(Title::new(x_label)));
    }

    if let Some(ref y_label) = config.y_label {
        layout = layout.y_axis(Axis::new().title(Title::new(y_label)));
    }

    // Apply theme
    layout = apply_theme_to_layout(layout, &config.theme)?;

    Ok(layout)
}

/// Apply theme styling to layout
fn apply_theme_to_layout(mut layout: Layout, theme: &Theme) -> Result<Layout, Box<dyn std::error::Error>> {
    match theme {
        Theme::Default => {
            layout = layout.paper_background_color("#FFFFFF")
                .plot_background_color("#FFFFFF");
        },
        Theme::Dark => {
            layout = layout.paper_background_color("#2F2F2F")
                .plot_background_color("#3F3F3F")
                .font(Font::new().color("#FFFFFF"));
        },
        Theme::Publication => {
            layout = layout.paper_background_color("#FFFFFF")
                .plot_background_color("#FFFFFF")
                .font(Font::new().family("Times New Roman").size(12));
        },
        Theme::HighContrast => {
            layout = layout.paper_background_color("#000000")
                .plot_background_color("#000000")
                .font(Font::new().color("#FFFF00"));
        },
        Theme::Custom(theme_config) => {
            layout = layout.paper_background_color(theme_config.background.clone())
                .plot_background_color(theme_config.background.clone())
                .font(Font::new().color(theme_config.text_color.clone()));
        },
    }

    Ok(layout)
}

/// Extract x-values for plotting
fn extract_x_values(plot_points: &[PlotPoint]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut x_values = Vec::new();

    for point in plot_points {
        let value = match point {
            PlotPoint::Timestamp(dt, _) => dt.format("%Y-%m-%d %H:%M:%S").to_string(),
            PlotPoint::Numeric(n) => n.to_string(),
            PlotPoint::Index(i) => i.to_string(),
            PlotPoint::Category(s) => s.clone(),
        };
        x_values.push(value);
    }

    Ok(x_values)
}

/// Get color for series based on index and theme
fn get_color_for_series(index: usize, theme: &Theme) -> NamedColor {
    let colors = match theme {
        Theme::Default => vec![
            NamedColor::Blue, NamedColor::Orange, NamedColor::Green,
            NamedColor::Red, NamedColor::Purple, NamedColor::Brown,
        ],
        Theme::Dark => vec![
            NamedColor::CornflowerBlue, NamedColor::Orange, NamedColor::LimeGreen,
            NamedColor::Crimson, NamedColor::Violet, NamedColor::SaddleBrown,
        ],
        Theme::Publication => vec![
            NamedColor::Black, NamedColor::Gray, NamedColor::DarkGray,
            NamedColor::DimGray, NamedColor::LightGray, NamedColor::SlateGray,
        ],
        Theme::HighContrast => vec![
            NamedColor::Yellow, NamedColor::Cyan, NamedColor::Magenta,
            NamedColor::Lime, NamedColor::Red, NamedColor::White,
        ],
        Theme::Custom(_) => vec![
            NamedColor::Blue, NamedColor::Orange, NamedColor::Green,
            NamedColor::Red, NamedColor::Purple, NamedColor::Brown,
        ],
    };

    colors[index % colors.len()]
}

/// Calculate correlation matrix for heatmap
fn calculate_correlation_matrix(data: &HashMap<String, Vec<f64>>) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let n = variables.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 1.0;
            } else {
                let series1 = &data[&variables[i]];
                let series2 = &data[&variables[j]];
                matrix[i][j] = calculate_pearson_correlation(series1, series2)?;
            }
        }
    }

    Ok(matrix)
}

/// Calculate Pearson correlation coefficient
fn calculate_pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if x.len() != y.len() || x.is_empty() {
        return Err("Series must have the same non-zero length".into());
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|b| b * b).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        Ok(0.0) // No correlation when there's no variance
    } else {
        Ok(numerator / denominator)
    }
}

/// Convenience function to create a plot
pub fn create_plot(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    crate::plotting::plot(data, timestamps, config)
}

/// Convenience function to render a plot
pub fn render_plot(plot_result: &PlotResult) -> Result<String, Box<dyn std::error::Error>> {
    match &plot_result.content {
        PlotContent::HTML(html) => Ok(html.clone()),
        PlotContent::JSON(json) => Ok(serde_json::to_string_pretty(json)?),
        PlotContent::FilePath(path) => Ok(format!("Plot saved to: {}", path)),
        PlotContent::ImageData(data) => Ok(format!("Image data (length: {})", data.len())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_data() -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("series2".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        data
    }

    #[test]
    fn test_plot_engine_creation() {
        let config = PlotConfig::default();
        let engine = PlotEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_calculate_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = calculate_pearson_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    fn test_calculate_correlation_matrix() {
        let data = create_test_data();
        let matrix = calculate_correlation_matrix(&data).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10); // Diagonal should be 1.0
    }

    #[test]
    fn test_extract_x_values() {
        let points = vec![
            PlotPoint::Index(0),
            PlotPoint::Index(1),
            PlotPoint::Numeric(2.5),
        ];

        let x_values = extract_x_values(&points).unwrap();
        assert_eq!(x_values, vec!["0", "1", "2.5"]);
    }
}