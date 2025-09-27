//! # Data Visualization and Plotting Module
//!
//! Comprehensive data visualization capabilities for time series analysis,
//! providing interactive and static plots with multiple export formats.

pub mod engine;
pub mod types;
pub mod timeseries;
pub mod statistical;
pub mod correlation;
pub mod decomposition;
pub mod styling;
pub mod export;

// Re-export main types and functions
pub use types::{
    PlotConfig, PlotType, ExportFormat, Theme,
    PlotResult, PlotData, PlotPoint, PlotSeries
};

pub use engine::{PlotEngine, create_plot, render_plot};

pub use timeseries::{
    create_line_plot, create_scatter_plot, create_multiple_series_plot,
    create_subplot_layout
};

pub use statistical::{
    create_histogram, create_box_plot, create_violin_plot, create_qq_plot,
    create_acf_plot, create_pacf_plot, create_density_plot
};

pub use correlation::{
    create_correlation_heatmap, create_scatter_matrix, create_correlation_plot
};

pub use decomposition::{
    create_decomposition_plot, create_seasonal_plot, create_trend_plot,
    create_forecast_plot, create_anomaly_plot
};

pub use styling::{
    DefaultTheme, ProfessionalTheme, PublicationTheme, DarkTheme,
    apply_theme, customize_styling
};

pub use export::{
    export_to_file, export_to_html, export_to_png, export_to_svg, export_to_pdf,
    ExportOptions
};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Main plotting function for creating various types of plots
pub fn plot(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let engine = PlotEngine::new(config.clone())?;

    match config.plot_type {
        PlotType::Line => {
            let series = prepare_time_series_data(data, timestamps)?;
            engine.create_line_plot(&series)
        },
        PlotType::Scatter => {
            let series = prepare_time_series_data(data, timestamps)?;
            engine.create_scatter_plot(&series)
        },
        PlotType::Histogram => {
            let values = get_single_series_values(data, &config.primary_column)?;
            engine.create_histogram(&values)
        },
        PlotType::BoxPlot => {
            let values = collect_all_series_values(data)?;
            engine.create_box_plot(&values)
        },
        PlotType::Heatmap => {
            engine.create_correlation_heatmap(data)
        },
        PlotType::Decomposition => {
            let series = prepare_time_series_data(data, timestamps)?;
            engine.create_decomposition_plot(&series)
        },
        _ => Err("Plot type not yet implemented".into()),
    }
}

/// Prepare time series data for plotting
fn prepare_time_series_data(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
) -> Result<Vec<PlotSeries>, Box<dyn std::error::Error>> {
    let mut series = Vec::new();

    for (name, values) in data {
        let x_values = if let Some(ts) = timestamps {
            if ts.len() != values.len() {
                return Err(format!("Timestamp length ({}) doesn't match data length ({}) for series '{}'",
                                   ts.len(), values.len(), name).into());
            }
            ts.iter().enumerate().map(|(i, &t)| PlotPoint::Timestamp(t, i)).collect()
        } else {
            (0..values.len()).map(PlotPoint::Index).collect()
        };

        series.push(PlotSeries {
            name: name.clone(),
            x_values,
            y_values: values.clone(),
            series_type: types::SeriesType::Line,
            color: None,
            style: None,
        });
    }

    Ok(series)
}

/// Get values for a single series
fn get_single_series_values(
    data: &HashMap<String, Vec<f64>>,
    column: &str,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    data.get(column)
        .ok_or_else(|| format!("Column '{}' not found in data", column).into())
        .map(|v| v.clone())
}

/// Collect all series values for multi-series plots
fn collect_all_series_values(
    data: &HashMap<String, Vec<f64>>,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    Ok(data.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_prepare_time_series_data() {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.0, 3.0]);
        data.insert("series2".to_string(), vec![4.0, 5.0, 6.0]);

        let series = prepare_time_series_data(&data, None).unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].y_values, vec![1.0, 2.0, 3.0]);
        assert_eq!(series[1].y_values, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_prepare_time_series_data_with_timestamps() {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.0, 3.0]);

        let timestamps = vec![
            Utc::now(),
            Utc::now() + chrono::Duration::days(1),
            Utc::now() + chrono::Duration::days(2),
        ];

        let series = prepare_time_series_data(&data, Some(&timestamps)).unwrap();
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].x_values.len(), 3);
    }

    #[test]
    fn test_get_single_series_values() {
        let mut data = HashMap::new();
        data.insert("test".to_string(), vec![1.0, 2.0, 3.0]);

        let values = get_single_series_values(&data, "test").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        let result = get_single_series_values(&data, "nonexistent");
        assert!(result.is_err());
    }
}