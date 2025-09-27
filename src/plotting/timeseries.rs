//! # Time Series Plotting Functions
//!
//! Specialized functions for creating time series plots.

use crate::plotting::types::*;
use crate::plotting::engine::PlotEngine;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Create a line plot for time series data
pub fn create_line_plot(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Line;

    let engine = PlotEngine::new(config)?;
    let series = prepare_time_series_data(data, timestamps)?;
    engine.create_line_plot(&series)
}

/// Create a scatter plot for time series data
pub fn create_scatter_plot(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Scatter;

    let engine = PlotEngine::new(config)?;
    let series = prepare_time_series_data(data, timestamps)?;
    engine.create_scatter_plot(&series)
}

/// Create a multiple series plot
pub fn create_multiple_series_plot(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    create_line_plot(data, timestamps, config)
}

/// Create subplot layout for multiple time series
pub fn create_subplot_layout(
    data: &[HashMap<String, Vec<f64>>],
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    // For now, create a simple plot with the first dataset
    // In a full implementation, this would create actual subplots
    if let Some(first_data) = data.first() {
        create_line_plot(first_data, timestamps, config)
    } else {
        Err("No data provided for subplot layout".into())
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
            series_type: SeriesType::Line,
            color: None,
            style: None,
        });
    }

    Ok(series)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("series2".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        data
    }

    #[test]
    fn test_create_line_plot() {
        let data = create_test_data();
        let config = PlotConfig::default();
        let result = create_line_plot(&data, None, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_scatter_plot() {
        let data = create_test_data();
        let config = PlotConfig::default();
        let result = create_scatter_plot(&data, None, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prepare_time_series_data() {
        let data = create_test_data();
        let series = prepare_time_series_data(&data, None).unwrap();

        assert_eq!(series.len(), 2);
        assert_eq!(series[0].x_values.len(), 5);
        assert_eq!(series[0].y_values.len(), 5);
    }
}