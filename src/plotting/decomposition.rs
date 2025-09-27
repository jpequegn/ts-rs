//! # Decomposition and Forecast Plotting Functions
//!
//! Functions for creating trend decomposition plots, seasonal pattern visualizations,
//! anomaly highlighting, and forecast plots with confidence intervals.

use crate::plotting::types::*;
use crate::plotting::engine::PlotEngine;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Create a decomposition plot showing trend, seasonal, and residual components
pub fn create_decomposition_plot(
    components: &DecompositionComponents,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Decomposition;

    let series = prepare_decomposition_series(components, timestamps)?;
    let engine = PlotEngine::new(config)?;
    engine.create_decomposition_plot(&series)
}

/// Create a seasonal pattern visualization
pub fn create_seasonal_plot(
    seasonal_data: &[f64],
    period: usize,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::SeasonalPattern;
    config.title = Some("Seasonal Pattern".to_string());

    let mut data = HashMap::new();
    data.insert("Seasonal".to_string(), seasonal_data.to_vec());

    let series = prepare_time_series_data(&data, timestamps)?;
    let engine = PlotEngine::new(config)?;
    engine.create_line_plot(&series)
}

/// Create a trend plot highlighting the long-term trend component
pub fn create_trend_plot(
    trend_data: &[f64],
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Line;
    config.title = Some("Trend Component".to_string());

    let mut data = HashMap::new();
    data.insert("Trend".to_string(), trend_data.to_vec());

    let series = prepare_time_series_data(&data, timestamps)?;
    let engine = PlotEngine::new(config)?;
    engine.create_line_plot(&series)
}

/// Create a forecast plot with confidence intervals
pub fn create_forecast_plot(
    historical_data: &[f64],
    forecast_data: &[f64],
    confidence_intervals: Option<&ConfidenceIntervals>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Forecast;
    config.title = Some("Forecast with Confidence Intervals".to_string());

    let series = prepare_forecast_series(
        historical_data,
        forecast_data,
        confidence_intervals,
        timestamps,
    )?;

    let engine = PlotEngine::new(config)?;
    engine.create_line_plot(&series)
}

/// Create an anomaly highlighting plot
pub fn create_anomaly_plot(
    data: &[f64],
    anomaly_indices: &[usize],
    timestamps: Option<&[DateTime<Utc>]>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::AnomalyHighlight;
    config.title = Some("Data with Anomalies Highlighted".to_string());

    let series = prepare_anomaly_series(data, anomaly_indices, timestamps)?;
    let engine = PlotEngine::new(config)?;
    engine.create_scatter_plot(&series)
}

/// Decomposition components structure
pub struct DecompositionComponents {
    pub original: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

/// Confidence intervals for forecasts
pub struct ConfidenceIntervals {
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub confidence_level: f64,
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
                return Err(format!(
                    "Timestamp length ({}) doesn't match data length ({}) for series '{}'",
                    ts.len(),
                    values.len(),
                    name
                )
                .into());
            }
            ts.iter()
                .enumerate()
                .map(|(i, &t)| PlotPoint::Timestamp(t, i))
                .collect()
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

/// Prepare decomposition components for plotting
fn prepare_decomposition_series(
    components: &DecompositionComponents,
    timestamps: Option<&[DateTime<Utc>]>,
) -> Result<Vec<PlotSeries>, Box<dyn std::error::Error>> {
    let mut data = HashMap::new();
    data.insert("Original".to_string(), components.original.clone());
    data.insert("Trend".to_string(), components.trend.clone());
    data.insert("Seasonal".to_string(), components.seasonal.clone());
    data.insert("Residual".to_string(), components.residual.clone());

    prepare_time_series_data(&data, timestamps)
}

/// Prepare forecast data with confidence intervals for plotting
fn prepare_forecast_series(
    historical_data: &[f64],
    forecast_data: &[f64],
    confidence_intervals: Option<&ConfidenceIntervals>,
    timestamps: Option<&[DateTime<Utc>]>,
) -> Result<Vec<PlotSeries>, Box<dyn std::error::Error>> {
    let mut series = Vec::new();
    let total_length = historical_data.len() + forecast_data.len();

    // Prepare historical data
    let historical_x_values = if let Some(ts) = timestamps {
        if ts.len() < historical_data.len() {
            return Err("Not enough timestamps for historical data".into());
        }
        ts[..historical_data.len()]
            .iter()
            .enumerate()
            .map(|(i, &t)| PlotPoint::Timestamp(t, i))
            .collect()
    } else {
        (0..historical_data.len()).map(PlotPoint::Index).collect()
    };

    series.push(PlotSeries {
        name: "Historical".to_string(),
        x_values: historical_x_values,
        y_values: historical_data.to_vec(),
        series_type: SeriesType::Line,
        color: Some("#1f77b4".to_string()),
        style: None,
    });

    // Prepare forecast data
    let forecast_start_idx = historical_data.len();
    let forecast_x_values = if let Some(ts) = timestamps {
        if ts.len() < total_length {
            return Err("Not enough timestamps for forecast data".into());
        }
        ts[forecast_start_idx..total_length]
            .iter()
            .enumerate()
            .map(|(i, &t)| PlotPoint::Timestamp(t, forecast_start_idx + i))
            .collect()
    } else {
        (forecast_start_idx..total_length)
            .map(PlotPoint::Index)
            .collect()
    };

    series.push(PlotSeries {
        name: "Forecast".to_string(),
        x_values: forecast_x_values.clone(),
        y_values: forecast_data.to_vec(),
        series_type: SeriesType::Line,
        color: Some("#ff7f0e".to_string()),
        style: Some(SeriesStyle {
            line_style: Some("dashed".to_string()),
            line_width: Some(2.0),
            marker_size: None,
            opacity: None,
            fill_color: None,
        }),
    });

    // Add confidence intervals if provided
    if let Some(ci) = confidence_intervals {
        if ci.lower_bound.len() != forecast_data.len() || ci.upper_bound.len() != forecast_data.len() {
            return Err("Confidence interval length doesn't match forecast data length".into());
        }

        series.push(PlotSeries {
            name: format!("{}% Confidence Lower", (ci.confidence_level * 100.0) as i32),
            x_values: forecast_x_values.clone(),
            y_values: ci.lower_bound.clone(),
            series_type: SeriesType::Line,
            color: Some("#d62728".to_string()),
            style: Some(SeriesStyle {
                line_style: Some("dotted".to_string()),
                line_width: Some(1.0),
                marker_size: None,
                opacity: Some(0.7),
                fill_color: None,
            }),
        });

        series.push(PlotSeries {
            name: format!("{}% Confidence Upper", (ci.confidence_level * 100.0) as i32),
            x_values: forecast_x_values,
            y_values: ci.upper_bound.clone(),
            series_type: SeriesType::Line,
            color: Some("#d62728".to_string()),
            style: Some(SeriesStyle {
                line_style: Some("dotted".to_string()),
                line_width: Some(1.0),
                marker_size: None,
                opacity: Some(0.7),
                fill_color: None,
            }),
        });
    }

    Ok(series)
}

/// Prepare anomaly data for plotting
fn prepare_anomaly_series(
    data: &[f64],
    anomaly_indices: &[usize],
    timestamps: Option<&[DateTime<Utc>]>,
) -> Result<Vec<PlotSeries>, Box<dyn std::error::Error>> {
    let mut series = Vec::new();

    // Normal data points
    let normal_x_values = if let Some(ts) = timestamps {
        if ts.len() != data.len() {
            return Err("Timestamp length doesn't match data length".into());
        }
        ts.iter()
            .enumerate()
            .map(|(i, &t)| PlotPoint::Timestamp(t, i))
            .collect()
    } else {
        (0..data.len()).map(PlotPoint::Index).collect()
    };

    series.push(PlotSeries {
        name: "Normal Data".to_string(),
        x_values: normal_x_values,
        y_values: data.to_vec(),
        series_type: SeriesType::Line,
        color: Some("#1f77b4".to_string()),
        style: None,
    });

    // Anomaly points
    if !anomaly_indices.is_empty() {
        let anomaly_x_values: Vec<PlotPoint> = anomaly_indices
            .iter()
            .filter_map(|&idx| {
                if idx < data.len() {
                    if let Some(ts) = timestamps {
                        Some(PlotPoint::Timestamp(ts[idx], idx))
                    } else {
                        Some(PlotPoint::Index(idx))
                    }
                } else {
                    None
                }
            })
            .collect();

        let anomaly_y_values: Vec<f64> = anomaly_indices
            .iter()
            .filter_map(|&idx| if idx < data.len() { Some(data[idx]) } else { None })
            .collect();

        series.push(PlotSeries {
            name: "Anomalies".to_string(),
            x_values: anomaly_x_values,
            y_values: anomaly_y_values,
            series_type: SeriesType::Scatter,
            color: Some("#d62728".to_string()),
            style: Some(SeriesStyle {
                marker_size: Some(8.0),
                line_width: None,
                opacity: Some(0.8),
                line_style: None,
                fill_color: None,
            }),
        });
    }

    Ok(series)
}

/// Calculate simple moving average for trend estimation
pub fn calculate_moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > data.len() {
        return Vec::new();
    }

    let mut result = Vec::new();

    for i in 0..data.len() {
        let start = if i >= window_size / 2 {
            i - window_size / 2
        } else {
            0
        };

        let end = if i + window_size / 2 + 1 <= data.len() {
            i + window_size / 2 + 1
        } else {
            data.len()
        };

        let window_data = &data[start..end];
        let avg = window_data.iter().sum::<f64>() / window_data.len() as f64;
        result.push(avg);
    }

    result
}

/// Simple seasonal decomposition (placeholder implementation)
pub fn simple_seasonal_decomposition(
    data: &[f64],
    period: usize,
) -> Result<DecompositionComponents, Box<dyn std::error::Error>> {
    if data.is_empty() || period == 0 {
        return Err("Invalid data or period for decomposition".into());
    }

    // Simple trend estimation using moving average
    let trend = calculate_moving_average(data, period);

    // Detrended data
    let detrended: Vec<f64> = data.iter().zip(trend.iter()).map(|(d, t)| d - t).collect();

    // Simple seasonal component estimation
    let mut seasonal_sums = vec![0.0; period];
    let mut seasonal_counts = vec![0; period];

    for (i, &value) in detrended.iter().enumerate() {
        let season_idx = i % period;
        seasonal_sums[season_idx] += value;
        seasonal_counts[season_idx] += 1;
    }

    let seasonal_averages: Vec<f64> = seasonal_sums
        .iter()
        .zip(seasonal_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();

    let seasonal: Vec<f64> = (0..data.len())
        .map(|i| seasonal_averages[i % period])
        .collect();

    // Residual component
    let residual: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&d, &t), &s)| d - t - s)
        .collect();

    Ok(DecompositionComponents {
        original: data.to_vec(),
        trend,
        seasonal,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ma = calculate_moving_average(&data, 3);

        assert_eq!(ma.len(), data.len());
        // Check that moving average smooths the data
        assert!(ma[4] > 3.0 && ma[4] < 7.0); // Should be around 5.0
    }

    #[test]
    fn test_simple_seasonal_decomposition() {
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let period = 3;

        let decomp = simple_seasonal_decomposition(&data, period).unwrap();

        assert_eq!(decomp.original.len(), data.len());
        assert_eq!(decomp.trend.len(), data.len());
        assert_eq!(decomp.seasonal.len(), data.len());
        assert_eq!(decomp.residual.len(), data.len());
    }

    #[test]
    fn test_prepare_anomaly_series() {
        let data = vec![1.0, 2.0, 10.0, 3.0, 4.0, 15.0, 5.0];
        let anomalies = vec![2, 5]; // Indices with values 10.0 and 15.0

        let series = prepare_anomaly_series(&data, &anomalies, None).unwrap();

        assert_eq!(series.len(), 2); // Normal data + anomalies
        assert_eq!(series[1].y_values, vec![10.0, 15.0]); // Anomaly values
    }

    #[test]
    fn test_create_trend_plot() {
        let trend_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = PlotConfig::default();
        let result = create_trend_plot(&trend_data, None, config);
        assert!(result.is_ok());
    }
}