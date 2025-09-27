//! # Correlation Visualization Functions
//!
//! Functions for creating correlation heatmaps, scatter matrices, and related visualizations.

use crate::plotting::types::*;
use crate::plotting::engine::PlotEngine;
use std::collections::HashMap;

/// Create a correlation heatmap
pub fn create_correlation_heatmap(
    data: &HashMap<String, Vec<f64>>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Heatmap;

    let engine = PlotEngine::new(config)?;
    engine.create_correlation_heatmap(data)
}

/// Create a scatter matrix for correlation analysis
pub fn create_scatter_matrix(
    data: &HashMap<String, Vec<f64>>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::ScatterMatrix;

    // For now, create individual scatter plots for each pair
    // In a full implementation, this would create a proper matrix layout
    let variables: Vec<String> = data.keys().cloned().collect();
    if variables.len() < 2 {
        return Err("Need at least 2 variables for scatter matrix".into());
    }

    // Create scatter plot for first two variables as example
    let first_var = &variables[0];
    let second_var = &variables[1];

    let mut series_data = HashMap::new();
    series_data.insert(format!("{} vs {}", first_var, second_var), data[first_var].clone());

    let engine = PlotEngine::new(config)?;
    let series = prepare_scatter_matrix_data(&series_data, data)?;
    engine.create_scatter_plot(&series)
}

/// Create a correlation plot with customizable correlation method
pub fn create_correlation_plot(
    data: &HashMap<String, Vec<f64>>,
    correlation_config: CorrelationPlotConfig,
    plot_config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    match correlation_config.method {
        CorrelationMethod::Pearson => create_correlation_heatmap(data, plot_config),
        CorrelationMethod::Spearman => {
            // For now, use Pearson as placeholder
            // Full implementation would calculate Spearman rank correlation
            create_correlation_heatmap(data, plot_config)
        },
        CorrelationMethod::Kendall => {
            // For now, use Pearson as placeholder
            // Full implementation would calculate Kendall's tau
            create_correlation_heatmap(data, plot_config)
        },
    }
}

/// Prepare data for scatter matrix plotting
fn prepare_scatter_matrix_data(
    series_data: &HashMap<String, Vec<f64>>,
    _all_data: &HashMap<String, Vec<f64>>,
) -> Result<Vec<PlotSeries>, Box<dyn std::error::Error>> {
    let mut series = Vec::new();

    for (name, values) in series_data {
        let x_values: Vec<PlotPoint> = (0..values.len()).map(PlotPoint::Index).collect();

        series.push(PlotSeries {
            name: name.clone(),
            x_values,
            y_values: values.clone(),
            series_type: SeriesType::Scatter,
            color: None,
            style: None,
        });
    }

    Ok(series)
}

/// Calculate correlation coefficient between two series
pub fn calculate_correlation(
    x: &[f64],
    y: &[f64],
    method: CorrelationMethod,
) -> Result<f64, Box<dyn std::error::Error>> {
    if x.len() != y.len() || x.is_empty() {
        return Err("Series must have the same non-zero length".into());
    }

    match method {
        CorrelationMethod::Pearson => calculate_pearson_correlation(x, y),
        CorrelationMethod::Spearman => {
            // Placeholder implementation
            // Full implementation would rank the data first
            calculate_pearson_correlation(x, y)
        },
        CorrelationMethod::Kendall => {
            // Placeholder implementation
            // Full implementation would calculate Kendall's tau
            calculate_pearson_correlation(x, y)
        },
    }
}

/// Calculate Pearson correlation coefficient
fn calculate_pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
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

/// Calculate correlation matrix for multiple series
pub fn calculate_correlation_matrix(
    data: &HashMap<String, Vec<f64>>,
    method: CorrelationMethod,
) -> Result<HashMap<String, HashMap<String, f64>>, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let mut matrix = HashMap::new();

    for var1 in &variables {
        let mut row = HashMap::new();
        for var2 in &variables {
            if var1 == var2 {
                row.insert(var2.clone(), 1.0);
            } else {
                let correlation = calculate_correlation(&data[var1], &data[var2], method.clone())?;
                row.insert(var2.clone(), correlation);
            }
        }
        matrix.insert(var1.clone(), row);
    }

    Ok(matrix)
}

/// Get significant correlations above a threshold
pub fn get_significant_correlations(
    correlation_matrix: &HashMap<String, HashMap<String, f64>>,
    threshold: f64,
) -> Vec<(String, String, f64)> {
    let mut significant = Vec::new();

    for (var1, row) in correlation_matrix {
        for (var2, &correlation) in row {
            if var1 < var2 && correlation.abs() >= threshold {
                significant.push((var1.clone(), var2.clone(), correlation));
            }
        }
    }

    // Sort by absolute correlation value, descending
    significant.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    significant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = calculate_pearson_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_calculate_correlation_matrix() {
        let mut data = HashMap::new();
        data.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("y".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let matrix = calculate_correlation_matrix(&data, CorrelationMethod::Pearson).unwrap();

        assert_eq!(matrix.len(), 2);
        assert!((matrix["x"]["x"] - 1.0).abs() < 1e-10);
        assert!((matrix["y"]["y"] - 1.0).abs() < 1e-10);
        assert!((matrix["x"]["y"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_significant_correlations() {
        let mut matrix = HashMap::new();
        let mut row1 = HashMap::new();
        row1.insert("var1".to_string(), 1.0);
        row1.insert("var2".to_string(), 0.8);
        row1.insert("var3".to_string(), 0.3);
        matrix.insert("var1".to_string(), row1);

        let mut row2 = HashMap::new();
        row2.insert("var1".to_string(), 0.8);
        row2.insert("var2".to_string(), 1.0);
        row2.insert("var3".to_string(), 0.9);
        matrix.insert("var2".to_string(), row2);

        let mut row3 = HashMap::new();
        row3.insert("var1".to_string(), 0.3);
        row3.insert("var2".to_string(), 0.9);
        row3.insert("var3".to_string(), 1.0);
        matrix.insert("var3".to_string(), row3);

        let significant = get_significant_correlations(&matrix, 0.7);
        assert_eq!(significant.len(), 2); // var1-var2 (0.8) and var2-var3 (0.9)
        assert!((significant[0].2 - 0.9).abs() < 1e-10); // Should be sorted by absolute value
    }

    #[test]
    fn test_create_correlation_heatmap() {
        let mut data = HashMap::new();
        data.insert("series1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("series2".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let config = PlotConfig::default();
        let result = create_correlation_heatmap(&data, config);
        assert!(result.is_ok());
    }
}