//! # Statistical Plotting Functions
//!
//! Functions for creating statistical plots like histograms, box plots, Q-Q plots, etc.

use crate::plotting::types::*;
use crate::plotting::engine::PlotEngine;
use std::collections::HashMap;

/// Create a histogram
pub fn create_histogram(
    values: &[f64],
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::Histogram;

    let engine = PlotEngine::new(config)?;
    engine.create_histogram(values)
}

/// Create a box plot
pub fn create_box_plot(
    data: &HashMap<String, Vec<f64>>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let mut config = config;
    config.plot_type = PlotType::BoxPlot;

    let engine = PlotEngine::new(config)?;
    engine.create_box_plot(data)
}

/// Create a violin plot
pub fn create_violin_plot(
    data: &HashMap<String, Vec<f64>>,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    // For now, create a box plot as violin plots require more complex implementation
    create_box_plot(data, config)
}

/// Create a Q-Q plot for normality testing
pub fn create_qq_plot(
    values: &[f64],
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let qq_data = calculate_qq_data(values)?;

    let mut data = HashMap::new();
    data.insert("theoretical".to_string(), qq_data.theoretical.clone());
    data.insert("sample".to_string(), qq_data.sample.clone());

    let mut config = config;
    config.plot_type = PlotType::Scatter;
    config.title = Some("Q-Q Plot".to_string());
    config.x_label = Some("Theoretical Quantiles".to_string());
    config.y_label = Some("Sample Quantiles".to_string());

    let engine = PlotEngine::new(config)?;
    let series = vec![PlotSeries {
        name: "Q-Q Plot".to_string(),
        x_values: qq_data.theoretical.iter().enumerate().map(|(i, _)| PlotPoint::Index(i)).collect(),
        y_values: qq_data.sample,
        series_type: SeriesType::Scatter,
        color: None,
        style: None,
    }];

    engine.create_scatter_plot(&series)
}

/// Create an autocorrelation function (ACF) plot
pub fn create_acf_plot(
    values: &[f64],
    max_lags: usize,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let acf_data = calculate_acf(values, max_lags)?;

    let mut data = HashMap::new();
    data.insert("ACF".to_string(), acf_data.clone());

    let mut config = config;
    config.plot_type = PlotType::Line;
    config.title = Some("Autocorrelation Function".to_string());
    config.x_label = Some("Lag".to_string());
    config.y_label = Some("Autocorrelation".to_string());

    let engine = PlotEngine::new(config)?;
    let series = vec![PlotSeries {
        name: "ACF".to_string(),
        x_values: (0..acf_data.len()).map(PlotPoint::Index).collect(),
        y_values: acf_data,
        series_type: SeriesType::Line,
        color: None,
        style: None,
    }];

    engine.create_line_plot(&series)
}

/// Create a partial autocorrelation function (PACF) plot
pub fn create_pacf_plot(
    values: &[f64],
    max_lags: usize,
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let pacf_data = calculate_pacf(values, max_lags)?;

    let mut data = HashMap::new();
    data.insert("PACF".to_string(), pacf_data.clone());

    let mut config = config;
    config.plot_type = PlotType::Line;
    config.title = Some("Partial Autocorrelation Function".to_string());
    config.x_label = Some("Lag".to_string());
    config.y_label = Some("Partial Autocorrelation".to_string());

    let engine = PlotEngine::new(config)?;
    let series = vec![PlotSeries {
        name: "PACF".to_string(),
        x_values: (0..pacf_data.len()).map(PlotPoint::Index).collect(),
        y_values: pacf_data,
        series_type: SeriesType::Line,
        color: None,
        style: None,
    }];

    engine.create_line_plot(&series)
}

/// Create a density plot
pub fn create_density_plot(
    values: &[f64],
    config: PlotConfig,
) -> Result<PlotResult, Box<dyn std::error::Error>> {
    let density_data = calculate_density(values)?;

    let mut data = HashMap::new();
    data.insert("Density".to_string(), density_data.densities.clone());

    let mut config = config;
    config.plot_type = PlotType::Line;
    config.title = Some("Density Plot".to_string());
    config.x_label = Some("Value".to_string());
    config.y_label = Some("Density".to_string());

    let engine = PlotEngine::new(config)?;
    let series = vec![PlotSeries {
        name: "Density".to_string(),
        x_values: density_data.x_values.iter().enumerate().map(|(i, _)| PlotPoint::Index(i)).collect(),
        y_values: density_data.densities,
        series_type: SeriesType::Line,
        color: None,
        style: None,
    }];

    engine.create_line_plot(&series)
}

/// Q-Q plot data
struct QQData {
    theoretical: Vec<f64>,
    sample: Vec<f64>,
}

/// Density plot data
struct DensityData {
    x_values: Vec<f64>,
    densities: Vec<f64>,
}

/// Calculate Q-Q plot data
fn calculate_qq_data(values: &[f64]) -> Result<QQData, Box<dyn std::error::Error>> {
    if values.is_empty() {
        return Err("Cannot create Q-Q plot for empty data".into());
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_values.len();
    let mut theoretical = Vec::new();
    let mut sample = Vec::new();

    for i in 0..n {
        // Calculate theoretical quantiles (standard normal)
        let p = (i + 1) as f64 / (n + 1) as f64;
        let z = inverse_normal_cdf(p);
        theoretical.push(z);
        sample.push(sorted_values[i]);
    }

    Ok(QQData { theoretical, sample })
}

/// Calculate autocorrelation function
fn calculate_acf(values: &[f64], max_lags: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if values.is_empty() {
        return Err("Cannot calculate ACF for empty data".into());
    }

    let n = values.len();
    let max_lags = max_lags.min(n - 1);
    let mut acf = Vec::new();

    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    for lag in 0..=max_lags {
        let mut covariance = 0.0;
        let count = n - lag;

        for i in 0..count {
            covariance += (values[i] - mean) * (values[i + lag] - mean);
        }

        covariance /= n as f64;
        let correlation = if variance > 0.0 { covariance / variance } else { 0.0 };
        acf.push(correlation);
    }

    Ok(acf)
}

/// Calculate partial autocorrelation function
fn calculate_pacf(values: &[f64], max_lags: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if values.is_empty() {
        return Err("Cannot calculate PACF for empty data".into());
    }

    let acf = calculate_acf(values, max_lags)?;
    let mut pacf = Vec::new();

    // PACF[0] = 1
    pacf.push(1.0);

    if max_lags == 0 {
        return Ok(pacf);
    }

    // PACF[1] = ACF[1]
    if acf.len() > 1 {
        pacf.push(acf[1]);
    }

    // Calculate higher order PACF values using Durbin-Levinson algorithm
    for k in 2..=max_lags.min(acf.len() - 1) {
        let mut numerator = acf[k];
        let mut denominator = 1.0;

        for j in 1..k {
            numerator -= pacf[j] * acf[k - j];
            denominator -= pacf[j] * acf[j];
        }

        let pacf_k = if denominator.abs() > 1e-10 { numerator / denominator } else { 0.0 };
        pacf.push(pacf_k);
    }

    Ok(pacf)
}

/// Calculate density using kernel density estimation
fn calculate_density(values: &[f64]) -> Result<DensityData, Box<dyn std::error::Error>> {
    if values.is_empty() {
        return Err("Cannot calculate density for empty data".into());
    }

    let n = values.len();
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if min_val == max_val {
        return Err("Cannot calculate density for constant data".into());
    }

    // Simple bandwidth estimation (Scott's rule)
    let std_dev = {
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        variance.sqrt()
    };
    let bandwidth = std_dev * (n as f64).powf(-1.0 / 5.0);

    // Create evaluation points
    let num_points = 100;
    let range = max_val - min_val;
    let padding = range * 0.1;
    let x_min = min_val - padding;
    let x_max = max_val + padding;

    let mut x_values = Vec::new();
    let mut densities = Vec::new();

    for i in 0..num_points {
        let x = x_min + (x_max - x_min) * i as f64 / (num_points - 1) as f64;
        x_values.push(x);

        // Gaussian kernel density estimation
        let density = values.iter()
            .map(|&xi| {
                let u = (x - xi) / bandwidth;
                (-0.5 * u * u).exp() / (bandwidth * (2.0 * std::f64::consts::PI).sqrt())
            })
            .sum::<f64>() / n as f64;

        densities.push(density);
    }

    Ok(DensityData { x_values, densities })
}

/// Approximate inverse normal CDF using Beasley-Springer-Moro algorithm
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-8 {
        return 0.0;
    }

    // Simplified implementation using Box-Muller transformation approximation
    if p < 0.5 {
        -inverse_normal_cdf(1.0 - p)
    } else {
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_histogram() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = PlotConfig::default();
        let result = create_histogram(&values, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_calculate_acf() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = calculate_acf(&values, 3).unwrap();
        assert!(acf.len() == 4); // lags 0, 1, 2, 3
        assert!((acf[0] - 1.0).abs() < 1e-10); // ACF at lag 0 should be 1
    }

    #[test]
    fn test_calculate_qq_data() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qq_data = calculate_qq_data(&values).unwrap();
        assert_eq!(qq_data.theoretical.len(), 5);
        assert_eq!(qq_data.sample.len(), 5);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        let z = inverse_normal_cdf(0.5);
        assert!((z - 0.0).abs() < 1e-6);

        let z = inverse_normal_cdf(0.9772);
        assert!((z - 2.0).abs() < 0.1); // Approximately 2 standard deviations
    }
}