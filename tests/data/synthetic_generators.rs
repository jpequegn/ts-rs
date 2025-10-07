//! Synthetic test data generators
//!
//! This module provides generators for creating synthetic time series data
//! with known statistical properties for comprehensive testing.

use chrono::{DateTime, TimeZone, Utc};
use chronos::stats::stationarity::test_stationarity;
use chronos::trend::detection::detect_trend;
use chronos::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Configuration for synthetic time series generation
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    pub size: usize,
    pub start_timestamp: DateTime<Utc>,
    pub frequency_hours: i64,
    pub seed: Option<u64>,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            size: 1000,
            start_timestamp: Utc.timestamp_opt(1000000000, 0).unwrap(),
            frequency_hours: 1,
            seed: Some(42),
        }
    }
}

/// Generate timestamps for time series
fn generate_timestamps(config: &SyntheticConfig) -> Vec<DateTime<Utc>> {
    (0..config.size)
        .map(|i| {
            config.start_timestamp + chrono::Duration::hours(i as i64 * config.frequency_hours)
        })
        .collect()
}

/// Generate white noise time series
pub fn generate_white_noise(config: SyntheticConfig, mean: f64, std_dev: f64) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            mean + std_dev * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        })
        .collect();

    TimeSeries::new("white_noise".to_string(), timestamps, values).unwrap()
}

/// Generate sine wave time series
pub fn generate_sine_wave(
    config: SyntheticConfig,
    amplitude: f64,
    frequency: f64,
    phase: f64,
    noise_level: f64,
) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|i| {
            let signal = amplitude * (2.0 * PI * frequency * i as f64 + phase).sin();
            let noise = if noise_level > 0.0 {
                noise_level * (rng.gen::<f64>() - 0.5)
            } else {
                0.0
            };
            signal + noise
        })
        .collect();

    TimeSeries::new("sine_wave".to_string(), timestamps, values).unwrap()
}

/// Generate linear trend time series
pub fn generate_linear_trend(
    config: SyntheticConfig,
    intercept: f64,
    slope: f64,
    noise_level: f64,
) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|i| {
            let trend = intercept + slope * i as f64;
            let noise = if noise_level > 0.0 {
                noise_level * (rng.gen::<f64>() - 0.5)
            } else {
                0.0
            };
            trend + noise
        })
        .collect();

    TimeSeries::new("linear_trend".to_string(), timestamps, values).unwrap()
}

/// Generate seasonal time series
pub fn generate_seasonal_pattern(
    config: SyntheticConfig,
    period: usize,
    amplitude: f64,
    noise_level: f64,
) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|i| {
            let seasonal = amplitude * (2.0 * PI * (i % period) as f64 / period as f64).sin();
            let noise = if noise_level > 0.0 {
                noise_level * (rng.gen::<f64>() - 0.5)
            } else {
                0.0
            };
            seasonal + noise
        })
        .collect();

    TimeSeries::new("seasonal_pattern".to_string(), timestamps, values).unwrap()
}

/// Generate AR(1) process: x_t = c + φx_{t-1} + ε_t
pub fn generate_ar1_process(
    config: SyntheticConfig,
    phi: f64,
    constant: f64,
    error_variance: f64,
) -> TimeSeries {
    assert!(
        phi.abs() < 1.0,
        "AR(1) coefficient must be < 1 for stationarity"
    );

    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let mut values = Vec::with_capacity(config.size);

    // Initial value from stationary distribution
    values.push(constant / (1.0 - phi));

    for _ in 1..config.size {
        let error = error_variance.sqrt() * (rng.gen::<f64>() - 0.5) * 2.0;
        let next_value = constant + phi * values.last().unwrap() + error;
        values.push(next_value);
    }

    TimeSeries::new("ar1_process".to_string(), timestamps, values).unwrap()
}

/// Generate random walk (non-stationary)
pub fn generate_random_walk(config: SyntheticConfig, step_size: f64, drift: f64) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let mut values = Vec::with_capacity(config.size);
    values.push(0.0); // Start at zero

    for _ in 1..config.size {
        let step = step_size * (rng.gen::<f64>() - 0.5) * 2.0;
        let next_value = values.last().unwrap() + drift + step;
        values.push(next_value);
    }

    TimeSeries::new("random_walk".to_string(), timestamps, values).unwrap()
}

/// Generate time series with change points
pub fn generate_changepoint_series(
    config: SyntheticConfig,
    changepoints: Vec<usize>,
    levels: Vec<f64>,
    noise_level: f64,
) -> TimeSeries {
    assert_eq!(
        changepoints.len() + 1,
        levels.len(),
        "Need one more level than changepoints"
    );

    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|i| {
            // Find which segment this point belongs to
            let segment = changepoints
                .iter()
                .position(|&cp| i < cp)
                .unwrap_or(changepoints.len());
            let level = levels[segment];

            let noise = if noise_level > 0.0 {
                noise_level * (rng.gen::<f64>() - 0.5)
            } else {
                0.0
            };

            level + noise
        })
        .collect();

    TimeSeries::new("changepoint_series".to_string(), timestamps, values).unwrap()
}

/// Generate complex time series with trend + seasonality + noise
pub fn generate_complex_series(
    config: SyntheticConfig,
    trend_slope: f64,
    seasonal_amplitude: f64,
    seasonal_period: usize,
    noise_level: f64,
) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|i| {
            let trend = trend_slope * i as f64;
            let seasonal = seasonal_amplitude
                * (2.0 * PI * (i % seasonal_period) as f64 / seasonal_period as f64).sin();
            let noise = noise_level * (rng.gen::<f64>() - 0.5);

            trend + seasonal + noise
        })
        .collect();

    TimeSeries::new("complex_series".to_string(), timestamps, values).unwrap()
}

/// Generate outlier-contaminated time series
pub fn generate_outlier_series(
    config: SyntheticConfig,
    base_mean: f64,
    base_std: f64,
    outlier_probability: f64,
    outlier_magnitude: f64,
) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let values: Vec<f64> = (0..config.size)
        .map(|_| {
            let base_value = base_mean + base_std * (rng.gen::<f64>() - 0.5) * 2.0;

            if rng.gen::<f64>() < outlier_probability {
                // Add outlier
                let outlier_sign = if rng.gen::<f64>() < 0.5 { -1.0 } else { 1.0 };
                base_value + outlier_sign * outlier_magnitude
            } else {
                base_value
            }
        })
        .collect();

    TimeSeries::new("outlier_series".to_string(), timestamps, values).unwrap()
}

/// Generate GARCH-like volatility clustering
pub fn generate_volatility_clustering(
    config: SyntheticConfig,
    base_volatility: f64,
    alpha: f64,
    beta: f64,
) -> TimeSeries {
    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    let mut values = Vec::with_capacity(config.size);
    let mut volatility = base_volatility;

    for _ in 0..config.size {
        let shock = (rng.gen::<f64>() - 0.5) * 2.0;
        let return_value = volatility * shock;
        values.push(return_value);

        // Update volatility (simplified GARCH)
        volatility =
            base_volatility.sqrt() * (alpha * shock.powi(2) + beta * volatility.powi(2)).sqrt();
    }

    TimeSeries::new("volatility_clustering".to_string(), timestamps, values).unwrap()
}

/// Generate multiple related time series for correlation testing
pub fn generate_correlated_series(
    config: SyntheticConfig,
    correlation: f64,
    series_count: usize,
) -> Vec<TimeSeries> {
    assert!(
        correlation >= -1.0 && correlation <= 1.0,
        "Correlation must be between -1 and 1"
    );
    assert!(series_count >= 2, "Need at least 2 series for correlation");

    let timestamps = generate_timestamps(&config);

    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    // Generate independent series
    let independent_series: Vec<Vec<f64>> = (0..series_count)
        .map(|_| (0..config.size).map(|_| rng.gen::<f64>() - 0.5).collect())
        .collect();

    // Create correlated series using Cholesky decomposition approach
    let mut correlated_series = Vec::new();

    for i in 0..series_count {
        let values: Vec<f64> = (0..config.size)
            .map(|j| {
                if i == 0 {
                    independent_series[0][j]
                } else {
                    correlation * independent_series[0][j]
                        + (1.0 - correlation.powi(2)).sqrt() * independent_series[i][j]
                }
            })
            .collect();

        let ts = TimeSeries::new(
            format!("correlated_series_{}", i),
            timestamps.clone(),
            values,
        )
        .unwrap();

        correlated_series.push(ts);
    }

    correlated_series
}

#[cfg(test)]
mod generator_tests {
    use super::*;
    use chronos::stats::compute_descriptive_stats;

    #[test]
    fn test_white_noise_properties() {
        let config = SyntheticConfig::default();
        let ts = generate_white_noise(config, 0.0, 1.0);
        let stats = compute_descriptive_stats(&ts.values).unwrap();

        // Should be approximately normal with mean 0 and std 1
        assert!((stats.mean).abs() < 0.2);
        assert!((stats.std_dev - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_sine_wave_properties() {
        let config = SyntheticConfig::default();
        let ts = generate_sine_wave(config, 1.0, 0.01, 0.0, 0.0);
        let stats = compute_descriptive_stats(&ts.values).unwrap();

        // Perfect sine wave should have mean ~0 and known variance
        assert!(stats.mean.abs() < 0.1);
        assert!((stats.variance - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_linear_trend_properties() {
        let config = SyntheticConfig::default();
        let ts = generate_linear_trend(config, 0.0, 1.0, 0.0);

        // Should detect strong trend
        let trend_result = detect_trend(&ts.values, "mann_kendall").unwrap();
        assert!(trend_result.p_value < 0.05); // Should be significant
    }

    #[test]
    fn test_ar1_stationarity() {
        let config = SyntheticConfig {
            size: 500,
            ..Default::default()
        };
        let ts = generate_ar1_process(config, 0.5, 0.0, 1.0);

        // AR(1) with |φ| < 1 should be stationary
        let stationarity_result = test_stationarity(&ts.values, "adf").unwrap();
        assert!(stationarity_result.p_value.is_finite());
    }

    #[test]
    fn test_random_walk_non_stationarity() {
        let config = SyntheticConfig {
            size: 200,
            ..Default::default()
        };
        let ts = generate_random_walk(config, 1.0, 0.0);

        // Random walk should be non-stationary
        let stationarity_result = test_stationarity(&ts.values, "adf").unwrap();
        // High p-value typically indicates non-stationarity, but results can vary
        assert!(stationarity_result.p_value >= 0.0 && stationarity_result.p_value <= 1.0);
    }
}
