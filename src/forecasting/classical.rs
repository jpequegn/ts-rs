//! # Classical Forecasting Methods
//!
//! Implementation of traditional time series forecasting methods including
//! moving averages, exponential smoothing, and Holt-Winters methods.

use crate::{TimeSeries, Result};
use crate::forecasting::{ForecastResult, SeasonalType};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;

/// Simple moving average forecast
pub fn simple_moving_average_forecast(
    timeseries: &TimeSeries,
    window: usize,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.len() < window {
        return Err("Time series too short for the specified window size".into());
    }

    let values = &timeseries.values;
    let n = values.len();

    // Calculate the simple moving average for the last window
    let last_window = &values[n - window..];
    let forecast_value = last_window.iter().sum::<f64>() / window as f64;

    // Generate forecasts (constant forecast)
    let forecasts = vec![forecast_value; horizon];

    // Generate future timestamps
    let last_timestamp = timeseries.timestamps[n - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    // Calculate fitted values
    let mut fitted_values = vec![f64::NAN; window - 1];
    for i in window - 1..n {
        let start_idx = i + 1 - window;
        let window_mean = values[start_idx..=i].iter().sum::<f64>() / window as f64;
        fitted_values.push(window_mean);
    }

    // Calculate residuals
    let residuals: Vec<f64> = values.iter()
        .zip(fitted_values.iter())
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("window_size".to_string(), serde_json::Value::Number(serde_json::Number::from(window)));
    metadata.insert("method".to_string(), serde_json::Value::String("Simple Moving Average".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Simple Moving Average".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Weighted moving average forecast
pub fn weighted_moving_average_forecast(
    timeseries: &TimeSeries,
    weights: &[f64],
    horizon: usize,
) -> Result<ForecastResult> {
    let window = weights.len();

    if timeseries.values.len() < window {
        return Err("Time series too short for the specified window size".into());
    }

    // Normalize weights to sum to 1
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= 0.0 {
        return Err("Weights must sum to a positive value".into());
    }

    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

    let values = &timeseries.values;
    let n = values.len();

    // Calculate weighted moving average for the last window
    let last_window = &values[n - window..];
    let forecast_value = last_window.iter()
        .zip(normalized_weights.iter())
        .map(|(val, weight)| val * weight)
        .sum::<f64>();

    let forecasts = vec![forecast_value; horizon];

    let last_timestamp = timeseries.timestamps[n - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    // Calculate fitted values
    let mut fitted_values = vec![f64::NAN; window - 1];
    for i in window - 1..n {
        let weighted_avg = values[i - window + 1..=i].iter()
            .zip(normalized_weights.iter())
            .map(|(val, weight)| val * weight)
            .sum::<f64>();
        fitted_values.push(weighted_avg);
    }

    let residuals: Vec<f64> = values.iter()
        .zip(fitted_values.iter())
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("window_size".to_string(), serde_json::Value::Number(serde_json::Number::from(window)));
    metadata.insert("weights".to_string(), serde_json::to_value(&normalized_weights)?);
    metadata.insert("method".to_string(), serde_json::Value::String("Weighted Moving Average".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Weighted Moving Average".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Exponential smoothing forecast (Simple Exponential Smoothing)
pub fn exponential_smoothing_forecast(
    timeseries: &TimeSeries,
    alpha: f64,
    horizon: usize,
) -> Result<ForecastResult> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err("Alpha parameter must be between 0 and 1".into());
    }

    if timeseries.values.is_empty() {
        return Err("Time series cannot be empty".into());
    }

    let values = &timeseries.values;
    let n = values.len();

    // Initialize with first value
    let mut level = values[0];
    let mut fitted_values = vec![level];

    // Apply exponential smoothing
    for i in 1..n {
        level = alpha * values[i] + (1.0 - alpha) * level;
        fitted_values.push(level);
    }

    // Forecast (constant level)
    let forecasts = vec![level; horizon];

    let last_timestamp = timeseries.timestamps[n - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    let residuals: Vec<f64> = values.iter()
        .zip(fitted_values.iter())
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("alpha".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(alpha).unwrap()));
    metadata.insert("final_level".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(level).unwrap()));
    metadata.insert("method".to_string(), serde_json::Value::String("Exponential Smoothing".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Exponential Smoothing".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Linear trend extrapolation forecast
pub fn linear_trend_forecast(
    timeseries: &TimeSeries,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.len() < 2 {
        return Err("Time series must have at least 2 points for trend analysis".into());
    }

    let values = &timeseries.values;
    let n = values.len();

    // Calculate linear trend using least squares
    let x_values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_values = values;

    let x_mean = x_values.iter().sum::<f64>() / n as f64;
    let y_mean = y_values.iter().sum::<f64>() / n as f64;

    let numerator: f64 = x_values.iter()
        .zip(y_values.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();

    let denominator: f64 = x_values.iter()
        .map(|x| (x - x_mean).powi(2))
        .sum();

    if denominator == 0.0 {
        return Err("Cannot calculate trend: no variation in time series".into());
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    // Generate forecasts
    let mut forecasts = Vec::new();
    for i in 0..horizon {
        let future_x = n as f64 + i as f64;
        let forecast = intercept + slope * future_x;
        forecasts.push(forecast);
    }

    let last_timestamp = timeseries.timestamps[n - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    // Calculate fitted values
    let fitted_values: Vec<f64> = x_values.iter()
        .map(|x| intercept + slope * x)
        .collect();

    let residuals: Vec<f64> = values.iter()
        .zip(fitted_values.iter())
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("slope".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(slope).unwrap()));
    metadata.insert("intercept".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(intercept).unwrap()));
    metadata.insert("method".to_string(), serde_json::Value::String("Linear Trend".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Linear Trend".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Seasonal naive forecast
pub fn seasonal_naive_forecast(
    timeseries: &TimeSeries,
    seasonal_period: usize,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.len() < seasonal_period {
        return Err("Time series too short for the specified seasonal period".into());
    }

    let values = &timeseries.values;
    let n = values.len();

    // Generate forecasts by repeating the last seasonal cycle
    let mut forecasts = Vec::new();
    for i in 0..horizon {
        let seasonal_index = (n - seasonal_period) + (i % seasonal_period);
        forecasts.push(values[seasonal_index]);
    }

    let last_timestamp = timeseries.timestamps[n - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    // Calculate fitted values (naive forecast for each period)
    let mut fitted_values = vec![f64::NAN; seasonal_period];
    for i in seasonal_period..n {
        let seasonal_lag_value = values[i - seasonal_period];
        fitted_values.push(seasonal_lag_value);
    }

    let residuals: Vec<f64> = values.iter()
        .zip(fitted_values.iter())
        .map(|(actual, fitted)| if fitted.is_nan() { f64::NAN } else { actual - fitted })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("seasonal_period".to_string(), serde_json::Value::Number(serde_json::Number::from(seasonal_period)));
    metadata.insert("method".to_string(), serde_json::Value::String("Seasonal Naive".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Seasonal Naive".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Holt-Winters exponential smoothing forecast
pub fn holt_winters_forecast(
    timeseries: &TimeSeries,
    alpha: f64,
    beta: f64,
    gamma: f64,
    seasonal_period: usize,
    seasonal_type: SeasonalType,
    horizon: usize,
) -> Result<ForecastResult> {
    // Validate parameters
    if ![alpha, beta, gamma].iter().all(|&x| (0.0..=1.0).contains(&x)) {
        return Err("Alpha, beta, and gamma must be between 0 and 1".into());
    }

    if timeseries.values.len() < 2 * seasonal_period {
        return Err("Time series too short for Holt-Winters method".into());
    }

    let values = &timeseries.values;
    let n = values.len();

    // Initialize level, trend, and seasonal components
    let initial_level = values[0..seasonal_period].iter().sum::<f64>() / seasonal_period as f64;

    let first_season_avg = values[0..seasonal_period].iter().sum::<f64>() / seasonal_period as f64;
    let second_season_avg = values[seasonal_period..2*seasonal_period].iter().sum::<f64>() / seasonal_period as f64;
    let initial_trend = (second_season_avg - first_season_avg) / seasonal_period as f64;

    // Initialize seasonal indices
    let mut seasonal = vec![0.0; seasonal_period];
    for i in 0..seasonal_period {
        seasonal[i] = match seasonal_type {
            SeasonalType::Additive => values[i] - initial_level,
            SeasonalType::Multiplicative => {
                if initial_level != 0.0 {
                    values[i] / initial_level
                } else {
                    1.0
                }
            }
        };
    }

    let mut level = initial_level;
    let mut trend = initial_trend;
    let mut fitted_values = Vec::new();

    // Apply Holt-Winters smoothing
    for i in 0..n {
        let seasonal_index = i % seasonal_period;

        // Calculate fitted value
        let fitted = match seasonal_type {
            SeasonalType::Additive => level + trend + seasonal[seasonal_index],
            SeasonalType::Multiplicative => (level + trend) * seasonal[seasonal_index],
        };
        fitted_values.push(fitted);

        if i > 0 {
            // Update level
            let prev_level = level;
            level = match seasonal_type {
                SeasonalType::Additive => {
                    alpha * (values[i] - seasonal[seasonal_index]) + (1.0 - alpha) * (level + trend)
                }
                SeasonalType::Multiplicative => {
                    alpha * (values[i] / seasonal[seasonal_index]) + (1.0 - alpha) * (level + trend)
                }
            };

            // Update trend
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;

            // Update seasonal component
            seasonal[seasonal_index] = match seasonal_type {
                SeasonalType::Additive => {
                    gamma * (values[i] - level) + (1.0 - gamma) * seasonal[seasonal_index]
                }
                SeasonalType::Multiplicative => {
                    gamma * (values[i] / level) + (1.0 - gamma) * seasonal[seasonal_index]
                }
            };
        }
    }

    // Generate forecasts
    let mut forecasts = Vec::new();
    for h in 1..=horizon {
        let seasonal_index = (n - 1 + h) % seasonal_period;
        let forecast = match seasonal_type {
            SeasonalType::Additive => level + h as f64 * trend + seasonal[seasonal_index],
            SeasonalType::Multiplicative => (level + h as f64 * trend) * seasonal[seasonal_index],
        };
        forecasts.push(forecast);
    }

    let last_timestamp = timeseries.timestamps[n - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    let residuals: Vec<f64> = values.iter()
        .zip(fitted_values.iter())
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("alpha".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(alpha).unwrap()));
    metadata.insert("beta".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(beta).unwrap()));
    metadata.insert("gamma".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(gamma).unwrap()));
    metadata.insert("seasonal_period".to_string(), serde_json::Value::Number(serde_json::Number::from(seasonal_period)));
    metadata.insert("seasonal_type".to_string(), serde_json::to_value(seasonal_type)?);
    metadata.insert("final_level".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(level).unwrap()));
    metadata.insert("final_trend".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(trend).unwrap()));
    metadata.insert("method".to_string(), serde_json::Value::String("Holt-Winters".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Holt-Winters".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Helper function to generate future timestamps
fn generate_future_timestamps(
    last_timestamp: DateTime<Utc>,
    horizon: usize,
    historical_timestamps: &[DateTime<Utc>],
) -> Result<Vec<DateTime<Utc>>> {
    if historical_timestamps.len() < 2 {
        return Err("Need at least 2 historical timestamps to infer frequency".into());
    }

    // Infer the time frequency from the last few timestamps
    let intervals: Vec<i64> = historical_timestamps.windows(2)
        .map(|pair| (pair[1] - pair[0]).num_seconds())
        .collect();

    // Use the most common interval or the last interval
    let interval_seconds = intervals.last().copied().unwrap_or(3600); // Default to 1 hour
    let interval = Duration::seconds(interval_seconds);

    let mut timestamps = Vec::new();
    let mut current = last_timestamp;

    for _ in 0..horizon {
        current = current + interval;
        timestamps.push(current);
    }

    Ok(timestamps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn create_test_timeseries() -> TimeSeries {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 11.0, 13.0, 14.0];

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_simple_moving_average() {
        let ts = create_test_timeseries();
        let result = simple_moving_average_forecast(&ts, 3, 5).unwrap();

        assert_eq!(result.forecasts.len(), 5);
        assert_eq!(result.timestamps.len(), 5);
        assert!(result.fitted_values.is_some());
        assert!(result.residuals.is_some());
    }

    #[test]
    fn test_exponential_smoothing() {
        let ts = create_test_timeseries();
        let result = exponential_smoothing_forecast(&ts, 0.3, 3).unwrap();

        assert_eq!(result.forecasts.len(), 3);
        assert_eq!(result.timestamps.len(), 3);
        assert!(result.fitted_values.is_some());
    }

    #[test]
    fn test_linear_trend() {
        let ts = create_test_timeseries();
        let result = linear_trend_forecast(&ts, 4).unwrap();

        assert_eq!(result.forecasts.len(), 4);
        assert_eq!(result.timestamps.len(), 4);
        assert!(result.fitted_values.is_some());
    }

    #[test]
    fn test_seasonal_naive() {
        let ts = create_test_timeseries();
        let result = seasonal_naive_forecast(&ts, 2, 6).unwrap();

        assert_eq!(result.forecasts.len(), 6);
        assert_eq!(result.timestamps.len(), 6);
    }
}