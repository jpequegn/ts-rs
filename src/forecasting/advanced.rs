//! Advanced forecasting methods including ETS, Theta, and Prophet-like models

use crate::{TimeSeries, Result, TimeSeriesError};
use crate::forecasting::{ForecastResult, ETSComponent, GrowthType, SeasonalityMode};
use chrono::Duration;
use std::collections::HashMap;
use serde_json;

/// Exponential Smoothing State Space (ETS) model forecast
pub fn ets_forecast(
    timeseries: &TimeSeries,
    error_type: ETSComponent,
    trend_type: ETSComponent,
    seasonal_type: ETSComponent,
    seasonal_period: Option<usize>,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Empty time series".to_string())));
    }

    let values = &timeseries.values;
    let n = values.len();

    // Initialize parameters with basic heuristics
    let alpha = 0.3; // Level smoothing parameter
    let beta = if matches!(trend_type, ETSComponent::None) { 0.0 } else { 0.1 }; // Trend smoothing parameter
    let gamma = if matches!(seasonal_type, ETSComponent::None) { 0.0 } else { 0.1 }; // Seasonal smoothing parameter

    let seasonal_period = seasonal_period.unwrap_or(12);

    // Initialize state variables
    let mut level = values[0];
    let mut trend = if n > 1 { values[1] - values[0] } else { 0.0 };
    let mut seasonal_indices = vec![1.0; seasonal_period];

    // Initialize seasonal indices if seasonal component exists
    if !matches!(seasonal_type, ETSComponent::None) && n >= seasonal_period {
        for i in 0..seasonal_period {
            if i < n {
                seasonal_indices[i] = values[i] / (values[0] + i as f64 * trend);
            }
        }
    }

    let mut fitted_values = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);

    // Fit the model
    for (t, &observation) in values.iter().enumerate() {
        let seasonal_index = if matches!(seasonal_type, ETSComponent::None) {
            1.0
        } else {
            seasonal_indices[t % seasonal_period]
        };

        // Calculate fitted value
        let fitted = match seasonal_type {
            ETSComponent::Additive => level + trend + seasonal_index,
            ETSComponent::Multiplicative => (level + trend) * seasonal_index,
            _ => level + trend,
        };

        fitted_values.push(fitted);
        residuals.push(observation - fitted);

        // Update state variables
        let error = observation - fitted;

        // Update level
        let level_update = match error_type {
            ETSComponent::Additive => alpha * error,
            ETSComponent::Multiplicative => alpha * error / fitted,
            _ => alpha * error,
        };

        let new_level = match seasonal_type {
            ETSComponent::Additive => level + level_update,
            ETSComponent::Multiplicative => level + level_update / seasonal_index,
            _ => level + level_update,
        };

        // Update trend
        let new_trend = if matches!(trend_type, ETSComponent::None) {
            0.0
        } else {
            match trend_type {
                ETSComponent::Damped => 0.98 * trend + beta * (new_level - level),
                _ => trend + beta * (new_level - level),
            }
        };

        // Update seasonal
        if !matches!(seasonal_type, ETSComponent::None) {
            let seasonal_error = match seasonal_type {
                ETSComponent::Additive => gamma * (observation - new_level - new_trend),
                ETSComponent::Multiplicative => gamma * (observation / (new_level + new_trend) - seasonal_index),
                _ => 0.0,
            };
            seasonal_indices[t % seasonal_period] += seasonal_error;
        }

        level = new_level;
        trend = new_trend;
    }

    // Generate forecasts
    let mut forecasts = Vec::with_capacity(horizon);
    let mut timestamps = Vec::with_capacity(horizon);

    let last_timestamp = timeseries.timestamps.last().unwrap();
    let interval = if timeseries.timestamps.len() > 1 {
        timeseries.timestamps[1] - timeseries.timestamps[0]
    } else {
        Duration::days(1)
    };

    for h in 1..=horizon {
        let seasonal_index = if matches!(seasonal_type, ETSComponent::None) {
            1.0
        } else {
            seasonal_indices[(n - 1 + h) % seasonal_period]
        };

        let trend_component = match trend_type {
            ETSComponent::None => 0.0,
            ETSComponent::Damped => {
                let phi: f64 = 0.98;
                trend * (1.0 - phi.powi(h as i32)) / (1.0 - phi)
            },
            _ => trend * h as f64,
        };

        let forecast = match seasonal_type {
            ETSComponent::Additive => level + trend_component + seasonal_index,
            ETSComponent::Multiplicative => (level + trend_component) * seasonal_index,
            _ => level + trend_component,
        };

        forecasts.push(forecast);
        timestamps.push(*last_timestamp + interval * h as i32);
    }

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("alpha".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(alpha).unwrap()));
    metadata.insert("beta".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(beta).unwrap()));
    metadata.insert("gamma".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(gamma).unwrap()));
    metadata.insert("error_type".to_string(), serde_json::Value::String(format!("{:?}", error_type)));
    metadata.insert("trend_type".to_string(), serde_json::Value::String(format!("{:?}", trend_type)));
    metadata.insert("seasonal_type".to_string(), serde_json::Value::String(format!("{:?}", seasonal_type)));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "ETS".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Theta method forecast
pub fn theta_forecast(
    timeseries: &TimeSeries,
    theta: f64,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Empty time series".to_string())));
    }

    let values = &timeseries.values;
    let n = values.len();

    // First differencing to create trend line (Theta line 0)
    let mut first_diff = Vec::with_capacity(n - 1);
    for i in 1..n {
        first_diff.push(values[i] - values[i - 1]);
    }

    // Linear regression on first differences to estimate trend
    let sum_x: f64 = (1..n).map(|i| i as f64).sum();
    let sum_y: f64 = first_diff.iter().sum();
    let sum_xy: f64 = first_diff.iter().enumerate().map(|(i, &y)| (i + 1) as f64 * y).sum();
    let sum_x2: f64 = (1..n).map(|i| (i as f64).powi(2)).sum();

    let n_diff = (n - 1) as f64;
    let slope = (n_diff * sum_xy - sum_x * sum_y) / (n_diff * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n_diff;

    // Create Theta line (weighted combination of original and trend)
    let mut theta_line = Vec::with_capacity(n);
    for i in 0..n {
        let trend_value = values[0] + i as f64 * intercept + (i as f64 * (i + 1) as f64 / 2.0) * slope;
        theta_line.push(theta * values[i] + (1.0 - theta) * trend_value);
    }

    // Apply simple exponential smoothing to theta line
    let alpha = 2.0 / (n as f64 + 1.0); // Optimize alpha
    let mut level = theta_line[0];
    let mut fitted_values = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);

    for &obs in &theta_line {
        fitted_values.push(level);
        residuals.push(obs - level);
        level = alpha * obs + (1.0 - alpha) * level;
    }

    // Generate forecasts
    let mut forecasts = Vec::with_capacity(horizon);
    let mut timestamps = Vec::with_capacity(horizon);

    let last_timestamp = timeseries.timestamps.last().unwrap();
    let interval = if timeseries.timestamps.len() > 1 {
        timeseries.timestamps[1] - timeseries.timestamps[0]
    } else {
        Duration::days(1)
    };

    // Calculate drift (long-term trend)
    let drift = if n > 1 {
        (values[n - 1] - values[0]) / (n - 1) as f64
    } else {
        0.0
    };

    for h in 1..=horizon {
        let forecast = level + h as f64 * drift;
        forecasts.push(forecast);
        timestamps.push(*last_timestamp + interval * h as i32);
    }

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("theta".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(theta).unwrap()));
    metadata.insert("alpha".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(alpha).unwrap()));
    metadata.insert("drift".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(drift).unwrap()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Theta".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Prophet-like decomposable time series forecast
pub fn prophet_forecast(
    timeseries: &TimeSeries,
    growth: GrowthType,
    seasonality_mode: SeasonalityMode,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    daily_seasonality: bool,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Empty time series".to_string())));
    }

    let values = &timeseries.values;
    let n = values.len();

    // Simple trend estimation
    let mut trend = Vec::with_capacity(n);
    match growth {
        GrowthType::Linear => {
            let slope = if n > 1 {
                (values[n - 1] - values[0]) / (n - 1) as f64
            } else {
                0.0
            };
            let intercept = values[0];

            for i in 0..n {
                trend.push(intercept + slope * i as f64);
            }
        },
        GrowthType::Logistic { capacity } => {
            // Simplified logistic growth
            let k = 0.1; // Growth rate parameter
            let m = n as f64 / 2.0; // Midpoint

            for i in 0..n {
                let t = i as f64;
                let growth_val = capacity / (1.0 + (-k * (t - m)).exp());
                trend.push(growth_val);
            }
        }
    }

    // Simple seasonal components
    let mut seasonal = vec![0.0; n];

    if yearly_seasonality && n >= 365 {
        for i in 0..n {
            let day_of_year = i % 365;
            seasonal[i] += 0.1 * (2.0 * std::f64::consts::PI * day_of_year as f64 / 365.0).sin();
        }
    }

    if weekly_seasonality && n >= 7 {
        for i in 0..n {
            let day_of_week = i % 7;
            seasonal[i] += 0.05 * (2.0 * std::f64::consts::PI * day_of_week as f64 / 7.0).sin();
        }
    }

    if daily_seasonality && n >= 24 {
        for i in 0..n {
            let hour_of_day = i % 24;
            seasonal[i] += 0.02 * (2.0 * std::f64::consts::PI * hour_of_day as f64 / 24.0).sin();
        }
    }

    // Combine components
    let mut fitted_values = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let fitted = match seasonality_mode {
            SeasonalityMode::Additive => trend[i] + seasonal[i],
            SeasonalityMode::Multiplicative => trend[i] * (1.0 + seasonal[i]),
        };
        fitted_values.push(fitted);
        residuals.push(values[i] - fitted);
    }

    // Generate forecasts
    let mut forecasts = Vec::with_capacity(horizon);
    let mut timestamps = Vec::with_capacity(horizon);

    let last_timestamp = timeseries.timestamps.last().unwrap();
    let interval = if timeseries.timestamps.len() > 1 {
        timeseries.timestamps[1] - timeseries.timestamps[0]
    } else {
        Duration::days(1)
    };

    let trend_slope = if n > 1 {
        trend[n - 1] - trend[n - 2]
    } else {
        0.0
    };

    for h in 1..=horizon {
        let future_trend = match growth {
            GrowthType::Linear => trend[n - 1] + h as f64 * trend_slope,
            GrowthType::Logistic { capacity } => {
                // Simple logistic extrapolation
                let k = 0.1;
                let t = (n + h - 1) as f64;
                let m = n as f64 / 2.0;
                capacity / (1.0 + (-k * (t - m)).exp())
            }
        };

        let mut future_seasonal = 0.0;

        if yearly_seasonality {
            let day_of_year = (n + h - 1) % 365;
            future_seasonal += 0.1 * (2.0 * std::f64::consts::PI * day_of_year as f64 / 365.0).sin();
        }

        if weekly_seasonality {
            let day_of_week = (n + h - 1) % 7;
            future_seasonal += 0.05 * (2.0 * std::f64::consts::PI * day_of_week as f64 / 7.0).sin();
        }

        if daily_seasonality {
            let hour_of_day = (n + h - 1) % 24;
            future_seasonal += 0.02 * (2.0 * std::f64::consts::PI * hour_of_day as f64 / 24.0).sin();
        }

        let forecast = match seasonality_mode {
            SeasonalityMode::Additive => future_trend + future_seasonal,
            SeasonalityMode::Multiplicative => future_trend * (1.0 + future_seasonal),
        };

        forecasts.push(forecast);
        timestamps.push(*last_timestamp + interval * h as i32);
    }

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("growth".to_string(), serde_json::Value::String(format!("{:?}", growth)));
    metadata.insert("seasonality_mode".to_string(), serde_json::Value::String(format!("{:?}", seasonality_mode)));
    metadata.insert("yearly_seasonality".to_string(), serde_json::Value::Bool(yearly_seasonality));
    metadata.insert("weekly_seasonality".to_string(), serde_json::Value::Bool(weekly_seasonality));
    metadata.insert("daily_seasonality".to_string(), serde_json::Value::Bool(daily_seasonality));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: "Prophet".to_string(),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}