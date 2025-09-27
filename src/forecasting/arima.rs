//! # ARIMA Family Models
//!
//! Implementation of ARIMA, SARIMA, and Auto-ARIMA forecasting methods
//! for time series analysis and prediction.

use crate::{TimeSeries, Result};
use crate::forecasting::ForecastResult;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;

/// ARIMA (AutoRegressive Integrated Moving Average) forecast
pub fn arima_forecast(
    timeseries: &TimeSeries,
    p: usize,  // Autoregressive order
    d: usize,  // Differencing order
    q: usize,  // Moving average order
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.len() < p + d + q + 10 {
        return Err("Time series too short for ARIMA model".into());
    }

    // Apply differencing
    let mut series = timeseries.values.clone();
    let mut differenced_series = Vec::new();

    for _ in 0..d {
        if series.len() < 2 {
            return Err("Cannot apply more differencing: series too short".into());
        }

        let mut diff = Vec::new();
        for i in 1..series.len() {
            diff.push(series[i] - series[i - 1]);
        }
        differenced_series = diff.clone();
        series = diff;
    }

    let working_series = if d > 0 { &differenced_series } else { &timeseries.values };

    // Estimate AR and MA parameters using method of moments (simplified)
    let ar_params = estimate_ar_parameters(working_series, p)?;
    let ma_params = estimate_ma_parameters(working_series, q)?;

    // Calculate fitted values and residuals
    let (fitted_values, residuals) = fit_arima_model(
        &timeseries.values, &ar_params, &ma_params, p, d, q
    )?;

    // Generate forecasts
    let mut forecasts = Vec::new();
    let mut forecast_errors = vec![0.0; q.max(1)]; // Store recent errors for MA part

    // Initialize with recent values for AR part
    let mut recent_values: Vec<f64> = if d > 0 {
        differenced_series[differenced_series.len().saturating_sub(p)..]
            .to_vec()
    } else {
        timeseries.values[timeseries.values.len().saturating_sub(p)..]
            .to_vec()
    };

    let mut last_level = *timeseries.values.last().unwrap_or(&0.0);

    for h in 0..horizon {
        // AR component
        let ar_component = if p > 0 {
            ar_params.iter()
                .zip(recent_values.iter().rev())
                .map(|(param, value)| param * value)
                .sum::<f64>()
        } else {
            0.0
        };

        // MA component
        let ma_component = if q > 0 {
            ma_params.iter()
                .zip(forecast_errors.iter().rev())
                .map(|(param, error)| param * error)
                .sum::<f64>()
        } else {
            0.0
        };

        let forecast_diff = ar_component + ma_component;

        // If differencing was applied, integrate back
        let forecast = if d > 0 {
            last_level + forecast_diff
        } else {
            forecast_diff
        };

        forecasts.push(forecast);

        // Update for next forecast
        if p > 0 {
            recent_values.push(if d > 0 { forecast_diff } else { forecast });
            if recent_values.len() > p {
                recent_values.remove(0);
            }
        }

        // Update forecast errors (assume zero for future forecasts)
        if q > 0 {
            forecast_errors.push(0.0);
            if forecast_errors.len() > q {
                forecast_errors.remove(0);
            }
        }

        if d > 0 {
            last_level = forecast;
        }
    }

    let last_timestamp = timeseries.timestamps[timeseries.timestamps.len() - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    let mut metadata = HashMap::new();
    metadata.insert("p".to_string(), serde_json::Value::Number(serde_json::Number::from(p)));
    metadata.insert("d".to_string(), serde_json::Value::Number(serde_json::Number::from(d)));
    metadata.insert("q".to_string(), serde_json::Value::Number(serde_json::Number::from(q)));
    metadata.insert("ar_params".to_string(), serde_json::to_value(&ar_params)?);
    metadata.insert("ma_params".to_string(), serde_json::to_value(&ma_params)?);
    metadata.insert("method".to_string(), serde_json::Value::String("ARIMA".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: format!("ARIMA({},{},{})", p, d, q),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Seasonal ARIMA (SARIMA) forecast
pub fn sarima_forecast(
    timeseries: &TimeSeries,
    p: usize, d: usize, q: usize,          // Non-seasonal parameters
    seasonal_p: usize, seasonal_d: usize, seasonal_q: usize,  // Seasonal parameters
    seasonal_period: usize,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.len() < (p + d + q + seasonal_p + seasonal_d + seasonal_q) * 2 + seasonal_period {
        return Err("Time series too short for SARIMA model".into());
    }

    // Apply non-seasonal differencing
    let mut series = timeseries.values.clone();
    for _ in 0..d {
        if series.len() < 2 {
            return Err("Cannot apply more differencing: series too short".into());
        }
        let mut diff = Vec::new();
        for i in 1..series.len() {
            diff.push(series[i] - series[i - 1]);
        }
        series = diff;
    }

    // Apply seasonal differencing
    for _ in 0..seasonal_d {
        if series.len() < seasonal_period + 1 {
            return Err("Cannot apply more seasonal differencing: series too short".into());
        }
        let mut seasonal_diff = Vec::new();
        for i in seasonal_period..series.len() {
            seasonal_diff.push(series[i] - series[i - seasonal_period]);
        }
        series = seasonal_diff;
    }

    // Estimate parameters (simplified approach)
    let ar_params = estimate_ar_parameters(&series, p)?;
    let ma_params = estimate_ma_parameters(&series, q)?;
    let seasonal_ar_params = estimate_seasonal_ar_parameters(&series, seasonal_p, seasonal_period)?;
    let seasonal_ma_params = estimate_seasonal_ma_parameters(&series, seasonal_q, seasonal_period)?;

    // Generate forecasts using SARIMA logic
    let mut forecasts = Vec::new();
    let n = timeseries.values.len();

    // Initialize with the original series values for seasonal patterns
    let mut extended_series = timeseries.values.clone();

    for h in 1..=horizon {
        let mut forecast = 0.0;

        // Non-seasonal AR component
        for i in 1..=p {
            if n >= i {
                forecast += ar_params.get(i - 1).unwrap_or(&0.0) * extended_series[n - i];
            }
        }

        // Seasonal AR component
        for i in 1..=seasonal_p {
            let lag = i * seasonal_period;
            if n >= lag {
                forecast += seasonal_ar_params.get(i - 1).unwrap_or(&0.0) * extended_series[n - lag];
            }
        }

        // MA components would require error terms, simplified here
        // In a full implementation, we'd track and use the residual errors

        // For seasonal naive component as backup
        if forecast == 0.0 {
            let seasonal_lag = ((h - 1) % seasonal_period) + 1;
            if n >= seasonal_lag {
                forecast = extended_series[n - seasonal_lag];
            } else {
                forecast = extended_series.last().copied().unwrap_or(0.0);
            }
        }

        forecasts.push(forecast);
        extended_series.push(forecast);
    }

    let last_timestamp = timeseries.timestamps[timeseries.timestamps.len() - 1];
    let timestamps = generate_future_timestamps(last_timestamp, horizon, &timeseries.timestamps)?;

    // Calculate fitted values (simplified)
    let fitted_values = vec![0.0; timeseries.values.len()]; // Placeholder
    let residuals = vec![0.0; timeseries.values.len()]; // Placeholder

    let mut metadata = HashMap::new();
    metadata.insert("p".to_string(), serde_json::Value::Number(serde_json::Number::from(p)));
    metadata.insert("d".to_string(), serde_json::Value::Number(serde_json::Number::from(d)));
    metadata.insert("q".to_string(), serde_json::Value::Number(serde_json::Number::from(q)));
    metadata.insert("seasonal_p".to_string(), serde_json::Value::Number(serde_json::Number::from(seasonal_p)));
    metadata.insert("seasonal_d".to_string(), serde_json::Value::Number(serde_json::Number::from(seasonal_d)));
    metadata.insert("seasonal_q".to_string(), serde_json::Value::Number(serde_json::Number::from(seasonal_q)));
    metadata.insert("seasonal_period".to_string(), serde_json::Value::Number(serde_json::Number::from(seasonal_period)));
    metadata.insert("method".to_string(), serde_json::Value::String("SARIMA".to_string()));

    Ok(ForecastResult {
        forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: format!("SARIMA({},{},{})x({},{},{})_{}", p, d, q, seasonal_p, seasonal_d, seasonal_q, seasonal_period),
        metadata,
        fitted_values: Some(fitted_values),
        residuals: Some(residuals),
        evaluation: None,
        feature_importance: None,
    })
}

/// Auto-ARIMA with automatic parameter selection
pub fn auto_arima_forecast(
    timeseries: &TimeSeries,
    max_p: usize, max_d: usize, max_q: usize,
    max_seasonal_p: usize, max_seasonal_d: usize, max_seasonal_q: usize,
    seasonal_period: Option<usize>,
    horizon: usize,
) -> Result<ForecastResult> {
    let mut best_aic = f64::INFINITY;
    let mut best_params = (0, 0, 0, 0, 0, 0);
    let mut best_result: Option<ForecastResult> = None;

    // Grid search over parameter space (simplified)
    for p in 0..=max_p.min(3) {  // Limit search space for performance
        for d in 0..=max_d.min(2) {
            for q in 0..=max_q.min(3) {
                // For seasonal models
                let seasonal_combinations = if let Some(period) = seasonal_period {
                    vec![(0, 0, 0, period), (1, 0, 0, period), (0, 1, 0, period), (1, 1, 0, period)]
                } else {
                    vec![(0, 0, 0, 0)]
                };

                for (sp, sd, sq, period) in seasonal_combinations {
                    if sp > max_seasonal_p || sd > max_seasonal_d || sq > max_seasonal_q {
                        continue;
                    }

                    let result = if period > 0 {
                        sarima_forecast(timeseries, p, d, q, sp, sd, sq, period, horizon)
                    } else {
                        arima_forecast(timeseries, p, d, q, horizon)
                    };

                    if let Ok(forecast_result) = result {
                        // Calculate AIC (simplified)
                        let aic = calculate_aic(&forecast_result, p + d + q + sp + sd + sq);

                        if aic < best_aic {
                            best_aic = aic;
                            best_params = (p, d, q, sp, sd, sq);
                            best_result = Some(forecast_result);
                        }
                    }
                }
            }
        }
    }

    if let Some(mut result) = best_result {
        result.method = format!("Auto-ARIMA({},{},{})x({},{},{})_{}",
                               best_params.0, best_params.1, best_params.2,
                               best_params.3, best_params.4, best_params.5,
                               seasonal_period.unwrap_or(0));

        // Add Auto-ARIMA metadata
        result.metadata.insert("auto_selected_params".to_string(),
                              serde_json::to_value(best_params)?);
        result.metadata.insert("best_aic".to_string(),
                              serde_json::Value::Number(serde_json::Number::from_f64(best_aic).unwrap()));

        Ok(result)
    } else {
        Err("Auto-ARIMA failed to find suitable parameters".into())
    }
}

/// Estimate AR parameters using Yule-Walker equations (simplified)
fn estimate_ar_parameters(series: &[f64], p: usize) -> Result<Vec<f64>> {
    if p == 0 {
        return Ok(Vec::new());
    }

    if series.len() < p + 1 {
        return Ok(vec![0.1; p]); // Default small values
    }

    // Calculate autocorrelations
    let autocorrs = calculate_autocorrelations(series, p)?;

    // Solve Yule-Walker equations (simplified - use autocorrelations directly)
    let mut ar_params = Vec::new();
    for i in 0..p {
        let param = autocorrs.get(i + 1).copied().unwrap_or(0.0);
        ar_params.push(param.clamp(-0.99, 0.99)); // Ensure stationarity
    }

    Ok(ar_params)
}

/// Estimate MA parameters (simplified method)
fn estimate_ma_parameters(series: &[f64], q: usize) -> Result<Vec<f64>> {
    if q == 0 {
        return Ok(Vec::new());
    }

    // Simplified MA parameter estimation
    // In practice, this would use maximum likelihood estimation
    let ma_params = vec![0.1; q]; // Default small values

    Ok(ma_params)
}

/// Estimate seasonal AR parameters
fn estimate_seasonal_ar_parameters(series: &[f64], seasonal_p: usize, seasonal_period: usize) -> Result<Vec<f64>> {
    if seasonal_p == 0 {
        return Ok(Vec::new());
    }

    // Calculate seasonal autocorrelations
    let seasonal_autocorr = calculate_seasonal_autocorrelation(series, seasonal_period)?;

    let mut params = Vec::new();
    for i in 0..seasonal_p {
        params.push(seasonal_autocorr.clamp(-0.99, 0.99));
    }

    Ok(params)
}

/// Estimate seasonal MA parameters
fn estimate_seasonal_ma_parameters(series: &[f64], seasonal_q: usize, _seasonal_period: usize) -> Result<Vec<f64>> {
    if seasonal_q == 0 {
        return Ok(Vec::new());
    }

    // Simplified seasonal MA parameter estimation
    let params = vec![0.1; seasonal_q];

    Ok(params)
}

/// Calculate autocorrelations for AR parameter estimation
fn calculate_autocorrelations(series: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;

    // Calculate variance
    let variance = series.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n as f64;

    if variance == 0.0 {
        return Ok(vec![0.0; max_lag + 1]);
    }

    let mut autocorrs = Vec::new();
    autocorrs.push(1.0); // Lag 0 autocorrelation is always 1

    for lag in 1..=max_lag {
        if lag >= n {
            autocorrs.push(0.0);
            continue;
        }

        let covariance = (0..n - lag)
            .map(|i| (series[i] - mean) * (series[i + lag] - mean))
            .sum::<f64>() / (n - lag) as f64;

        let autocorr = covariance / variance;
        autocorrs.push(autocorr);
    }

    Ok(autocorrs)
}

/// Calculate seasonal autocorrelation
fn calculate_seasonal_autocorrelation(series: &[f64], seasonal_period: usize) -> Result<f64> {
    if series.len() < seasonal_period + 1 {
        return Ok(0.0);
    }

    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;

    let variance = series.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n as f64;

    if variance == 0.0 {
        return Ok(0.0);
    }

    let covariance = (0..n - seasonal_period)
        .map(|i| (series[i] - mean) * (series[i + seasonal_period] - mean))
        .sum::<f64>() / (n - seasonal_period) as f64;

    Ok(covariance / variance)
}

/// Fit ARIMA model and return fitted values and residuals
fn fit_arima_model(
    original_series: &[f64],
    ar_params: &[f64],
    ma_params: &[f64],
    p: usize,
    d: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = original_series.len();
    let mut fitted_values = vec![0.0; n];
    let mut residuals = vec![0.0; n];
    let mut errors = vec![0.0; q.max(1)];

    // Apply differencing if needed
    let mut working_series = original_series.to_vec();
    for _ in 0..d {
        if working_series.len() < 2 {
            break;
        }
        let mut diff = Vec::new();
        for i in 1..working_series.len() {
            diff.push(working_series[i] - working_series[i - 1]);
        }
        working_series = diff;
    }

    let start_index = p.max(q).max(d);
    for i in start_index..n {
        let mut prediction = 0.0;

        // AR component
        for j in 1..=p {
            if i >= j && j <= working_series.len() {
                let idx = working_series.len() - (n - i) - 1 + j;
                if idx < working_series.len() {
                    prediction += ar_params[j - 1] * working_series[idx];
                }
            }
        }

        // MA component
        for j in 1..=q {
            if j <= errors.len() {
                prediction += ma_params[j - 1] * errors[errors.len() - j];
            }
        }

        // Integrate back if differencing was applied
        let fitted = if d > 0 && i > 0 {
            fitted_values[i - 1] + prediction
        } else {
            prediction
        };

        fitted_values[i] = fitted;
        residuals[i] = original_series[i] - fitted;

        // Update errors for MA component
        if q > 0 {
            errors.push(residuals[i]);
            if errors.len() > q {
                errors.remove(0);
            }
        }
    }

    Ok((fitted_values, residuals))
}

/// Calculate AIC for model selection
fn calculate_aic(forecast_result: &ForecastResult, num_params: usize) -> f64 {
    if let Some(residuals) = &forecast_result.residuals {
        let n = residuals.len() as f64;
        let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
        let mse = sse / n;

        if mse > 0.0 {
            n * mse.ln() + 2.0 * num_params as f64
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    }
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

    let interval_seconds = intervals.last().copied().unwrap_or(3600);
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
        let timestamps = (0..20)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1 + i, 0, 0, 0).unwrap())
            .collect();

        // Create a simple AR(1) process
        let mut values = vec![0.0];
        for i in 1..20 {
            values.push(0.5 * values[i - 1] + (i as f64).sin());
        }

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_arima_forecast() {
        let ts = create_test_timeseries();
        let result = arima_forecast(&ts, 1, 0, 1, 5).unwrap();

        assert_eq!(result.forecasts.len(), 5);
        assert_eq!(result.timestamps.len(), 5);
        assert!(result.fitted_values.is_some());
        assert!(result.residuals.is_some());
    }

    #[test]
    fn test_auto_arima() {
        let ts = create_test_timeseries();
        let result = auto_arima_forecast(&ts, 2, 1, 2, 1, 1, 1, None, 3).unwrap();

        assert_eq!(result.forecasts.len(), 3);
        assert!(result.method.starts_with("Auto-ARIMA"));
    }

    #[test]
    fn test_autocorrelation_calculation() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let autocorrs = calculate_autocorrelations(&series, 2).unwrap();

        assert_eq!(autocorrs.len(), 3); // Lags 0, 1, 2
        assert_eq!(autocorrs[0], 1.0); // Lag 0 is always 1
    }
}