//! Model evaluation and validation for forecasting models

use crate::{TimeSeries, Result, TimeSeriesError};
use crate::forecasting::{ForecastConfig, ForecastResult, ModelEvaluation, EvaluationMetric, forecast_timeseries};
use std::collections::HashMap;

/// Evaluate a forecasting model using various metrics
pub fn evaluate_model(
    timeseries: &TimeSeries,
    config: &ForecastConfig,
) -> Result<ModelEvaluation> {
    if timeseries.values.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Empty time series".to_string())));
    }

    let n = timeseries.values.len();
    if n < config.evaluation.min_train_size + config.horizon {
        return Err(Box::new(TimeSeriesError::InvalidInput(
            "Time series too short for evaluation".to_string()
        )));
    }

    if config.evaluation.cross_validation {
        cross_validation_evaluate(timeseries, config)
    } else {
        simple_holdout_evaluate(timeseries, config)
    }
}

/// Simple train-test split evaluation
fn simple_holdout_evaluate(
    timeseries: &TimeSeries,
    config: &ForecastConfig,
) -> Result<ModelEvaluation> {
    let n = timeseries.values.len();
    let test_size = config.horizon;
    let train_size = n - test_size;

    // Split the data
    let train_ts = TimeSeries {
        name: timeseries.name.clone(),
        values: timeseries.values[..train_size].to_vec(),
        timestamps: timeseries.timestamps[..train_size].to_vec(),
        metadata: timeseries.metadata.clone(),
        frequency: timeseries.frequency.clone(),
        missing_value_policy: timeseries.missing_value_policy.clone(),
    };

    let test_values = &timeseries.values[train_size..];

    // Generate forecast
    let forecast_result = forecast_timeseries(&train_ts, config)?;

    // Calculate metrics
    calculate_metrics(&forecast_result.forecasts, test_values, &config.evaluation.metrics, Some(&train_ts))
}

/// Cross-validation evaluation
fn cross_validation_evaluate(
    timeseries: &TimeSeries,
    config: &ForecastConfig,
) -> Result<ModelEvaluation> {
    let n = timeseries.values.len();
    let min_train_size = config.evaluation.min_train_size;
    let horizon = config.horizon;
    let folds = config.evaluation.cv_folds;

    if n < min_train_size + horizon * folds {
        return Err(Box::new(TimeSeriesError::InvalidInput(
            "Time series too short for cross-validation".to_string()
        )));
    }

    let mut all_forecasts = Vec::new();
    let mut all_actuals = Vec::new();
    let mut mae_scores = Vec::new();
    let mut rmse_scores = Vec::new();
    let mut mape_scores = Vec::new();
    let mut smape_scores = Vec::new();

    if config.evaluation.walk_forward {
        // Walk-forward validation
        let step_size = (n - min_train_size - horizon) / folds;

        for fold in 0..folds {
            let train_end = min_train_size + fold * step_size;
            let test_start = train_end;
            let test_end = test_start + horizon;

            if test_end > n {
                break;
            }

            let train_ts = TimeSeries {
                name: timeseries.name.clone(),
                values: timeseries.values[..train_end].to_vec(),
                timestamps: timeseries.timestamps[..train_end].to_vec(),
                metadata: timeseries.metadata.clone(),
                frequency: timeseries.frequency.clone(),
                missing_value_policy: timeseries.missing_value_policy.clone(),
            };

            let test_values = &timeseries.values[test_start..test_end];

            // Generate forecast
            if let Ok(forecast_result) = forecast_timeseries(&train_ts, config) {
                let forecast_values = &forecast_result.forecasts[..test_values.len().min(forecast_result.forecasts.len())];

                all_forecasts.extend_from_slice(forecast_values);
                all_actuals.extend_from_slice(test_values);

                // Calculate fold metrics
                if let Ok(fold_eval) = calculate_metrics(forecast_values, test_values, &config.evaluation.metrics, Some(&train_ts)) {
                    mae_scores.push(fold_eval.mae);
                    rmse_scores.push(fold_eval.rmse);
                    mape_scores.push(fold_eval.mape);
                    smape_scores.push(fold_eval.smape);
                }
            }
        }
    } else {
        // Traditional k-fold cross-validation adapted for time series
        let fold_size = (n - min_train_size) / folds;

        for fold in 0..folds {
            let test_start = min_train_size + fold * fold_size;
            let test_end = (test_start + horizon).min(n);

            if test_end > n {
                break;
            }

            let train_ts = TimeSeries {
                name: timeseries.name.clone(),
                values: timeseries.values[..test_start].to_vec(),
                timestamps: timeseries.timestamps[..test_start].to_vec(),
                metadata: timeseries.metadata.clone(),
                frequency: timeseries.frequency.clone(),
                missing_value_policy: timeseries.missing_value_policy.clone(),
            };

            let test_values = &timeseries.values[test_start..test_end];

            // Generate forecast
            if let Ok(forecast_result) = forecast_timeseries(&train_ts, config) {
                let forecast_values = &forecast_result.forecasts[..test_values.len().min(forecast_result.forecasts.len())];

                all_forecasts.extend_from_slice(forecast_values);
                all_actuals.extend_from_slice(test_values);

                // Calculate fold metrics
                if let Ok(fold_eval) = calculate_metrics(forecast_values, test_values, &config.evaluation.metrics, Some(&train_ts)) {
                    mae_scores.push(fold_eval.mae);
                    rmse_scores.push(fold_eval.rmse);
                    mape_scores.push(fold_eval.mape);
                    smape_scores.push(fold_eval.smape);
                }
            }
        }
    }

    if all_forecasts.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No valid folds in cross-validation".to_string())));
    }

    // Calculate overall metrics from all folds
    calculate_metrics(&all_forecasts, &all_actuals, &config.evaluation.metrics, None)
}

/// Calculate evaluation metrics
fn calculate_metrics(
    forecasts: &[f64],
    actuals: &[f64],
    metrics: &[EvaluationMetric],
    train_data: Option<&TimeSeries>,
) -> Result<ModelEvaluation> {
    if forecasts.len() != actuals.len() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Forecast and actual lengths don't match".to_string())));
    }

    let n = forecasts.len() as f64;
    if n == 0.0 {
        return Err(Box::new(TimeSeriesError::InvalidInput("No data to evaluate".to_string())));
    }

    // Calculate basic metrics
    let mut mae = 0.0;
    let mut mse = 0.0;
    let mut mape = 0.0;
    let mut smape = 0.0;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    let actual_mean: f64 = actuals.iter().sum::<f64>() / n;

    for (pred, actual) in forecasts.iter().zip(actuals.iter()) {
        let error = actual - pred;
        let abs_error = error.abs();

        mae += abs_error;
        mse += error * error;
        ss_res += error * error;
        ss_tot += (actual - actual_mean).powi(2);

        if actual.abs() > 1e-10 {
            mape += (abs_error / actual.abs()) * 100.0;
        }

        let denom = (actual.abs() + pred.abs()) / 2.0;
        if denom > 1e-10 {
            smape += (abs_error / denom) * 100.0;
        }
    }

    mae /= n;
    mse /= n;
    let rmse = mse.sqrt();
    mape /= n;
    smape /= n;

    let r_squared = if ss_tot > 1e-10 {
        Some(1.0 - (ss_res / ss_tot))
    } else {
        None
    };

    // Calculate MASE if training data is available
    let mase = if let Some(train_ts) = train_data {
        calculate_mase(forecasts, actuals, &train_ts.values)
    } else {
        None
    };

    // Calculate information criteria (simplified versions)
    let log_likelihood = if mse > 1e-10 {
        Some(-n * (2.0 * std::f64::consts::PI * mse).ln() / 2.0 - n / 2.0)
    } else {
        None
    };

    let k = 3.0; // Simplified parameter count
    let aic = log_likelihood.map(|ll| -2.0 * ll + 2.0 * k);
    let bic = log_likelihood.map(|ll| -2.0 * ll + k * n.ln());

    Ok(ModelEvaluation {
        mae,
        rmse,
        mape,
        smape,
        mase,
        aic,
        bic,
        log_likelihood,
        r_squared,
        custom_metrics: HashMap::new(),
    })
}

/// Calculate Mean Absolute Scaled Error
fn calculate_mase(forecasts: &[f64], actuals: &[f64], train_values: &[f64]) -> Option<f64> {
    if train_values.len() < 2 {
        return None;
    }

    // Calculate naive forecast error (seasonal naive with period=1)
    let mut naive_errors = 0.0;
    for i in 1..train_values.len() {
        naive_errors += (train_values[i] - train_values[i - 1]).abs();
    }
    let mae_naive = naive_errors / (train_values.len() - 1) as f64;

    if mae_naive < 1e-10 {
        return None;
    }

    // Calculate MAE of forecasts
    let mae_forecast: f64 = forecasts.iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| (actual - pred).abs())
        .sum::<f64>() / forecasts.len() as f64;

    Some(mae_forecast / mae_naive)
}

/// Compute prediction intervals using residual-based approach
pub fn compute_prediction_intervals(
    _timeseries: &TimeSeries,
    forecasts: &[f64],
    confidence_level: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if forecasts.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No forecasts provided".to_string())));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(Box::new(TimeSeriesError::InvalidInput("Confidence level must be between 0 and 1".to_string())));
    }

    // Simplified prediction intervals
    // In practice, this would use residual analysis and statistical methods
    let alpha = 1.0 - confidence_level;
    let z_score = match confidence_level {
        x if x >= 0.99 => 2.576,
        x if x >= 0.95 => 1.96,
        x if x >= 0.90 => 1.645,
        _ => 1.96, // Default to 95%
    };

    // Estimate forecast error standard deviation
    // This is a simplified approach - real implementation would use model residuals
    let forecast_mean = forecasts.iter().sum::<f64>() / forecasts.len() as f64;
    let forecast_std = (forecasts.iter()
        .map(|x| (x - forecast_mean).powi(2))
        .sum::<f64>() / forecasts.len() as f64)
        .sqrt();

    // Use a minimum standard deviation to avoid zero intervals
    let std_dev = forecast_std.max(forecast_mean.abs() * 0.1);

    let lower_bounds: Vec<f64> = forecasts.iter()
        .enumerate()
        .map(|(i, &forecast)| {
            let interval_width = z_score * std_dev * (1.0 + i as f64 * 0.1).sqrt();
            forecast - interval_width
        })
        .collect();

    let upper_bounds: Vec<f64> = forecasts.iter()
        .enumerate()
        .map(|(i, &forecast)| {
            let interval_width = z_score * std_dev * (1.0 + i as f64 * 0.1).sqrt();
            forecast + interval_width
        })
        .collect();

    Ok((lower_bounds, upper_bounds))
}

/// Perform residual analysis
pub fn residual_analysis(residuals: &[f64]) -> HashMap<String, f64> {
    let mut analysis = HashMap::new();

    if residuals.is_empty() {
        return analysis;
    }

    let n = residuals.len() as f64;

    // Basic statistics
    let mean: f64 = residuals.iter().sum::<f64>() / n;
    let variance = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    analysis.insert("mean".to_string(), mean);
    analysis.insert("std_dev".to_string(), std_dev);
    analysis.insert("variance".to_string(), variance);

    // Skewness
    let skewness = if std_dev > 1e-10 {
        residuals.iter().map(|r| ((r - mean) / std_dev).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };
    analysis.insert("skewness".to_string(), skewness);

    // Kurtosis
    let kurtosis = if std_dev > 1e-10 {
        residuals.iter().map(|r| ((r - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0
    } else {
        0.0
    };
    analysis.insert("kurtosis".to_string(), kurtosis);

    // Autocorrelation at lag 1
    if residuals.len() > 1 {
        let lag1_corr = calculate_autocorrelation(residuals, 1);
        analysis.insert("autocorr_lag1".to_string(), lag1_corr);
    }

    // Ljung-Box test statistic (simplified)
    let ljung_box = calculate_ljung_box_statistic(residuals, 10);
    analysis.insert("ljung_box".to_string(), ljung_box);

    analysis
}

/// Calculate autocorrelation at a given lag
fn calculate_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..(n - lag) {
        numerator += (data[i] - mean) * (data[i + lag] - mean);
    }

    for val in data {
        denominator += (val - mean).powi(2);
    }

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate Ljung-Box test statistic
fn calculate_ljung_box_statistic(residuals: &[f64], max_lag: usize) -> f64 {
    let n = residuals.len() as f64;
    let mut statistic = 0.0;

    for lag in 1..=max_lag.min(residuals.len() / 4) {
        let rho = calculate_autocorrelation(residuals, lag);
        statistic += rho * rho / (n - lag as f64);
    }

    n * (n + 2.0) * statistic
}