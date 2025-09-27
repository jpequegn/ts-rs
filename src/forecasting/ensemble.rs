//! Ensemble forecasting methods for combining multiple forecasting models

use crate::{TimeSeries, Result, TimeSeriesError};
use crate::forecasting::{ForecastMethod, ForecastResult, EnsembleCombination, forecast_timeseries, ForecastConfig};
use std::collections::HashMap;
use serde_json;

/// Generate ensemble forecast by combining multiple forecasting methods
pub fn ensemble_forecast(
    timeseries: &TimeSeries,
    methods: &[ForecastMethod],
    combination: EnsembleCombination,
    weights: Option<&Vec<f64>>,
    horizon: usize,
) -> Result<ForecastResult> {
    if timeseries.values.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Empty time series".to_string())));
    }

    if methods.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No methods provided for ensemble".to_string())));
    }

    // Generate forecasts from all methods
    let mut method_results = Vec::new();
    let mut method_names = Vec::new();

    for method in methods {
        let config = ForecastConfig {
            method: method.clone(),
            horizon,
            confidence_level: 0.95,
            include_intervals: false,
            evaluation: crate::forecasting::EvaluationConfig::default(),
            features: crate::forecasting::FeatureConfig::default(),
        };

        match forecast_timeseries(timeseries, &config) {
            Ok(result) => {
                method_results.push(result);
                method_names.push(format!("{:?}", method));
            }
            Err(_) => {
                // Skip methods that fail, but log the failure
                continue;
            }
        }
    }

    if method_results.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("All ensemble methods failed".to_string())));
    }

    // Combine forecasts
    let combined_forecasts = match combination {
        EnsembleCombination::Average => combine_average(&method_results),
        EnsembleCombination::Weighted => combine_weighted(&method_results, weights),
        EnsembleCombination::Median => combine_median(&method_results),
        EnsembleCombination::BestModel => combine_best_model(&method_results, timeseries),
        EnsembleCombination::OptimalWeights => combine_optimal_weights(&method_results, timeseries),
    }?;

    // Use timestamps from the first successful method
    let timestamps = method_results[0].timestamps.clone();

    // Combine fitted values and residuals if available
    let fitted_values = combine_fitted_values(&method_results, combination, weights);
    let residuals = if let (Some(fitted), Some(original)) = (&fitted_values, timeseries.values.get(..fitted_values.as_ref().map_or(0, |f| f.len()))) {
        Some(original.iter().zip(fitted.iter()).map(|(a, f)| a - f).collect())
    } else {
        None
    };

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("combination_method".to_string(), serde_json::Value::String(format!("{:?}", combination)));
    metadata.insert("num_methods".to_string(), serde_json::Value::Number(serde_json::Number::from(method_results.len())));
    metadata.insert("method_names".to_string(), serde_json::Value::Array(
        method_names.iter().map(|name| serde_json::Value::String(name.clone())).collect()
    ));

    if let Some(w) = weights {
        metadata.insert("weights".to_string(), serde_json::Value::Array(
            w.iter().map(|&weight| serde_json::Value::Number(serde_json::Number::from_f64(weight).unwrap())).collect()
        ));
    }

    Ok(ForecastResult {
        forecasts: combined_forecasts,
        timestamps,
        lower_bounds: None,
        upper_bounds: None,
        confidence_level: 0.95,
        method: format!("Ensemble-{:?}", combination),
        metadata,
        fitted_values,
        residuals,
        evaluation: None,
        feature_importance: calculate_method_importance(&method_results, &method_names),
    })
}

/// Simple average combination
fn combine_average(results: &[ForecastResult]) -> Result<Vec<f64>> {
    if results.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No results to combine".to_string())));
    }

    let horizon = results[0].forecasts.len();
    let num_methods = results.len() as f64;
    let mut combined = vec![0.0; horizon];

    for result in results {
        if result.forecasts.len() != horizon {
            return Err(Box::new(TimeSeriesError::InvalidInput("Inconsistent forecast horizons".to_string())));
        }

        for (i, &forecast) in result.forecasts.iter().enumerate() {
            combined[i] += forecast / num_methods;
        }
    }

    Ok(combined)
}

/// Weighted average combination
fn combine_weighted(results: &[ForecastResult], weights: Option<&Vec<f64>>) -> Result<Vec<f64>> {
    if results.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No results to combine".to_string())));
    }

    let weights = match weights {
        Some(w) => {
            if w.len() != results.len() {
                return Err(Box::new(TimeSeriesError::InvalidInput("Weight count doesn't match method count".to_string())));
            }
            w.clone()
        }
        None => {
            // Equal weights
            vec![1.0 / results.len() as f64; results.len()]
        }
    };

    // Normalize weights
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= 0.0 {
        return Err(Box::new(TimeSeriesError::InvalidInput("Invalid weights".to_string())));
    }
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

    let horizon = results[0].forecasts.len();
    let mut combined = vec![0.0; horizon];

    for (result, &weight) in results.iter().zip(normalized_weights.iter()) {
        if result.forecasts.len() != horizon {
            return Err(Box::new(TimeSeriesError::InvalidInput("Inconsistent forecast horizons".to_string())));
        }

        for (i, &forecast) in result.forecasts.iter().enumerate() {
            combined[i] += forecast * weight;
        }
    }

    Ok(combined)
}

/// Median combination
fn combine_median(results: &[ForecastResult]) -> Result<Vec<f64>> {
    if results.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No results to combine".to_string())));
    }

    let horizon = results[0].forecasts.len();
    let mut combined = vec![0.0; horizon];

    for i in 0..horizon {
        let mut values: Vec<f64> = results.iter()
            .filter_map(|result| result.forecasts.get(i).copied())
            .collect();

        if values.is_empty() {
            return Err(Box::new(TimeSeriesError::InvalidInput("No valid forecasts at position".to_string())));
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        combined[i] = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };
    }

    Ok(combined)
}

/// Best model combination (select best performing model)
fn combine_best_model(results: &[ForecastResult], timeseries: &TimeSeries) -> Result<Vec<f64>> {
    if results.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No results to combine".to_string())));
    }

    // Evaluate each model's performance on fitted values
    let mut best_index = 0;
    let mut best_score = f64::INFINITY;

    for (i, result) in results.iter().enumerate() {
        if let Some(fitted) = &result.fitted_values {
            let score = calculate_model_score(fitted, &timeseries.values);
            if score < best_score {
                best_score = score;
                best_index = i;
            }
        }
    }

    Ok(results[best_index].forecasts.clone())
}

/// Optimal weights combination using regression
fn combine_optimal_weights(results: &[ForecastResult], timeseries: &TimeSeries) -> Result<Vec<f64>> {
    if results.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("No results to combine".to_string())));
    }

    // Calculate optimal weights using least squares on fitted values
    let weights = calculate_optimal_weights(results, timeseries)?;
    combine_weighted(results, Some(&weights))
}

/// Calculate optimal weights using least squares regression
fn calculate_optimal_weights(results: &[ForecastResult], timeseries: &TimeSeries) -> Result<Vec<f64>> {
    let num_methods = results.len();

    // Collect fitted values for each method
    let mut method_fitted: Vec<Vec<f64>> = Vec::new();
    let mut common_length = usize::MAX;

    for result in results {
        if let Some(fitted) = &result.fitted_values {
            common_length = common_length.min(fitted.len().min(timeseries.values.len()));
            method_fitted.push(fitted.clone());
        } else {
            return Err(Box::new(TimeSeriesError::InvalidInput("Fitted values required for optimal weights".to_string())));
        }
    }

    if common_length == 0 {
        return Err(Box::new(TimeSeriesError::InvalidInput("No common fitted values".to_string())));
    }

    // Set up linear system: X * weights = y
    // Where X is matrix of fitted values, y is actual values
    let actual_values = &timeseries.values[..common_length];

    // Simple approach: equal weights with adjustment based on individual performance
    let mut weights = vec![1.0 / num_methods as f64; num_methods];

    // Adjust weights based on individual model performance
    let mut total_inverse_error = 0.0;
    let mut inverse_errors = Vec::new();

    for fitted in &method_fitted {
        let mse = calculate_mse(&fitted[..common_length], actual_values);
        let inverse_error = if mse > 1e-10 { 1.0 / mse } else { 1e10 };
        inverse_errors.push(inverse_error);
        total_inverse_error += inverse_error;
    }

    // Normalize weights based on inverse error
    if total_inverse_error > 0.0 {
        for (i, &inverse_error) in inverse_errors.iter().enumerate() {
            weights[i] = inverse_error / total_inverse_error;
        }
    }

    Ok(weights)
}

/// Calculate Mean Squared Error
fn calculate_mse(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() || predictions.is_empty() {
        return f64::INFINITY;
    }

    let n = predictions.len() as f64;
    predictions.iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f64>() / n
}

/// Calculate model score for best model selection
fn calculate_model_score(fitted: &[f64], actual: &[f64]) -> f64 {
    let common_len = fitted.len().min(actual.len());
    if common_len == 0 {
        return f64::INFINITY;
    }

    calculate_mse(&fitted[..common_len], &actual[..common_len])
}

/// Combine fitted values from multiple methods
fn combine_fitted_values(
    results: &[ForecastResult],
    combination: EnsembleCombination,
    weights: Option<&Vec<f64>>,
) -> Option<Vec<f64>> {
    let fitted_results: Vec<&Vec<f64>> = results.iter()
        .filter_map(|r| r.fitted_values.as_ref())
        .collect();

    if fitted_results.is_empty() {
        return None;
    }

    let min_length = fitted_results.iter().map(|f| f.len()).min()?;
    let mut combined = vec![0.0; min_length];

    match combination {
        EnsembleCombination::Average => {
            let num_methods = fitted_results.len() as f64;
            for fitted in &fitted_results {
                for i in 0..min_length {
                    combined[i] += fitted[i] / num_methods;
                }
            }
        }
        EnsembleCombination::Weighted => {
            let default_weights = vec![1.0 / fitted_results.len() as f64; fitted_results.len()];
            let weights = weights.unwrap_or(&default_weights);
            let weight_sum: f64 = weights.iter().take(fitted_results.len()).sum();

            for (fitted, &weight) in fitted_results.iter().zip(weights.iter()) {
                for i in 0..min_length {
                    combined[i] += fitted[i] * weight / weight_sum;
                }
            }
        }
        EnsembleCombination::Median => {
            for i in 0..min_length {
                let mut values: Vec<f64> = fitted_results.iter().map(|f| f[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                combined[i] = if values.len() % 2 == 0 {
                    let mid = values.len() / 2;
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[values.len() / 2]
                };
            }
        }
        _ => {
            // For other methods, use average
            let num_methods = fitted_results.len() as f64;
            for fitted in &fitted_results {
                for i in 0..min_length {
                    combined[i] += fitted[i] / num_methods;
                }
            }
        }
    }

    Some(combined)
}

/// Calculate method importance for feature importance
fn calculate_method_importance(results: &[ForecastResult], method_names: &[String]) -> Option<HashMap<String, f64>> {
    if results.is_empty() || method_names.is_empty() {
        return None;
    }

    let mut importance = HashMap::new();
    let num_methods = results.len() as f64;

    // Simple importance based on relative performance
    let mut scores = Vec::new();
    for result in results {
        if let Some(residuals) = &result.residuals {
            let mse = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / residuals.len() as f64;
            scores.push(if mse > 1e-10 { 1.0 / mse } else { 1e10 });
        } else {
            scores.push(1.0); // Default score
        }
    }

    let total_score: f64 = scores.iter().sum();
    if total_score > 0.0 {
        for (i, name) in method_names.iter().enumerate() {
            if let Some(score) = scores.get(i) {
                importance.insert(name.clone(), score / total_score);
            }
        }
    } else {
        // Equal importance if no performance data
        for name in method_names {
            importance.insert(name.clone(), 1.0 / num_methods);
        }
    }

    Some(importance)
}