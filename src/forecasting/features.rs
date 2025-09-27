//! Feature engineering utilities for time series forecasting

use crate::{TimeSeries, Result, TimeSeriesError};
use crate::forecasting::{FeatureConfig, LagConfig, RollingConfig, CalendarConfig, RollingStatistic};
use chrono::{DateTime, Utc, Datelike, Timelike};
use std::collections::HashMap;

/// Enhanced TimeSeries with engineered features
#[derive(Debug, Clone)]
pub struct EnhancedTimeSeries {
    pub original: TimeSeries,
    pub features: HashMap<String, Vec<f64>>,
    pub feature_names: Vec<String>,
}

/// Create enhanced time series with engineered features
pub fn create_enhanced_timeseries(
    timeseries: &TimeSeries,
    config: &FeatureConfig,
) -> Result<EnhancedTimeSeries> {
    if timeseries.values.is_empty() {
        return Err(Box::new(TimeSeriesError::InvalidInput("Empty time series".to_string())));
    }

    let mut features = HashMap::new();
    let mut feature_names = Vec::new();

    // Add lag features
    if let Some(lag_config) = &config.lag_features {
        let (lag_features, lag_names) = create_lag_features(timeseries, lag_config)?;
        for (name, values) in lag_features {
            features.insert(name.clone(), values);
            feature_names.push(name);
        }
        feature_names.extend(lag_names);
    }

    // Add rolling features
    if let Some(rolling_config) = &config.rolling_features {
        let (rolling_features, rolling_names) = create_rolling_features(timeseries, rolling_config)?;
        for (name, values) in rolling_features {
            features.insert(name.clone(), values);
        }
        feature_names.extend(rolling_names);
    }

    // Add calendar features
    if let Some(calendar_config) = &config.calendar_features {
        let (calendar_features, calendar_names) = create_calendar_features(timeseries, calendar_config)?;
        for (name, values) in calendar_features {
            features.insert(name.clone(), values);
        }
        feature_names.extend(calendar_names);
    }

    // Add trend features
    if config.trend_features {
        let (trend_features, trend_names) = create_trend_features(timeseries)?;
        for (name, values) in trend_features {
            features.insert(name.clone(), values);
        }
        feature_names.extend(trend_names);
    }

    // Add seasonal features
    if config.seasonal_features {
        let (seasonal_features, seasonal_names) = create_seasonal_features(timeseries)?;
        for (name, values) in seasonal_features {
            features.insert(name.clone(), values);
        }
        feature_names.extend(seasonal_names);
    }

    Ok(EnhancedTimeSeries {
        original: timeseries.clone(),
        features,
        feature_names,
    })
}

/// Create lag features
fn create_lag_features(
    timeseries: &TimeSeries,
    config: &LagConfig,
) -> Result<(HashMap<String, Vec<f64>>, Vec<String>)> {
    let mut features = HashMap::new();
    let mut feature_names = Vec::new();
    let values = &timeseries.values;
    let n = values.len();

    // Determine which lags to create
    let lags = if let Some(specific_lags) = &config.specific_lags {
        specific_lags.clone()
    } else {
        (1..=config.max_lags).collect()
    };

    // Create basic lag features
    for lag in &lags {
        if *lag >= n {
            continue;
        }

        let mut lag_values = vec![f64::NAN; *lag];
        lag_values.extend(&values[..(n - lag)]);

        let feature_name = format!("lag_{}", lag);
        features.insert(feature_name.clone(), lag_values);
        feature_names.push(feature_name);
    }

    // Create seasonal lag features if requested
    if config.seasonal_lags {
        if let Some(seasonal_period) = config.seasonal_period {
            for multiplier in 1..=3 {
                let seasonal_lag = seasonal_period * multiplier;
                if seasonal_lag >= n {
                    continue;
                }

                let mut seasonal_values = vec![f64::NAN; seasonal_lag];
                seasonal_values.extend(&values[..(n - seasonal_lag)]);

                let feature_name = format!("seasonal_lag_{}", seasonal_lag);
                features.insert(feature_name.clone(), seasonal_values);
                feature_names.push(feature_name);
            }
        }
    }

    Ok((features, feature_names))
}

/// Create rolling window features
fn create_rolling_features(
    timeseries: &TimeSeries,
    config: &RollingConfig,
) -> Result<(HashMap<String, Vec<f64>>, Vec<String>)> {
    let mut features = HashMap::new();
    let mut feature_names = Vec::new();
    let values = &timeseries.values;
    let n = values.len();

    for &window in &config.windows {
        if window >= n {
            continue;
        }

        for &statistic in &config.statistics {
            let (feature_values, feature_name) = calculate_rolling_statistic(values, window, statistic)?;
            features.insert(feature_name.clone(), feature_values);
            feature_names.push(feature_name);
        }
    }

    Ok((features, feature_names))
}

/// Calculate rolling statistic
fn calculate_rolling_statistic(
    values: &[f64],
    window: usize,
    statistic: RollingStatistic,
) -> Result<(Vec<f64>, String)> {
    let n = values.len();
    let mut result = vec![f64::NAN; window - 1];

    for i in (window - 1)..n {
        let window_values = &values[(i + 1 - window)..(i + 1)];

        let stat_value = match statistic {
            RollingStatistic::Mean => {
                window_values.iter().sum::<f64>() / window as f64
            }
            RollingStatistic::Median => {
                let mut sorted = window_values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if sorted.len() % 2 == 0 {
                    let mid = sorted.len() / 2;
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                }
            }
            RollingStatistic::Std => {
                let mean = window_values.iter().sum::<f64>() / window as f64;
                let variance = window_values.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / window as f64;
                variance.sqrt()
            }
            RollingStatistic::Min => {
                window_values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
            }
            RollingStatistic::Max => {
                window_values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
            }
            RollingStatistic::Quantile(q) => {
                let mut sorted = window_values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let index = (q * (sorted.len() - 1) as f64).round() as usize;
                sorted[index.min(sorted.len() - 1)]
            }
            RollingStatistic::Skewness => {
                calculate_skewness(window_values)
            }
            RollingStatistic::Kurtosis => {
                calculate_kurtosis(window_values)
            }
        };

        result.push(stat_value);
    }

    let feature_name = match statistic {
        RollingStatistic::Quantile(q) => format!("rolling_{}_{:.0}q", window, q * 100.0),
        _ => format!("rolling_{}_{:?}", window, statistic).to_lowercase(),
    };

    Ok((result, feature_name))
}

/// Calculate skewness
fn calculate_skewness(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 3.0 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return 0.0;
    }

    let skew = values.iter()
        .map(|v| ((v - mean) / std_dev).powi(3))
        .sum::<f64>() / n;

    skew
}

/// Calculate kurtosis
fn calculate_kurtosis(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 4.0 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return 0.0;
    }

    let kurt = values.iter()
        .map(|v| ((v - mean) / std_dev).powi(4))
        .sum::<f64>() / n - 3.0; // Excess kurtosis

    kurt
}

/// Create calendar features
fn create_calendar_features(
    timeseries: &TimeSeries,
    config: &CalendarConfig,
) -> Result<(HashMap<String, Vec<f64>>, Vec<String>)> {
    let mut features = HashMap::new();
    let mut feature_names = Vec::new();
    let timestamps = &timeseries.timestamps;

    for timestamp in timestamps {
        // Day of week (0 = Sunday, 6 = Saturday)
        if config.day_of_week {
            let dow = timestamp.weekday().num_days_from_sunday() as f64;
            features.entry("day_of_week".to_string())
                .or_insert_with(Vec::new)
                .push(dow);

            // Sine/cosine encoding for cyclical nature
            features.entry("day_of_week_sin".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * dow / 7.0).sin());
            features.entry("day_of_week_cos".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * dow / 7.0).cos());
        }

        // Month (1-12)
        if config.month {
            let month = timestamp.month() as f64;
            features.entry("month".to_string())
                .or_insert_with(Vec::new)
                .push(month);

            // Sine/cosine encoding
            features.entry("month_sin".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * (month - 1.0) / 12.0).sin());
            features.entry("month_cos".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * (month - 1.0) / 12.0).cos());
        }

        // Quarter (1-4)
        if config.quarter {
            let quarter = ((timestamp.month() - 1) / 3 + 1) as f64;
            features.entry("quarter".to_string())
                .or_insert_with(Vec::new)
                .push(quarter);
        }

        // Year
        if config.year {
            let year = timestamp.year() as f64;
            features.entry("year".to_string())
                .or_insert_with(Vec::new)
                .push(year);
        }

        // Day of year (1-366)
        if config.day_of_year {
            let doy = timestamp.ordinal() as f64;
            features.entry("day_of_year".to_string())
                .or_insert_with(Vec::new)
                .push(doy);

            // Sine/cosine encoding
            features.entry("day_of_year_sin".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * (doy - 1.0) / 365.0).sin());
            features.entry("day_of_year_cos".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * (doy - 1.0) / 365.0).cos());
        }

        // Week of year (1-53)
        if config.week_of_year {
            let week = timestamp.iso_week().week() as f64;
            features.entry("week_of_year".to_string())
                .or_insert_with(Vec::new)
                .push(week);
        }

        // Hour (0-23) for intraday data
        if config.hour {
            let hour = timestamp.hour() as f64;
            features.entry("hour".to_string())
                .or_insert_with(Vec::new)
                .push(hour);

            // Sine/cosine encoding
            features.entry("hour_sin".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * hour / 24.0).sin());
            features.entry("hour_cos".to_string())
                .or_insert_with(Vec::new)
                .push((2.0 * std::f64::consts::PI * hour / 24.0).cos());
        }

        // Holiday indicators (simplified implementation)
        if config.holidays {
            let is_holiday = is_holiday_simple(timestamp);
            features.entry("is_holiday".to_string())
                .or_insert_with(Vec::new)
                .push(if is_holiday { 1.0 } else { 0.0 });
        }
    }

    feature_names.extend(features.keys().cloned());

    Ok((features, feature_names))
}

/// Simple holiday detection (can be expanded with holiday calendar)
fn is_holiday_simple(timestamp: &DateTime<Utc>) -> bool {
    let month = timestamp.month();
    let day = timestamp.day();

    // Simple holidays (New Year's Day, Christmas)
    (month == 1 && day == 1) || (month == 12 && day == 25)
}

/// Create trend features
fn create_trend_features(
    timeseries: &TimeSeries,
) -> Result<(HashMap<String, Vec<f64>>, Vec<String>)> {
    let mut features = HashMap::new();
    let mut feature_names = Vec::new();
    let values = &timeseries.values;
    let n = values.len();

    // Linear trend (time index)
    let trend: Vec<f64> = (0..n).map(|i| i as f64).collect();
    features.insert("trend".to_string(), trend);
    feature_names.push("trend".to_string());

    // Quadratic trend
    let trend_squared: Vec<f64> = (0..n).map(|i| (i as f64).powi(2)).collect();
    features.insert("trend_squared".to_string(), trend_squared);
    feature_names.push("trend_squared".to_string());

    // First differences
    let mut first_diff = vec![f64::NAN];
    for i in 1..n {
        first_diff.push(values[i] - values[i - 1]);
    }
    features.insert("first_diff".to_string(), first_diff);
    feature_names.push("first_diff".to_string());

    // Second differences
    let mut second_diff = vec![f64::NAN, f64::NAN];
    for i in 2..n {
        second_diff.push(values[i] - 2.0 * values[i - 1] + values[i - 2]);
    }
    features.insert("second_diff".to_string(), second_diff);
    feature_names.push("second_diff".to_string());

    Ok((features, feature_names))
}

/// Create seasonal features
fn create_seasonal_features(
    timeseries: &TimeSeries,
) -> Result<(HashMap<String, Vec<f64>>, Vec<String>)> {
    let mut features = HashMap::new();
    let mut feature_names = Vec::new();
    let n = timeseries.values.len();

    // Detect potential seasonal periods
    let seasonal_periods = detect_seasonal_periods(timeseries)?;

    for period in seasonal_periods {
        // Seasonal indicators
        let seasonal_indicators: Vec<f64> = (0..n)
            .map(|i| (i % period) as f64)
            .collect();

        let feature_name = format!("seasonal_{}", period);
        features.insert(feature_name.clone(), seasonal_indicators);
        feature_names.push(feature_name);

        // Sine/cosine encoding for seasonal patterns
        let seasonal_sin: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin())
            .collect();

        let seasonal_cos: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).cos())
            .collect();

        let sin_name = format!("seasonal_{}_sin", period);
        let cos_name = format!("seasonal_{}_cos", period);

        features.insert(sin_name.clone(), seasonal_sin);
        features.insert(cos_name.clone(), seasonal_cos);
        feature_names.push(sin_name);
        feature_names.push(cos_name);
    }

    Ok((features, feature_names))
}

/// Detect potential seasonal periods
fn detect_seasonal_periods(timeseries: &TimeSeries) -> Result<Vec<usize>> {
    let n = timeseries.values.len();
    let mut periods = Vec::new();

    // Common seasonal periods to test
    let candidate_periods = vec![7, 12, 24, 30, 52, 365];

    for period in candidate_periods {
        if period < n / 2 {
            // Simple test for seasonality using autocorrelation
            let autocorr = calculate_autocorrelation(&timeseries.values, period);
            if autocorr > 0.3 { // Threshold for significant seasonality
                periods.push(period);
            }
        }
    }

    // If no clear seasonality detected, include some common periods
    if periods.is_empty() && n > 24 {
        periods.push(12); // Monthly seasonality
    }

    Ok(periods)
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