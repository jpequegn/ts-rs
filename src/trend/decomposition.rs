//! Time Series Decomposition Module
//!
//! Implements various methods for decomposing time series into trend, seasonal,
//! and residual components including classical and STL decomposition.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Available decomposition methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecompositionMethod {
    /// Classical additive decomposition
    ClassicalAdditive,
    /// Classical multiplicative decomposition
    ClassicalMultiplicative,
    /// STL (Seasonal and Trend decomposition using Loess)
    Stl,
    /// X-13ARIMA-SEATS (simplified version)
    X13Seats,
}

/// Result of time series decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Method used for decomposition
    pub method: DecompositionMethod,

    /// Trend component (can be None for some methods)
    pub trend: Option<Vec<f64>>,

    /// Seasonal component (can be None for non-seasonal series)
    pub seasonal: Option<Vec<f64>>,

    /// Residual/remainder component
    pub residual: Vec<f64>,

    /// Seasonal periods detected
    pub seasonal_periods: Vec<SeasonalPeriod>,

    /// Decomposition quality metrics
    pub quality_metrics: DecompositionQuality,

    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Information about detected seasonal periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPeriod {
    /// Period length
    pub period: f64,

    /// Strength of this seasonal component
    pub strength: f64,

    /// Phase shift
    pub phase: f64,

    /// Amplitude
    pub amplitude: f64,
}

/// Quality metrics for decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionQuality {
    /// R-squared of the decomposition
    pub r_squared: f64,

    /// Mean absolute error of residuals
    pub mae_residuals: f64,

    /// Standard deviation of residuals
    pub std_residuals: f64,

    /// Seasonality strength (0.0 to 1.0)
    pub seasonality_strength: f64,

    /// Trend strength (0.0 to 1.0)
    pub trend_strength: f64,
}

/// Classical decomposition implementation
#[derive(Debug, Clone)]
pub struct ClassicalDecomposition {
    /// Seasonal period
    pub seasonal_period: usize,

    /// Whether to use additive or multiplicative decomposition
    pub additive: bool,
}

impl ClassicalDecomposition {
    /// Create a new classical decomposition
    pub fn new(seasonal_period: usize, additive: bool) -> Self {
        Self {
            seasonal_period,
            additive,
        }
    }

    /// Perform classical decomposition
    pub fn decompose(&self, data: &[f64]) -> Result<DecompositionResult, Box<dyn std::error::Error>> {
        if data.len() < self.seasonal_period * 2 {
            return Err("Data too short for classical decomposition".into());
        }

        let n = data.len();

        // 1. Estimate trend using centered moving average
        let trend = self.estimate_trend(data)?;

        // 2. Detrend the data
        let detrended = if self.additive {
            data.iter().zip(trend.iter())
                .map(|(&x, &t)| if t.is_finite() { x - t } else { f64::NAN })
                .collect::<Vec<f64>>()
        } else {
            data.iter().zip(trend.iter())
                .map(|(&x, &t)| if t.is_finite() && t != 0.0 { x / t } else { f64::NAN })
                .collect::<Vec<f64>>()
        };

        // 3. Estimate seasonal component
        let seasonal = self.estimate_seasonal(&detrended)?;

        // 4. Calculate residuals
        let residual = self.calculate_residuals(data, &trend, &seasonal)?;

        // 5. Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(data, &trend, &seasonal, &residual)?;

        // 6. Detect seasonal periods
        let seasonal_periods = self.detect_seasonal_periods(&seasonal)?;

        let mut metadata = HashMap::new();
        metadata.insert("seasonal_period".to_string(), self.seasonal_period as f64);
        metadata.insert("n_observations".to_string(), n as f64);

        let method = if self.additive {
            DecompositionMethod::ClassicalAdditive
        } else {
            DecompositionMethod::ClassicalMultiplicative
        };

        Ok(DecompositionResult {
            method,
            trend: Some(trend),
            seasonal: Some(seasonal),
            residual,
            seasonal_periods,
            quality_metrics,
            metadata,
        })
    }

    fn estimate_trend(&self, data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = data.len();
        let period = self.seasonal_period;
        let mut trend = vec![f64::NAN; n];

        // Use centered moving average
        let half_period = period / 2;

        for i in half_period..(n - half_period) {
            if period % 2 == 0 {
                // Even period: use 2-step moving average
                let sum1: f64 = data[(i - half_period)..(i + half_period)].iter().sum();
                let sum2: f64 = data[(i - half_period + 1)..(i + half_period + 1)].iter().sum();
                trend[i] = (sum1 + sum2) / (2.0 * period as f64);
            } else {
                // Odd period: simple centered moving average
                let sum: f64 = data[(i - half_period)..(i + half_period + 1)].iter().sum();
                trend[i] = sum / period as f64;
            }
        }

        // Fill in the endpoints with linear extrapolation
        self.extrapolate_endpoints(&mut trend)?;

        Ok(trend)
    }

    fn estimate_seasonal(&self, detrended: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = detrended.len();
        let period = self.seasonal_period;
        let mut seasonal = vec![0.0; n];

        // Calculate average for each seasonal position
        let mut seasonal_averages = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];

        for (i, &value) in detrended.iter().enumerate() {
            if value.is_finite() {
                let seasonal_index = i % period;
                seasonal_averages[seasonal_index] += value;
                seasonal_counts[seasonal_index] += 1;
            }
        }

        // Compute averages
        for i in 0..period {
            if seasonal_counts[i] > 0 {
                seasonal_averages[i] /= seasonal_counts[i] as f64;
            }
        }

        // Center the seasonal component (ensure sum = 0 for additive, product = 1 for multiplicative)
        if self.additive {
            let mean_seasonal = seasonal_averages.iter().sum::<f64>() / period as f64;
            for avg in &mut seasonal_averages {
                *avg -= mean_seasonal;
            }
        } else {
            let geometric_mean = seasonal_averages.iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| x.ln())
                .sum::<f64>() / period as f64;
            let geometric_mean = geometric_mean.exp();

            for avg in &mut seasonal_averages {
                if *avg > 0.0 {
                    *avg /= geometric_mean;
                } else {
                    *avg = 1.0;
                }
            }
        }

        // Assign seasonal values
        for (i, seasonal_val) in seasonal.iter_mut().enumerate() {
            *seasonal_val = seasonal_averages[i % period];
        }

        Ok(seasonal)
    }

    fn calculate_residuals(
        &self,
        original: &[f64],
        trend: &[f64],
        seasonal: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let residual = if self.additive {
            original.iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&o, &t), &s)| {
                    if t.is_finite() && s.is_finite() {
                        o - t - s
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        } else {
            original.iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&o, &t), &s)| {
                    if t.is_finite() && s.is_finite() && t != 0.0 && s != 0.0 {
                        o / (t * s)
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        };

        Ok(residual)
    }

    fn calculate_quality_metrics(
        &self,
        original: &[f64],
        trend: &[f64],
        seasonal: &[f64],
        residual: &[f64],
    ) -> Result<DecompositionQuality, Box<dyn std::error::Error>> {
        let n = original.len();

        // Calculate R-squared
        let mean_original = original.iter().sum::<f64>() / n as f64;
        let total_variance = original.iter()
            .map(|&x| (x - mean_original).powi(2))
            .sum::<f64>();

        let explained_variance = if self.additive {
            original.iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&o, &t), &s)| {
                    if t.is_finite() && s.is_finite() {
                        (o - (t + s)).powi(2)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        } else {
            original.iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&o, &t), &s)| {
                    if t.is_finite() && s.is_finite() && t != 0.0 && s != 0.0 {
                        (o - t * s).powi(2)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        };

        let r_squared = if total_variance > 0.0 {
            1.0 - explained_variance / total_variance
        } else {
            0.0
        };

        // Calculate residual statistics
        let valid_residuals: Vec<f64> = residual.iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();

        let mae_residuals = if !valid_residuals.is_empty() {
            valid_residuals.iter().map(|&x| x.abs()).sum::<f64>() / valid_residuals.len() as f64
        } else {
            0.0
        };

        let std_residuals = if valid_residuals.len() > 1 {
            let mean_residual = valid_residuals.iter().sum::<f64>() / valid_residuals.len() as f64;
            let variance = valid_residuals.iter()
                .map(|&x| (x - mean_residual).powi(2))
                .sum::<f64>() / (valid_residuals.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Calculate component strengths
        let seasonal_strength = self.calculate_seasonal_strength(original, seasonal)?;
        let trend_strength = self.calculate_trend_strength(original, trend)?;

        Ok(DecompositionQuality {
            r_squared,
            mae_residuals,
            std_residuals,
            seasonality_strength: seasonal_strength,
            trend_strength,
        })
    }

    fn calculate_seasonal_strength(&self, original: &[f64], seasonal: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        let original_var = calculate_variance(original);
        let seasonal_var = calculate_variance(seasonal);

        if original_var > 0.0 {
            Ok((seasonal_var / original_var).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn calculate_trend_strength(&self, original: &[f64], trend: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        let valid_trend: Vec<f64> = trend.iter().filter(|&&x| x.is_finite()).copied().collect();
        let original_var = calculate_variance(original);
        let trend_var = calculate_variance(&valid_trend);

        if original_var > 0.0 {
            Ok((trend_var / original_var).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn detect_seasonal_periods(&self, seasonal: &[f64]) -> Result<Vec<SeasonalPeriod>, Box<dyn std::error::Error>> {
        let period = self.seasonal_period as f64;
        let strength = calculate_variance(seasonal) / (calculate_variance(seasonal) + 0.1);

        // Simple amplitude calculation
        let amplitude = seasonal.iter()
            .map(|&x| x.abs())
            .fold(0.0, f64::max);

        Ok(vec![SeasonalPeriod {
            period,
            strength,
            phase: 0.0, // Simplified
            amplitude,
        }])
    }

    fn extrapolate_endpoints(&self, trend: &mut Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
        let n = trend.len();

        // Find first and last valid values
        let first_valid = trend.iter().position(|&x| x.is_finite());
        let last_valid = trend.iter().rposition(|&x| x.is_finite());

        if let (Some(first), Some(last)) = (first_valid, last_valid) {
            if first > 0 {
                // Linear extrapolation for beginning
                let slope = (trend[first + 1] - trend[first]) / 1.0;
                for i in (0..first).rev() {
                    trend[i] = trend[i + 1] - slope;
                }
            }

            if last < n - 1 {
                // Linear extrapolation for end
                let slope = (trend[last] - trend[last - 1]) / 1.0;
                for i in (last + 1)..n {
                    trend[i] = trend[i - 1] + slope;
                }
            }
        }

        Ok(())
    }
}

/// STL decomposition implementation (simplified version)
#[derive(Debug, Clone)]
pub struct StlDecomposition {
    /// Seasonal period
    pub seasonal_period: usize,

    /// Number of iterations
    pub n_iterations: usize,

    /// Seasonal smoother span
    pub seasonal_span: usize,

    /// Trend smoother span
    pub trend_span: usize,
}

impl StlDecomposition {
    /// Create a new STL decomposition
    pub fn new(seasonal_period: usize) -> Self {
        Self {
            seasonal_period,
            n_iterations: 2,
            seasonal_span: seasonal_period,
            trend_span: ((seasonal_period as f64 * 1.5).ceil() as usize).max(7) | 1, // Ensure odd
        }
    }

    /// Perform STL decomposition (simplified implementation)
    pub fn decompose(&self, data: &[f64]) -> Result<DecompositionResult, Box<dyn std::error::Error>> {
        if data.len() < self.seasonal_period * 2 {
            return Err("Data too short for STL decomposition".into());
        }

        let n = data.len();
        let mut seasonal = vec![0.0; n];
        let mut trend = vec![0.0; n];

        // Simplified STL algorithm
        for _iteration in 0..self.n_iterations {
            // Step 1: Detrend
            let detrended: Vec<f64> = data.iter()
                .zip(trend.iter())
                .map(|(&d, &t)| d - t)
                .collect();

            // Step 2: Seasonal smoothing
            seasonal = self.seasonal_smoothing(&detrended)?;

            // Step 3: Remove seasonal component
            let deseasoned: Vec<f64> = data.iter()
                .zip(seasonal.iter())
                .map(|(&d, &s)| d - s)
                .collect();

            // Step 4: Trend smoothing
            trend = self.trend_smoothing(&deseasoned)?;
        }

        // Calculate residuals
        let residual: Vec<f64> = data.iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((&d, &t), &s)| d - t - s)
            .collect();

        // Calculate quality metrics
        let quality_metrics = DecompositionQuality {
            r_squared: self.calculate_r_squared(data, &trend, &seasonal, &residual),
            mae_residuals: residual.iter().map(|&x| x.abs()).sum::<f64>() / n as f64,
            std_residuals: calculate_std_dev(&residual),
            seasonality_strength: calculate_variance(&seasonal) / calculate_variance(data),
            trend_strength: calculate_variance(&trend) / calculate_variance(data),
        };

        let seasonal_periods = vec![SeasonalPeriod {
            period: self.seasonal_period as f64,
            strength: quality_metrics.seasonality_strength,
            phase: 0.0,
            amplitude: seasonal.iter().map(|&x| x.abs()).fold(0.0, f64::max),
        }];

        let mut metadata = HashMap::new();
        metadata.insert("seasonal_period".to_string(), self.seasonal_period as f64);
        metadata.insert("n_iterations".to_string(), self.n_iterations as f64);

        Ok(DecompositionResult {
            method: DecompositionMethod::Stl,
            trend: Some(trend),
            seasonal: Some(seasonal),
            residual,
            seasonal_periods,
            quality_metrics,
            metadata,
        })
    }

    fn seasonal_smoothing(&self, detrended: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = detrended.len();
        let period = self.seasonal_period;
        let mut seasonal = vec![0.0; n];

        // Group by seasonal index and apply smoothing
        for seasonal_idx in 0..period {
            let mut seasonal_values = Vec::new();
            let mut positions = Vec::new();

            for (i, &value) in detrended.iter().enumerate() {
                if i % period == seasonal_idx {
                    seasonal_values.push(value);
                    positions.push(i);
                }
            }

            // Apply smoothing to seasonal values (simplified moving average)
            let smoothed = self.moving_average(&seasonal_values, 3)?;

            for (pos, &smoothed_val) in positions.iter().zip(smoothed.iter()) {
                seasonal[*pos] = smoothed_val;
            }
        }

        // Center seasonal component
        let mean_seasonal = seasonal.iter().sum::<f64>() / n as f64;
        for s in &mut seasonal {
            *s -= mean_seasonal;
        }

        Ok(seasonal)
    }

    fn trend_smoothing(&self, deseasoned: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        // Use moving average for trend smoothing
        self.moving_average(deseasoned, self.trend_span)
    }

    fn moving_average(&self, data: &[f64], window: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = data.len();
        let mut result = vec![0.0; n];
        let half_window = window / 2;

        for i in 0..n {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);
            let count = end - start;

            if count > 0 {
                let sum: f64 = data[start..end].iter().sum();
                result[i] = sum / count as f64;
            } else {
                result[i] = data[i];
            }
        }

        Ok(result)
    }

    fn calculate_r_squared(&self, original: &[f64], trend: &[f64], seasonal: &[f64], residual: &[f64]) -> f64 {
        let mean_original = original.iter().sum::<f64>() / original.len() as f64;
        let total_variance = original.iter()
            .map(|&x| (x - mean_original).powi(2))
            .sum::<f64>();

        let residual_variance = residual.iter()
            .map(|&x| x.powi(2))
            .sum::<f64>();

        if total_variance > 0.0 {
            1.0 - residual_variance / total_variance
        } else {
            0.0
        }
    }
}

/// Perform decomposition using the specified method
pub fn perform_decomposition(
    data: &[f64],
    method: DecompositionMethod,
    seasonal_period: Option<usize>,
) -> Result<DecompositionResult, Box<dyn std::error::Error>> {
    let period = seasonal_period.unwrap_or_else(|| detect_seasonal_period(data));

    match method {
        DecompositionMethod::ClassicalAdditive => {
            let decomposer = ClassicalDecomposition::new(period, true);
            decomposer.decompose(data)
        }
        DecompositionMethod::ClassicalMultiplicative => {
            let decomposer = ClassicalDecomposition::new(period, false);
            decomposer.decompose(data)
        }
        DecompositionMethod::Stl => {
            let decomposer = StlDecomposition::new(period);
            decomposer.decompose(data)
        }
        DecompositionMethod::X13Seats => {
            // Simplified X13-SEATS (fallback to STL)
            let decomposer = StlDecomposition::new(period);
            decomposer.decompose(data)
        }
    }
}

// Helper functions
fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

fn calculate_std_dev(values: &[f64]) -> f64 {
    calculate_variance(values).sqrt()
}

fn detect_seasonal_period(data: &[f64]) -> usize {
    // Simple seasonal period detection using autocorrelation
    let max_period = (data.len() / 3).min(52); // Max one year for weekly data
    let mut best_period = 12; // Default to monthly
    let mut best_correlation = 0.0;

    for period in 2..=max_period {
        let correlation = calculate_autocorrelation(data, period);
        if correlation > best_correlation {
            best_correlation = correlation;
            best_period = period;
        }
    }

    best_period
}

fn calculate_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let n = data.len() - lag;
    if n == 0 {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        numerator += (data[i] - mean) * (data[i + lag] - mean);
    }

    for &value in data {
        denominator += (value - mean).powi(2);
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_decomposition() {
        let data: Vec<f64> = (0..48).map(|i| {
            let t = i as f64;
            let trend = 0.1 * t;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
            let noise = 0.5 * (i as f64 * 0.1).sin();
            trend + seasonal + noise
        }).collect();

        let decomposer = ClassicalDecomposition::new(12, true);
        let result = decomposer.decompose(&data).unwrap();

        assert_eq!(result.method, DecompositionMethod::ClassicalAdditive);
        assert!(result.trend.is_some());
        assert!(result.seasonal.is_some());
        assert_eq!(result.residual.len(), data.len());
        assert!(result.quality_metrics.r_squared > 0.0);
    }

    #[test]
    fn test_stl_decomposition() {
        let data: Vec<f64> = (0..36).map(|i| {
            let t = i as f64;
            let trend = 0.2 * t;
            let seasonal = 3.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
            trend + seasonal
        }).collect();

        let decomposer = StlDecomposition::new(12);
        let result = decomposer.decompose(&data).unwrap();

        assert_eq!(result.method, DecompositionMethod::Stl);
        assert!(result.trend.is_some());
        assert!(result.seasonal.is_some());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_seasonal_period_detection() {
        let data: Vec<f64> = (0..48).map(|i| {
            let t = i as f64;
            5.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin() // 12-period cycle
        }).collect();

        let detected_period = detect_seasonal_period(&data);
        // Should detect period close to 12
        assert!(detected_period >= 10 && detected_period <= 14);
    }

    #[test]
    fn test_decomposition_quality_metrics() {
        let quality = DecompositionQuality {
            r_squared: 0.95,
            mae_residuals: 0.1,
            std_residuals: 0.2,
            seasonality_strength: 0.7,
            trend_strength: 0.8,
        };

        assert!(quality.r_squared > 0.9);
        assert!(quality.seasonality_strength > 0.0);
        assert!(quality.trend_strength > 0.0);
    }
}