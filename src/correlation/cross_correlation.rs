//! Enhanced cross-correlation analysis module
//!
//! Implements comprehensive cross-correlation analysis including lagged correlations,
//! lead-lag relationship detection, and maximum correlation lag identification.

use serde::{Serialize, Deserialize};

/// Cross-correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCorrelationAnalysis {
    /// Cross-correlation function values
    pub ccf_values: Vec<f64>,

    /// Lag values (negative, zero, positive)
    pub lags: Vec<i32>,

    /// Confidence intervals for each lag
    pub confidence_intervals: Vec<(f64, f64)>,

    /// Lead-lag analysis result
    pub lead_lag: LeadLagResult,

    /// Maximum correlation information
    pub max_correlation: MaxCorrelationInfo,

    /// Variable names
    pub variable1: String,
    pub variable2: String,

    /// Number of observations used
    pub n_observations: usize,

    /// Statistical significance threshold
    pub significance_threshold: f64,
}

/// Lead-lag relationship analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadLagResult {
    /// Optimal lag (negative means var1 leads var2, positive means var2 leads var1)
    pub optimal_lag: i32,

    /// Correlation at optimal lag
    pub optimal_correlation: f64,

    /// Confidence interval for optimal correlation
    pub confidence_interval: (f64, f64),

    /// Lead-lag relationship type
    pub relationship_type: LeadLagType,

    /// Statistical significance
    pub is_significant: bool,

    /// P-value for the optimal correlation
    pub p_value: f64,

    /// All significant lags with their correlations
    pub significant_lags: Vec<(i32, f64)>,
}

/// Maximum correlation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCorrelationInfo {
    /// Lag at which maximum absolute correlation occurs
    pub lag: i32,

    /// Maximum correlation value
    pub correlation: f64,

    /// Type of maximum correlation (positive or negative)
    pub correlation_type: MaxCorrelationType,

    /// Confidence interval for maximum correlation
    pub confidence_interval: (f64, f64),

    /// Whether the maximum correlation is statistically significant
    pub is_significant: bool,
}

/// Type of lead-lag relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeadLagType {
    /// First variable leads second variable
    FirstLeadsSecond,
    /// Second variable leads first variable
    SecondLeadsFirst,
    /// Variables are synchronous (lag = 0)
    Synchronous,
    /// No significant relationship
    NoRelationship,
}

/// Type of maximum correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaxCorrelationType {
    Positive,
    Negative,
}

impl CrossCorrelationAnalysis {
    /// Get correlation at specific lag
    pub fn get_correlation_at_lag(&self, lag: i32) -> Option<f64> {
        self.lags.iter()
            .position(|&l| l == lag)
            .and_then(|pos| self.ccf_values.get(pos))
            .copied()
    }

    /// Get all significant correlations
    pub fn get_significant_correlations(&self) -> Vec<(i32, f64)> {
        self.lags.iter()
            .zip(self.ccf_values.iter())
            .zip(self.confidence_intervals.iter())
            .filter_map(|((&lag, &corr), &(lower, upper))| {
                if corr.abs() > self.significance_threshold || (lower > 0.0 || upper < 0.0) {
                    Some((lag, corr))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get correlations within a lag range
    pub fn get_correlations_in_lag_range(&self, min_lag: i32, max_lag: i32) -> Vec<(i32, f64)> {
        self.lags.iter()
            .zip(self.ccf_values.iter())
            .filter(|(&lag, _)| lag >= min_lag && lag <= max_lag)
            .map(|(&lag, &corr)| (lag, corr))
            .collect()
    }
}

/// Compute comprehensive cross-correlation analysis
pub fn compute_cross_correlation_analysis(
    series1: &[f64],
    series2: &[f64],
    max_lag: usize,
    var1_name: &str,
    var2_name: &str,
) -> Result<CrossCorrelationAnalysis, Box<dyn std::error::Error>> {
    if series1.len() != series2.len() {
        return Err("Series must have the same length".into());
    }

    if series1.len() < max_lag + 3 {  // Need at least max_lag + 3 for meaningful analysis
        return Err("Series too short for requested maximum lag".into());
    }

    // Filter out missing values
    let valid_pairs: Vec<(f64, f64)> = series1.iter()
        .zip(series2.iter())
        .filter(|(&x, &y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();

    if valid_pairs.len() < max_lag * 2 + 10 {
        return Err("Too many missing values for cross-correlation analysis".into());
    }

    let series1_clean: Vec<f64> = valid_pairs.iter().map(|(x, _)| *x).collect();
    let series2_clean: Vec<f64> = valid_pairs.iter().map(|(_, y)| *y).collect();

    // Compute cross-correlation function
    let (lags, ccf_values) = compute_ccf(&series1_clean, &series2_clean, max_lag)?;
    let confidence_intervals = compute_ccf_confidence_intervals(&ccf_values, valid_pairs.len());

    // Find maximum correlation
    let max_correlation = find_max_correlation(&lags, &ccf_values, &confidence_intervals);

    // Analyze lead-lag relationships
    let lead_lag = analyze_lead_lag_relationships(&lags, &ccf_values, &confidence_intervals, 0.05);

    // Set significance threshold (95% confidence level)
    let significance_threshold = 1.96 / (valid_pairs.len() as f64).sqrt();

    Ok(CrossCorrelationAnalysis {
        ccf_values,
        lags,
        confidence_intervals,
        lead_lag,
        max_correlation,
        variable1: var1_name.to_string(),
        variable2: var2_name.to_string(),
        n_observations: valid_pairs.len(),
        significance_threshold,
    })
}

/// Compute cross-correlation function
fn compute_ccf(
    series1: &[f64],
    series2: &[f64],
    max_lag: usize,
) -> Result<(Vec<i32>, Vec<f64>), Box<dyn std::error::Error>> {
    let n = series1.len();
    let mut lags = Vec::new();
    let mut ccf_values = Vec::new();

    // Compute means
    let mean1 = series1.iter().sum::<f64>() / n as f64;
    let mean2 = series2.iter().sum::<f64>() / n as f64;

    // Compute standard deviations
    let var1 = series1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / n as f64;
    let var2 = series2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / n as f64;
    let std1 = var1.sqrt();
    let std2 = var2.sqrt();

    if std1 == 0.0 || std2 == 0.0 {
        return Err("One or both series have zero variance".into());
    }

    // Compute cross-correlation for negative lags (series1 lags series2)
    for lag in (1..=max_lag).rev() {
        let lag_i32 = -(lag as i32);
        let corr = compute_ccf_at_lag(series1, series2, lag_i32, mean1, mean2, std1, std2);
        lags.push(lag_i32);
        ccf_values.push(corr);
    }

    // Compute cross-correlation at lag 0
    let corr_0 = compute_ccf_at_lag(series1, series2, 0, mean1, mean2, std1, std2);
    lags.push(0);
    ccf_values.push(corr_0);

    // Compute cross-correlation for positive lags (series2 lags series1)
    for lag in 1..=max_lag {
        let lag_i32 = lag as i32;
        let corr = compute_ccf_at_lag(series1, series2, lag_i32, mean1, mean2, std1, std2);
        lags.push(lag_i32);
        ccf_values.push(corr);
    }

    Ok((lags, ccf_values))
}

/// Compute cross-correlation at a specific lag
fn compute_ccf_at_lag(
    series1: &[f64],
    series2: &[f64],
    lag: i32,
    mean1: f64,
    mean2: f64,
    std1: f64,
    std2: f64,
) -> f64 {
    let n = series1.len();
    let mut sum = 0.0;
    let mut count = 0;

    if lag >= 0 {
        // Positive lag: series2 lags series1
        let lag_u = lag as usize;
        for i in lag_u..n {
            sum += (series1[i - lag_u] - mean1) * (series2[i] - mean2);
            count += 1;
        }
    } else {
        // Negative lag: series1 lags series2
        let lag_u = (-lag) as usize;
        for i in lag_u..n {
            sum += (series1[i] - mean1) * (series2[i - lag_u] - mean2);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    sum / (count as f64 * std1 * std2)
}

/// Compute confidence intervals for cross-correlation function
fn compute_ccf_confidence_intervals(ccf_values: &[f64], n: usize) -> Vec<(f64, f64)> {
    let std_error = 1.0 / (n as f64).sqrt();
    let z_critical = 1.96; // 95% confidence level

    ccf_values.iter()
        .map(|&ccf| {
            let margin = z_critical * std_error;
            (ccf - margin, ccf + margin)
        })
        .collect()
}

/// Find maximum correlation and its properties
fn find_max_correlation(
    lags: &[i32],
    ccf_values: &[f64],
    confidence_intervals: &[(f64, f64)],
) -> MaxCorrelationInfo {
    let mut max_abs_corr = 0.0;
    let mut max_lag = 0;
    let mut max_corr = 0.0;
    let mut max_confidence_interval = (0.0, 0.0);

    for (i, (&lag, &corr)) in lags.iter().zip(ccf_values.iter()).enumerate() {
        if corr.abs() > max_abs_corr {
            max_abs_corr = corr.abs();
            max_lag = lag;
            max_corr = corr;
            if let Some(&ci) = confidence_intervals.get(i) {
                max_confidence_interval = ci;
            }
        }
    }

    let correlation_type = if max_corr >= 0.0 {
        MaxCorrelationType::Positive
    } else {
        MaxCorrelationType::Negative
    };

    let is_significant = max_confidence_interval.0 > 0.0 || max_confidence_interval.1 < 0.0;

    MaxCorrelationInfo {
        lag: max_lag,
        correlation: max_corr,
        correlation_type,
        confidence_interval: max_confidence_interval,
        is_significant,
    }
}

/// Analyze lead-lag relationships
fn analyze_lead_lag_relationships(
    lags: &[i32],
    ccf_values: &[f64],
    confidence_intervals: &[(f64, f64)],
    alpha: f64,
) -> LeadLagResult {
    let significance_threshold = 1.96 / (ccf_values.len() as f64).sqrt();

    // Find all significant correlations
    let mut significant_lags = Vec::new();
    for (i, (&lag, &corr)) in lags.iter().zip(ccf_values.iter()).enumerate() {
        if let Some(&(lower, upper)) = confidence_intervals.get(i) {
            if corr.abs() > significance_threshold || (lower > 0.0 || upper < 0.0) {
                significant_lags.push((lag, corr));
            }
        }
    }

    // Find optimal lag (highest absolute correlation among significant lags)
    let (optimal_lag, optimal_correlation, confidence_interval) = if let Some(&(lag, corr)) = significant_lags.iter()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()) {
        let ci = confidence_intervals.iter()
            .zip(lags.iter())
            .find(|(_, &l)| l == lag)
            .map(|(ci, _)| *ci)
            .unwrap_or((corr, corr));
        (lag, corr, ci)
    } else {
        (0, 0.0, (0.0, 0.0))
    };

    // Determine relationship type
    let relationship_type = if optimal_correlation.abs() < significance_threshold {
        LeadLagType::NoRelationship
    } else if optimal_lag < 0 {
        LeadLagType::FirstLeadsSecond
    } else if optimal_lag > 0 {
        LeadLagType::SecondLeadsFirst
    } else {
        LeadLagType::Synchronous
    };

    let is_significant = optimal_correlation.abs() > significance_threshold;
    let p_value = compute_correlation_p_value(optimal_correlation, ccf_values.len());

    LeadLagResult {
        optimal_lag,
        optimal_correlation,
        confidence_interval,
        relationship_type,
        is_significant,
        p_value,
        significant_lags,
    }
}

/// Compute p-value for correlation (simplified)
fn compute_correlation_p_value(correlation: f64, n: usize) -> f64 {
    if n <= 2 || correlation.abs() >= 1.0 {
        return if correlation.abs() >= 1.0 { 0.0 } else { 1.0 };
    }

    // Simplified p-value calculation using normal approximation
    let z_score = correlation * (n as f64 - 3.0).sqrt() / (1.0 - correlation * correlation).sqrt();
    2.0 * (1.0 - approximate_normal_cdf(z_score.abs()))
}

/// Approximate normal CDF
fn approximate_normal_cdf(z: f64) -> f64 {
    if z < -5.0 {
        0.0
    } else if z > 5.0 {
        1.0
    } else {
        0.5 * (1.0 + (z / (2.0_f64).sqrt()).tanh())
    }
}

/// Wrapper function for lead-lag analysis
pub fn compute_lead_lag_analysis(
    series1: &[f64],
    series2: &[f64],
    max_lag: usize,
    var1_name: &str,
    var2_name: &str,
) -> Result<LeadLagResult, Box<dyn std::error::Error>> {
    let analysis = compute_cross_correlation_analysis(series1, series2, max_lag, var1_name, var2_name)?;
    Ok(analysis.lead_lag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_correlation_perfect_positive() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let result = compute_cross_correlation_analysis(&series1, &series2, 3, "x", "y").unwrap();

        // Should have perfect correlation at lag 0
        let corr_at_0 = result.get_correlation_at_lag(0).unwrap();
        assert!((corr_at_0 - 1.0).abs() < 1e-10);

        assert_eq!(result.lead_lag.optimal_lag, 0);
        assert!(matches!(result.lead_lag.relationship_type, LeadLagType::Synchronous));
    }

    #[test]
    fn test_cross_correlation_with_lag() {
        // Create series where series2 is series1 shifted by 1 period
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let series2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let result = compute_cross_correlation_analysis(&series1, &series2, 3, "x", "y").unwrap();

        // Maximum correlation should be at lag -1 (series1 leads series2)
        assert!(result.max_correlation.lag < 0);
        assert!(matches!(result.lead_lag.relationship_type, LeadLagType::FirstLeadsSecond));
    }

    #[test]
    fn test_ccf_computation() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let (lags, ccf_values) = compute_ccf(&series1, &series2, 2).unwrap();

        assert_eq!(lags.len(), 5); // -2, -1, 0, 1, 2
        assert_eq!(ccf_values.len(), 5);

        // Should include negative and positive lags plus zero
        assert!(lags.contains(&-2));
        assert!(lags.contains(&-1));
        assert!(lags.contains(&0));
        assert!(lags.contains(&1));
        assert!(lags.contains(&2));
    }

    #[test]
    fn test_significant_correlations() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let series2 = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = compute_cross_correlation_analysis(&series1, &series2, 2, "x", "y").unwrap();

        let significant_corrs = result.get_significant_correlations();
        assert!(significant_corrs.len() > 0);

        // Should detect strong negative correlation
        let corr_at_0 = result.get_correlation_at_lag(0).unwrap();
        assert!(corr_at_0 < -0.9);
    }
}