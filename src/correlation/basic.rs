//! Basic correlation analysis module
//!
//! Implements Pearson, Spearman, and Kendall's tau correlation coefficients
//! for pairwise and matrix-based correlation analysis.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Types of correlation coefficients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrelationType {
    /// Pearson product-moment correlation (linear relationships)
    Pearson,
    /// Spearman rank correlation (monotonic relationships)
    Spearman,
    /// Kendall's tau correlation (rank-based, robust to outliers)
    Kendall,
}

/// Correlation matrix containing multiple correlation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    /// Variable names
    pub variables: Vec<String>,

    /// Correlation matrices by type
    pub correlations: HashMap<CorrelationType, Vec<Vec<f64>>>,

    /// P-values for significance testing
    pub p_values: HashMap<CorrelationType, Vec<Vec<f64>>>,

    /// Correlation types computed
    pub correlation_types: Vec<CorrelationType>,

    /// Number of observations used
    pub n_observations: usize,
}

/// Pairwise correlation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseCorrelation {
    /// Variable names
    pub variable1: String,
    pub variable2: String,

    /// Correlation coefficient
    pub correlation: f64,

    /// P-value for significance test
    pub p_value: f64,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Number of observations
    pub n_observations: usize,

    /// Confidence interval (95% by default)
    pub confidence_interval: (f64, f64),
}

impl CorrelationMatrix {
    /// Get correlation between two variables
    pub fn get_correlation(&self, var1: &str, var2: &str, corr_type: CorrelationType) -> Option<f64> {
        let idx1 = self.variables.iter().position(|v| v == var1)?;
        let idx2 = self.variables.iter().position(|v| v == var2)?;

        self.correlations.get(&corr_type)?.get(idx1)?.get(idx2).copied()
    }

    /// Get p-value for correlation between two variables
    pub fn get_p_value(&self, var1: &str, var2: &str, corr_type: CorrelationType) -> Option<f64> {
        let idx1 = self.variables.iter().position(|v| v == var1)?;
        let idx2 = self.variables.iter().position(|v| v == var2)?;

        self.p_values.get(&corr_type)?.get(idx1)?.get(idx2).copied()
    }

    /// Get strongest correlations above a threshold
    pub fn get_strong_correlations(&self, threshold: f64, corr_type: CorrelationType) -> Vec<(String, String, f64)> {
        let mut strong_correlations = Vec::new();

        if let Some(matrix) = self.correlations.get(&corr_type) {
            for (i, var1) in self.variables.iter().enumerate() {
                for (j, var2) in self.variables.iter().enumerate() {
                    if i < j {  // Avoid duplicates and self-correlations
                        if let Some(&corr) = matrix.get(i).and_then(|row| row.get(j)) {
                            if corr.abs() >= threshold {
                                strong_correlations.push((var1.clone(), var2.clone(), corr));
                            }
                        }
                    }
                }
            }
        }

        // Sort by correlation strength
        strong_correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        strong_correlations
    }
}

/// Compute correlation matrix for multiple time series
pub fn compute_correlation_matrix(
    data: &HashMap<String, Vec<f64>>,
    correlation_types: &[CorrelationType],
) -> Result<CorrelationMatrix, Box<dyn std::error::Error>> {
    let variables: Vec<String> = data.keys().cloned().collect();
    let n_vars = variables.len();
    let n_obs = data.values().next().map(|v| v.len()).unwrap_or(0);

    if n_vars < 2 {
        return Err("At least 2 variables are required for correlation analysis".into());
    }

    let mut correlations = HashMap::new();
    let mut p_values = HashMap::new();

    for &corr_type in correlation_types {
        let mut corr_matrix = vec![vec![0.0; n_vars]; n_vars];
        let mut p_value_matrix = vec![vec![0.0; n_vars]; n_vars];

        for (i, var1) in variables.iter().enumerate() {
            for (j, var2) in variables.iter().enumerate() {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                    p_value_matrix[i][j] = 0.0;
                } else {
                    let series1 = &data[var1];
                    let series2 = &data[var2];

                    let result = compute_pairwise_correlation(
                        series1, series2, corr_type, var1, var2
                    )?;

                    corr_matrix[i][j] = result.correlation;
                    p_value_matrix[i][j] = result.p_value;
                }
            }
        }

        correlations.insert(corr_type, corr_matrix);
        p_values.insert(corr_type, p_value_matrix);
    }

    Ok(CorrelationMatrix {
        variables,
        correlations,
        p_values,
        correlation_types: correlation_types.to_vec(),
        n_observations: n_obs,
    })
}

/// Compute pairwise correlation between two time series
pub fn compute_pairwise_correlation(
    series1: &[f64],
    series2: &[f64],
    correlation_type: CorrelationType,
    var1_name: &str,
    var2_name: &str,
) -> Result<PairwiseCorrelation, Box<dyn std::error::Error>> {
    if series1.len() != series2.len() {
        return Err("Series must have the same length".into());
    }

    // Filter out missing values
    let pairs: Vec<(f64, f64)> = series1.iter()
        .zip(series2.iter())
        .filter(|(&x, &y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();

    if pairs.len() < 3 {
        return Err("Need at least 3 valid observations for correlation".into());
    }

    let n = pairs.len();
    let correlation = match correlation_type {
        CorrelationType::Pearson => compute_pearson_correlation(&pairs)?,
        CorrelationType::Spearman => compute_spearman_correlation(&pairs)?,
        CorrelationType::Kendall => compute_kendall_correlation(&pairs)?,
    };

    // Compute p-value and confidence interval
    let p_value = compute_correlation_p_value(correlation, n, correlation_type);
    let confidence_interval = compute_correlation_confidence_interval(correlation, n);

    Ok(PairwiseCorrelation {
        variable1: var1_name.to_string(),
        variable2: var2_name.to_string(),
        correlation,
        p_value,
        correlation_type,
        n_observations: n,
        confidence_interval,
    })
}

/// Compute Pearson product-moment correlation
pub(crate) fn compute_pearson_correlation(pairs: &[(f64, f64)]) -> Result<f64, Box<dyn std::error::Error>> {
    let n = pairs.len() as f64;

    // Calculate means
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

    // Calculate correlation coefficient
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for &(x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator == 0.0 {
        return Ok(f64::NAN);  // No variance in one or both variables - correlation is undefined
    }

    Ok(numerator / denominator)
}

/// Compute Spearman rank correlation
fn compute_spearman_correlation(pairs: &[(f64, f64)]) -> Result<f64, Box<dyn std::error::Error>> {
    let n = pairs.len();

    // Create ranks for both variables
    let x_values: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let y_values: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();

    let x_ranks = compute_ranks(&x_values);
    let y_ranks = compute_ranks(&y_values);

    // Create rank pairs
    let rank_pairs: Vec<(f64, f64)> = x_ranks.into_iter().zip(y_ranks).collect();

    // Compute Pearson correlation on ranks
    compute_pearson_correlation(&rank_pairs)
}

/// Compute Kendall's tau correlation
fn compute_kendall_correlation(pairs: &[(f64, f64)]) -> Result<f64, Box<dyn std::error::Error>> {
    let n = pairs.len();
    let mut concordant = 0;
    let mut discordant = 0;

    // Count concordant and discordant pairs
    for i in 0..n {
        for j in (i + 1)..n {
            let (x1, y1) = pairs[i];
            let (x2, y2) = pairs[j];

            let x_diff = x2 - x1;
            let y_diff = y2 - y1;

            if (x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0) {
                concordant += 1;
            } else if (x_diff > 0.0 && y_diff < 0.0) || (x_diff < 0.0 && y_diff > 0.0) {
                discordant += 1;
            }
            // Ties are ignored in this simple implementation
        }
    }

    let total_pairs = n * (n - 1) / 2;
    if total_pairs == 0 {
        return Ok(0.0);
    }

    Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
}

/// Compute ranks for a series of values
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed_values: Vec<(usize, f64)> = values.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    // Sort by value
    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];

    // Assign ranks (handling ties with average ranks)
    let mut i = 0;
    while i < n {
        let current_value = indexed_values[i].1;
        let mut j = i;

        // Find all values equal to current_value
        while j < n && indexed_values[j].1 == current_value {
            j += 1;
        }

        // Assign average rank to all tied values
        let average_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            let original_index = indexed_values[k].0;
            ranks[original_index] = average_rank;
        }

        i = j;
    }

    ranks
}

/// Compute p-value for correlation coefficient
fn compute_correlation_p_value(correlation: f64, n: usize, corr_type: CorrelationType) -> f64 {
    match corr_type {
        CorrelationType::Pearson => {
            if n <= 2 || correlation.abs() >= 1.0 {
                return if correlation.abs() >= 1.0 { 0.0 } else { 1.0 };
            }

            // t-test for Pearson correlation
            let t_stat = correlation * ((n - 2) as f64).sqrt() / (1.0 - correlation * correlation).sqrt();

            // Approximate p-value using t-distribution
            // This is a simplified implementation
            let df = n - 2;
            approximate_t_test_p_value(t_stat.abs(), df)
        },
        CorrelationType::Spearman | CorrelationType::Kendall => {
            // Simplified p-value computation for rank correlations
            // In practice, these would use more sophisticated methods
            if n <= 3 {
                return 1.0;
            }

            let z_score = correlation * (n as f64 - 1.0).sqrt();
            2.0 * (1.0 - approximate_normal_cdf(z_score.abs()))
        }
    }
}

/// Compute confidence interval for correlation coefficient
fn compute_correlation_confidence_interval(correlation: f64, n: usize) -> (f64, f64) {
    if n <= 3 || correlation.abs() >= 1.0 {
        return (correlation, correlation);
    }

    // Fisher's z-transformation for Pearson correlation
    let z = 0.5 * ((1.0 + correlation) / (1.0 - correlation)).ln();
    let se_z = 1.0 / (n as f64 - 3.0).sqrt();

    // 95% confidence interval
    let z_critical = 1.96;
    let z_lower = z - z_critical * se_z;
    let z_upper = z + z_critical * se_z;

    // Transform back to correlation scale
    let r_lower = (z_lower.exp() * 2.0 - 1.0) / (z_lower.exp() * 2.0 + 1.0);
    let r_upper = (z_upper.exp() * 2.0 - 1.0) / (z_upper.exp() * 2.0 + 1.0);

    (r_lower, r_upper)
}

/// Approximate p-value for t-test (simplified implementation)
fn approximate_t_test_p_value(t_stat: f64, df: usize) -> f64 {
    // Simplified approximation - in practice would use proper t-distribution
    if df == 0 {
        return 1.0;
    }

    let p: f64 = if t_stat > 6.0 {
        0.0001
    } else if t_stat > 4.0 {
        0.001
    } else if t_stat > 3.0 {
        0.01
    } else if t_stat > 2.0 {
        0.05
    } else if t_stat > 1.0 {
        0.2
    } else {
        0.5
    };

    p.min(1.0)
}

/// Approximate normal CDF (simplified implementation)
fn approximate_normal_cdf(z: f64) -> f64 {
    // Simplified normal CDF approximation
    if z < -5.0 {
        0.0
    } else if z > 5.0 {
        1.0
    } else {
        0.5 * (1.0 + (z / (2.0_f64).sqrt()).tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        ).unwrap();

        assert!((result.correlation - 1.0).abs() < 1e-10);
        assert_eq!(result.n_observations, 5);
    }

    #[test]
    fn test_correlation_matrix() {
        let mut data = HashMap::new();
        data.insert("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("B".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        data.insert("C".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        let types = vec![CorrelationType::Pearson, CorrelationType::Spearman];
        let matrix = compute_correlation_matrix(&data, &types).unwrap();

        assert_eq!(matrix.variables.len(), 3);
        assert_eq!(matrix.correlation_types.len(), 2);

        // Check that diagonal is 1.0
        for i in 0..3 {
            assert!((matrix.correlations[&CorrelationType::Pearson][i][i] - 1.0).abs() < 1e-10);
        }

        // Check strong positive correlation between A and B
        let corr_ab = matrix.get_correlation("A", "B", CorrelationType::Pearson).unwrap();
        assert!(corr_ab > 0.99);

        // Check strong negative correlation between A and C
        let corr_ac = matrix.get_correlation("A", "C", CorrelationType::Pearson).unwrap();
        assert!(corr_ac < -0.99);
    }

    #[test]
    fn test_ranks() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = compute_ranks(&values);

        // Expected ranks: 3.0 -> 3, 1.0 -> 1.5, 4.0 -> 4, 1.0 -> 1.5, 5.0 -> 5
        assert_eq!(ranks, vec![3.0, 1.5, 4.0, 1.5, 5.0]);
    }

    #[test]
    fn test_kendall_correlation() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Kendall, "x", "y"
        ).unwrap();

        // Should be perfect positive correlation
        assert!((result.correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_correlation() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Spearman, "x", "y"
        ).unwrap();

        // Should be perfect positive correlation
        assert!((result.correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_correlation() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        ).unwrap();

        // Should be perfect negative correlation
        assert!((result.correlation + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_correlation() {
        let series1 = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let series2 = vec![1.0, 1.0, 2.0, 2.0, 3.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        ).unwrap();

        // Should be approximately zero correlation
        assert!(result.correlation.abs() < 0.5);
    }

    #[test]
    fn test_with_missing_values() {
        let series1 = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        ).unwrap();

        // Should handle NaN by excluding those pairs
        assert_eq!(result.n_observations, 4);
        assert!((result.correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_data() {
        let series1 = vec![1.0];
        let series2 = vec![2.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_lengths() {
        let series1 = vec![1.0, 2.0, 3.0];
        let series2 = vec![1.0, 2.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_constant_series() {
        let series1 = vec![5.0, 5.0, 5.0, 5.0];
        let series2 = vec![1.0, 2.0, 3.0, 4.0];

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        ).unwrap();

        // Correlation with constant should be NaN
        assert!(result.correlation.is_nan());
    }

    #[test]
    fn test_significance_values() {
        let series1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let series2: Vec<f64> = series1.iter().map(|&x| x * 2.0 + 1.0).collect();

        let result = compute_pairwise_correlation(
            &series1, &series2, CorrelationType::Pearson, "x", "y"
        ).unwrap();

        // Perfect correlation should have very small p-value
        assert!(result.p_value < 0.001);
        assert!(result.p_value < 0.05); // Test significance at 5% level
    }

    #[test]
    fn test_matrix_with_single_variable() {
        let mut data = HashMap::new();
        data.insert("A".to_string(), vec![1.0, 2.0, 3.0]);

        let result = compute_correlation_matrix(&data, &[CorrelationType::Pearson]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_with_empty_data() {
        let data = HashMap::new();
        let result = compute_correlation_matrix(&data, &[CorrelationType::Pearson]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_with_unequal_lengths() {
        let mut data = HashMap::new();
        data.insert("A".to_string(), vec![1.0, 2.0, 3.0]);
        data.insert("B".to_string(), vec![1.0, 2.0]); // Different length

        let result = compute_correlation_matrix(&data, &[CorrelationType::Pearson]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_correlation_types() {
        let mut data = HashMap::new();
        data.insert("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("B".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let types = vec![CorrelationType::Pearson, CorrelationType::Spearman, CorrelationType::Kendall];
        let matrix = compute_correlation_matrix(&data, &types).unwrap();

        assert_eq!(matrix.correlation_types.len(), 3);
        assert!(matrix.correlation_types.contains(&CorrelationType::Pearson));
        assert!(matrix.correlation_types.contains(&CorrelationType::Spearman));
        assert!(matrix.correlation_types.contains(&CorrelationType::Kendall));

        // All should show perfect positive correlation
        for corr_type in &types {
            let corr = matrix.get_correlation("A", "B", *corr_type).unwrap();
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_symmetry() {
        let mut data = HashMap::new();
        data.insert("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("B".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        data.insert("C".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        let matrix = compute_correlation_matrix(&data, &[CorrelationType::Pearson]).unwrap();

        // Test symmetry: correlation(A,B) = correlation(B,A)
        let corr_ab = matrix.get_correlation("A", "B", CorrelationType::Pearson).unwrap();
        let corr_ba = matrix.get_correlation("B", "A", CorrelationType::Pearson).unwrap();
        assert!((corr_ab - corr_ba).abs() < 1e-10);

        let corr_ac = matrix.get_correlation("A", "C", CorrelationType::Pearson).unwrap();
        let corr_ca = matrix.get_correlation("C", "A", CorrelationType::Pearson).unwrap();
        assert!((corr_ac - corr_ca).abs() < 1e-10);
    }
}