//! Distribution analysis module for statistical testing and distribution fitting
//!
//! Provides normality tests, histogram generation, and Q-Q plot data computation.

use serde::{Serialize, Deserialize};
use statrs::distribution::{Normal, ContinuousCDF, Continuous};
use statrs::statistics::Statistics;

/// Distribution analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Histogram data
    pub histogram: Histogram,

    /// Normality test results
    pub normality_test: Option<NormalityTest>,

    /// Q-Q plot data for normal distribution
    pub qq_plot: Option<QQPlotData>,

    /// Skewness
    pub skewness: f64,

    /// Kurtosis (excess kurtosis)
    pub kurtosis: f64,

    /// Distribution parameters if fitted
    pub distribution_fit: Option<DistributionFit>,
}

/// Histogram representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    /// Bin edges (n+1 edges for n bins)
    pub bin_edges: Vec<f64>,

    /// Bin counts
    pub counts: Vec<usize>,

    /// Bin centers for plotting
    pub bin_centers: Vec<f64>,

    /// Bin width
    pub bin_width: f64,

    /// Total number of observations
    pub total_count: usize,
}

/// Normality test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTest {
    /// Test name (e.g., "Shapiro-Wilk", "Kolmogorov-Smirnov")
    pub test_name: String,

    /// Test statistic
    pub statistic: f64,

    /// P-value
    pub p_value: f64,

    /// Is the distribution likely normal? (p > 0.05)
    pub is_normal: bool,

    /// Critical value (if applicable)
    pub critical_value: Option<f64>,
}

/// Q-Q plot data against normal distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QQPlotData {
    /// Theoretical quantiles (from normal distribution)
    pub theoretical_quantiles: Vec<f64>,

    /// Sample quantiles
    pub sample_quantiles: Vec<f64>,

    /// Expected line slope (for perfect normal fit)
    pub expected_line_slope: f64,

    /// Expected line intercept (for perfect normal fit)
    pub expected_line_intercept: f64,
}

/// Distribution fit parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    /// Distribution name
    pub distribution_name: String,

    /// Fitted parameters
    pub parameters: std::collections::HashMap<String, f64>,

    /// Goodness of fit measure (e.g., R-squared, AIC)
    pub goodness_of_fit: f64,

    /// Log-likelihood
    pub log_likelihood: f64,
}

/// Perform comprehensive distribution analysis
pub fn compute_distribution_analysis(data: &[f64]) -> Result<DistributionAnalysis, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err("Cannot analyze distribution of empty dataset".into());
    }

    // Compute histogram
    let histogram = compute_histogram(data, None)?;

    // Compute skewness and kurtosis
    let mean = data.mean();
    let std_dev = data.std_dev();
    let skewness = compute_skewness_moment(data, mean, std_dev);
    let kurtosis = compute_kurtosis_moment(data, mean, std_dev);

    // Perform normality test
    let normality_test = if data.len() >= 8 && data.len() <= 5000 {
        Some(shapiro_wilk_test(data)?)
    } else if data.len() > 50 {
        Some(kolmogorov_smirnov_test(data)?)
    } else {
        None
    };

    // Generate Q-Q plot data
    let qq_plot = if data.len() >= 10 {
        Some(compute_qq_plot_data(data)?)
    } else {
        None
    };

    // Fit normal distribution
    let distribution_fit = fit_normal_distribution(data)?;

    Ok(DistributionAnalysis {
        histogram,
        normality_test,
        qq_plot,
        skewness,
        kurtosis,
        distribution_fit: Some(distribution_fit),
    })
}

/// Compute histogram with automatic or specified bin count
pub fn compute_histogram(data: &[f64], bin_count: Option<usize>) -> Result<Histogram, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err("Cannot compute histogram for empty dataset".into());
    }

    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if min_val == max_val {
        // All values are the same
        return Ok(Histogram {
            bin_edges: vec![min_val - 0.5, min_val + 0.5],
            counts: vec![data.len()],
            bin_centers: vec![min_val],
            bin_width: 1.0,
            total_count: data.len(),
        });
    }

    // Determine number of bins using Sturges' rule or Scott's rule
    let n_bins = bin_count.unwrap_or_else(|| {
        let n = data.len() as f64;
        let sturges = (1.0 + n.log2()).ceil() as usize;
        let scott = {
            let range = max_val - min_val;
            let std_dev = data.std_dev();
            let bin_width = 3.49 * std_dev / n.cbrt();
            (range / bin_width).ceil() as usize
        };
        sturges.min(scott).max(5).min(50) // Reasonable bounds
    });

    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut bin_edges = Vec::with_capacity(n_bins + 1);
    let mut bin_centers = Vec::with_capacity(n_bins);

    // Create bin edges and centers
    for i in 0..=n_bins {
        bin_edges.push(min_val + i as f64 * bin_width);
    }

    for i in 0..n_bins {
        bin_centers.push(min_val + (i as f64 + 0.5) * bin_width);
    }

    // Count observations in each bin
    let mut counts = vec![0; n_bins];
    for &value in data {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1); // Handle edge case where value == max_val
        counts[bin_idx] += 1;
    }

    Ok(Histogram {
        bin_edges,
        counts,
        bin_centers,
        bin_width,
        total_count: data.len(),
    })
}

/// Perform Shapiro-Wilk normality test
fn shapiro_wilk_test(data: &[f64]) -> Result<NormalityTest, Box<dyn std::error::Error>> {
    if data.len() < 3 || data.len() > 5000 {
        return Err("Shapiro-Wilk test requires between 3 and 5000 observations".into());
    }

    // This is a simplified implementation
    // In practice, you would use a proper statistical library
    let n = data.len();
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate test statistic (simplified version)
    let mean = data.mean();
    let ss = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();

    // Approximate W statistic calculation
    let w_statistic = {
        let mut numerator = 0.0;
        for i in 0..n {
            let coef = expected_normal_order_statistic(i + 1, n);
            numerator += coef * sorted_data[i];
        }
        numerator.powi(2) / ss
    };

    // Approximate p-value (this is a simplified calculation)
    let p_value = approximate_shapiro_wilk_p_value(w_statistic, n);
    let is_normal = p_value > 0.05;

    Ok(NormalityTest {
        test_name: "Shapiro-Wilk".to_string(),
        statistic: w_statistic,
        p_value,
        is_normal,
        critical_value: None,
    })
}

/// Perform Kolmogorov-Smirnov normality test
fn kolmogorov_smirnov_test(data: &[f64]) -> Result<NormalityTest, Box<dyn std::error::Error>> {
    if data.len() < 8 {
        return Err("Kolmogorov-Smirnov test requires at least 8 observations".into());
    }

    let n = data.len() as f64;
    let mean = data.mean();
    let std_dev = data.std_dev();

    if std_dev == 0.0 {
        return Err("Cannot perform KS test on data with zero variance".into());
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let normal = Normal::new(mean, std_dev)?;

    // Calculate the maximum difference between empirical and theoretical CDFs
    let mut max_diff: f64 = 0.0;
    for (i, &value) in sorted_data.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n;
        let theoretical_cdf = normal.cdf(value);
        let diff = (empirical_cdf - theoretical_cdf).abs();
        max_diff = max_diff.max(diff);
    }

    // Critical value for α = 0.05
    let critical_value = 1.36 / n.sqrt();

    // Approximate p-value
    let p_value = approximate_ks_p_value(max_diff, n as usize);
    let is_normal = max_diff < critical_value;

    Ok(NormalityTest {
        test_name: "Kolmogorov-Smirnov".to_string(),
        statistic: max_diff,
        p_value,
        is_normal,
        critical_value: Some(critical_value),
    })
}

/// Compute Q-Q plot data against normal distribution
fn compute_qq_plot_data(data: &[f64]) -> Result<QQPlotData, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err("Cannot compute Q-Q plot for empty dataset".into());
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len();
    let mut theoretical_quantiles = Vec::with_capacity(n);
    let mut sample_quantiles = Vec::with_capacity(n);

    // Standard normal distribution
    let standard_normal = Normal::new(0.0, 1.0)?;

    // Calculate quantiles
    for i in 0..n {
        // Use (i + 0.5) / n for plotting position to avoid extreme quantiles
        let p = (i as f64 + 0.5) / n as f64;
        let theoretical_q = standard_normal.inverse_cdf(p);
        theoretical_quantiles.push(theoretical_q);
        sample_quantiles.push(sorted_data[i]);
    }

    // Calculate expected line (if data were perfectly normal)
    let sample_mean = data.mean();
    let sample_std = data.std_dev();

    Ok(QQPlotData {
        theoretical_quantiles,
        sample_quantiles,
        expected_line_slope: sample_std,
        expected_line_intercept: sample_mean,
    })
}

/// Fit normal distribution to data
fn fit_normal_distribution(data: &[f64]) -> Result<DistributionFit, Box<dyn std::error::Error>> {
    let mean = data.mean();
    let std_dev = data.std_dev();

    let mut parameters = std::collections::HashMap::new();
    parameters.insert("mean".to_string(), mean);
    parameters.insert("std_dev".to_string(), std_dev);

    // Calculate log-likelihood
    let normal = Normal::new(mean, std_dev)?;
    let log_likelihood = data.iter()
        .map(|&x| normal.ln_pdf(x))
        .sum::<f64>();

    // Calculate AIC (Akaike Information Criterion)
    let k = 2.0; // Number of parameters for normal distribution
    let n = data.len() as f64;
    let aic = 2.0 * k - 2.0 * log_likelihood;

    Ok(DistributionFit {
        distribution_name: "Normal".to_string(),
        parameters,
        goodness_of_fit: -aic, // Use negative AIC so higher is better
        log_likelihood,
    })
}

// Helper functions

fn compute_skewness_moment(data: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }

    let n = data.len() as f64;
    let m3 = data.iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n;

    m3
}

fn compute_kurtosis_moment(data: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }

    let n = data.len() as f64;
    let m4 = data.iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / n;

    m4 - 3.0 // Excess kurtosis
}

fn expected_normal_order_statistic(rank: usize, n: usize) -> f64 {
    // Simplified approximation for expected value of order statistics
    let p = (rank as f64 - 0.375) / (n as f64 + 0.25);
    approximate_normal_inverse(p)
}

fn approximate_normal_inverse(p: f64) -> f64 {
    // Approximation of the inverse normal CDF using Beasley-Springer-Moro algorithm
    if p <= 0.0 || p >= 1.0 {
        return if p <= 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
    }

    let a0 = 2.50662823884;
    let a1 = -18.61500062529;
    let a2 = 41.39119773534;
    let a3 = -25.44106049637;

    let b0 = -8.47351093090;
    let b1 = 23.08336743743;
    let b2 = -21.06224101826;
    let b3 = 3.13082909833;

    let c0 = 0.3374754822726147;
    let c1 = 0.9761690190917186;
    let c2 = 0.1607979714918209;
    let c3 = 0.0276438810333863;
    let c4 = 0.0038405729373609;
    let c5 = 0.0003951896511919;
    let c6 = 0.0000321767881768;
    let c7 = 0.0000002888167364;
    let c8 = 0.0000003960315187;

    let x = p - 0.5;

    if x.abs() < 0.42 {
        let r = x * x;
        return x * (((a3 * r + a2) * r + a1) * r + a0) /
                   ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
    }

    let r = if p < 0.5 { p } else { 1.0 - p };
    let s = (-2.0 * r.ln()).sqrt();
    let t = s - (c0 + c1 * s + c2 * s.powi(2) + c3 * s.powi(3) + c4 * s.powi(4) +
                 c5 * s.powi(5) + c6 * s.powi(6) + c7 * s.powi(7) + c8 * s.powi(8));

    if p < 0.5 { -t } else { t }
}

fn approximate_shapiro_wilk_p_value(w: f64, n: usize) -> f64 {
    // Very simplified p-value approximation
    // In practice, use lookup tables or more sophisticated approximations
    if w > 0.95 {
        0.5 + (w - 0.95) * 2.0
    } else if w > 0.90 {
        0.1 + (w - 0.90) * 8.0
    } else {
        0.01 + w * 0.09
    }
}

fn approximate_ks_p_value(d: f64, n: usize) -> f64 {
    // Approximation of KS p-value
    let lambda = d * (n as f64).sqrt();
    let mut p = 0.0;

    for i in 1..=10 {
        let term = (-2.0 * (i as f64).powi(2) * lambda.powi(2)).exp();
        if i % 2 == 1 {
            p += term;
        } else {
            p -= term;
        }
    }

    2.0 * p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5];
        let histogram = compute_histogram(&data, Some(5)).unwrap();

        assert_eq!(histogram.bin_edges.len(), 6);
        assert_eq!(histogram.counts.len(), 5);
        assert_eq!(histogram.bin_centers.len(), 5);
        assert_eq!(histogram.total_count, 9);
    }

    #[test]
    fn test_qq_plot_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qq_plot = compute_qq_plot_data(&data).unwrap();

        assert_eq!(qq_plot.theoretical_quantiles.len(), 5);
        assert_eq!(qq_plot.sample_quantiles.len(), 5);
        assert!(qq_plot.expected_line_slope > 0.0);
    }

    #[test]
    fn test_distribution_analysis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let analysis = compute_distribution_analysis(&data).unwrap();

        assert!(analysis.histogram.counts.len() > 0);
        assert!(analysis.qq_plot.is_some());
        assert!(analysis.distribution_fit.is_some());
    }
}