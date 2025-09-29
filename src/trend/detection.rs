//! # Trend Detection Module
//!
//! Statistical tests for detecting trends and change points in time series data.
//! Includes Mann-Kendall test, Sen's slope estimator, and Pettitt's test.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Trait for trend detection tests
pub trait TrendTest {
    /// Execute the trend test
    fn test(&self, data: &[f64]) -> Result<TrendTestResult, Box<dyn std::error::Error>>;

    /// Get test name
    fn name(&self) -> &'static str;
}

/// Result of a trend detection test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendTestResult {
    /// Name of the test performed
    pub test_name: String,

    /// Test statistic value
    pub statistic: f64,

    /// P-value of the test
    pub p_value: f64,

    /// Critical value at specified significance level
    pub critical_value: Option<f64>,

    /// Significance level used
    pub alpha: f64,

    /// Whether the trend is statistically significant
    pub is_significant: bool,

    /// Direction of the trend (-1: decreasing, 0: no trend, 1: increasing)
    pub trend_direction: i8,

    /// Slope estimate (if applicable)
    pub slope: Option<f64>,

    /// Additional test-specific results
    pub metadata: HashMap<String, f64>,
}

impl TrendTestResult {
    /// Create a new trend test result
    pub fn new(test_name: String, statistic: f64, p_value: f64, alpha: f64) -> Self {
        Self {
            test_name,
            statistic,
            p_value,
            critical_value: None,
            alpha,
            is_significant: p_value < alpha,
            trend_direction: 0,
            slope: None,
            metadata: HashMap::new(),
        }
    }

    /// Set trend direction
    pub fn with_direction(mut self, direction: i8) -> Self {
        self.trend_direction = direction;
        self
    }

    /// Set slope estimate
    pub fn with_slope(mut self, slope: f64) -> Self {
        self.slope = Some(slope);
        self
    }

    /// Set critical value
    pub fn with_critical_value(mut self, critical_value: f64) -> Self {
        self.critical_value = Some(critical_value);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: f64) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Mann-Kendall test for monotonic trend detection
#[derive(Debug, Clone)]
pub struct MannKendallTest {
    /// Significance level (default: 0.05)
    pub alpha: f64,

    /// Whether to apply continuity correction
    pub continuity_correction: bool,
}

impl Default for MannKendallTest {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            continuity_correction: true,
        }
    }
}

impl MannKendallTest {
    /// Create new Mann-Kendall test with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Create Mann-Kendall test with custom significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable/disable continuity correction
    pub fn with_continuity_correction(mut self, correction: bool) -> Self {
        self.continuity_correction = correction;
        self
    }

    /// Calculate Mann-Kendall S statistic
    fn calculate_s(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len();
        let mut s = 0.0;

        // Calculate S statistic
        for i in 0..n {
            for j in (i + 1)..n {
                if data[j] > data[i] {
                    s += 1.0;
                } else if data[j] < data[i] {
                    s -= 1.0;
                }
            }
        }

        // Calculate variance of S
        let n_f64 = n as f64;
        let var_s = (n_f64 * (n_f64 - 1.0) * (2.0 * n_f64 + 5.0)) / 18.0;

        // Adjust for ties (simplified - assumes no ties for now)
        // In a full implementation, would need to account for tied values

        (s, var_s)
    }

    /// Calculate normalized test statistic (Z)
    fn calculate_z(&self, s: f64, var_s: f64) -> f64 {
        if var_s == 0.0 {
            return 0.0;
        }

        let std_s = var_s.sqrt();

        if self.continuity_correction {
            if s > 0.0 {
                (s - 1.0) / std_s
            } else if s < 0.0 {
                (s + 1.0) / std_s
            } else {
                0.0
            }
        } else {
            s / std_s
        }
    }

    /// Calculate p-value from Z statistic (two-tailed)
    fn calculate_p_value(&self, z: f64) -> f64 {
        // Approximate normal distribution p-value calculation
        // Using complementary error function approximation
        let abs_z = z.abs();

        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let x = abs_z / (2.0_f64).sqrt();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        2.0 * (1.0 - y)
    }
}

impl TrendTest for MannKendallTest {
    fn test(&self, data: &[f64]) -> Result<TrendTestResult, Box<dyn std::error::Error>> {
        if data.len() < 4 {
            return Err("Mann-Kendall test requires at least 4 observations".into());
        }

        // Filter out non-finite values
        let valid_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if valid_data.len() < 4 {
            return Err("Mann-Kendall test requires at least 4 valid observations".into());
        }

        let (s, var_s) = self.calculate_s(&valid_data);
        let z = self.calculate_z(s, var_s);
        let p_value = self.calculate_p_value(z);

        let trend_direction = if z > 0.0 { 1 } else if z < 0.0 { -1 } else { 0 };

        Ok(TrendTestResult::new("Mann-Kendall".to_string(), z, p_value, self.alpha)
            .with_direction(trend_direction)
            .with_metadata("s_statistic".to_string(), s)
            .with_metadata("variance_s".to_string(), var_s))
    }

    fn name(&self) -> &'static str {
        "Mann-Kendall"
    }
}

/// Sen's slope estimator for trend magnitude
#[derive(Debug, Clone)]
pub struct SensSlope {
    /// Significance level for confidence interval
    pub alpha: f64,
}

impl Default for SensSlope {
    fn default() -> Self {
        Self { alpha: 0.05 }
    }
}

impl SensSlope {
    /// Create new Sen's slope estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create Sen's slope estimator with custom significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Calculate all pairwise slopes
    fn calculate_slopes(&self, data: &[f64]) -> Vec<f64> {
        let mut slopes = Vec::new();
        let n = data.len();

        for i in 0..n {
            for j in (i + 1)..n {
                if j != i {
                    let slope = (data[j] - data[i]) / ((j - i) as f64);
                    if slope.is_finite() {
                        slopes.push(slope);
                    }
                }
            }
        }

        slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        slopes
    }

    /// Calculate median slope
    fn median_slope(&self, slopes: &[f64]) -> f64 {
        if slopes.is_empty() {
            return 0.0;
        }

        let n = slopes.len();
        if n % 2 == 1 {
            slopes[n / 2]
        } else {
            (slopes[n / 2 - 1] + slopes[n / 2]) / 2.0
        }
    }

    /// Calculate confidence interval for slope
    fn confidence_interval(&self, slopes: &[f64], data_len: usize) -> (f64, f64) {
        if slopes.is_empty() {
            return (0.0, 0.0);
        }

        let n = slopes.len() as f64;
        let data_len_f64 = data_len as f64;

        // Approximate critical value for confidence interval
        let z_alpha_2 = 1.96; // For 95% confidence interval (alpha = 0.05)
        let var_s = (data_len_f64 * (data_len_f64 - 1.0) * (2.0 * data_len_f64 + 5.0)) / 18.0;
        let c_alpha = z_alpha_2 * var_s.sqrt();

        let m1 = ((n - c_alpha) / 2.0).floor() as usize;
        let m2 = ((n + c_alpha) / 2.0).ceil() as usize;

        let lower = if m1 > 0 && m1 <= slopes.len() {
            slopes[m1 - 1]
        } else {
            slopes[0]
        };

        let upper = if m2 > 0 && m2 <= slopes.len() {
            slopes[m2 - 1]
        } else {
            slopes[slopes.len() - 1]
        };

        (lower, upper)
    }
}

impl TrendTest for SensSlope {
    fn test(&self, data: &[f64]) -> Result<TrendTestResult, Box<dyn std::error::Error>> {
        if data.len() < 4 {
            return Err("Sen's slope estimator requires at least 4 observations".into());
        }

        // Filter out non-finite values
        let valid_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if valid_data.len() < 4 {
            return Err("Sen's slope estimator requires at least 4 valid observations".into());
        }

        let slopes = self.calculate_slopes(&valid_data);
        if slopes.is_empty() {
            return Err("Could not calculate any valid slopes".into());
        }

        let median_slope = self.median_slope(&slopes);
        let (ci_lower, ci_upper) = self.confidence_interval(&slopes, valid_data.len());

        // Test if slope is significantly different from zero
        let is_significant = !(ci_lower <= 0.0 && ci_upper >= 0.0);

        let trend_direction = if median_slope > 0.0 { 1 } else if median_slope < 0.0 { -1 } else { 0 };

        // P-value approximation (simplified)
        let p_value = if is_significant { self.alpha * 0.5 } else { 0.5 };

        Ok(TrendTestResult::new("Sen's Slope".to_string(), median_slope, p_value, self.alpha)
            .with_direction(trend_direction)
            .with_slope(median_slope)
            .with_metadata("confidence_lower".to_string(), ci_lower)
            .with_metadata("confidence_upper".to_string(), ci_upper)
            .with_metadata("num_slopes".to_string(), slopes.len() as f64))
    }

    fn name(&self) -> &'static str {
        "Sen's Slope"
    }
}

/// Pettitt's test for change point detection
#[derive(Debug, Clone)]
pub struct PettittTest {
    /// Significance level
    pub alpha: f64,
}

impl Default for PettittTest {
    fn default() -> Self {
        Self { alpha: 0.05 }
    }
}

impl PettittTest {
    /// Create new Pettitt's test
    pub fn new() -> Self {
        Self::default()
    }

    /// Create Pettitt's test with custom significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Calculate Pettitt's U statistic
    fn calculate_u_statistic(&self, data: &[f64]) -> (f64, usize) {
        let n = data.len();
        let mut max_u: f64 = 0.0;
        let mut change_point = 0;

        for t in 1..n {
            let mut u_t: f64 = 0.0;

            for i in 0..t {
                for j in t..n {
                    if data[i] < data[j] {
                        u_t += 1.0;
                    } else if data[i] > data[j] {
                        u_t -= 1.0;
                    }
                }
            }

            if u_t.abs() > max_u.abs() {
                max_u = u_t;
                change_point = t;
            }
        }

        (max_u, change_point)
    }

    /// Calculate approximate p-value
    fn calculate_p_value(&self, u_max: f64, n: usize) -> f64 {
        let n_f64 = n as f64;

        // Approximate formula for p-value
        let k_max = u_max.abs();
        let p_approx = 2.0 * (-6.0 * k_max * k_max / (n_f64 * n_f64 * n_f64 + n_f64 * n_f64)).exp();

        p_approx.min(1.0)
    }
}

impl TrendTest for PettittTest {
    fn test(&self, data: &[f64]) -> Result<TrendTestResult, Box<dyn std::error::Error>> {
        if data.len() < 10 {
            return Err("Pettitt's test requires at least 10 observations".into());
        }

        // Filter out non-finite values
        let valid_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if valid_data.len() < 10 {
            return Err("Pettitt's test requires at least 10 valid observations".into());
        }

        let (u_max, change_point) = self.calculate_u_statistic(&valid_data);
        let p_value = self.calculate_p_value(u_max, valid_data.len());

        let trend_direction = if u_max > 0.0 { 1 } else if u_max < 0.0 { -1 } else { 0 };

        Ok(TrendTestResult::new("Pettitt".to_string(), u_max, p_value, self.alpha)
            .with_direction(trend_direction)
            .with_metadata("change_point".to_string(), change_point as f64)
            .with_metadata("change_point_ratio".to_string(), change_point as f64 / valid_data.len() as f64))
    }

    fn name(&self) -> &'static str {
        "Pettitt"
    }
}

/// Detect trend using specified method
pub fn detect_trend(data: &[f64], method: &str) -> Result<TrendTestResult, Box<dyn std::error::Error>> {
    match method.to_lowercase().as_str() {
        "mann_kendall" | "mk" => {
            let test = MannKendallTest::new();
            test.test(data)
        },
        "sens_slope" | "sen" => {
            let test = SensSlope::new();
            test.test(data)
        },
        "pettitt" => {
            let test = PettittTest::new();
            test.test(data)
        },
        _ => Err(format!("Unknown trend detection method: {}", method).into())
    }
}

/// Test trend significance for multiple methods
pub fn test_trend_significance(data: &[f64], alpha: f64) -> Result<HashMap<String, TrendTestResult>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();

    // Mann-Kendall test
    let mk_test = MannKendallTest::new().with_alpha(alpha);
    if let Ok(result) = mk_test.test(data) {
        results.insert("Mann-Kendall".to_string(), result);
    }

    // Sen's slope
    let sen_test = SensSlope::new().with_alpha(alpha);
    if let Ok(result) = sen_test.test(data) {
        results.insert("Sen's Slope".to_string(), result);
    }

    // Pettitt's test
    let pettitt_test = PettittTest::new().with_alpha(alpha);
    if let Ok(result) = pettitt_test.test(data) {
        results.insert("Pettitt".to_string(), result);
    }

    if results.is_empty() {
        return Err("No trend tests could be performed on the provided data".into());
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_trend_data(n: usize, slope: f64, noise: f64) -> Vec<f64> {
        (0..n)
            .map(|i| slope * i as f64 + noise * (i as f64 * 0.1).sin())
            .collect()
    }

    #[test]
    fn test_mann_kendall_increasing_trend() {
        let data = generate_trend_data(50, 1.0, 0.1);
        let test = MannKendallTest::new();
        let result = test.test(&data).unwrap();

        assert_eq!(result.test_name, "Mann-Kendall");
        assert_eq!(result.trend_direction, 1);
        assert!(result.p_value < 0.05);
        assert!(result.is_significant);
    }

    #[test]
    fn test_mann_kendall_decreasing_trend() {
        let data = generate_trend_data(50, -1.0, 0.1);
        let test = MannKendallTest::new();
        let result = test.test(&data).unwrap();

        assert_eq!(result.trend_direction, -1);
        assert!(result.p_value < 0.05);
        assert!(result.is_significant);
    }

    #[test]
    fn test_mann_kendall_no_trend() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let test = MannKendallTest::new();
        let result = test.test(&data).unwrap();

        assert!(result.p_value > 0.05);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_sens_slope_positive() {
        let data = generate_trend_data(30, 2.0, 0.5);
        let test = SensSlope::new();
        let result = test.test(&data).unwrap();

        assert_eq!(result.test_name, "Sen's Slope");
        assert!(result.slope.unwrap() > 0.0);
        assert_eq!(result.trend_direction, 1);
    }

    #[test]
    fn test_sens_slope_negative() {
        let data = generate_trend_data(30, -1.5, 0.3);
        let test = SensSlope::new();
        let result = test.test(&data).unwrap();

        assert!(result.slope.unwrap() < 0.0);
        assert_eq!(result.trend_direction, -1);
    }

    #[test]
    fn test_pettitt_change_point() {
        // Generate data with a change point
        let mut data = Vec::new();
        // First half: mean = 10
        for _ in 0..25 {
            data.push(10.0);
        }
        // Second half: mean = 15
        for _ in 0..25 {
            data.push(15.0);
        }

        let test = PettittTest::new();
        let result = test.test(&data).unwrap();

        assert_eq!(result.test_name, "Pettitt");
        assert!(result.is_significant);
        assert!(result.metadata.contains_key("change_point"));
    }

    #[test]
    fn test_detect_trend_function() {
        let data = generate_trend_data(40, 1.0, 0.2);

        let mk_result = detect_trend(&data, "mann_kendall").unwrap();
        assert_eq!(mk_result.test_name, "Mann-Kendall");

        let sen_result = detect_trend(&data, "sens_slope").unwrap();
        assert_eq!(sen_result.test_name, "Sen's Slope");

        let pettitt_result = detect_trend(&data, "pettitt").unwrap();
        assert_eq!(pettitt_result.test_name, "Pettitt");
    }

    #[test]
    fn test_trend_significance_multiple() {
        let data = generate_trend_data(35, 1.5, 0.3);
        let results = test_trend_significance(&data, 0.05).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results.contains_key("Mann-Kendall"));
        assert!(results.contains_key("Sen's Slope"));
        assert!(results.contains_key("Pettitt"));
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![1.0, 2.0];
        let test = MannKendallTest::new();
        let result = test.test(&data);
        assert!(result.is_err());

        let test2 = SensSlope::new();
        let result2 = test2.test(&data);
        assert!(result2.is_err());
    }

    #[test]
    fn test_trend_test_result_builder() {
        let result = TrendTestResult::new("Test".to_string(), 1.5, 0.02, 0.05)
            .with_direction(1)
            .with_slope(0.5)
            .with_critical_value(1.96)
            .with_metadata("custom_metric".to_string(), 42.0);

        assert_eq!(result.test_name, "Test");
        assert_eq!(result.statistic, 1.5);
        assert_eq!(result.p_value, 0.02);
        assert_eq!(result.alpha, 0.05);
        assert!(result.is_significant);
        assert_eq!(result.trend_direction, 1);
        assert_eq!(result.slope, Some(0.5));
        assert_eq!(result.critical_value, Some(1.96));
        assert_eq!(result.metadata.get("custom_metric"), Some(&42.0));
    }
}