//! Data profiling and completeness metrics module
//!
//! This module provides comprehensive data profiling capabilities for time series,
//! including completeness analysis, temporal coverage assessment, and statistical profiling.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::quality::{QualityError, QualityResult};
use crate::stats::{compute_descriptive_stats, DescriptiveStats};
use crate::timeseries::TimeSeries;
use crate::types::Frequency;

/// Configuration for data profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Whether to analyze completeness
    pub analyze_completeness: bool,
    /// Whether to generate statistical profile
    pub generate_statistical_profile: bool,
    /// Whether to analyze temporal coverage
    pub analyze_temporal_coverage: bool,
    /// Expected frequency for gap detection (None for auto-detection)
    pub expected_frequency: Option<Frequency>,
    /// Minimum gap duration to report (in seconds)
    pub min_gap_duration_seconds: Option<i64>,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        ProfilingConfig {
            analyze_completeness: true,
            generate_statistical_profile: true,
            analyze_temporal_coverage: true,
            expected_frequency: None,
            min_gap_duration_seconds: None,
        }
    }
}

impl ProfilingConfig {
    /// Creates a new profiling configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether to analyze completeness
    pub fn with_completeness(mut self, analyze: bool) -> Self {
        self.analyze_completeness = analyze;
        self
    }

    /// Sets whether to generate statistical profile
    pub fn with_statistical_profile(mut self, generate: bool) -> Self {
        self.generate_statistical_profile = generate;
        self
    }

    /// Sets whether to analyze temporal coverage
    pub fn with_temporal_coverage(mut self, analyze: bool) -> Self {
        self.analyze_temporal_coverage = analyze;
        self
    }

    /// Sets the expected frequency
    pub fn with_expected_frequency(mut self, frequency: Frequency) -> Self {
        self.expected_frequency = Some(frequency);
        self
    }

    /// Sets the minimum gap duration to report
    pub fn with_min_gap_duration(mut self, seconds: i64) -> Self {
        self.min_gap_duration_seconds = Some(seconds);
        self
    }
}

/// Represents a gap in the time series data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataGap {
    /// Start time of the gap
    pub start_time: DateTime<Utc>,
    /// End time of the gap
    pub end_time: DateTime<Utc>,
    /// Duration of the gap
    pub duration: Duration,
    /// Expected number of points in this gap
    pub expected_points: usize,
}

impl DataGap {
    /// Creates a new data gap
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>, expected_points: usize) -> Self {
        DataGap {
            start_time: start,
            end_time: end,
            duration: end - start,
            expected_points,
        }
    }

    /// Returns the duration in seconds
    pub fn duration_seconds(&self) -> i64 {
        self.duration.num_seconds()
    }
}

/// Completeness analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessReport {
    /// Total expected number of data points
    pub total_expected_points: usize,
    /// Actual number of data points
    pub actual_points: usize,
    /// Number of missing data points
    pub missing_points: usize,
    /// Completeness ratio (0.0-1.0)
    pub completeness_ratio: f64,
    /// List of detected gaps
    pub gaps: Vec<DataGap>,
    /// Duration of the largest gap
    pub largest_gap_duration: Duration,
    /// Number of gaps detected
    pub gap_count: usize,
}

impl CompletenessReport {
    /// Returns true if completeness ratio meets the threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.completeness_ratio >= threshold
    }

    /// Returns the average gap duration in seconds
    pub fn average_gap_duration_seconds(&self) -> f64 {
        if self.gaps.is_empty() {
            return 0.0;
        }
        let total: i64 = self.gaps.iter().map(|g| g.duration_seconds()).sum();
        total as f64 / self.gaps.len() as f64
    }
}

/// Temporal coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoverage {
    /// Start time of the time series
    pub start_time: DateTime<Utc>,
    /// End time of the time series
    pub end_time: DateTime<Utc>,
    /// Total duration covered
    pub total_duration: Duration,
    /// Detected sampling frequency
    pub detected_frequency: Option<Frequency>,
    /// Average interval between consecutive points (seconds)
    pub average_interval_seconds: f64,
    /// Standard deviation of intervals (seconds)
    pub interval_std_dev: f64,
    /// Coefficient of variation for intervals
    pub interval_cv: f64,
    /// Whether the time series has regular intervals
    pub is_regular: bool,
}

impl TemporalCoverage {
    /// Returns the total duration in seconds
    pub fn duration_seconds(&self) -> i64 {
        self.total_duration.num_seconds()
    }

    /// Returns true if intervals are highly regular (CV < 0.1)
    pub fn is_highly_regular(&self) -> bool {
        self.interval_cv < 0.1
    }
}

/// Statistical profile for time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalProfile {
    /// Basic descriptive statistics
    pub descriptive_stats: DescriptiveStats,
    /// Value range information
    pub value_range: ValueRange,
    /// Trend indicators
    pub trend_indicators: TrendIndicators,
}

/// Value range analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueRange {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
    /// Lower outlier boundary (Q1 - 1.5*IQR)
    pub lower_outlier_boundary: f64,
    /// Upper outlier boundary (Q3 + 1.5*IQR)
    pub upper_outlier_boundary: f64,
}

/// Trend indicators for the time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendIndicators {
    /// Overall direction (positive, negative, or neutral)
    pub direction: String,
    /// Strength of trend (0.0-1.0)
    pub strength: f64,
    /// Simple linear regression slope
    pub slope: f64,
}

/// Quality indicators from profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIndicators {
    /// Overall data quality score (0.0-1.0)
    pub overall_quality: f64,
    /// Completeness score (0.0-1.0)
    pub completeness_score: f64,
    /// Regularity score (0.0-1.0)
    pub regularity_score: f64,
    /// Validity score (0.0-1.0)
    pub validity_score: f64,
}

impl QualityIndicators {
    /// Creates quality indicators from individual scores
    pub fn new(completeness: f64, regularity: f64, validity: f64) -> Self {
        let overall = (completeness + regularity + validity) / 3.0;
        QualityIndicators {
            overall_quality: overall,
            completeness_score: completeness,
            regularity_score: regularity,
            validity_score: validity,
        }
    }
}

/// Comprehensive data profile for a time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProfile {
    /// Basic descriptive statistics
    pub basic_stats: DescriptiveStats,
    /// Completeness analysis
    pub completeness: CompletenessReport,
    /// Temporal coverage analysis
    pub temporal_coverage: TemporalCoverage,
    /// Statistical profile
    pub statistical_profile: StatisticalProfile,
    /// Quality indicators
    pub quality_indicators: QualityIndicators,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl DataProfile {
    /// Returns true if the profile meets minimum quality standards
    pub fn meets_quality_threshold(&self, threshold: f64) -> bool {
        self.quality_indicators.overall_quality >= threshold
    }

    /// Returns a summary of the profile
    pub fn summary(&self) -> String {
        format!(
            "DataProfile: {} points, {:.2}% complete, {:.2} overall quality",
            self.basic_stats.count,
            self.completeness.completeness_ratio * 100.0,
            self.quality_indicators.overall_quality
        )
    }
}

/// Main profiling function that generates a comprehensive data profile
pub fn profile_timeseries(
    data: &TimeSeries,
    config: &ProfilingConfig,
) -> QualityResult<DataProfile> {
    // Validate input
    if data.timestamps.is_empty() || data.values.is_empty() {
        return Err(QualityError::insufficient_data(
            "Time series must have at least one data point",
        ));
    }

    // Generate descriptive statistics
    let basic_stats = compute_descriptive_stats(&data.values)
        .map_err(|e| QualityError::computation(format!("Failed to compute statistics: {}", e)))?;

    // Analyze completeness
    let completeness = if config.analyze_completeness {
        analyze_completeness(data, config)?
    } else {
        CompletenessReport {
            total_expected_points: data.timestamps.len(),
            actual_points: data.timestamps.len(),
            missing_points: 0,
            completeness_ratio: 1.0,
            gaps: vec![],
            largest_gap_duration: Duration::zero(),
            gap_count: 0,
        }
    };

    // Analyze temporal coverage
    let temporal_coverage = if config.analyze_temporal_coverage {
        analyze_temporal_coverage(data, config.expected_frequency.as_ref())?
    } else {
        let duration = *data.timestamps.last().unwrap() - *data.timestamps.first().unwrap();
        TemporalCoverage {
            start_time: *data.timestamps.first().unwrap(),
            end_time: *data.timestamps.last().unwrap(),
            total_duration: duration,
            detected_frequency: None,
            average_interval_seconds: 0.0,
            interval_std_dev: 0.0,
            interval_cv: 0.0,
            is_regular: false,
        }
    };

    // Generate statistical profile
    let statistical_profile = if config.generate_statistical_profile {
        generate_statistical_profile(data)?
    } else {
        let value_range = calculate_value_range(&basic_stats);
        StatisticalProfile {
            descriptive_stats: basic_stats.clone(),
            value_range,
            trend_indicators: TrendIndicators {
                direction: "unknown".to_string(),
                strength: 0.0,
                slope: 0.0,
            },
        }
    };

    // Calculate quality indicators
    let quality_indicators = QualityIndicators::new(
        completeness.completeness_ratio,
        1.0 - temporal_coverage.interval_cv.min(1.0),
        calculate_validity_score(&basic_stats),
    );

    Ok(DataProfile {
        basic_stats,
        completeness,
        temporal_coverage,
        statistical_profile,
        quality_indicators,
        custom_metrics: HashMap::new(),
    })
}

/// Analyzes completeness of the time series
pub fn analyze_completeness(
    data: &TimeSeries,
    config: &ProfilingConfig,
) -> QualityResult<CompletenessReport> {
    if data.timestamps.len() < 2 {
        return Ok(CompletenessReport {
            total_expected_points: data.timestamps.len(),
            actual_points: data.timestamps.len(),
            missing_points: 0,
            completeness_ratio: 1.0,
            gaps: vec![],
            largest_gap_duration: Duration::zero(),
            gap_count: 0,
        });
    }

    // Determine expected frequency
    let inferred_freq = Frequency::infer_from_timestamps(&data.timestamps);
    let expected_freq = config
        .expected_frequency
        .as_ref()
        .or_else(|| data.frequency.as_ref())
        .or(inferred_freq.as_ref());

    // Detect gaps
    let gaps = detect_gaps(data, expected_freq, config.min_gap_duration_seconds)?;

    // Calculate expected points
    let total_duration = *data.timestamps.last().unwrap() - *data.timestamps.first().unwrap();
    let total_expected_points = if let Some(freq) = expected_freq {
        estimate_expected_points(total_duration, freq)
    } else {
        data.timestamps.len()
    };

    let actual_points = data.timestamps.len();
    let missing_points = total_expected_points.saturating_sub(actual_points);
    let completeness_ratio = if total_expected_points > 0 {
        actual_points as f64 / total_expected_points as f64
    } else {
        1.0
    };

    let largest_gap_duration = gaps
        .iter()
        .map(|g| g.duration)
        .max()
        .unwrap_or(Duration::zero());

    Ok(CompletenessReport {
        total_expected_points,
        actual_points,
        missing_points,
        completeness_ratio,
        gap_count: gaps.len(),
        gaps,
        largest_gap_duration,
    })
}

/// Generates a statistical profile for the time series
pub fn generate_statistical_profile(data: &TimeSeries) -> QualityResult<StatisticalProfile> {
    let descriptive_stats = compute_descriptive_stats(&data.values)
        .map_err(|e| QualityError::computation(format!("Failed to compute statistics: {}", e)))?;

    let value_range = calculate_value_range(&descriptive_stats);
    let trend_indicators = calculate_trend_indicators(data)?;

    Ok(StatisticalProfile {
        descriptive_stats,
        value_range,
        trend_indicators,
    })
}

/// Analyzes temporal coverage of the time series
pub fn analyze_temporal_coverage(
    data: &TimeSeries,
    expected_frequency: Option<&Frequency>,
) -> QualityResult<TemporalCoverage> {
    if data.timestamps.len() < 2 {
        return Err(QualityError::insufficient_data(
            "Need at least 2 timestamps for temporal coverage analysis",
        ));
    }

    let start_time = *data.timestamps.first().unwrap();
    let end_time = *data.timestamps.last().unwrap();
    let total_duration = end_time - start_time;

    // Calculate intervals between consecutive timestamps
    let intervals: Vec<f64> = data
        .timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).num_seconds() as f64)
        .collect();

    let average_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;

    // Calculate standard deviation
    let variance: f64 = intervals
        .iter()
        .map(|&x| {
            let diff = x - average_interval;
            diff * diff
        })
        .sum::<f64>()
        / intervals.len() as f64;
    let interval_std_dev = variance.sqrt();

    // Calculate coefficient of variation
    let interval_cv = if average_interval > 0.0 {
        interval_std_dev / average_interval
    } else {
        0.0
    };

    // Determine if regular (CV < 0.1 is considered regular)
    let is_regular = interval_cv < 0.1;

    // Try to detect frequency
    let detected_frequency = expected_frequency
        .cloned()
        .or_else(|| data.frequency.clone())
        .or_else(|| Frequency::infer_from_timestamps(&data.timestamps));

    Ok(TemporalCoverage {
        start_time,
        end_time,
        total_duration,
        detected_frequency,
        average_interval_seconds: average_interval,
        interval_std_dev,
        interval_cv,
        is_regular,
    })
}

// Helper functions

fn detect_gaps(
    data: &TimeSeries,
    expected_freq: Option<&Frequency>,
    min_duration: Option<i64>,
) -> QualityResult<Vec<DataGap>> {
    let mut gaps = Vec::new();

    if data.timestamps.len() < 2 {
        return Ok(gaps);
    }

    // Determine threshold for gap detection
    let threshold_seconds = if let Some(freq) = expected_freq {
        // Use 2x the expected interval as threshold
        freq.to_duration()
            .map(|d| d.as_secs() as i64 * 2)
            .unwrap_or(0)
    } else {
        // Use median interval * 2 as threshold
        let mut intervals: Vec<i64> = data
            .timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_seconds())
            .collect();
        intervals.sort_unstable();
        let median = intervals[intervals.len() / 2];
        median * 2
    };

    // Detect gaps
    for window in data.timestamps.windows(2) {
        let gap_duration = window[1] - window[0];
        let gap_seconds = gap_duration.num_seconds();

        if gap_seconds > threshold_seconds {
            // Check minimum duration filter
            if let Some(min) = min_duration {
                if gap_seconds < min {
                    continue;
                }
            }

            let expected_points = if let Some(freq) = expected_freq {
                if let Some(interval) = freq.to_duration() {
                    (gap_seconds as f64 / interval.as_secs() as f64) as usize
                } else {
                    0
                }
            } else {
                0
            };

            gaps.push(DataGap::new(window[0], window[1], expected_points));
        }
    }

    Ok(gaps)
}

fn estimate_expected_points(duration: Duration, freq: &Frequency) -> usize {
    if let Some(interval) = freq.to_duration() {
        let seconds = duration.num_seconds() as f64;
        let interval_seconds = interval.as_secs() as f64;
        (seconds / interval_seconds).ceil() as usize
    } else {
        // For variable frequencies (Month, Quarter, Year), estimate conservatively
        match freq {
            Frequency::Month => (duration.num_days() / 30) as usize,
            Frequency::Quarter => (duration.num_days() / 90) as usize,
            Frequency::Year => (duration.num_days() / 365) as usize,
            _ => 0,
        }
    }
}

fn calculate_value_range(stats: &DescriptiveStats) -> ValueRange {
    let iqr = stats.quantiles.iqr;
    let lower_outlier_boundary = stats.quantiles.q25 - 1.5 * iqr;
    let upper_outlier_boundary = stats.quantiles.q75 + 1.5 * iqr;

    ValueRange {
        min: stats.min,
        max: stats.max,
        range: stats.range,
        lower_outlier_boundary,
        upper_outlier_boundary,
    }
}

fn calculate_trend_indicators(data: &TimeSeries) -> QualityResult<TrendIndicators> {
    if data.values.len() < 2 {
        return Ok(TrendIndicators {
            direction: "unknown".to_string(),
            strength: 0.0,
            slope: 0.0,
        });
    }

    // Simple linear regression to estimate trend
    let n = data.values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = data.values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &value) in data.values.iter().enumerate() {
        let x_diff = i as f64 - x_mean;
        let y_diff = value - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    let slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    // Determine direction and strength
    let direction = if slope.abs() < 0.01 {
        "neutral"
    } else if slope > 0.0 {
        "positive"
    } else {
        "negative"
    }
    .to_string();

    // Calculate R-squared for strength
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for (i, &value) in data.values.iter().enumerate() {
        let predicted = y_mean + slope * (i as f64 - x_mean);
        ss_res += (value - predicted).powi(2);
        ss_tot += (value - y_mean).powi(2);
    }

    let strength = if ss_tot > 0.0 {
        (1.0 - (ss_res / ss_tot)).max(0.0)
    } else {
        0.0
    };

    Ok(TrendIndicators {
        direction,
        strength,
        slope,
    })
}

fn calculate_validity_score(stats: &DescriptiveStats) -> f64 {
    // Check for NaN, infinity, or other invalid values
    if stats.mean.is_nan() || stats.mean.is_infinite() {
        return 0.0;
    }
    if stats.std_dev.is_nan() || stats.std_dev.is_infinite() {
        return 0.0;
    }

    // Score based on missing count ratio
    let missing_ratio = stats.missing_count as f64 / (stats.count + stats.missing_count) as f64;
    1.0 - missing_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_regular_timeseries() -> TimeSeries {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<DateTime<Utc>> = (0..100)
            .map(|i| start + Duration::seconds(i * 60))
            .collect();
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    fn create_timeseries_with_gaps() -> TimeSeries {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        // Add first 20 points
        for i in 0..20 {
            timestamps.push(start + Duration::seconds(i * 60));
            values.push(i as f64);
        }

        // Gap of 1 hour
        let gap_start = start + Duration::seconds(20 * 60);
        let after_gap = gap_start + Duration::hours(1);

        // Add next 20 points after gap
        for i in 0..20 {
            timestamps.push(after_gap + Duration::seconds(i * 60));
            values.push((20 + i) as f64);
        }

        TimeSeries::new("test_with_gaps".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_profiling_config_default() {
        let config = ProfilingConfig::default();
        assert!(config.analyze_completeness);
        assert!(config.generate_statistical_profile);
        assert!(config.analyze_temporal_coverage);
    }

    #[test]
    fn test_profiling_config_builder() {
        let config = ProfilingConfig::new()
            .with_completeness(false)
            .with_expected_frequency(Frequency::Minute);

        assert!(!config.analyze_completeness);
        assert!(config.expected_frequency.is_some());
    }

    #[test]
    fn test_profile_regular_timeseries() {
        let data = create_regular_timeseries();
        let config = ProfilingConfig::default();

        let profile = profile_timeseries(&data, &config).unwrap();

        assert_eq!(profile.basic_stats.count, 100);
        assert!(profile.completeness.completeness_ratio > 0.99);
        assert!(profile.temporal_coverage.is_regular);
        assert_eq!(profile.temporal_coverage.average_interval_seconds, 60.0);
    }

    #[test]
    fn test_analyze_completeness_no_gaps() {
        let data = create_regular_timeseries();
        let config = ProfilingConfig::new().with_expected_frequency(Frequency::Minute);

        let completeness = analyze_completeness(&data, &config).unwrap();

        assert_eq!(completeness.gap_count, 0);
        assert!(completeness.completeness_ratio > 0.99);
    }

    #[test]
    fn test_analyze_completeness_with_gaps() {
        let data = create_timeseries_with_gaps();
        let config = ProfilingConfig::new().with_expected_frequency(Frequency::Minute);

        let completeness = analyze_completeness(&data, &config).unwrap();

        assert!(completeness.gap_count > 0);
        assert!(completeness.completeness_ratio < 1.0);
        assert!(completeness.largest_gap_duration.num_seconds() > 0);
    }

    #[test]
    fn test_temporal_coverage_regular() {
        let data = create_regular_timeseries();

        let coverage = analyze_temporal_coverage(&data, None).unwrap();

        assert!(coverage.is_regular);
        assert!(coverage.interval_cv < 0.1);
        assert_eq!(coverage.average_interval_seconds, 60.0);
    }

    #[test]
    fn test_statistical_profile() {
        let data = create_regular_timeseries();

        let profile = generate_statistical_profile(&data).unwrap();

        assert!(profile.descriptive_stats.mean > 0.0);
        assert!(profile.value_range.range > 0.0);
        assert_eq!(profile.trend_indicators.direction, "positive");
    }

    #[test]
    fn test_data_gap() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = start + Duration::hours(2);
        let gap = DataGap::new(start, end, 120);

        assert_eq!(gap.duration_seconds(), 7200);
        assert_eq!(gap.expected_points, 120);
    }

    #[test]
    fn test_completeness_report_threshold() {
        let report = CompletenessReport {
            total_expected_points: 100,
            actual_points: 95,
            missing_points: 5,
            completeness_ratio: 0.95,
            gaps: vec![],
            largest_gap_duration: Duration::zero(),
            gap_count: 0,
        };

        assert!(report.meets_threshold(0.9));
        assert!(!report.meets_threshold(0.96));
    }

    #[test]
    fn test_quality_indicators() {
        let indicators = QualityIndicators::new(0.95, 0.9, 0.85);

        assert_eq!(indicators.completeness_score, 0.95);
        assert_eq!(indicators.regularity_score, 0.9);
        assert_eq!(indicators.validity_score, 0.85);
        assert!((indicators.overall_quality - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_profile_summary() {
        let data = create_regular_timeseries();
        let config = ProfilingConfig::default();
        let profile = profile_timeseries(&data, &config).unwrap();

        let summary = profile.summary();
        assert!(summary.contains("100 points"));
    }

    #[test]
    fn test_temporal_coverage_highly_regular() {
        let data = create_regular_timeseries();
        let coverage = analyze_temporal_coverage(&data, None).unwrap();

        assert!(coverage.is_highly_regular());
    }

    #[test]
    fn test_profile_empty_timeseries() {
        let data = TimeSeries::empty("empty".to_string());
        let config = ProfilingConfig::default();

        let result = profile_timeseries(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_performance_10k_datapoints() {
        use std::time::Instant;

        // Create 10K datapoints
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<DateTime<Utc>> = (0..10000)
            .map(|i| start + Duration::seconds(i * 60))
            .collect();
        let values: Vec<f64> = (0..10000).map(|i| (i as f64).sin()).collect();
        let data = TimeSeries::new("performance_test".to_string(), timestamps, values).unwrap();

        let config = ProfilingConfig::default();

        let start_time = Instant::now();
        let profile = profile_timeseries(&data, &config).unwrap();
        let elapsed = start_time.elapsed();

        // Requirement: <50ms for 10K datapoints
        println!("Profiling 10K datapoints took: {:?}", elapsed);
        assert!(
            elapsed.as_millis() < 50,
            "Profiling took {}ms, expected <50ms",
            elapsed.as_millis()
        );

        // Verify profile was generated correctly
        assert_eq!(profile.basic_stats.count, 10000);
        assert!(profile.completeness.completeness_ratio > 0.99);
    }

    #[test]
    fn test_irregular_timeseries() {
        // Create irregular time series
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        for i in 0..50 {
            // Irregular intervals
            let offset = i * i; // Quadratic growth
            timestamps.push(start + Duration::seconds(offset * 10));
            values.push(i as f64);
        }

        let data = TimeSeries::new("irregular".to_string(), timestamps, values).unwrap();
        let coverage = analyze_temporal_coverage(&data, None).unwrap();

        assert!(!coverage.is_regular);
        assert!(coverage.interval_cv > 0.1);
    }
}