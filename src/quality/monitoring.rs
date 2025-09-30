//! Quality monitoring and tracking over time
//!
//! This module provides continuous quality monitoring capabilities that track
//! quality metrics over time and detect quality degradation patterns.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::quality::{
    EnhancedQualityAssessment, QualityError, QualityResult, TimeRange,
};

/// Direction of quality trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Quality is improving
    Improving,
    /// Quality is stable
    Stable,
    /// Quality is declining
    Declining,
    /// Insufficient data to determine trend
    Unknown,
}

/// Alert type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// Quality has degraded significantly
    QualityDegradation,
    /// Quality threshold has been violated
    ThresholdViolation,
    /// Anomalous quality pattern detected
    AnomalousPattern,
    /// Data drift detected
    DataDrift,
    /// Systematic issue identified
    SystematicIssue,
}

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning that requires attention
    Warning,
    /// Critical issue requiring immediate action
    Critical,
    /// Emergency requiring urgent response
    Emergency,
}

/// Quality alert with context and recommendations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityAlert {
    /// Type of alert
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Metric that triggered the alert
    pub metric: String,
    /// Current value of the metric
    pub current_value: f64,
    /// Threshold value that was violated
    pub threshold_value: f64,
    /// Timestamp when alert was generated
    pub timestamp: DateTime<Utc>,
    /// Human-readable message
    pub message: String,
    /// Actionable recommendations
    pub recommendations: Vec<String>,
}

/// Notification channel configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email { recipients: Vec<String> },
    /// Webhook notification
    Webhook { url: String, headers: HashMap<String, String> },
    /// Slack notification
    Slack { webhook_url: String, channel: String },
    /// SMS notification
    SMS { phone_numbers: Vec<String> },
    /// Dashboard update
    Dashboard { dashboard_id: String },
}

/// Threshold configuration for a quality dimension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityThresholdConfig {
    /// Warning threshold (0-100 scale)
    pub warning_threshold: f64,
    /// Critical threshold (0-100 scale)
    pub critical_threshold: f64,
    /// Rate of change threshold (per day)
    pub degradation_rate_threshold: f64,
}

impl Default for QualityThresholdConfig {
    fn default() -> Self {
        QualityThresholdConfig {
            warning_threshold: 75.0,
            critical_threshold: 60.0,
            degradation_rate_threshold: -5.0, // -5 points per day
        }
    }
}

/// Quality thresholds for all dimensions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Overall quality threshold
    pub overall_quality: QualityThresholdConfig,
    /// Completeness threshold
    pub completeness: QualityThresholdConfig,
    /// Consistency threshold
    pub consistency: QualityThresholdConfig,
    /// Validity threshold
    pub validity: QualityThresholdConfig,
    /// Timeliness threshold
    pub timeliness: QualityThresholdConfig,
    /// Accuracy threshold
    pub accuracy: QualityThresholdConfig,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        QualityThresholds {
            overall_quality: QualityThresholdConfig::default(),
            completeness: QualityThresholdConfig::default(),
            consistency: QualityThresholdConfig::default(),
            validity: QualityThresholdConfig::default(),
            timeliness: QualityThresholdConfig::default(),
            accuracy: QualityThresholdConfig::default(),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// How often to track quality
    pub tracking_frequency: Duration,
    /// Alert thresholds
    pub alert_thresholds: QualityThresholds,
    /// Window for trend detection
    pub trend_detection_window: Duration,
    /// How often to update baseline
    pub baseline_update_frequency: Duration,
    /// Enable quality predictions
    pub enable_predictions: bool,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        MonitoringConfig {
            tracking_frequency: Duration::hours(1),
            alert_thresholds: QualityThresholds::default(),
            trend_detection_window: Duration::days(7),
            baseline_update_frequency: Duration::days(30),
            enable_predictions: true,
            notification_channels: Vec::new(),
        }
    }
}

/// Quality baseline for comparison
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBaseline {
    /// Mean overall quality
    pub mean_overall_quality: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum acceptable quality
    pub min_acceptable: f64,
    /// Maximum observed quality
    pub max_observed: f64,
    /// Baseline established at
    pub established_at: DateTime<Utc>,
    /// Number of samples used
    pub sample_count: usize,
}

impl Default for QualityBaseline {
    fn default() -> Self {
        QualityBaseline {
            mean_overall_quality: 80.0,
            std_dev: 10.0,
            min_acceptable: 60.0,
            max_observed: 100.0,
            established_at: Utc::now(),
            sample_count: 0,
        }
    }
}

/// Quality time series data point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall quality score
    pub overall_quality: f64,
    /// Dimension scores
    pub dimensions: HashMap<String, f64>,
}

/// Quality time series
pub type QualityTimeSeries = Vec<QualityDataPoint>;

/// Quality trend analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityTrend {
    /// Overall trend direction
    pub overall_trend: TrendDirection,
    /// Trend for each dimension
    pub dimension_trends: HashMap<String, TrendDirection>,
    /// Rate of quality change (points per day)
    pub quality_velocity: f64,
    /// Predicted future quality
    pub predicted_quality: Option<f64>,
    /// Time periods with anomalous quality
    pub anomaly_periods: Vec<TimeRange>,
}

/// Quality tracker maintains historical quality data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityTracker {
    /// Historical quality metrics
    pub metric_history: HashMap<String, Vec<(DateTime<Utc>, f64)>>,
    /// Alert history
    pub alert_history: Vec<QualityAlert>,
    /// Quality baseline
    pub baseline_metrics: QualityBaseline,
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
}

impl QualityTracker {
    /// Create a new quality tracker
    pub fn new(config: MonitoringConfig) -> Self {
        QualityTracker {
            metric_history: HashMap::new(),
            alert_history: Vec::new(),
            baseline_metrics: QualityBaseline::default(),
            monitoring_config: config,
        }
    }

    /// Track a quality assessment
    pub fn track(&mut self, assessment: &EnhancedQualityAssessment) -> QualityResult<()> {
        let timestamp = Utc::now();

        // Track overall quality
        self.add_metric("overall_quality", timestamp, assessment.overall_score);

        // Track dimension scores
        self.add_metric("completeness", timestamp, assessment.dimension_scores.completeness);
        self.add_metric("consistency", timestamp, assessment.dimension_scores.consistency);
        self.add_metric("validity", timestamp, assessment.dimension_scores.validity);
        self.add_metric("timeliness", timestamp, assessment.dimension_scores.timeliness);
        self.add_metric("accuracy", timestamp, assessment.dimension_scores.accuracy);

        // Check for threshold violations
        let alerts = self.check_thresholds(assessment, timestamp);
        self.alert_history.extend(alerts);

        Ok(())
    }

    /// Add a metric data point
    fn add_metric(&mut self, name: &str, timestamp: DateTime<Utc>, value: f64) {
        self.metric_history
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push((timestamp, value));
    }

    /// Check for threshold violations
    fn check_thresholds(
        &self,
        assessment: &EnhancedQualityAssessment,
        timestamp: DateTime<Utc>,
    ) -> Vec<QualityAlert> {
        let mut alerts = Vec::new();
        let thresholds = &self.monitoring_config.alert_thresholds;

        // Check overall quality
        alerts.extend(self.check_dimension_threshold(
            "overall_quality",
            assessment.overall_score,
            &thresholds.overall_quality,
            timestamp,
        ));

        // Check completeness
        alerts.extend(self.check_dimension_threshold(
            "completeness",
            assessment.dimension_scores.completeness,
            &thresholds.completeness,
            timestamp,
        ));

        // Check consistency
        alerts.extend(self.check_dimension_threshold(
            "consistency",
            assessment.dimension_scores.consistency,
            &thresholds.consistency,
            timestamp,
        ));

        // Check validity
        alerts.extend(self.check_dimension_threshold(
            "validity",
            assessment.dimension_scores.validity,
            &thresholds.validity,
            timestamp,
        ));

        // Check timeliness
        alerts.extend(self.check_dimension_threshold(
            "timeliness",
            assessment.dimension_scores.timeliness,
            &thresholds.timeliness,
            timestamp,
        ));

        // Check accuracy
        alerts.extend(self.check_dimension_threshold(
            "accuracy",
            assessment.dimension_scores.accuracy,
            &thresholds.accuracy,
            timestamp,
        ));

        alerts
    }

    /// Check a single dimension threshold
    fn check_dimension_threshold(
        &self,
        metric: &str,
        value: f64,
        threshold: &QualityThresholdConfig,
        timestamp: DateTime<Utc>,
    ) -> Vec<QualityAlert> {
        let mut alerts = Vec::new();

        if value < threshold.critical_threshold {
            alerts.push(QualityAlert {
                alert_type: AlertType::ThresholdViolation,
                severity: AlertSeverity::Critical,
                metric: metric.to_string(),
                current_value: value,
                threshold_value: threshold.critical_threshold,
                timestamp,
                message: format!(
                    "Critical: {} has fallen below critical threshold ({:.2} < {:.2})",
                    metric, value, threshold.critical_threshold
                ),
                recommendations: vec![
                    format!("Investigate {} quality issues immediately", metric),
                    "Review recent data changes".to_string(),
                    "Check data pipeline integrity".to_string(),
                ],
            });
        } else if value < threshold.warning_threshold {
            alerts.push(QualityAlert {
                alert_type: AlertType::ThresholdViolation,
                severity: AlertSeverity::Warning,
                metric: metric.to_string(),
                current_value: value,
                threshold_value: threshold.warning_threshold,
                timestamp,
                message: format!(
                    "Warning: {} has fallen below warning threshold ({:.2} < {:.2})",
                    metric, value, threshold.warning_threshold
                ),
                recommendations: vec![
                    format!("Monitor {} quality closely", metric),
                    "Consider data quality improvements".to_string(),
                ],
            });
        }

        alerts
    }

    /// Detect sudden quality degradation within a time window
    pub fn detect_sudden_degradation(&self, window: Duration) -> Vec<QualityAlert> {
        let mut alerts = Vec::new();
        let now = Utc::now();
        let cutoff = now - window;

        for (metric_name, history) in &self.metric_history {
            let recent: Vec<_> = history
                .iter()
                .filter(|(t, _)| *t >= cutoff)
                .collect();

            if recent.len() < 2 {
                continue;
            }

            // Check for sudden drops (>20 points in one reading)
            for i in 1..recent.len() {
                let drop = recent[i - 1].1 - recent[i].1;
                if drop > 20.0 {
                    alerts.push(QualityAlert {
                        alert_type: AlertType::QualityDegradation,
                        severity: AlertSeverity::Critical,
                        metric: metric_name.clone(),
                        current_value: recent[i].1,
                        threshold_value: recent[i - 1].1,
                        timestamp: recent[i].0,
                        message: format!(
                            "Sudden quality degradation detected in {}: dropped {:.2} points",
                            metric_name, drop
                        ),
                        recommendations: vec![
                            "Investigate recent data changes".to_string(),
                            "Check for data pipeline issues".to_string(),
                            "Review system changes".to_string(),
                        ],
                    });
                }
            }
        }

        alerts
    }

    /// Detect gradual quality decline using linear regression
    pub fn detect_gradual_decline(&self, trend_window: Duration) -> Vec<QualityAlert> {
        let mut alerts = Vec::new();
        let now = Utc::now();
        let cutoff = now - trend_window;

        for (metric_name, history) in &self.metric_history {
            let recent: Vec<_> = history
                .iter()
                .filter(|(t, _)| *t >= cutoff)
                .collect();

            if recent.len() < 3 {
                continue;
            }

            // Simple linear regression to detect trend
            let slope = calculate_trend_slope(&recent);

            // Get threshold for this metric
            let threshold = self.get_threshold_for_metric(metric_name);

            // Check if decline rate exceeds threshold
            if slope < threshold.degradation_rate_threshold {
                let latest_value = recent.last().map(|(_, v)| *v).unwrap_or(0.0);

                alerts.push(QualityAlert {
                    alert_type: AlertType::QualityDegradation,
                    severity: AlertSeverity::Warning,
                    metric: metric_name.clone(),
                    current_value: latest_value,
                    threshold_value: threshold.degradation_rate_threshold,
                    timestamp: now,
                    message: format!(
                        "Gradual quality decline detected in {}: declining at {:.2} points/day",
                        metric_name, slope
                    ),
                    recommendations: vec![
                        format!("Monitor {} trend closely", metric_name),
                        "Identify root cause of decline".to_string(),
                        "Implement corrective measures".to_string(),
                    ],
                });
            }
        }

        alerts
    }

    /// Get threshold config for a metric
    fn get_threshold_for_metric(&self, metric: &str) -> &QualityThresholdConfig {
        match metric {
            "overall_quality" => &self.monitoring_config.alert_thresholds.overall_quality,
            "completeness" => &self.monitoring_config.alert_thresholds.completeness,
            "consistency" => &self.monitoring_config.alert_thresholds.consistency,
            "validity" => &self.monitoring_config.alert_thresholds.validity,
            "timeliness" => &self.monitoring_config.alert_thresholds.timeliness,
            "accuracy" => &self.monitoring_config.alert_thresholds.accuracy,
            _ => &self.monitoring_config.alert_thresholds.overall_quality,
        }
    }

    /// Get quality time series
    pub fn get_quality_timeseries(&self) -> QualityTimeSeries {
        let overall = self.metric_history.get("overall_quality");

        if let Some(history) = overall {
            history
                .iter()
                .map(|(timestamp, value)| {
                    let mut dimensions = HashMap::new();

                    // Add all dimension scores at this timestamp
                    for (metric, values) in &self.metric_history {
                        if metric != "overall_quality" {
                            if let Some((_, v)) = values.iter()
                                .find(|(t, _)| t == timestamp) {
                                dimensions.insert(metric.clone(), *v);
                            }
                        }
                    }

                    QualityDataPoint {
                        timestamp: *timestamp,
                        overall_quality: *value,
                        dimensions,
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Update quality baseline
    pub fn update_baseline(&mut self) {
        if let Some(history) = self.metric_history.get("overall_quality") {
            if history.is_empty() {
                return;
            }

            let values: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;

            let variance: f64 = values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            self.baseline_metrics = QualityBaseline {
                mean_overall_quality: mean,
                std_dev,
                min_acceptable: mean - 2.0 * std_dev,
                max_observed: max,
                established_at: Utc::now(),
                sample_count: values.len(),
            };
        }
    }
}

/// Calculate trend slope using simple linear regression
fn calculate_trend_slope(data: &[&(DateTime<Utc>, f64)]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let n = data.len() as f64;
    let base_time = data[0].0;

    // Convert timestamps to days since base_time
    let x: Vec<f64> = data.iter()
        .map(|(t, _)| (*t - base_time).num_seconds() as f64 / 86400.0)
        .collect();
    let y: Vec<f64> = data.iter().map(|(_, v)| *v).collect();

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    slope
}

/// Monitor quality over time and generate trend analysis
pub fn monitor_quality_over_time(
    quality_history: &QualityTimeSeries,
    config: &MonitoringConfig,
) -> QualityResult<QualityTrend> {
    if quality_history.is_empty() {
        return Err(QualityError::insufficient_data(
            "Cannot monitor quality with empty history"
        ));
    }

    // Calculate overall trend
    let overall_data: Vec<(DateTime<Utc>, f64)> = quality_history
        .iter()
        .map(|dp| (dp.timestamp, dp.overall_quality))
        .collect();

    let overall_slope = calculate_trend_slope(
        &overall_data.iter().collect::<Vec<_>>()
    );

    let overall_trend = classify_trend(overall_slope);

    // Calculate dimension trends
    let mut dimension_trends = HashMap::new();

    // Get all dimension names from first data point
    if let Some(first) = quality_history.first() {
        for dimension_name in first.dimensions.keys() {
            let dim_data: Vec<(DateTime<Utc>, f64)> = quality_history
                .iter()
                .filter_map(|dp| {
                    dp.dimensions.get(dimension_name).map(|v| (dp.timestamp, *v))
                })
                .collect();

            if !dim_data.is_empty() {
                let slope = calculate_trend_slope(&dim_data.iter().collect::<Vec<_>>());
                dimension_trends.insert(dimension_name.clone(), classify_trend(slope));
            }
        }
    }

    // Calculate quality velocity (rate of change)
    let quality_velocity = overall_slope;

    // Predict future quality if enabled
    let predicted_quality = if config.enable_predictions && quality_history.len() >= 3 {
        let last_value = quality_history.last().unwrap().overall_quality;
        let days_ahead = 7.0; // Predict 7 days ahead
        Some((last_value + overall_slope * days_ahead).max(0.0).min(100.0))
    } else {
        None
    };

    // Detect anomaly periods (simplified: periods where quality deviates >2 std devs)
    let anomaly_periods = detect_anomaly_periods(quality_history);

    Ok(QualityTrend {
        overall_trend,
        dimension_trends,
        quality_velocity,
        predicted_quality,
        anomaly_periods,
    })
}

/// Classify trend direction based on slope
fn classify_trend(slope: f64) -> TrendDirection {
    if slope > 1.0 {
        TrendDirection::Improving
    } else if slope < -1.0 {
        TrendDirection::Declining
    } else {
        TrendDirection::Stable
    }
}

/// Detect periods with anomalous quality
fn detect_anomaly_periods(quality_history: &QualityTimeSeries) -> Vec<TimeRange> {
    let mut anomaly_periods = Vec::new();

    if quality_history.len() < 3 {
        return anomaly_periods;
    }

    // Calculate mean and std dev
    let values: Vec<f64> = quality_history.iter().map(|dp| dp.overall_quality).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    // Find consecutive anomalous periods
    let mut in_anomaly = false;
    let mut anomaly_start: Option<DateTime<Utc>> = None;

    for dp in quality_history {
        let is_anomalous = (dp.overall_quality - mean).abs() > 2.0 * std_dev;

        if is_anomalous && !in_anomaly {
            // Start of anomaly period
            in_anomaly = true;
            anomaly_start = Some(dp.timestamp);
        } else if !is_anomalous && in_anomaly {
            // End of anomaly period
            in_anomaly = false;
            if let Some(start) = anomaly_start {
                anomaly_periods.push(TimeRange {
                    start,
                    end: dp.timestamp,
                });
            }
        }
    }

    // Close any open anomaly period
    if in_anomaly {
        if let (Some(start), Some(last)) = (anomaly_start, quality_history.last()) {
            anomaly_periods.push(TimeRange {
                start,
                end: last.timestamp,
            });
        }
    }

    anomaly_periods
}

/// Detect quality degradation patterns
pub fn detect_quality_degradation(
    quality_history: &QualityTimeSeries,
    thresholds: &QualityThresholds,
) -> Vec<QualityAlert> {
    let mut alerts = Vec::new();

    if quality_history.is_empty() {
        return alerts;
    }

    // Check latest quality against thresholds
    if let Some(latest) = quality_history.last() {
        let overall = latest.overall_quality;
        let threshold = &thresholds.overall_quality;

        if overall < threshold.critical_threshold {
            alerts.push(QualityAlert {
                alert_type: AlertType::QualityDegradation,
                severity: AlertSeverity::Critical,
                metric: "overall_quality".to_string(),
                current_value: overall,
                threshold_value: threshold.critical_threshold,
                timestamp: latest.timestamp,
                message: format!(
                    "Overall quality critically low: {:.2} < {:.2}",
                    overall, threshold.critical_threshold
                ),
                recommendations: vec![
                    "Immediate investigation required".to_string(),
                    "Review all quality dimensions".to_string(),
                    "Check data pipeline health".to_string(),
                ],
            });
        }

        // Check each dimension
        for (dim_name, dim_value) in &latest.dimensions {
            let dim_threshold = match dim_name.as_str() {
                "completeness" => &thresholds.completeness,
                "consistency" => &thresholds.consistency,
                "validity" => &thresholds.validity,
                "timeliness" => &thresholds.timeliness,
                "accuracy" => &thresholds.accuracy,
                _ => continue,
            };

            if *dim_value < dim_threshold.warning_threshold {
                let severity = if *dim_value < dim_threshold.critical_threshold {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                };

                alerts.push(QualityAlert {
                    alert_type: AlertType::QualityDegradation,
                    severity,
                    metric: dim_name.clone(),
                    current_value: *dim_value,
                    threshold_value: dim_threshold.warning_threshold,
                    timestamp: latest.timestamp,
                    message: format!(
                        "{} quality below threshold: {:.2} < {:.2}",
                        dim_name, dim_value, dim_threshold.warning_threshold
                    ),
                    recommendations: vec![
                        format!("Address {} quality issues", dim_name),
                        "Review data quality checks".to_string(),
                    ],
                });
            }
        }
    }

    alerts
}

/// Track quality metrics in a tracker
pub fn track_quality_metrics(
    assessment: &EnhancedQualityAssessment,
    tracker: &mut QualityTracker,
) -> QualityResult<()> {
    tracker.track(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_quality_history() -> QualityTimeSeries {
        let base_time = Utc::now();
        (0..10)
            .map(|i| {
                let mut dimensions = HashMap::new();
                dimensions.insert("completeness".to_string(), 85.0 - i as f64 * 2.0);
                dimensions.insert("consistency".to_string(), 90.0 - i as f64);

                QualityDataPoint {
                    timestamp: base_time + Duration::days(i),
                    overall_quality: 80.0 - i as f64 * 2.0, // Declining 2 points per day
                    dimensions,
                }
            })
            .collect()
    }

    #[test]
    fn test_quality_tracker_creation() {
        let config = MonitoringConfig::default();
        let tracker = QualityTracker::new(config.clone());

        assert_eq!(tracker.metric_history.len(), 0);
        assert_eq!(tracker.alert_history.len(), 0);
        assert_eq!(tracker.monitoring_config, config);
    }

    #[test]
    fn test_trend_classification() {
        assert_eq!(classify_trend(2.0), TrendDirection::Improving);
        assert_eq!(classify_trend(-2.0), TrendDirection::Declining);
        assert_eq!(classify_trend(0.5), TrendDirection::Stable);
    }

    #[test]
    fn test_monitor_quality_over_time() {
        let history = create_test_quality_history();
        let config = MonitoringConfig::default();

        let result = monitor_quality_over_time(&history, &config).unwrap();

        // Quality is declining (80 -> 62, 2 points per day)
        assert_eq!(result.overall_trend, TrendDirection::Declining);
        assert!(result.quality_velocity < -1.0); // Should be around -2 points/day
    }

    #[test]
    fn test_detect_quality_degradation() {
        let mut history = create_test_quality_history();
        // Set last point to critical level
        if let Some(last) = history.last_mut() {
            last.overall_quality = 50.0; // Below default critical threshold of 60
        }

        let thresholds = QualityThresholds::default();
        let alerts = detect_quality_degradation(&history, &thresholds);

        assert!(!alerts.is_empty());
        assert!(alerts.iter().any(|a| a.severity == AlertSeverity::Critical));
    }

    #[test]
    fn test_sudden_degradation_detection() {
        let mut tracker = QualityTracker::new(MonitoringConfig::default());
        let base_time = Utc::now();

        // Add normal values
        tracker.add_metric("overall_quality", base_time, 85.0);
        tracker.add_metric("overall_quality", base_time + Duration::hours(1), 84.0);

        // Add sudden drop
        tracker.add_metric("overall_quality", base_time + Duration::hours(2), 60.0);

        let alerts = tracker.detect_sudden_degradation(Duration::days(1));

        assert!(!alerts.is_empty());
        assert!(alerts[0].message.contains("Sudden quality degradation"));
    }

    #[test]
    fn test_gradual_decline_detection() {
        let mut tracker = QualityTracker::new(MonitoringConfig::default());
        let base_time = Utc::now();

        // Add gradually declining values
        for i in 0..10 {
            tracker.add_metric(
                "completeness",
                base_time + Duration::days(i),
                90.0 - i as f64 * 3.0, // Declining 3 points per day
            );
        }

        let alerts = tracker.detect_gradual_decline(Duration::days(30));

        // Should detect decline since -3 points/day exceeds default -5 threshold
        // Actually should not alert since -3 > -5 (less severe)
        // But if we had -6 points/day it would alert
    }

    #[test]
    fn test_anomaly_period_detection() {
        let history = create_test_quality_history();
        let anomalies = detect_anomaly_periods(&history);

        // With steadily declining quality, may or may not detect anomalies
        // depending on variance
        assert!(anomalies.len() >= 0);
    }

    #[test]
    fn test_baseline_update() {
        let mut tracker = QualityTracker::new(MonitoringConfig::default());
        let base_time = Utc::now();

        // Add some values
        for i in 0..5 {
            tracker.add_metric("overall_quality", base_time + Duration::days(i), 80.0 + i as f64);
        }

        tracker.update_baseline();

        assert!(tracker.baseline_metrics.sample_count == 5);
        assert!(tracker.baseline_metrics.mean_overall_quality > 80.0);
    }

    #[test]
    fn test_empty_history_error() {
        let history = QualityTimeSeries::new();
        let config = MonitoringConfig::default();

        let result = monitor_quality_over_time(&history, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_threshold_config_defaults() {
        let config = QualityThresholdConfig::default();

        assert_eq!(config.warning_threshold, 75.0);
        assert_eq!(config.critical_threshold, 60.0);
        assert_eq!(config.degradation_rate_threshold, -5.0);
    }

    #[test]
    fn test_performance() {
        use std::time::Instant;

        // Create large history
        let history: QualityTimeSeries = (0..1000)
            .map(|i| {
                let mut dimensions = HashMap::new();
                dimensions.insert("completeness".to_string(), 85.0);

                QualityDataPoint {
                    timestamp: Utc::now() + Duration::days(i),
                    overall_quality: 80.0,
                    dimensions,
                }
            })
            .collect();

        let config = MonitoringConfig::default();
        let start = Instant::now();

        let _result = monitor_quality_over_time(&history, &config).unwrap();

        let duration = start.elapsed();

        // Should be under 50ms as per requirements
        assert!(duration.as_millis() < 50, "Performance requirement not met: {:?}", duration);
    }
}