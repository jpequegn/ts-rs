//! Core types and structures for data quality assessment

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Time range representing a gap or issue period
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start timestamp of the range
    pub start: DateTime<Utc>,
    /// End timestamp of the range
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Creates a new TimeRange
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        TimeRange { start, end }
    }

    /// Returns the duration of the time range in seconds
    pub fn duration_seconds(&self) -> i64 {
        (self.end - self.start).num_seconds()
    }
}

/// Methods for outlier detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutlierMethod {
    /// Z-score method (standard deviations from mean)
    ZScore,
    /// Interquartile range method
    IQR,
    /// Modified Z-score using median absolute deviation
    ModifiedZScore,
    /// Isolation forest algorithm
    IsolationForest,
    /// Local outlier factor
    LOF,
    /// DBSCAN clustering
    DBSCAN,
}

impl OutlierMethod {
    /// Returns a human-readable description of the method
    pub fn description(&self) -> &'static str {
        match self {
            OutlierMethod::ZScore => "Z-score based on standard deviations from mean",
            OutlierMethod::IQR => "Interquartile range method",
            OutlierMethod::ModifiedZScore => "Modified Z-score using median absolute deviation",
            OutlierMethod::IsolationForest => "Isolation forest algorithm",
            OutlierMethod::LOF => "Local outlier factor",
            OutlierMethod::DBSCAN => "DBSCAN density-based clustering",
        }
    }

    /// Returns the short name of the method
    pub fn as_str(&self) -> &'static str {
        match self {
            OutlierMethod::ZScore => "zscore",
            OutlierMethod::IQR => "iqr",
            OutlierMethod::ModifiedZScore => "modified_zscore",
            OutlierMethod::IsolationForest => "isolation_forest",
            OutlierMethod::LOF => "lof",
            OutlierMethod::DBSCAN => "dbscan",
        }
    }
}

impl std::fmt::Display for OutlierMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Represents a data quality issue detected in the time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityIssue {
    /// Missing data detected
    MissingData {
        /// Percentage of missing data
        percentage: f64,
        /// Gaps in the time series
        gaps: Vec<TimeRange>,
    },
    /// Outlier detected
    OutlierDetected {
        /// Method used to detect the outlier
        method: OutlierMethod,
        /// Value of the outlier
        value: f64,
        /// Timestamp of the outlier
        timestamp: DateTime<Utc>,
        /// Severity score (0.0-1.0)
        severity: f64,
    },
    /// Consistency violation detected
    ConsistencyViolation {
        /// Description of the violation
        description: String,
        /// Affected timestamps
        timestamps: Vec<DateTime<Utc>>,
    },
    /// Duplicate timestamps detected
    DuplicateTimestamps {
        /// Number of duplicates
        count: usize,
        /// Affected timestamps
        timestamps: Vec<DateTime<Utc>>,
    },
    /// Unexpected frequency change
    FrequencyAnomaly {
        /// Expected frequency in seconds
        expected_frequency: f64,
        /// Actual frequency in seconds
        actual_frequency: f64,
        /// Timestamp where anomaly occurred
        timestamp: DateTime<Utc>,
    },
    /// Value out of expected range
    RangeViolation {
        /// Value that violated the range
        value: f64,
        /// Expected minimum value
        expected_min: f64,
        /// Expected maximum value
        expected_max: f64,
        /// Timestamp of violation
        timestamp: DateTime<Utc>,
    },
}

impl QualityIssue {
    /// Returns the severity of the quality issue (0.0-1.0)
    pub fn severity(&self) -> f64 {
        match self {
            QualityIssue::MissingData { percentage, .. } => percentage / 100.0,
            QualityIssue::OutlierDetected { severity, .. } => *severity,
            QualityIssue::ConsistencyViolation { .. } => 0.7,
            QualityIssue::DuplicateTimestamps { .. } => 0.8,
            QualityIssue::FrequencyAnomaly { .. } => 0.6,
            QualityIssue::RangeViolation { .. } => 0.5,
        }
    }

    /// Returns a human-readable description of the issue
    pub fn description(&self) -> String {
        match self {
            QualityIssue::MissingData { percentage, gaps } => {
                format!(
                    "Missing data: {:.2}% ({} gaps)",
                    percentage,
                    gaps.len()
                )
            }
            QualityIssue::OutlierDetected { method, value, timestamp, .. } => {
                format!(
                    "Outlier detected ({}): value={:.4} at {}",
                    method, value, timestamp
                )
            }
            QualityIssue::ConsistencyViolation { description, timestamps } => {
                format!(
                    "Consistency violation: {} ({} occurrences)",
                    description,
                    timestamps.len()
                )
            }
            QualityIssue::DuplicateTimestamps { count, .. } => {
                format!("Duplicate timestamps: {} duplicates found", count)
            }
            QualityIssue::FrequencyAnomaly { expected_frequency, actual_frequency, timestamp } => {
                format!(
                    "Frequency anomaly: expected {:.2}s, got {:.2}s at {}",
                    expected_frequency, actual_frequency, timestamp
                )
            }
            QualityIssue::RangeViolation { value, expected_min, expected_max, timestamp } => {
                format!(
                    "Range violation: value={:.4} outside [{:.4}, {:.4}] at {}",
                    value, expected_min, expected_max, timestamp
                )
            }
        }
    }

    /// Returns the category of the quality issue
    pub fn category(&self) -> &'static str {
        match self {
            QualityIssue::MissingData { .. } => "completeness",
            QualityIssue::OutlierDetected { .. } => "validity",
            QualityIssue::ConsistencyViolation { .. } => "consistency",
            QualityIssue::DuplicateTimestamps { .. } => "consistency",
            QualityIssue::FrequencyAnomaly { .. } => "timeliness",
            QualityIssue::RangeViolation { .. } => "validity",
        }
    }
}

/// Detailed metrics for quality assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Total number of data points
    pub total_points: usize,
    /// Number of missing data points
    pub missing_points: usize,
    /// Number of outliers detected
    pub outlier_count: usize,
    /// Number of consistency violations
    pub consistency_violations: usize,
    /// Number of duplicate timestamps
    pub duplicate_timestamps: usize,
    /// Number of frequency anomalies
    pub frequency_anomalies: usize,
    /// Number of range violations
    pub range_violations: usize,
    /// Average data quality score (0.0-1.0)
    pub average_quality: f64,
    /// Standard deviation of values
    pub std_dev: f64,
    /// Mean of values
    pub mean: f64,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        QualityMetrics {
            total_points: 0,
            missing_points: 0,
            outlier_count: 0,
            consistency_violations: 0,
            duplicate_timestamps: 0,
            frequency_anomalies: 0,
            range_violations: 0,
            average_quality: 0.0,
            std_dev: 0.0,
            mean: 0.0,
            custom_metrics: HashMap::new(),
        }
    }
}

impl QualityMetrics {
    /// Creates a new QualityMetrics with basic values
    pub fn new(total_points: usize) -> Self {
        QualityMetrics {
            total_points,
            ..Default::default()
        }
    }

    /// Calculates the completeness score (0.0-1.0)
    pub fn completeness_score(&self) -> f64 {
        if self.total_points == 0 {
            return 1.0;
        }
        1.0 - (self.missing_points as f64 / self.total_points as f64)
    }

    /// Calculates the validity score (0.0-1.0)
    pub fn validity_score(&self) -> f64 {
        if self.total_points == 0 {
            return 1.0;
        }
        let invalid = self.outlier_count + self.range_violations;
        1.0 - (invalid as f64 / self.total_points as f64).min(1.0)
    }

    /// Calculates the consistency score (0.0-1.0)
    pub fn consistency_score(&self) -> f64 {
        if self.total_points == 0 {
            return 1.0;
        }
        let inconsistent = self.consistency_violations + self.duplicate_timestamps;
        1.0 - (inconsistent as f64 / self.total_points as f64).min(1.0)
    }

    /// Calculates the timeliness score (0.0-1.0)
    pub fn timeliness_score(&self) -> f64 {
        if self.total_points == 0 {
            return 1.0;
        }
        1.0 - (self.frequency_anomalies as f64 / self.total_points as f64).min(1.0)
    }

    /// Calculates the overall quality score (0.0-1.0)
    pub fn overall_score(&self) -> f64 {
        (self.completeness_score()
            + self.validity_score()
            + self.consistency_score()
            + self.timeliness_score()) / 4.0
    }
}

/// Result of a comprehensive quality assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0-1.0)
    pub overall_score: f64,
    /// Completeness score (0.0-1.0)
    pub completeness_score: f64,
    /// Consistency score (0.0-1.0)
    pub consistency_score: f64,
    /// Validity score (0.0-1.0)
    pub validity_score: f64,
    /// Timeliness score (0.0-1.0)
    pub timeliness_score: f64,
    /// Detailed quality metrics
    pub metrics: QualityMetrics,
    /// List of detected quality issues
    pub issues: Vec<QualityIssue>,
    /// Timestamp when assessment was performed
    pub assessment_timestamp: DateTime<Utc>,
}

impl QualityAssessment {
    /// Creates a new quality assessment from metrics and issues
    pub fn new(metrics: QualityMetrics, issues: Vec<QualityIssue>) -> Self {
        let completeness_score = metrics.completeness_score();
        let consistency_score = metrics.consistency_score();
        let validity_score = metrics.validity_score();
        let timeliness_score = metrics.timeliness_score();
        let overall_score = metrics.overall_score();

        QualityAssessment {
            overall_score,
            completeness_score,
            consistency_score,
            validity_score,
            timeliness_score,
            metrics,
            issues,
            assessment_timestamp: Utc::now(),
        }
    }

    /// Returns the number of critical issues (severity >= 0.7)
    pub fn critical_issue_count(&self) -> usize {
        self.issues.iter().filter(|issue| issue.severity() >= 0.7).count()
    }

    /// Returns issues grouped by category
    pub fn issues_by_category(&self) -> HashMap<String, Vec<&QualityIssue>> {
        let mut grouped: HashMap<String, Vec<&QualityIssue>> = HashMap::new();
        for issue in &self.issues {
            grouped.entry(issue.category().to_string())
                .or_insert_with(Vec::new)
                .push(issue);
        }
        grouped
    }

    /// Returns true if the quality assessment passes the given threshold
    pub fn passes_threshold(&self, threshold: f64) -> bool {
        self.overall_score >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_time_range_duration() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap();
        let range = TimeRange::new(start, end);
        assert_eq!(range.duration_seconds(), 3600);
    }

    #[test]
    fn test_outlier_method_display() {
        assert_eq!(format!("{}", OutlierMethod::ZScore), "zscore");
        assert_eq!(format!("{}", OutlierMethod::IQR), "iqr");
    }

    #[test]
    fn test_quality_issue_severity() {
        let issue = QualityIssue::MissingData {
            percentage: 50.0,
            gaps: vec![],
        };
        assert_eq!(issue.severity(), 0.5);

        let timestamp = Utc::now();
        let outlier = QualityIssue::OutlierDetected {
            method: OutlierMethod::ZScore,
            value: 100.0,
            timestamp,
            severity: 0.9,
        };
        assert_eq!(outlier.severity(), 0.9);
    }

    #[test]
    fn test_quality_issue_category() {
        let missing = QualityIssue::MissingData {
            percentage: 10.0,
            gaps: vec![],
        };
        assert_eq!(missing.category(), "completeness");

        let timestamp = Utc::now();
        let outlier = QualityIssue::OutlierDetected {
            method: OutlierMethod::ZScore,
            value: 100.0,
            timestamp,
            severity: 0.9,
        };
        assert_eq!(outlier.category(), "validity");
    }

    #[test]
    fn test_quality_metrics_scores() {
        let mut metrics = QualityMetrics::new(100);
        metrics.missing_points = 10;
        metrics.outlier_count = 5;
        metrics.consistency_violations = 3;
        metrics.frequency_anomalies = 2;

        assert_eq!(metrics.completeness_score(), 0.9);
        assert_eq!(metrics.validity_score(), 0.95);
        assert_eq!(metrics.consistency_score(), 0.97);
        assert_eq!(metrics.timeliness_score(), 0.98);

        let overall = (0.9 + 0.95 + 0.97 + 0.98) / 4.0;
        assert!((metrics.overall_score() - overall).abs() < 0.001);
    }

    #[test]
    fn test_quality_assessment_creation() {
        let metrics = QualityMetrics::new(100);
        let issues = vec![
            QualityIssue::MissingData {
                percentage: 10.0,
                gaps: vec![],
            },
        ];

        let assessment = QualityAssessment::new(metrics, issues);
        assert_eq!(assessment.issues.len(), 1);
        assert!(assessment.overall_score >= 0.0 && assessment.overall_score <= 1.0);
    }

    #[test]
    fn test_quality_assessment_critical_issues() {
        let metrics = QualityMetrics::new(100);
        let timestamp = Utc::now();
        let issues = vec![
            QualityIssue::OutlierDetected {
                method: OutlierMethod::ZScore,
                value: 100.0,
                timestamp,
                severity: 0.5,
            },
            QualityIssue::OutlierDetected {
                method: OutlierMethod::IQR,
                value: 200.0,
                timestamp,
                severity: 0.9,
            },
            QualityIssue::DuplicateTimestamps {
                count: 3,
                timestamps: vec![],
            },
        ];

        let assessment = QualityAssessment::new(metrics, issues);
        assert_eq!(assessment.critical_issue_count(), 2); // severity >= 0.7
    }

    #[test]
    fn test_quality_assessment_passes_threshold() {
        let mut metrics = QualityMetrics::new(100);
        metrics.missing_points = 5; // 95% completeness
        let assessment = QualityAssessment::new(metrics, vec![]);

        assert!(assessment.passes_threshold(0.8));
        assert!(!assessment.passes_threshold(0.99));
    }
}