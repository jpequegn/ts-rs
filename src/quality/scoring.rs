//! Quality scoring and assessment framework
//!
//! This module provides comprehensive quality scoring capabilities that combine
//! multiple quality dimensions into actionable quality scores and assessments.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::quality::{
    DataProfile, OutlierMethod, OutlierReport, QualityConfig, QualityError, QualityIssue,
    QualityResult,
};
use crate::timeseries::TimeSeries;
use crate::types::Frequency;

/// Quality dimension scores (0-100 scale)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionScores {
    /// Completeness score: data availability (0-100)
    pub completeness: f64,
    /// Consistency score: internal consistency (0-100)
    pub consistency: f64,
    /// Validity score: value validity (0-100)
    pub validity: f64,
    /// Timeliness score: temporal regularity (0-100)
    pub timeliness: f64,
    /// Accuracy score: outlier/error rate (0-100)
    pub accuracy: f64,
}

impl DimensionScores {
    /// Creates dimension scores with all values set to zero
    pub fn zero() -> Self {
        DimensionScores {
            completeness: 0.0,
            consistency: 0.0,
            validity: 0.0,
            timeliness: 0.0,
            accuracy: 0.0,
        }
    }

    /// Creates dimension scores with all values set to perfect (100)
    pub fn perfect() -> Self {
        DimensionScores {
            completeness: 100.0,
            consistency: 100.0,
            validity: 100.0,
            timeliness: 100.0,
            accuracy: 100.0,
        }
    }
}

/// Configurable weights for quality dimensions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityWeights {
    /// Weight for completeness dimension
    pub completeness: f64,
    /// Weight for consistency dimension
    pub consistency: f64,
    /// Weight for validity dimension
    pub validity: f64,
    /// Weight for timeliness dimension
    pub timeliness: f64,
    /// Weight for accuracy dimension
    pub accuracy: f64,
}

impl Default for QualityWeights {
    fn default() -> Self {
        QualityWeights {
            completeness: 0.25,
            consistency: 0.20,
            validity: 0.25,
            timeliness: 0.15,
            accuracy: 0.15,
        }
    }
}

impl QualityWeights {
    /// Returns the total weight
    pub fn total(&self) -> f64 {
        self.completeness + self.consistency + self.validity + self.timeliness + self.accuracy
    }

    /// Normalizes weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.total();
        if total > 0.0 {
            self.completeness /= total;
            self.consistency /= total;
            self.validity /= total;
            self.timeliness /= total;
            self.accuracy /= total;
        }
    }
}

/// Quality benchmarks for comparison
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBenchmarks {
    /// Industry percentile (if available)
    pub industry_percentile: Option<f64>,
    /// Comparison to historical data (if available)
    pub historical_comparison: Option<f64>,
    /// Comparison to peer datasets (if available)
    pub peer_comparison: Option<f64>,
}

impl Default for QualityBenchmarks {
    fn default() -> Self {
        QualityBenchmarks {
            industry_percentile: None,
            historical_comparison: None,
            peer_comparison: None,
        }
    }
}

/// Priority level for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority - minor improvements
    Low,
    /// Medium priority - should be addressed
    Medium,
    /// High priority - important to address
    High,
    /// Critical priority - must be addressed immediately
    Critical,
}

impl Priority {
    /// Creates priority from a severity score (0.0-1.0)
    pub fn from_severity(severity: f64) -> Self {
        if severity >= 0.8 {
            Priority::Critical
        } else if severity >= 0.6 {
            Priority::High
        } else if severity >= 0.3 {
            Priority::Medium
        } else {
            Priority::Low
        }
    }
}

/// Imputation method for filling gaps
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImputationMethod {
    /// Forward fill (carry last observation forward)
    ForwardFill,
    /// Backward fill (carry next observation backward)
    BackwardFill,
    /// Linear interpolation
    LinearInterpolation,
    /// Spline interpolation
    SplineInterpolation,
    /// Mean imputation
    MeanImputation,
    /// Median imputation
    MedianImputation,
    /// Seasonal imputation (use same time point from previous periods)
    SeasonalImputation,
}

/// Quality improvement recommendations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityRecommendation {
    /// Improve data collection processes
    ImproveDataCollection {
        priority: Priority,
        description: String,
    },
    /// Implement validation rules
    ImplementValidation {
        priority: Priority,
        validation_rules: Vec<String>,
    },
    /// Clean detected outliers
    CleanOutliers {
        priority: Priority,
        method: OutlierMethod,
        count: usize,
    },
    /// Fill data gaps
    FillGaps {
        priority: Priority,
        method: ImputationMethod,
        gap_count: usize,
    },
    /// Standardize data frequency
    StandardizeFrequency {
        priority: Priority,
        target_frequency: Frequency,
    },
    /// Review and correct inconsistencies
    ReviewInconsistencies {
        priority: Priority,
        inconsistency_count: usize,
    },
}

impl QualityRecommendation {
    /// Returns the priority of the recommendation
    pub fn priority(&self) -> Priority {
        match self {
            QualityRecommendation::ImproveDataCollection { priority, .. } => *priority,
            QualityRecommendation::ImplementValidation { priority, .. } => *priority,
            QualityRecommendation::CleanOutliers { priority, .. } => *priority,
            QualityRecommendation::FillGaps { priority, .. } => *priority,
            QualityRecommendation::StandardizeFrequency { priority, .. } => *priority,
            QualityRecommendation::ReviewInconsistencies { priority, .. } => *priority,
        }
    }

    /// Returns a description of the recommendation
    pub fn description(&self) -> String {
        match self {
            QualityRecommendation::ImproveDataCollection { description, .. } => {
                description.clone()
            }
            QualityRecommendation::ImplementValidation { validation_rules, .. } => {
                format!("Implement {} validation rules", validation_rules.len())
            }
            QualityRecommendation::CleanOutliers { method, count, .. } => {
                format!("Clean {} outliers using {} method", count, method)
            }
            QualityRecommendation::FillGaps { method, gap_count, .. } => {
                format!("Fill {} gaps using {:?}", gap_count, method)
            }
            QualityRecommendation::StandardizeFrequency { target_frequency, .. } => {
                format!("Standardize frequency to {:?}", target_frequency)
            }
            QualityRecommendation::ReviewInconsistencies { inconsistency_count, .. } => {
                format!("Review {} inconsistencies", inconsistency_count)
            }
        }
    }
}

/// Quality profile preset configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityProfile {
    /// Lenient standards for exploratory analysis
    Exploratory,
    /// Balanced standards for production systems
    Production,
    /// Strict standards for regulatory compliance
    Regulatory,
    /// Optimized for real-time streaming data
    RealTime,
}

impl QualityProfile {
    /// Returns the quality weights for this profile
    pub fn get_weights(&self) -> QualityWeights {
        match self {
            QualityProfile::Exploratory => QualityWeights {
                completeness: 0.30,
                consistency: 0.15,
                validity: 0.20,
                timeliness: 0.10,
                accuracy: 0.25,
            },
            QualityProfile::Production => QualityWeights {
                completeness: 0.25,
                consistency: 0.25,
                validity: 0.25,
                timeliness: 0.15,
                accuracy: 0.10,
            },
            QualityProfile::Regulatory => QualityWeights {
                completeness: 0.30,
                consistency: 0.25,
                validity: 0.30,
                timeliness: 0.10,
                accuracy: 0.05,
            },
            QualityProfile::RealTime => QualityWeights {
                completeness: 0.20,
                consistency: 0.15,
                validity: 0.20,
                timeliness: 0.35,
                accuracy: 0.10,
            },
        }
    }

    /// Returns the quality threshold for this profile
    pub fn get_threshold(&self) -> f64 {
        match self {
            QualityProfile::Exploratory => 60.0,
            QualityProfile::Production => 75.0,
            QualityProfile::Regulatory => 90.0,
            QualityProfile::RealTime => 70.0,
        }
    }
}

/// Enhanced quality assessment with scoring framework
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnhancedQualityAssessment {
    /// Overall quality score (0-100)
    pub overall_score: f64,
    /// Dimension-specific scores
    pub dimension_scores: DimensionScores,
    /// Detected quality issues
    pub quality_issues: Vec<QualityIssue>,
    /// Actionable recommendations
    pub recommendations: Vec<QualityRecommendation>,
    /// Quality benchmarks
    pub benchmarks: QualityBenchmarks,
    /// Timestamp when assessment was performed
    pub assessment_timestamp: chrono::DateTime<Utc>,
    /// Configuration used for assessment
    pub config_summary: HashMap<String, String>,
}

impl EnhancedQualityAssessment {
    /// Creates a new enhanced quality assessment
    pub fn new(
        overall_score: f64,
        dimension_scores: DimensionScores,
        quality_issues: Vec<QualityIssue>,
        recommendations: Vec<QualityRecommendation>,
    ) -> Self {
        EnhancedQualityAssessment {
            overall_score,
            dimension_scores,
            quality_issues,
            recommendations,
            benchmarks: QualityBenchmarks::default(),
            assessment_timestamp: Utc::now(),
            config_summary: HashMap::new(),
        }
    }

    /// Adds benchmark information
    pub fn with_benchmarks(mut self, benchmarks: QualityBenchmarks) -> Self {
        self.benchmarks = benchmarks;
        self
    }

    /// Adds configuration summary
    pub fn with_config_summary(mut self, summary: HashMap<String, String>) -> Self {
        self.config_summary = summary;
        self
    }

    /// Returns true if the quality assessment passes the given threshold
    pub fn passes_threshold(&self, threshold: f64) -> bool {
        self.overall_score >= threshold
    }

    /// Returns recommendations filtered by minimum priority
    pub fn recommendations_by_priority(&self, min_priority: Priority) -> Vec<&QualityRecommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority() >= min_priority)
            .collect()
    }

    /// Returns the quality grade based on score
    pub fn quality_grade(&self) -> &'static str {
        if self.overall_score >= 90.0 {
            "Excellent"
        } else if self.overall_score >= 80.0 {
            "Good"
        } else if self.overall_score >= 70.0 {
            "Fair"
        } else if self.overall_score >= 60.0 {
            "Poor"
        } else {
            "Critical"
        }
    }
}

/// Calculates completeness score (0-100)
pub fn calculate_completeness_score(profile: &DataProfile, _config: &QualityConfig) -> f64 {
    let completeness_report = &profile.completeness;
    let coverage_percentage = completeness_report.completeness_ratio * 100.0;

    // Perfect: 100 - No missing data
    // Excellent: 95-99 - Minimal gaps (<5%)
    // Good: 85-94 - Small gaps (5-15%)
    // Fair: 70-84 - Moderate gaps (15-30%)
    // Poor: <70 - Significant gaps (>30%)

    if coverage_percentage >= 100.0 {
        100.0
    } else if coverage_percentage >= 95.0 {
        95.0 + (coverage_percentage - 95.0)
    } else if coverage_percentage >= 85.0 {
        85.0 + (coverage_percentage - 85.0)
    } else if coverage_percentage >= 70.0 {
        70.0 + (coverage_percentage - 70.0) * 0.8
    } else {
        coverage_percentage * 0.7
    }
}

/// Calculates consistency score (0-100)
pub fn calculate_consistency_score(profile: &DataProfile, outliers: &OutlierReport) -> f64 {
    let temporal_coverage = &profile.temporal_coverage;
    let total_points = profile.basic_stats.count as f64;

    if total_points == 0.0 {
        return 100.0;
    }

    // Temporal consistency (50% weight)
    let temporal_consistency = if temporal_coverage.is_regular {
        100.0
    } else {
        let gap_penalty = (profile.completeness.gaps.len() as f64 / total_points) * 100.0;
        (100.0 - gap_penalty).max(0.0_f64)
    };

    // Value consistency (30% weight) - based on outlier rate
    let outlier_rate = outliers.summary.outlier_percentage / 100.0;
    let value_consistency = (100.0 - (outlier_rate * 200.0)).max(0.0);

    // Pattern consistency (20% weight) - based on quality indicators
    let pattern_consistency = profile.quality_indicators.overall_quality * 100.0;

    temporal_consistency * 0.5 + value_consistency * 0.3 + pattern_consistency * 0.2
}

/// Calculates validity score (0-100)
pub fn calculate_validity_score(data: &TimeSeries, profile: &DataProfile) -> f64 {
    let total_points = data.values.len() as f64;

    if total_points == 0.0 {
        return 100.0;
    }

    // Range validation (40% weight)
    let value_range = &profile.statistical_profile.value_range;
    let expected_range = value_range.max - value_range.min;
    let range_validity = if expected_range > 0.0 && expected_range.is_finite() {
        100.0
    } else {
        50.0
    };

    // Data type correctness (30% weight) - all f64, so always valid
    let type_validity = 100.0;

    // Value reasonableness (30% weight) - based on standard deviation
    let std_dev = profile.basic_stats.std_dev;
    let mean = profile.basic_stats.mean;
    let cv = if mean.abs() > 1e-10 {
        (std_dev / mean.abs()).min(5.0)
    } else {
        0.0
    };
    let reasonableness = (100.0 - (cv * 10.0)).max(0.0);

    range_validity * 0.4 + type_validity * 0.3 + reasonableness * 0.3
}

/// Calculates timeliness score (0-100)
pub fn calculate_timeliness_score(data: &TimeSeries, expected_frequency: Frequency) -> f64 {
    if data.timestamps.len() < 2 {
        return 100.0;
    }

    // Check if data has consistent frequency
    let inferred_frequency = Frequency::infer_from_timestamps(&data.timestamps);
    let actual_frequency = data.frequency.as_ref().or(inferred_frequency.as_ref());

    // Frequency match (60% weight)
    let frequency_score = match actual_frequency {
        Some(actual) => {
            if actual == &expected_frequency {
                100.0
            } else {
                70.0 // Mismatched but regular
            }
        }
        None => 50.0, // Irregular frequency
    };

    // Regularity score (40% weight)
    let mut intervals = Vec::new();
    for i in 1..data.timestamps.len() {
        let interval = (data.timestamps[i] - data.timestamps[i - 1]).num_seconds();
        intervals.push(interval);
    }

    let mean_interval = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
    let variance = intervals
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean_interval;
            diff * diff
        })
        .sum::<f64>()
        / intervals.len() as f64;
    let std_dev = variance.sqrt();
    let cv = if mean_interval > 0.0 {
        std_dev / mean_interval
    } else {
        1.0
    };

    let regularity_score = (100.0 - (cv * 100.0).min(100.0)).max(0.0);

    frequency_score * 0.6 + regularity_score * 0.4
}

/// Calculates accuracy score (0-100) based on outlier density
pub fn calculate_accuracy_score(outliers: &OutlierReport) -> f64 {
    let outlier_percentage = outliers.summary.outlier_percentage;

    // Perfect: 100 - No outliers (<1%)
    // Excellent: 95-99 - Very few outliers (1-3%)
    // Good: 85-94 - Few outliers (3-7%)
    // Fair: 70-84 - Some outliers (7-15%)
    // Poor: <70 - Many outliers (>15%)

    if outlier_percentage < 1.0 {
        100.0 - outlier_percentage
    } else if outlier_percentage < 3.0 {
        97.0 - (outlier_percentage - 1.0) * 2.0
    } else if outlier_percentage < 7.0 {
        92.0 - (outlier_percentage - 3.0) * 1.75
    } else if outlier_percentage < 15.0 {
        77.0 - (outlier_percentage - 7.0) * 0.875
    } else {
        (70.0 - (outlier_percentage - 15.0) * 0.5).max(0.0)
    }
}

/// Calculates overall composite score using weighted average
pub fn calculate_overall_score(scores: &DimensionScores, weights: &QualityWeights) -> f64 {
    let weighted_sum = scores.completeness * weights.completeness
        + scores.consistency * weights.consistency
        + scores.validity * weights.validity
        + scores.timeliness * weights.timeliness
        + scores.accuracy * weights.accuracy;

    let total_weight = weights.total();

    if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        0.0
    }
}

/// Generates quality recommendations based on assessment
pub fn generate_recommendations(
    dimension_scores: &DimensionScores,
    profile: &DataProfile,
    outliers: &OutlierReport,
) -> Vec<QualityRecommendation> {
    let mut recommendations = Vec::new();

    // Completeness recommendations
    if dimension_scores.completeness < 85.0 {
        let gap_count = profile.completeness.gap_count;
        let priority = if dimension_scores.completeness < 70.0 {
            Priority::High
        } else {
            Priority::Medium
        };

        recommendations.push(QualityRecommendation::FillGaps {
            priority,
            method: ImputationMethod::LinearInterpolation,
            gap_count,
        });

        if dimension_scores.completeness < 60.0 {
            recommendations.push(QualityRecommendation::ImproveDataCollection {
                priority: Priority::High,
                description: format!(
                    "Data completeness is only {:.1}%. Review data collection processes.",
                    dimension_scores.completeness
                ),
            });
        }
    }

    // Accuracy recommendations (outliers)
    if dimension_scores.accuracy < 85.0 {
        let outlier_count = outliers.summary.total_outliers;
        let priority = if dimension_scores.accuracy < 70.0 {
            Priority::High
        } else {
            Priority::Medium
        };

        recommendations.push(QualityRecommendation::CleanOutliers {
            priority,
            method: OutlierMethod::IQR,
            count: outlier_count,
        });
    }

    // Consistency recommendations
    if dimension_scores.consistency < 85.0 {
        let priority = if dimension_scores.consistency < 70.0 {
            Priority::High
        } else {
            Priority::Medium
        };

        recommendations.push(QualityRecommendation::ReviewInconsistencies {
            priority,
            inconsistency_count: profile.completeness.gaps.len(),
        });
    }

    // Validity recommendations
    if dimension_scores.validity < 85.0 {
        let priority = if dimension_scores.validity < 70.0 {
            Priority::Critical
        } else {
            Priority::High
        };

        let validation_rules = vec![
            "Implement range checks".to_string(),
            "Add business rule validation".to_string(),
            "Set up data type constraints".to_string(),
        ];

        recommendations.push(QualityRecommendation::ImplementValidation {
            priority,
            validation_rules,
        });
    }

    // Timeliness recommendations
    if dimension_scores.timeliness < 85.0 {
        let priority = if dimension_scores.timeliness < 70.0 {
            Priority::High
        } else {
            Priority::Medium
        };

        if let Some(freq) = &profile.temporal_coverage.detected_frequency {
            recommendations.push(QualityRecommendation::StandardizeFrequency {
                priority,
                target_frequency: freq.clone(),
            });
        }
    }

    // Sort by priority
    recommendations.sort_by(|a, b| b.priority().cmp(&a.priority()));

    recommendations
}

/// Main quality assessment function
pub fn assess_quality(
    data: &TimeSeries,
    config: &QualityConfig,
) -> QualityResult<EnhancedQualityAssessment> {
    use crate::quality::{detect_outliers, profile_timeseries, OutlierConfig, ProfilingConfig};

    if data.is_empty() {
        return Err(QualityError::validation("Cannot assess quality of empty time series"));
    }

    // Step 1: Profile the data
    let profiling_config = ProfilingConfig::default();
    let profile = profile_timeseries(data, &profiling_config)?;

    // Step 2: Detect outliers
    let outlier_config = OutlierConfig::default();
    let outliers = detect_outliers(data, &outlier_config)?;

    // Step 3: Calculate dimension scores
    let completeness = calculate_completeness_score(&profile, config);
    let consistency = calculate_consistency_score(&profile, &outliers);
    let validity = calculate_validity_score(data, &profile);

    let expected_frequency = profile
        .temporal_coverage
        .detected_frequency
        .clone()
        .unwrap_or(Frequency::Day);
    let timeliness = calculate_timeliness_score(data, expected_frequency);
    let accuracy = calculate_accuracy_score(&outliers);

    let dimension_scores = DimensionScores {
        completeness,
        consistency,
        validity,
        timeliness,
        accuracy,
    };

    // Step 4: Calculate overall score
    let weights = QualityWeights::default(); // TODO: Make configurable
    let overall_score = calculate_overall_score(&dimension_scores, &weights);

    // Step 5: Generate recommendations
    let recommendations = generate_recommendations(&dimension_scores, &profile, &outliers);

    // Step 6: Collect quality issues (simplified for now)
    let quality_issues = Vec::new(); // TODO: Convert from profile/outlier reports

    // Step 7: Create assessment
    let assessment = EnhancedQualityAssessment::new(
        overall_score,
        dimension_scores,
        quality_issues,
        recommendations,
    );

    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_timeseries(values: Vec<f64>) -> TimeSeries {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..values.len())
            .map(|i| start + chrono::Duration::hours(i as i64))
            .collect();

        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_dimension_scores_creation() {
        let scores = DimensionScores::zero();
        assert_eq!(scores.completeness, 0.0);
        assert_eq!(scores.consistency, 0.0);

        let perfect = DimensionScores::perfect();
        assert_eq!(perfect.completeness, 100.0);
        assert_eq!(perfect.accuracy, 100.0);
    }

    #[test]
    fn test_quality_weights_default() {
        let weights = QualityWeights::default();
        let total = weights.total();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_weights_normalize() {
        let mut weights = QualityWeights {
            completeness: 2.0,
            consistency: 2.0,
            validity: 2.0,
            timeliness: 2.0,
            accuracy: 2.0,
        };
        weights.normalize();
        assert!((weights.completeness - 0.2).abs() < 0.01);
        assert!((weights.total() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_priority_from_severity() {
        assert_eq!(Priority::from_severity(0.9), Priority::Critical);
        assert_eq!(Priority::from_severity(0.7), Priority::High);
        assert_eq!(Priority::from_severity(0.4), Priority::Medium);
        assert_eq!(Priority::from_severity(0.1), Priority::Low);
    }

    #[test]
    fn test_quality_profile_weights() {
        let exploratory = QualityProfile::Exploratory.get_weights();
        assert!((exploratory.completeness - 0.3).abs() < 0.01);

        let production = QualityProfile::Production.get_weights();
        assert!((production.completeness - 0.25).abs() < 0.01);

        let regulatory = QualityProfile::Regulatory.get_weights();
        assert!((regulatory.validity - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_quality_profile_thresholds() {
        assert_eq!(QualityProfile::Exploratory.get_threshold(), 60.0);
        assert_eq!(QualityProfile::Production.get_threshold(), 75.0);
        assert_eq!(QualityProfile::Regulatory.get_threshold(), 90.0);
        assert_eq!(QualityProfile::RealTime.get_threshold(), 70.0);
    }

    #[test]
    fn test_calculate_overall_score() {
        let scores = DimensionScores {
            completeness: 90.0,
            consistency: 85.0,
            validity: 95.0,
            timeliness: 80.0,
            accuracy: 88.0,
        };
        let weights = QualityWeights::default();
        let overall = calculate_overall_score(&scores, &weights);

        assert!(overall >= 80.0 && overall <= 95.0);
    }

    #[test]
    fn test_enhanced_quality_assessment_grade() {
        let scores = DimensionScores::perfect();
        let assessment = EnhancedQualityAssessment::new(
            95.0,
            scores,
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(assessment.quality_grade(), "Excellent");
        assert!(assessment.passes_threshold(90.0));
    }

    #[test]
    fn test_recommendation_priority_filtering() {
        let recommendations = vec![
            QualityRecommendation::ImproveDataCollection {
                priority: Priority::High,
                description: "Test".to_string(),
            },
            QualityRecommendation::CleanOutliers {
                priority: Priority::Low,
                method: OutlierMethod::ZScore,
                count: 5,
            },
        ];

        let assessment = EnhancedQualityAssessment::new(
            75.0,
            DimensionScores::perfect(),
            Vec::new(),
            recommendations,
        );

        let high_priority = assessment.recommendations_by_priority(Priority::High);
        assert_eq!(high_priority.len(), 1);
    }

    #[test]
    fn test_assess_quality_high_quality_data() {
        // Create high-quality data: regular, complete, no outliers
        let values: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64 / 10.0)).collect();
        let data = create_test_timeseries(values);
        let config = QualityConfig::default();

        let result = assess_quality(&data, &config);
        assert!(result.is_ok());

        let assessment = result.unwrap();
        assert!(assessment.overall_score >= 80.0);
        assert_eq!(assessment.quality_grade(), "Excellent");
    }

    #[test]
    fn test_assess_quality_empty_data() {
        let data = TimeSeries::empty("empty".to_string());
        let config = QualityConfig::default();

        let result = assess_quality(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_assess_quality_performance() {
        // Test requirement: <30ms for quality assessment
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let mut timestamps = Vec::with_capacity(1000);
        let mut values = Vec::with_capacity(1000);

        // Generate 1K datapoints
        for i in 0..1000 {
            timestamps.push(start + chrono::Duration::hours(i as i64));
            values.push(50.0 + (i as f64 % 50.0));
        }

        let data = TimeSeries::new("performance_test".to_string(), timestamps, values).unwrap();
        let config = QualityConfig::default();

        // Measure performance
        let start_time = std::time::Instant::now();
        let result = assess_quality(&data, &config);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Assessment should succeed");
        assert!(
            duration.as_millis() < 30,
            "Assessment should complete in <30ms for 1K points, took {}ms",
            duration.as_millis()
        );

        let assessment = result.unwrap();
        assert!(assessment.overall_score >= 0.0 && assessment.overall_score <= 100.0);
        assert!(assessment.dimension_scores.completeness >= 0.0);
        assert!(assessment.dimension_scores.accuracy >= 0.0);
    }
}