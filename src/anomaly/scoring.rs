//! # Anomaly Scoring and Ranking
//!
//! Utilities for combining multiple anomaly detection methods,
//! scoring anomalies, and ranking them by severity.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::anomaly::{ScoringConfig, ScoringMethod, SeverityThresholds};
use std::collections::HashMap;
use chrono::{Timelike, Datelike};

/// Combine and score anomalies from multiple detection methods
pub fn combine_and_score_anomalies(
    all_anomalies: Vec<Anomaly>,
    method_results: &HashMap<String, Vec<Anomaly>>,
    config: &ScoringConfig,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    if all_anomalies.is_empty() {
        return Ok(Vec::new());
    }

    // Group anomalies by index (same data point detected by multiple methods)
    let mut anomaly_groups: HashMap<usize, Vec<Anomaly>> = HashMap::new();

    for anomaly in all_anomalies {
        anomaly_groups.entry(anomaly.index).or_insert_with(Vec::new).push(anomaly);
    }

    let mut combined_anomalies = Vec::new();

    for (index, anomalies) in anomaly_groups {
        let combined_anomaly = match config.method {
            ScoringMethod::Maximum => combine_using_maximum(anomalies)?,
            ScoringMethod::WeightedAverage => combine_using_weighted_average(anomalies, &config.method_weights)?,
            ScoringMethod::EnsembleVoting => combine_using_ensemble_voting(anomalies)?,
            ScoringMethod::Custom(_) => {
                // For now, fallback to maximum scoring
                combine_using_maximum(anomalies)?
            }
        };

        combined_anomalies.push(combined_anomaly);
    }

    // Rank anomalies if enabled
    if config.enable_ranking {
        combined_anomalies.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        combined_anomalies.truncate(config.max_top_anomalies);
    }

    Ok(combined_anomalies)
}

/// Classify anomaly severity based on score thresholds
pub fn classify_anomaly_severity(
    mut anomalies: Vec<Anomaly>,
    thresholds: &SeverityThresholds,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    for anomaly in &mut anomalies {
        anomaly.severity = if anomaly.score >= thresholds.critical {
            AnomalySeverity::Critical
        } else if anomaly.score >= thresholds.high {
            AnomalySeverity::High
        } else if anomaly.score >= thresholds.medium {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        };
    }

    Ok(anomalies)
}

/// Generate anomaly ranking and statistics
pub fn generate_anomaly_ranking(
    anomalies: &[Anomaly],
) -> Result<AnomalyRanking, Box<dyn std::error::Error>> {
    let mut sorted_anomalies = anomalies.to_vec();
    sorted_anomalies.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let total_count = anomalies.len();
    let mut severity_counts = HashMap::new();

    for anomaly in anomalies {
        let severity_str = match anomaly.severity {
            AnomalySeverity::Low => "low",
            AnomalySeverity::Medium => "medium",
            AnomalySeverity::High => "high",
            AnomalySeverity::Critical => "critical",
        };
        *severity_counts.entry(severity_str.to_string()).or_insert(0) += 1;
    }

    let scores: Vec<f64> = anomalies.iter().map(|a| a.score).collect();
    let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean_score = if !scores.is_empty() {
        scores.iter().sum::<f64>() / scores.len() as f64
    } else {
        0.0
    };

    let variance = if scores.len() > 1 {
        scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / scores.len() as f64
    } else {
        0.0
    };
    let std_score = variance.sqrt();

    Ok(AnomalyRanking {
        ranked_anomalies: sorted_anomalies,
        total_count,
        severity_counts,
        score_statistics: ScoreStatistics {
            min_score,
            max_score,
            mean_score,
            std_score,
        },
    })
}

/// Calculate anomaly patterns and insights
pub fn analyze_anomaly_patterns(
    anomalies: &[Anomaly],
) -> Result<AnomalyPatterns, Box<dyn std::error::Error>> {
    if anomalies.is_empty() {
        return Ok(AnomalyPatterns::default());
    }

    // Temporal patterns
    let mut hourly_counts = vec![0; 24];
    let mut daily_counts = vec![0; 7];
    let mut monthly_counts = vec![0; 12];

    for anomaly in anomalies {
        let hour = anomaly.timestamp.hour() as usize;
        let day = anomaly.timestamp.weekday().num_days_from_monday() as usize;
        let month = (anomaly.timestamp.month() - 1) as usize;

        hourly_counts[hour] += 1;
        daily_counts[day] += 1;
        monthly_counts[month] += 1;
    }

    // Find peak hours, days, months
    let peak_hour = hourly_counts.iter().enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(hour, _)| hour)
        .unwrap_or(0);

    let peak_day = daily_counts.iter().enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(day, _)| day)
        .unwrap_or(0);

    let peak_month = monthly_counts.iter().enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(month, _)| month + 1)
        .unwrap_or(1);

    // Clustering analysis (simple approach)
    let clusters = find_temporal_clusters(anomalies)?;

    Ok(AnomalyPatterns {
        hourly_distribution: hourly_counts,
        daily_distribution: daily_counts,
        monthly_distribution: monthly_counts,
        peak_hour,
        peak_day,
        peak_month,
        temporal_clusters: clusters,
    })
}

// Helper functions

/// Combine anomalies using maximum scoring
fn combine_using_maximum(anomalies: Vec<Anomaly>) -> Result<Anomaly, Box<dyn std::error::Error>> {
    if anomalies.is_empty() {
        return Err("No anomalies to combine".into());
    }

    let max_anomaly = anomalies.into_iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .unwrap();

    Ok(max_anomaly)
}

/// Combine anomalies using weighted average
fn combine_using_weighted_average(
    anomalies: Vec<Anomaly>,
    weights: &HashMap<String, f64>,
) -> Result<Anomaly, Box<dyn std::error::Error>> {
    if anomalies.is_empty() {
        return Err("No anomalies to combine".into());
    }

    if anomalies.len() == 1 {
        return Ok(anomalies[0].clone());
    }

    // For simplicity, assume equal weights if no weights provided
    let weight = 1.0 / anomalies.len() as f64;

    let weighted_score = anomalies.iter()
        .map(|a| a.score * weight)
        .sum::<f64>();

    let mut combined = anomalies[0].clone();
    combined.score = weighted_score;

    Ok(combined)
}

/// Combine anomalies using ensemble voting
fn combine_using_ensemble_voting(anomalies: Vec<Anomaly>) -> Result<Anomaly, Box<dyn std::error::Error>> {
    if anomalies.is_empty() {
        return Err("No anomalies to combine".into());
    }

    // Count votes by severity level
    let mut severity_votes = HashMap::new();

    for anomaly in &anomalies {
        let severity_key = match anomaly.severity {
            AnomalySeverity::Low => 1,
            AnomalySeverity::Medium => 2,
            AnomalySeverity::High => 3,
            AnomalySeverity::Critical => 4,
        };
        *severity_votes.entry(severity_key).or_insert(0) += 1;
    }

    // Find majority vote
    let majority_severity = severity_votes.into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(severity, _)| severity)
        .unwrap_or(1);

    let ensemble_severity = match majority_severity {
        1 => AnomalySeverity::Low,
        2 => AnomalySeverity::Medium,
        3 => AnomalySeverity::High,
        4 => AnomalySeverity::Critical,
        _ => AnomalySeverity::Low,
    };

    // Use average score
    let ensemble_score = anomalies.iter().map(|a| a.score).sum::<f64>() / anomalies.len() as f64;

    let mut combined = anomalies[0].clone();
    combined.score = ensemble_score;
    combined.severity = ensemble_severity;

    Ok(combined)
}

/// Find temporal clusters of anomalies
fn find_temporal_clusters(anomalies: &[Anomaly]) -> Result<Vec<TemporalCluster>, Box<dyn std::error::Error>> {
    if anomalies.len() < 2 {
        return Ok(Vec::new());
    }

    let mut sorted_anomalies = anomalies.to_vec();
    sorted_anomalies.sort_by_key(|a| a.timestamp);

    let mut clusters = Vec::new();
    let mut current_cluster = vec![sorted_anomalies[0].clone()];

    // Group anomalies that are within 1 hour of each other
    let cluster_threshold = chrono::Duration::hours(1);

    for anomaly in sorted_anomalies.into_iter().skip(1) {
        if let Some(last_anomaly) = current_cluster.last() {
            if anomaly.timestamp - last_anomaly.timestamp <= cluster_threshold {
                current_cluster.push(anomaly);
            } else {
                // Start new cluster
                if current_cluster.len() >= 2 {
                    clusters.push(TemporalCluster {
                        start_time: current_cluster[0].timestamp,
                        end_time: current_cluster.last().unwrap().timestamp,
                        anomaly_count: current_cluster.len(),
                        avg_severity: calculate_avg_severity(&current_cluster),
                    });
                }
                current_cluster = vec![anomaly];
            }
        }
    }

    // Add final cluster
    if current_cluster.len() >= 2 {
        clusters.push(TemporalCluster {
            start_time: current_cluster[0].timestamp,
            end_time: current_cluster.last().unwrap().timestamp,
            anomaly_count: current_cluster.len(),
            avg_severity: calculate_avg_severity(&current_cluster),
        });
    }

    Ok(clusters)
}

/// Calculate average severity for a group of anomalies
fn calculate_avg_severity(anomalies: &[Anomaly]) -> f64 {
    let severity_sum: i32 = anomalies.iter().map(|a| match a.severity {
        AnomalySeverity::Low => 1,
        AnomalySeverity::Medium => 2,
        AnomalySeverity::High => 3,
        AnomalySeverity::Critical => 4,
    }).sum();

    severity_sum as f64 / anomalies.len() as f64
}

// Data structures

/// Anomaly ranking result
#[derive(Debug, Clone)]
pub struct AnomalyRanking {
    pub ranked_anomalies: Vec<Anomaly>,
    pub total_count: usize,
    pub severity_counts: HashMap<String, usize>,
    pub score_statistics: ScoreStatistics,
}

/// Statistical summary of anomaly scores
#[derive(Debug, Clone)]
pub struct ScoreStatistics {
    pub min_score: f64,
    pub max_score: f64,
    pub mean_score: f64,
    pub std_score: f64,
}

/// Anomaly pattern analysis result
#[derive(Debug, Clone)]
pub struct AnomalyPatterns {
    pub hourly_distribution: Vec<usize>,
    pub daily_distribution: Vec<usize>,
    pub monthly_distribution: Vec<usize>,
    pub peak_hour: usize,
    pub peak_day: usize,
    pub peak_month: usize,
    pub temporal_clusters: Vec<TemporalCluster>,
}

impl Default for AnomalyPatterns {
    fn default() -> Self {
        Self {
            hourly_distribution: vec![0; 24],
            daily_distribution: vec![0; 7],
            monthly_distribution: vec![0; 12],
            peak_hour: 0,
            peak_day: 0,
            peak_month: 1,
            temporal_clusters: Vec::new(),
        }
    }
}

/// Temporal cluster of anomalies
#[derive(Debug, Clone)]
pub struct TemporalCluster {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub anomaly_count: usize,
    pub avg_severity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_anomalies() -> Vec<Anomaly> {
        vec![
            Anomaly {
                index: 0,
                timestamp: Utc::now(),
                value: 100.0,
                score: 5.0,
                severity: AnomalySeverity::High,
                expected_value: Some(10.0),
            },
            Anomaly {
                index: 1,
                timestamp: Utc::now() + chrono::Duration::hours(1),
                value: 200.0,
                score: 8.0,
                severity: AnomalySeverity::Critical,
                expected_value: Some(15.0),
            },
            Anomaly {
                index: 2,
                timestamp: Utc::now() + chrono::Duration::hours(2),
                value: 50.0,
                score: 3.0,
                severity: AnomalySeverity::Medium,
                expected_value: Some(20.0),
            },
        ]
    }

    #[test]
    fn test_combine_using_maximum() {
        let anomalies = create_test_anomalies();
        let combined = combine_using_maximum(anomalies).unwrap();

        assert_eq!(combined.score, 8.0);
        assert_eq!(combined.index, 1);
    }

    #[test]
    fn test_combine_using_weighted_average() {
        let anomalies = create_test_anomalies();
        let weights = HashMap::new();
        let combined = combine_using_weighted_average(anomalies, &weights).unwrap();

        // Should be average of 5.0, 8.0, 3.0 = 5.33...
        assert!((combined.score - 5.333).abs() < 0.01);
    }

    #[test]
    fn test_combine_using_ensemble_voting() {
        let anomalies = create_test_anomalies();
        let combined = combine_using_ensemble_voting(anomalies).unwrap();

        // Each severity appears once, so it should pick one
        assert!((combined.score - 5.333).abs() < 0.01);
    }

    #[test]
    fn test_classify_anomaly_severity() {
        let mut anomalies = create_test_anomalies();
        let thresholds = SeverityThresholds {
            low: 1.0,
            medium: 3.0,
            high: 6.0,
            critical: 9.0,
        };

        let classified = classify_anomaly_severity(anomalies, &thresholds).unwrap();

        assert_eq!(classified[0].severity, AnomalySeverity::Medium); // score 5.0
        assert_eq!(classified[1].severity, AnomalySeverity::High);   // score 8.0
        assert_eq!(classified[2].severity, AnomalySeverity::Medium); // score 3.0
    }

    #[test]
    fn test_generate_anomaly_ranking() {
        let anomalies = create_test_anomalies();
        let ranking = generate_anomaly_ranking(&anomalies).unwrap();

        assert_eq!(ranking.total_count, 3);
        assert_eq!(ranking.ranked_anomalies[0].score, 8.0); // Highest score first
        assert_eq!(ranking.score_statistics.max_score, 8.0);
        assert_eq!(ranking.score_statistics.min_score, 3.0);
    }

    #[test]
    fn test_analyze_anomaly_patterns() {
        let anomalies = create_test_anomalies();
        let patterns = analyze_anomaly_patterns(&anomalies).unwrap();

        assert_eq!(patterns.hourly_distribution.len(), 24);
        assert_eq!(patterns.daily_distribution.len(), 7);
        assert_eq!(patterns.monthly_distribution.len(), 12);
    }

    #[test]
    fn test_temporal_clustering() {
        let mut anomalies = Vec::new();
        let base_time = Utc::now();

        // Create cluster of 3 anomalies within 30 minutes
        for i in 0..3 {
            anomalies.push(Anomaly {
                index: i,
                timestamp: base_time + chrono::Duration::minutes(i * 10),
                value: 100.0,
                score: 5.0,
                severity: AnomalySeverity::High,
                expected_value: Some(10.0),
            });
        }

        // Add isolated anomaly 2 hours later
        anomalies.push(Anomaly {
            index: 3,
            timestamp: base_time + chrono::Duration::hours(2),
            value: 200.0,
            score: 8.0,
            severity: AnomalySeverity::Critical,
            expected_value: Some(15.0),
        });

        let clusters = find_temporal_clusters(&anomalies).unwrap();
        assert_eq!(clusters.len(), 1); // Should find one cluster
        assert_eq!(clusters[0].anomaly_count, 3);
    }
}