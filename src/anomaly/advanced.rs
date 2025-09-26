//! # Advanced ML-based Anomaly Detection
//!
//! Machine learning algorithms for anomaly detection including
//! Isolation Forest, Local Outlier Factor (LOF), and DBSCAN clustering.

use crate::analysis::{Anomaly, AnomalySeverity};
use crate::TimeSeries;
use std::collections::HashMap;

/// Detect anomalies using Isolation Forest algorithm
pub fn detect_isolation_forest_anomalies(
    timeseries: &TimeSeries,
    contamination: f64,
    n_trees: usize,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
        .values
        .iter()
        .enumerate()
        .zip(timeseries.timestamps.iter())
        .filter_map(|((i, &val), &timestamp)| {
            if !val.is_nan() {
                Some((i, val, timestamp))
            } else {
                None
            }
        })
        .collect();

    if valid_data.len() < 10 {
        return Ok(Vec::new());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, v, _)| *v).collect();
    let mut scores: Vec<(usize, f64)> = Vec::new();

    let max_depth = (values.len() as f64).log2().ceil() as usize;

    // Calculate isolation scores for each point
    for &(original_idx, value, _) in &valid_data {
        let mut path_lengths = Vec::new();

        // Generate multiple trees
        for _ in 0..n_trees {
            let path_length = calculate_isolation_path_length(value, &values, max_depth);
            path_lengths.push(path_length);
        }

        let avg_path_length = path_lengths.iter().sum::<f64>() / path_lengths.len() as f64;
        let isolation_score = 2.0_f64.powf(-avg_path_length / c_factor(values.len()));
        scores.push((original_idx, isolation_score));
    }

    // Sort by score (higher score = more anomalous)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top contamination% as anomalies
    let num_anomalies = (scores.len() as f64 * contamination).ceil() as usize;
    let mut anomalies = Vec::new();

    for (index, score) in scores.into_iter().take(num_anomalies) {
        let (_, value, timestamp) = valid_data.iter().find(|(i, _, _)| *i == index).unwrap();

        let severity = classify_isolation_severity(score);

        anomalies.push(Anomaly {
            index,
            timestamp: *timestamp,
            value: *value,
            score,
            severity,
            expected_value: None, // Isolation Forest doesn't provide expected values
        });
    }

    Ok(anomalies)
}

/// Detect anomalies using Local Outlier Factor (LOF)
pub fn detect_lof_anomalies(
    timeseries: &TimeSeries,
    n_neighbors: usize,
    contamination: f64,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
        .values
        .iter()
        .enumerate()
        .zip(timeseries.timestamps.iter())
        .filter_map(|((i, &val), &timestamp)| {
            if !val.is_nan() {
                Some((i, val, timestamp))
            } else {
                None
            }
        })
        .collect();

    if valid_data.len() < n_neighbors + 1 {
        return Ok(Vec::new());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, v, _)| *v).collect();
    let mut lof_scores: Vec<(usize, f64)> = Vec::new();

    // Calculate LOF for each point
    for (i, &(original_idx, value, _)) in valid_data.iter().enumerate() {
        // Find k-nearest neighbors
        let mut distances: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .map(|(j, &other_val)| (j, (value - other_val).abs()))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Get k+1 nearest neighbors (including self)
        let neighbors: Vec<(usize, f64)> = distances.into_iter().take(n_neighbors + 1).collect();

        // Calculate k-distance (distance to k-th neighbor)
        let k_distance = neighbors[n_neighbors].1;

        // Calculate reachability distance
        let mut reachability_distances = Vec::new();
        for (neighbor_idx, neighbor_dist) in &neighbors[1..] {
            // Skip self (index 0)
            let neighbor_k_distance = calculate_k_distance(&values, *neighbor_idx, n_neighbors);
            let reachability_dist = neighbor_dist.max(neighbor_k_distance);
            reachability_distances.push(reachability_dist);
        }

        // Calculate Local Reachability Density (LRD)
        let avg_reachability = if reachability_distances.is_empty() {
            1.0
        } else {
            reachability_distances.iter().sum::<f64>() / reachability_distances.len() as f64
        };

        let lrd = if avg_reachability > 0.0 {
            1.0 / avg_reachability
        } else {
            f64::INFINITY
        };

        // Calculate LOF
        let mut neighbor_lrds = Vec::new();
        for (neighbor_idx, _) in &neighbors[1..] {
            let neighbor_lrd = calculate_lrd(&values, *neighbor_idx, n_neighbors);
            neighbor_lrds.push(neighbor_lrd);
        }

        let lof = if lrd > 0.0 && !neighbor_lrds.is_empty() {
            let avg_neighbor_lrd = neighbor_lrds.iter().sum::<f64>() / neighbor_lrds.len() as f64;
            avg_neighbor_lrd / lrd
        } else {
            1.0
        };

        lof_scores.push((original_idx, lof));
    }

    // Sort by LOF score (higher = more anomalous)
    lof_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top contamination% as anomalies
    let num_anomalies = (lof_scores.len() as f64 * contamination).ceil() as usize;
    let mut anomalies = Vec::new();

    for (index, score) in lof_scores.into_iter().take(num_anomalies) {
        let (_, value, timestamp) = valid_data.iter().find(|(i, _, _)| *i == index).unwrap();

        let severity = classify_lof_severity(score);

        anomalies.push(Anomaly {
            index,
            timestamp: *timestamp,
            value: *value,
            score,
            severity,
            expected_value: None, // LOF doesn't provide expected values
        });
    }

    Ok(anomalies)
}

/// Detect anomalies using DBSCAN clustering
pub fn detect_dbscan_anomalies(
    timeseries: &TimeSeries,
    eps: f64,
    min_samples: usize,
) -> Result<Vec<Anomaly>, Box<dyn std::error::Error>> {
    let valid_data: Vec<(usize, f64, chrono::DateTime<chrono::Utc>)> = timeseries
        .values
        .iter()
        .enumerate()
        .zip(timeseries.timestamps.iter())
        .filter_map(|((i, &val), &timestamp)| {
            if !val.is_nan() {
                Some((i, val, timestamp))
            } else {
                None
            }
        })
        .collect();

    if valid_data.len() < min_samples {
        return Ok(Vec::new());
    }

    let values: Vec<f64> = valid_data.iter().map(|(_, v, _)| *v).collect();
    let mut clusters: HashMap<usize, i32> = HashMap::new();
    let mut cluster_id = 0;

    // DBSCAN algorithm
    for (i, &value) in values.iter().enumerate() {
        if clusters.contains_key(&i) {
            continue; // Already processed
        }

        // Find neighbors within eps distance
        let neighbors: Vec<usize> = values
            .iter()
            .enumerate()
            .filter_map(|(j, &other_val)| {
                if i != j && (value - other_val).abs() <= eps {
                    Some(j)
                } else {
                    None
                }
            })
            .collect();

        if neighbors.len() >= min_samples {
            // Core point - start new cluster
            clusters.insert(i, cluster_id);

            let mut seed_set = neighbors;
            let mut seed_index = 0;

            while seed_index < seed_set.len() {
                let current_point = seed_set[seed_index];

                if !clusters.contains_key(&current_point) {
                    clusters.insert(current_point, cluster_id);

                    // Find neighbors of current point
                    let current_neighbors: Vec<usize> = values
                        .iter()
                        .enumerate()
                        .filter_map(|(j, &other_val)| {
                            if current_point != j && (values[current_point] - other_val).abs() <= eps {
                                Some(j)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if current_neighbors.len() >= min_samples {
                        // Add new neighbors to seed set
                        for neighbor in current_neighbors {
                            if !seed_set.contains(&neighbor) {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }

                seed_index += 1;
            }

            cluster_id += 1;
        } else {
            // Noise point (anomaly)
            clusters.insert(i, -1);
        }
    }

    // Collect anomalies (noise points with cluster_id = -1)
    let mut anomalies = Vec::new();

    for (data_idx, (original_idx, value, timestamp)) in valid_data.iter().enumerate() {
        if let Some(&cluster) = clusters.get(&data_idx) {
            if cluster == -1 {
                // Calculate anomaly score based on distance to nearest cluster
                let score = calculate_cluster_distance_score(&values, data_idx, &clusters, eps);
                let severity = classify_dbscan_severity(score);

                anomalies.push(Anomaly {
                    index: *original_idx,
                    timestamp: *timestamp,
                    value: *value,
                    score,
                    severity,
                    expected_value: None, // DBSCAN doesn't provide expected values
                });
            }
        }
    }

    Ok(anomalies)
}

// Helper functions

/// Calculate isolation path length for a value
fn calculate_isolation_path_length(value: f64, data: &[f64], max_depth: usize) -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut current_data = data.to_vec();
    let mut depth = 0;

    while current_data.len() > 1 && depth < max_depth {
        if current_data.len() == 1 {
            break;
        }

        let min_val = current_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = current_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            break;
        }

        let split_point = rng.gen_range(min_val..max_val);

        if value < split_point {
            current_data.retain(|&x| x < split_point);
        } else {
            current_data.retain(|&x| x >= split_point);
        }

        depth += 1;
    }

    depth as f64
}

/// Calculate c factor for isolation forest normalization
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }

    let n_f64 = n as f64;
    2.0 * ((n_f64 - 1.0).ln() + std::f64::consts::E.ln()) - (2.0 * (n_f64 - 1.0) / n_f64)
}

/// Calculate k-distance for LOF
fn calculate_k_distance(values: &[f64], point_idx: usize, k: usize) -> f64 {
    if point_idx >= values.len() {
        return 0.0;
    }

    let point_value = values[point_idx];
    let mut distances: Vec<f64> = values
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != point_idx)
        .map(|(_, &val)| (point_value - val).abs())
        .collect();

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if distances.len() >= k {
        distances[k - 1]
    } else if !distances.is_empty() {
        distances[distances.len() - 1]
    } else {
        0.0
    }
}

/// Calculate Local Reachability Density for LOF
fn calculate_lrd(values: &[f64], point_idx: usize, k: usize) -> f64 {
    if point_idx >= values.len() {
        return 0.0;
    }

    let point_value = values[point_idx];
    let mut distances: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != point_idx)
        .map(|(i, &val)| (i, (point_value - val).abs()))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let neighbors: Vec<(usize, f64)> = distances.into_iter().take(k).collect();

    let mut reachability_distances = Vec::new();
    for (neighbor_idx, neighbor_dist) in neighbors {
        let neighbor_k_distance = calculate_k_distance(values, neighbor_idx, k);
        let reachability_dist = neighbor_dist.max(neighbor_k_distance);
        reachability_distances.push(reachability_dist);
    }

    let avg_reachability = if reachability_distances.is_empty() {
        1.0
    } else {
        reachability_distances.iter().sum::<f64>() / reachability_distances.len() as f64
    };

    if avg_reachability > 0.0 {
        1.0 / avg_reachability
    } else {
        f64::INFINITY
    }
}

/// Calculate distance score for DBSCAN anomalies
fn calculate_cluster_distance_score(
    values: &[f64],
    point_idx: usize,
    clusters: &HashMap<usize, i32>,
    eps: f64,
) -> f64 {
    let point_value = values[point_idx];

    // Find minimum distance to any clustered point
    let mut min_distance = f64::INFINITY;

    for (other_idx, &cluster_id) in clusters {
        if cluster_id >= 0 && *other_idx != point_idx {
            let distance = (point_value - values[*other_idx]).abs();
            min_distance = min_distance.min(distance);
        }
    }

    // Normalize by eps
    if min_distance.is_finite() {
        min_distance / eps
    } else {
        1.0
    }
}

/// Classify severity for Isolation Forest
fn classify_isolation_severity(score: f64) -> AnomalySeverity {
    if score >= 0.8 {
        AnomalySeverity::Critical
    } else if score >= 0.6 {
        AnomalySeverity::High
    } else if score >= 0.4 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity for LOF
fn classify_lof_severity(score: f64) -> AnomalySeverity {
    if score >= 3.0 {
        AnomalySeverity::Critical
    } else if score >= 2.0 {
        AnomalySeverity::High
    } else if score >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

/// Classify severity for DBSCAN
fn classify_dbscan_severity(score: f64) -> AnomalySeverity {
    if score >= 5.0 {
        AnomalySeverity::Critical
    } else if score >= 3.0 {
        AnomalySeverity::High
    } else if score >= 2.0 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_timeseries() -> TimeSeries {
        let timestamps = (0..20)
            .map(|i| Utc::now() + chrono::Duration::hours(i))
            .collect();
        // Most values normal, with a few outliers
        let values = vec![
            1.0, 2.0, 3.0, 2.5, 1.5, 2.2, 3.1, 2.8, 1.9, 2.4,
            100.0, // Outlier
            2.1, 3.3, 2.7, 1.8, 2.6, 3.0, 2.3, 1.7, 2.9,
        ];
        TimeSeries::new("test".to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_isolation_forest() {
        let ts = create_test_timeseries();
        let anomalies = detect_isolation_forest_anomalies(&ts, 0.1, 10).unwrap();

        assert!(!anomalies.is_empty());
        // Should detect the outlier at index 10 (value 100.0)
        assert!(anomalies.iter().any(|a| a.index == 10));
    }

    #[test]
    fn test_lof() {
        let ts = create_test_timeseries();
        let anomalies = detect_lof_anomalies(&ts, 3, 0.1).unwrap();

        assert!(!anomalies.is_empty());
        // Should detect the outlier
        assert!(anomalies.iter().any(|a| a.index == 10));
    }

    #[test]
    fn test_dbscan() {
        let ts = create_test_timeseries();
        let anomalies = detect_dbscan_anomalies(&ts, 1.0, 3).unwrap();

        assert!(!anomalies.is_empty());
        // Should detect the outlier as noise
        assert!(anomalies.iter().any(|a| a.index == 10));
    }

    #[test]
    fn test_helper_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let c_factor_val = c_factor(values.len());
        assert!(c_factor_val > 0.0);

        let k_distance = calculate_k_distance(&values, 0, 2);
        assert!(k_distance >= 0.0);

        let lrd = calculate_lrd(&values, 0, 2);
        assert!(lrd >= 0.0);
    }
}