//! Similarity metrics for time series comparison

use crate::ml::{MLResult, MLError};
use crate::timeseries::TimeSeries;
use super::SimilarityMethod;

/// Compute Euclidean distance between two time series
pub fn euclidean_distance(ts1: &TimeSeries, ts2: &TimeSeries) -> MLResult<f64> {
    if ts1.values.len() != ts2.values.len() {
        return Err(MLError::invalid_input(format!(
            "Time series lengths must match: {} vs {}",
            ts1.values.len(),
            ts2.values.len()
        )));
    }

    let sum_squared: f64 = ts1
        .values
        .iter()
        .zip(&ts2.values)
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    Ok(sum_squared.sqrt())
}

/// Compute Manhattan distance between two time series
pub fn manhattan_distance(ts1: &TimeSeries, ts2: &TimeSeries) -> MLResult<f64> {
    if ts1.values.len() != ts2.values.len() {
        return Err(MLError::invalid_input(format!(
            "Time series lengths must match: {} vs {}",
            ts1.values.len(),
            ts2.values.len()
        )));
    }

    let sum_abs: f64 = ts1
        .values
        .iter()
        .zip(&ts2.values)
        .map(|(a, b)| (a - b).abs())
        .sum();

    Ok(sum_abs)
}

/// Compute cosine similarity between two time series
pub fn cosine_similarity(ts1: &TimeSeries, ts2: &TimeSeries) -> MLResult<f64> {
    if ts1.values.len() != ts2.values.len() {
        return Err(MLError::invalid_input(format!(
            "Time series lengths must match: {} vs {}",
            ts1.values.len(),
            ts2.values.len()
        )));
    }

    let dot_product: f64 = ts1.values.iter().zip(&ts2.values).map(|(a, b)| a * b).sum();

    let magnitude1: f64 = ts1.values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let magnitude2: f64 = ts2.values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return Err(MLError::invalid_input(
            "Cannot compute cosine similarity with zero magnitude vector",
        ));
    }

    Ok(dot_product / (magnitude1 * magnitude2))
}

/// Compute Dynamic Time Warping distance between two time series
pub fn dtw_distance(
    ts1: &TimeSeries,
    ts2: &TimeSeries,
    window: Option<usize>,
) -> MLResult<f64> {
    let n = ts1.values.len();
    let m = ts2.values.len();

    if n == 0 || m == 0 {
        return Err(MLError::invalid_input("Cannot compute DTW on empty time series"));
    }

    // Create cost matrix
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;

    // Compute DTW with optional window constraint
    for i in 1..=n {
        let window_start = if let Some(w) = window {
            i.saturating_sub(w)
        } else {
            1
        };
        let window_end = if let Some(w) = window {
            (i + w).min(m)
        } else {
            m
        };

        for j in window_start..=window_end {
            let distance = (ts1.values[i - 1] - ts2.values[j - 1]).abs();
            let min_cost = cost[i - 1][j]
                .min(cost[i][j - 1])
                .min(cost[i - 1][j - 1]);
            cost[i][j] = distance + min_cost;
        }
    }

    Ok(cost[n][m])
}

/// Compute Soft-DTW distance (differentiable version of DTW)
pub fn soft_dtw_distance(
    ts1: &TimeSeries,
    ts2: &TimeSeries,
    gamma: f64,
) -> MLResult<f64> {
    let n = ts1.values.len();
    let m = ts2.values.len();

    if n == 0 || m == 0 {
        return Err(MLError::invalid_input(
            "Cannot compute Soft-DTW on empty time series",
        ));
    }

    if gamma <= 0.0 {
        return Err(MLError::invalid_input("Gamma must be positive"));
    }

    // Create cost matrix with soft-min operator
    let mut cost = vec![vec![f64::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;

    // Soft-min function - handles cases where some inputs are infinite
    let soft_min = |a: f64, b: f64, c: f64, gamma: f64| -> f64 {
        // Collect finite values
        let finite_vals: Vec<f64> = vec![a, b, c].into_iter().filter(|x| x.is_finite()).collect();

        if finite_vals.is_empty() {
            return f64::INFINITY;
        }

        if finite_vals.len() == 1 {
            return finite_vals[0];
        }

        let max_val = finite_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = finite_vals.iter().map(|x| (-(x - max_val) / gamma).exp()).sum();

        -gamma * sum_exp.ln() - max_val
    };

    for i in 1..=n {
        for j in 1..=m {
            let distance = (ts1.values[i - 1] - ts2.values[j - 1]).powi(2);
            let soft_min_cost = soft_min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1], gamma);
            cost[i][j] = distance + soft_min_cost;
        }
    }

    Ok(cost[n][m])
}

/// Compute cross-correlation between two time series
pub fn cross_correlation(
    ts1: &TimeSeries,
    ts2: &TimeSeries,
    max_lag: usize,
) -> MLResult<Vec<f64>> {
    let n = ts1.values.len();
    let m = ts2.values.len();

    if n == 0 || m == 0 {
        return Err(MLError::invalid_input(
            "Cannot compute cross-correlation on empty time series",
        ));
    }

    // Calculate means
    let mean1 = ts1.values.iter().sum::<f64>() / n as f64;
    let mean2 = ts2.values.iter().sum::<f64>() / m as f64;

    // Calculate standard deviations
    let std1 = (ts1
        .values
        .iter()
        .map(|x| (x - mean1).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let std2 = (ts2
        .values
        .iter()
        .map(|x| (x - mean2).powi(2))
        .sum::<f64>()
        / m as f64)
        .sqrt();

    if std1 == 0.0 || std2 == 0.0 {
        return Err(MLError::invalid_input(
            "Cannot compute cross-correlation with zero standard deviation",
        ));
    }

    let min_len = n.min(m);
    let actual_max_lag = max_lag.min(min_len - 1);

    let mut correlations = Vec::new();

    // Compute cross-correlation for each lag
    for lag in 0..=actual_max_lag {
        let valid_len = min_len - lag;
        let mut sum = 0.0;

        for i in 0..valid_len {
            let norm1 = (ts1.values[i] - mean1) / std1;
            let norm2 = (ts2.values[i + lag] - mean2) / std2;
            sum += norm1 * norm2;
        }

        correlations.push(sum / valid_len as f64);
    }

    Ok(correlations)
}

/// Compute maximum cross-correlation value
pub fn max_cross_correlation(
    ts1: &TimeSeries,
    ts2: &TimeSeries,
    max_lag: usize,
) -> MLResult<f64> {
    let correlations = cross_correlation(ts1, ts2, max_lag)?;
    correlations
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or_else(|| MLError::invalid_input("No correlations computed"))
}

/// Main similarity computation function
pub fn compute_similarity(
    ts1: &TimeSeries,
    ts2: &TimeSeries,
    method: &SimilarityMethod,
) -> MLResult<f64> {
    match method {
        SimilarityMethod::Euclidean => euclidean_distance(ts1, ts2),
        SimilarityMethod::Manhattan => manhattan_distance(ts1, ts2),
        SimilarityMethod::Cosine => {
            // Convert cosine similarity ([-1, 1]) to distance ([0, 2])
            let similarity = cosine_similarity(ts1, ts2)?;
            Ok(1.0 - similarity)
        }
        SimilarityMethod::DynamicTimeWarping { window } => dtw_distance(ts1, ts2, *window),
        SimilarityMethod::SoftDTW { gamma } => soft_dtw_distance(ts1, ts2, *gamma),
        SimilarityMethod::CrossCorrelation { max_lag } => {
            // Return 1 - max_correlation as distance (lower is better)
            let max_corr = max_cross_correlation(ts1, ts2, *max_lag)?;
            Ok(1.0 - max_corr)
        }
        _ => Err(MLError::invalid_input(format!(
            "Similarity method not yet implemented: {:?}",
            method
        ))),
    }
}

/// Convert distance to similarity score (0-1, higher is better)
pub fn distance_to_similarity(distance: f64, max_distance: f64) -> f64 {
    if max_distance == 0.0 {
        return 1.0;
    }
    (1.0 - (distance / max_distance).min(1.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_timeseries(name: &str, values: Vec<f64>) -> TimeSeries {
        let timestamps = (0..values.len())
            .map(|i| {
                Utc::now()
                    + chrono::Duration::seconds(i as i64)
            })
            .collect();

        TimeSeries::new(name.to_string(), timestamps, values).unwrap()
    }

    #[test]
    fn test_euclidean_distance() {
        let ts1 = create_test_timeseries("ts1", vec![1.0, 2.0, 3.0]);
        let ts2 = create_test_timeseries("ts2", vec![4.0, 5.0, 6.0]);

        let distance = euclidean_distance(&ts1, &ts2).unwrap();
        // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assert!((distance - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_manhattan_distance() {
        let ts1 = create_test_timeseries("ts1", vec![1.0, 2.0, 3.0]);
        let ts2 = create_test_timeseries("ts2", vec![4.0, 5.0, 6.0]);

        let distance = manhattan_distance(&ts1, &ts2).unwrap();
        // |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        assert_eq!(distance, 9.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let ts1 = create_test_timeseries("ts1", vec![1.0, 2.0, 3.0]);
        let ts2 = create_test_timeseries("ts2", vec![2.0, 4.0, 6.0]);

        let similarity = cosine_similarity(&ts1, &ts2).unwrap();
        // Parallel vectors should have similarity of 1.0
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dtw_distance() {
        let ts1 = create_test_timeseries("ts1", vec![1.0, 2.0, 3.0]);
        let ts2 = create_test_timeseries("ts2", vec![1.0, 2.0, 3.0]);

        let distance = dtw_distance(&ts1, &ts2, None).unwrap();
        // Identical series should have DTW distance of 0
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_distance_shifted() {
        let ts1 = create_test_timeseries("ts1", vec![0.0, 1.0, 2.0, 3.0]);
        let ts2 = create_test_timeseries("ts2", vec![1.0, 2.0, 3.0, 4.0]);

        let distance = dtw_distance(&ts1, &ts2, None).unwrap();
        // DTW finds optimal alignment: the minimum cost path gives distance 2.0
        // This is because DTW can match: 0->1 (cost 1), 1->2, 2->3, 3->4 (cost 1)
        assert_eq!(distance, 2.0);
    }

    #[test]
    fn test_cross_correlation() {
        let ts1 = create_test_timeseries("ts1", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let ts2 = create_test_timeseries("ts2", vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let correlations = cross_correlation(&ts1, &ts2, 2).unwrap();
        // Autocorrelation at lag 0 should be 1.0
        assert!((correlations[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_to_similarity() {
        assert_eq!(distance_to_similarity(0.0, 10.0), 1.0);
        assert_eq!(distance_to_similarity(5.0, 10.0), 0.5);
        assert_eq!(distance_to_similarity(10.0, 10.0), 0.0);
        assert_eq!(distance_to_similarity(15.0, 10.0), 0.0); // Capped at 0
    }

    #[test]
    fn test_soft_dtw_distance() {
        let ts1 = create_test_timeseries("ts1", vec![1.0, 2.0, 3.0]);
        let ts2 = create_test_timeseries("ts2", vec![1.0, 2.0, 3.0]);

        let distance = soft_dtw_distance(&ts1, &ts2, 1.0).unwrap();
        // Identical series should have very small Soft-DTW distance (close to 0)
        // With gamma=1.0, the soft-min smoothing adds some numerical overhead
        assert!(distance < 1.0, "Soft-DTW distance for identical series should be < 1.0, got {}", distance);
    }
}
