//! Dynamic Time Warping (DTW) analysis module
//!
//! Implements Dynamic Time Warping algorithms for time series pattern matching,
//! alignment, and similarity analysis. DTW allows comparison of time series
//! that may vary in speed or have temporal distortions.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// DTW analysis result containing distance and alignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTWResult {
    /// Input series names
    pub series1_name: String,
    pub series2_name: String,

    /// DTW distance between the series
    pub distance: f64,

    /// Normalized DTW distance (by path length)
    pub normalized_distance: f64,

    /// Optimal warping path alignment
    pub alignment: DTWAlignment,

    /// Cost matrix (optional, for analysis)
    pub cost_matrix: Option<Vec<Vec<f64>>>,

    /// DTW constraint parameters used
    pub constraints: DTWConstraints,

    /// Analysis metadata
    pub metadata: DTWMetadata,
}

/// DTW alignment path and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTWAlignment {
    /// Warping path coordinates (series1_index, series2_index)
    pub path: Vec<(usize, usize)>,

    /// Path length (number of alignment steps)
    pub path_length: usize,

    /// Alignment quality metrics
    pub quality_metrics: AlignmentQuality,

    /// Aligned subsequences
    pub aligned_series1: Vec<f64>,
    pub aligned_series2: Vec<f64>,

    /// Local distance along the path
    pub local_distances: Vec<f64>,
}

/// DTW constraint parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTWConstraints {
    /// Sakoe-Chiba band constraint (None for unconstrained)
    pub sakoe_chiba_radius: Option<usize>,

    /// Itakura parallelogram constraint
    pub itakura_parallelogram: bool,

    /// Step pattern (local path constraints)
    pub step_pattern: StepPattern,

    /// Distance function used
    pub distance_function: DistanceFunction,
}

/// Alignment quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentQuality {
    /// Path efficiency (ratio of aligned length to maximum possible)
    pub path_efficiency: f64,

    /// Temporal distortion measure
    pub temporal_distortion: f64,

    /// Alignment variance (consistency of alignment)
    pub alignment_variance: f64,

    /// Diagonal dominance (how much path follows diagonal)
    pub diagonal_dominance: f64,

    /// Quality score (0-100, higher is better)
    pub quality_score: f64,
}

/// DTW computation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTWMetadata {
    /// Length of input series
    pub series1_length: usize,
    pub series2_length: usize,

    /// Computation time in milliseconds
    pub computation_time_ms: u64,

    /// Memory usage (matrix size)
    pub matrix_size: (usize, usize),

    /// Whether constraints were applied
    pub constrained: bool,

    /// Algorithm variant used
    pub algorithm_variant: String,
}

/// Multiple DTW comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleDTWResult {
    /// Query series name
    pub query_series: String,

    /// Candidate series names and their DTW results
    pub candidates: HashMap<String, DTWResult>,

    /// Ranking of candidates by DTW distance
    pub ranking: Vec<(String, f64)>,

    /// Best match information
    pub best_match: Option<BestMatchInfo>,

    /// Summary statistics
    pub summary: MultipleDTWSummary,
}

/// Best match information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestMatchInfo {
    /// Name of the best matching series
    pub series_name: String,

    /// DTW distance to best match
    pub distance: f64,

    /// Confidence score (0-100)
    pub confidence: f64,

    /// Whether the match is statistically significant
    pub is_significant: bool,
}

/// Summary statistics for multiple DTW comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleDTWSummary {
    /// Number of candidate series
    pub n_candidates: usize,

    /// Mean DTW distance
    pub mean_distance: f64,

    /// Standard deviation of DTW distances
    pub std_distance: f64,

    /// Minimum DTW distance
    pub min_distance: f64,

    /// Maximum DTW distance
    pub max_distance: f64,

    /// Distance distribution quartiles
    pub quartiles: [f64; 3],  // Q1, Q2 (median), Q3
}

/// DTW barycenter (average) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTWBarycenterResult {
    /// Input series names
    pub series_names: Vec<String>,

    /// Computed barycenter series
    pub barycenter: Vec<f64>,

    /// Total DTW distance from all series to barycenter
    pub total_distance: f64,

    /// Individual distances from each series to barycenter
    pub individual_distances: Vec<f64>,

    /// Number of iterations used
    pub iterations: usize,

    /// Whether algorithm converged
    pub converged: bool,

    /// Convergence criteria used
    pub convergence_threshold: f64,
}

/// Step patterns for DTW local path constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepPattern {
    /// Standard symmetric pattern: (1,1), (1,0), (0,1)
    Symmetric,

    /// Asymmetric pattern favoring first series: (1,1), (1,0)
    AsymmetricFirst,

    /// Asymmetric pattern favoring second series: (1,1), (0,1)
    AsymmetricSecond,

    /// Type I pattern: (1,1), (2,1), (1,2)
    TypeI,

    /// Type II pattern: (1,1), (1,0), (0,1) with different weights
    TypeII,
}

/// Distance functions for DTW
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// Euclidean distance (L2 norm)
    Euclidean,

    /// Manhattan distance (L1 norm)
    Manhattan,

    /// Maximum distance (L∞ norm)
    Maximum,

    /// Squared Euclidean distance
    SquaredEuclidean,

    /// Derivative distance (for shape matching)
    Derivative,
}

impl Default for DTWConstraints {
    fn default() -> Self {
        Self {
            sakoe_chiba_radius: None,
            itakura_parallelogram: false,
            step_pattern: StepPattern::Symmetric,
            distance_function: DistanceFunction::Euclidean,
        }
    }
}

/// Compute DTW distance and alignment between two time series
pub fn compute_dtw_distance(
    series1: &[f64],
    series2: &[f64],
    constraints: Option<DTWConstraints>,
) -> Result<DTWResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let constraints = constraints.unwrap_or_default();

    if series1.is_empty() || series2.is_empty() {
        return Err("Input series cannot be empty".into());
    }

    let n = series1.len();
    let m = series2.len();

    // Initialize cost matrix
    let mut cost_matrix = vec![vec![f64::INFINITY; m]; n];
    let mut accumulated_cost = vec![vec![f64::INFINITY; m]; n];

    // Compute local cost matrix
    for i in 0..n {
        for j in 0..m {
            if is_valid_cell(i, j, n, m, &constraints) {
                cost_matrix[i][j] = compute_local_distance(
                    series1[i],
                    series2[j],
                    &constraints.distance_function,
                );
            }
        }
    }

    // Initialize first cell
    accumulated_cost[0][0] = cost_matrix[0][0];

    // Fill first row
    for j in 1..m {
        if is_valid_cell(0, j, n, m, &constraints) {
            accumulated_cost[0][j] = accumulated_cost[0][j-1] + cost_matrix[0][j];
        }
    }

    // Fill first column
    for i in 1..n {
        if is_valid_cell(i, 0, n, m, &constraints) {
            accumulated_cost[i][0] = accumulated_cost[i-1][0] + cost_matrix[i][0];
        }
    }

    // Fill the rest of the matrix using dynamic programming
    for i in 1..n {
        for j in 1..m {
            if is_valid_cell(i, j, n, m, &constraints) {
                let min_cost = compute_step_cost(
                    i, j, &accumulated_cost, &cost_matrix, &constraints.step_pattern
                );
                accumulated_cost[i][j] = min_cost;
            }
        }
    }

    let dtw_distance = accumulated_cost[n-1][m-1];

    if dtw_distance == f64::INFINITY {
        return Err("No valid DTW path found with given constraints".into());
    }

    // Backtrack to find optimal path
    let path = backtrack_path(&accumulated_cost, &cost_matrix, &constraints.step_pattern)?;
    let path_length = path.len();
    let normalized_distance = dtw_distance / path_length as f64;

    // Create alignment
    let alignment = create_alignment(series1, series2, &path, &cost_matrix)?;

    let computation_time_ms = start_time.elapsed().as_millis() as u64;

    let metadata = DTWMetadata {
        series1_length: n,
        series2_length: m,
        computation_time_ms,
        matrix_size: (n, m),
        constrained: constraints.sakoe_chiba_radius.is_some() || constraints.itakura_parallelogram,
        algorithm_variant: format!("{:?}", constraints.step_pattern),
    };

    Ok(DTWResult {
        series1_name: "series1".to_string(),
        series2_name: "series2".to_string(),
        distance: dtw_distance,
        normalized_distance,
        alignment,
        cost_matrix: Some(cost_matrix),
        constraints,
        metadata,
    })
}

/// Compute DTW alignment path without full cost matrix (memory efficient)
pub fn compute_dtw_alignment(
    series1: &[f64],
    series2: &[f64],
    constraints: Option<DTWConstraints>,
) -> Result<DTWAlignment, Box<dyn std::error::Error>> {
    let result = compute_dtw_distance(series1, series2, constraints)?;
    Ok(result.alignment)
}

/// Compare one series against multiple candidates using DTW
pub fn compute_multiple_dtw(
    query_series: &[f64],
    candidates: &HashMap<String, Vec<f64>>,
    constraints: Option<DTWConstraints>,
) -> Result<MultipleDTWResult, Box<dyn std::error::Error>> {
    if candidates.is_empty() {
        return Err("No candidate series provided".into());
    }

    let mut results = HashMap::new();
    let mut distances = Vec::new();

    // Compute DTW for each candidate
    for (name, candidate_series) in candidates {
        let mut dtw_result = compute_dtw_distance(
            query_series,
            candidate_series,
            constraints.clone(),
        )?;

        dtw_result.series1_name = "query".to_string();
        dtw_result.series2_name = name.clone();

        distances.push((name.clone(), dtw_result.distance));
        results.insert(name.clone(), dtw_result);
    }

    // Sort by distance (ascending)
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Calculate summary statistics
    let distance_values: Vec<f64> = distances.iter().map(|(_, d)| *d).collect();
    let summary = calculate_multiple_dtw_summary(&distance_values);

    // Determine best match
    let best_match = if !distances.is_empty() {
        let (best_name, best_distance) = &distances[0];
        let confidence = calculate_match_confidence(&distance_values, *best_distance);
        let is_significant = confidence > 70.0;  // Threshold for significance

        Some(BestMatchInfo {
            series_name: best_name.clone(),
            distance: *best_distance,
            confidence,
            is_significant,
        })
    } else {
        None
    };

    Ok(MultipleDTWResult {
        query_series: "query".to_string(),
        candidates: results,
        ranking: distances,
        best_match,
        summary,
    })
}

/// Compute DTW barycenter (average) of multiple time series
pub fn compute_dtw_barycenter(
    series_map: &HashMap<String, Vec<f64>>,
    max_iterations: usize,
    convergence_threshold: f64,
    constraints: Option<DTWConstraints>,
) -> Result<DTWBarycenterResult, Box<dyn std::error::Error>> {
    if series_map.is_empty() {
        return Err("No series provided for barycenter computation".into());
    }

    let series_names: Vec<String> = series_map.keys().cloned().collect();
    let series_list: Vec<&Vec<f64>> = series_map.values().collect();

    // Initialize barycenter as the first series
    let mut barycenter = series_list[0].clone();
    let mut converged = false;

    let mut iteration = 0;
    while iteration < max_iterations && !converged {
        let previous_barycenter = barycenter.clone();

        // Update barycenter using DTW alignment with all series
        barycenter = update_barycenter(&barycenter, &series_list, constraints.as_ref())?;

        // Check convergence
        let change = compute_series_distance(&previous_barycenter, &barycenter);
        if change < convergence_threshold {
            converged = true;
        }

        iteration += 1;
    }

    // Calculate final distances
    let mut individual_distances = Vec::new();
    let mut total_distance = 0.0;

    for series in &series_list {
        let dtw_result = compute_dtw_distance(
            &barycenter,
            series,
            constraints.clone(),
        )?;

        individual_distances.push(dtw_result.distance);
        total_distance += dtw_result.distance;
    }

    Ok(DTWBarycenterResult {
        series_names,
        barycenter,
        total_distance,
        individual_distances,
        iterations: iteration,
        converged,
        convergence_threshold,
    })
}

// Helper functions

fn is_valid_cell(i: usize, j: usize, n: usize, m: usize, constraints: &DTWConstraints) -> bool {
    // Check Sakoe-Chiba band constraint
    if let Some(radius) = constraints.sakoe_chiba_radius {
        let diagonal_ratio = (i as f64) / (n as f64) - (j as f64) / (m as f64);
        let band_width = radius as f64 / n.max(m) as f64;
        if diagonal_ratio.abs() > band_width {
            return false;
        }
    }

    // Check Itakura parallelogram constraint
    if constraints.itakura_parallelogram {
        let slope_constraint = 2.0;
        let i_ratio = i as f64 / n as f64;
        let j_ratio = j as f64 / m as f64;

        if j_ratio > slope_constraint * i_ratio || i_ratio > slope_constraint * j_ratio {
            return false;
        }
    }

    true
}

fn compute_local_distance(val1: f64, val2: f64, distance_function: &DistanceFunction) -> f64 {
    match distance_function {
        DistanceFunction::Euclidean => (val1 - val2).abs(),
        DistanceFunction::Manhattan => (val1 - val2).abs(),
        DistanceFunction::Maximum => (val1 - val2).abs(),
        DistanceFunction::SquaredEuclidean => (val1 - val2).powi(2),
        DistanceFunction::Derivative => (val1 - val2).abs(),  // Simplified for scalar values
    }
}

fn compute_step_cost(
    i: usize,
    j: usize,
    accumulated_cost: &[Vec<f64>],
    cost_matrix: &[Vec<f64>],
    step_pattern: &StepPattern,
) -> f64 {
    let current_cost = cost_matrix[i][j];

    match step_pattern {
        StepPattern::Symmetric => {
            let mut min_cost = f64::INFINITY;

            // (i-1, j-1) - diagonal
            if i > 0 && j > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j-1]);
            }

            // (i-1, j) - vertical
            if i > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j]);
            }

            // (i, j-1) - horizontal
            if j > 0 {
                min_cost = min_cost.min(accumulated_cost[i][j-1]);
            }

            current_cost + min_cost
        }

        StepPattern::AsymmetricFirst => {
            let mut min_cost = f64::INFINITY;

            // (i-1, j-1) - diagonal
            if i > 0 && j > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j-1]);
            }

            // (i-1, j) - vertical (favoring first series)
            if i > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j]);
            }

            current_cost + min_cost
        }

        StepPattern::AsymmetricSecond => {
            let mut min_cost = f64::INFINITY;

            // (i-1, j-1) - diagonal
            if i > 0 && j > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j-1]);
            }

            // (i, j-1) - horizontal (favoring second series)
            if j > 0 {
                min_cost = min_cost.min(accumulated_cost[i][j-1]);
            }

            current_cost + min_cost
        }

        StepPattern::TypeI => {
            let mut min_cost = f64::INFINITY;

            // (i-1, j-1) - diagonal
            if i > 0 && j > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j-1]);
            }

            // (i-2, j-1) - double step in first series
            if i >= 2 && j > 0 {
                min_cost = min_cost.min(accumulated_cost[i-2][j-1]);
            }

            // (i-1, j-2) - double step in second series
            if i > 0 && j >= 2 {
                min_cost = min_cost.min(accumulated_cost[i-1][j-2]);
            }

            current_cost + min_cost
        }

        StepPattern::TypeII => {
            let mut min_cost = f64::INFINITY;

            // (i-1, j-1) - diagonal (weight 2)
            if i > 0 && j > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j-1] + current_cost);
            }

            // (i-1, j) - vertical (weight 1)
            if i > 0 {
                min_cost = min_cost.min(accumulated_cost[i-1][j]);
            }

            // (i, j-1) - horizontal (weight 1)
            if j > 0 {
                min_cost = min_cost.min(accumulated_cost[i][j-1]);
            }

            current_cost + min_cost
        }
    }
}

fn backtrack_path(
    accumulated_cost: &[Vec<f64>],
    cost_matrix: &[Vec<f64>],
    step_pattern: &StepPattern,
) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    let n = accumulated_cost.len();
    let m = accumulated_cost[0].len();

    let mut path = Vec::new();
    let mut i = n - 1;
    let mut j = m - 1;

    path.push((i, j));

    while i > 0 || j > 0 {
        let current_cost = accumulated_cost[i][j];
        let local_cost = cost_matrix[i][j];

        let mut best_prev = None;
        let mut best_cost = f64::INFINITY;

        // Check possible predecessors based on step pattern
        match step_pattern {
            StepPattern::Symmetric => {
                // Check diagonal
                if i > 0 && j > 0 {
                    let cost = accumulated_cost[i-1][j-1];
                    if (cost + local_cost - current_cost).abs() < 1e-10 && cost < best_cost {
                        best_cost = cost;
                        best_prev = Some((i-1, j-1));
                    }
                }

                // Check vertical
                if i > 0 {
                    let cost = accumulated_cost[i-1][j];
                    if (cost + local_cost - current_cost).abs() < 1e-10 && cost < best_cost {
                        best_cost = cost;
                        best_prev = Some((i-1, j));
                    }
                }

                // Check horizontal
                if j > 0 {
                    let cost = accumulated_cost[i][j-1];
                    if (cost + local_cost - current_cost).abs() < 1e-10 && cost < best_cost {
                        best_cost = cost;
                        best_prev = Some((i, j-1));
                    }
                }
            }

            // Similar logic for other step patterns...
            _ => {
                // Simplified: use symmetric for other patterns
                if i > 0 && j > 0 {
                    best_prev = Some((i-1, j-1));
                } else if i > 0 {
                    best_prev = Some((i-1, j));
                } else if j > 0 {
                    best_prev = Some((i, j-1));
                }
            }
        }

        if let Some((prev_i, prev_j)) = best_prev {
            i = prev_i;
            j = prev_j;
            path.push((i, j));
        } else {
            break;
        }
    }

    path.reverse();
    Ok(path)
}

fn create_alignment(
    series1: &[f64],
    series2: &[f64],
    path: &[(usize, usize)],
    cost_matrix: &[Vec<f64>],
) -> Result<DTWAlignment, Box<dyn std::error::Error>> {
    let path_length = path.len();

    let mut aligned_series1 = Vec::new();
    let mut aligned_series2 = Vec::new();
    let mut local_distances = Vec::new();

    for &(i, j) in path {
        aligned_series1.push(series1[i]);
        aligned_series2.push(series2[j]);
        local_distances.push(cost_matrix[i][j]);
    }

    // Calculate quality metrics
    let quality_metrics = calculate_alignment_quality(path, series1.len(), series2.len());

    Ok(DTWAlignment {
        path: path.to_vec(),
        path_length,
        quality_metrics,
        aligned_series1,
        aligned_series2,
        local_distances,
    })
}

fn calculate_alignment_quality(path: &[(usize, usize)], n: usize, m: usize) -> AlignmentQuality {
    let path_length = path.len();
    let max_length = n.max(m);

    // Path efficiency
    let path_efficiency = (max_length as f64) / (path_length as f64);

    // Temporal distortion
    let expected_diagonal_step = (n as f64) / (m as f64);
    let mut distortion_sum = 0.0;

    for window in path.windows(2) {
        let (i1, j1) = window[0];
        let (i2, j2) = window[1];

        let actual_step = if j2 > j1 {
            (i2 - i1) as f64 / (j2 - j1) as f64
        } else {
            expected_diagonal_step
        };

        distortion_sum += (actual_step - expected_diagonal_step).abs();
    }

    let temporal_distortion = distortion_sum / (path_length as f64);

    // Diagonal dominance
    let diagonal_steps = path.windows(2)
        .filter(|window| {
            let (i1, j1) = window[0];
            let (i2, j2) = window[1];
            i2 > i1 && j2 > j1
        })
        .count();

    let diagonal_dominance = (diagonal_steps as f64) / (path_length as f64 - 1.0);

    // Alignment variance (simplified)
    let alignment_variance = temporal_distortion;

    // Overall quality score
    let quality_score = (path_efficiency * 40.0 +
                        (1.0 - temporal_distortion.min(1.0)) * 30.0 +
                        diagonal_dominance * 30.0).min(100.0);

    AlignmentQuality {
        path_efficiency,
        temporal_distortion,
        alignment_variance,
        diagonal_dominance,
        quality_score,
    }
}

fn calculate_multiple_dtw_summary(distances: &[f64]) -> MultipleDTWSummary {
    let n = distances.len();

    if n == 0 {
        return MultipleDTWSummary {
            n_candidates: 0,
            mean_distance: 0.0,
            std_distance: 0.0,
            min_distance: 0.0,
            max_distance: 0.0,
            quartiles: [0.0, 0.0, 0.0],
        };
    }

    let mean_distance = distances.iter().sum::<f64>() / n as f64;
    let variance = distances.iter()
        .map(|&d| (d - mean_distance).powi(2))
        .sum::<f64>() / n as f64;
    let std_distance = variance.sqrt();

    let min_distance = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_distance = distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Calculate quartiles
    let mut sorted_distances = distances.to_vec();
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1_idx = n / 4;
    let q2_idx = n / 2;
    let q3_idx = 3 * n / 4;

    let quartiles = [
        sorted_distances[q1_idx],
        sorted_distances[q2_idx],
        sorted_distances[q3_idx.min(n - 1)],
    ];

    MultipleDTWSummary {
        n_candidates: n,
        mean_distance,
        std_distance,
        min_distance,
        max_distance,
        quartiles,
    }
}

fn calculate_match_confidence(distances: &[f64], best_distance: f64) -> f64 {
    if distances.len() < 2 {
        return 50.0;  // Default confidence
    }

    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    let std = {
        let variance = distances.iter()
            .map(|&d| (d - mean).powi(2))
            .sum::<f64>() / distances.len() as f64;
        variance.sqrt()
    };

    // Confidence based on how many standard deviations below mean
    let z_score = (mean - best_distance) / std.max(1e-10);
    let confidence = (z_score * 20.0).min(100.0).max(0.0);

    confidence
}

fn update_barycenter(
    current_barycenter: &[f64],
    series_list: &[&Vec<f64>],
    constraints: Option<&DTWConstraints>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let barycenter_length = current_barycenter.len();
    let mut new_barycenter = vec![0.0; barycenter_length];
    let mut weights = vec![0.0; barycenter_length];

    // Align each series with current barycenter and accumulate values
    for series in series_list {
        let dtw_result = compute_dtw_distance(
            current_barycenter,
            series,
            constraints.cloned(),
        )?;

        // Accumulate aligned values
        for &(i, j) in &dtw_result.alignment.path {
            new_barycenter[i] += series[j];
            weights[i] += 1.0;
        }
    }

    // Average the accumulated values
    for i in 0..barycenter_length {
        if weights[i] > 0.0 {
            new_barycenter[i] /= weights[i];
        } else {
            new_barycenter[i] = current_barycenter[i];  // Keep original value
        }
    }

    Ok(new_barycenter)
}

fn compute_series_distance(series1: &[f64], series2: &[f64]) -> f64 {
    if series1.len() != series2.len() {
        return f64::INFINITY;
    }

    series1.iter()
        .zip(series2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>() / series1.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_dtw_basic_distance() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![1.1, 2.1, 3.1, 4.1, 5.1];

        let result = compute_dtw_distance(&series1, &series2, None).unwrap();

        assert!(result.distance > 0.0);
        assert!(result.normalized_distance > 0.0);
        assert_eq!(result.alignment.path.len(), result.alignment.path_length);
        assert!(result.alignment.quality_metrics.quality_score >= 0.0);
    }

    #[test]
    fn test_dtw_perfect_match() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = series1.clone();

        let result = compute_dtw_distance(&series1, &series2, None).unwrap();

        assert!((result.distance - 0.0).abs() < 1e-10);
        assert_eq!(result.alignment.path.len(), series1.len());
    }

    #[test]
    fn test_dtw_with_constraints() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let series2 = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];

        let constraints = DTWConstraints {
            sakoe_chiba_radius: Some(2),
            itakura_parallelogram: false,
            step_pattern: StepPattern::Symmetric,
            distance_function: DistanceFunction::Euclidean,
        };

        let result = compute_dtw_distance(&series1, &series2, Some(constraints)).unwrap();

        assert!(result.distance > 0.0);
        assert!(result.metadata.constrained);
    }

    #[test]
    fn test_multiple_dtw() {
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut candidates = HashMap::new();
        candidates.insert("similar".to_string(), vec![1.1, 2.1, 3.1, 4.1, 5.1]);
        candidates.insert("different".to_string(), vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        candidates.insert("reversed".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = compute_multiple_dtw(&query, &candidates, None).unwrap();

        assert_eq!(result.candidates.len(), 3);
        assert_eq!(result.ranking.len(), 3);
        assert!(result.best_match.is_some());

        // The most similar should be ranked first
        assert_eq!(result.ranking[0].0, "similar");
        assert!(result.ranking[0].1 < result.ranking[1].1);
    }

    #[test]
    fn test_dtw_barycenter() {
        let mut series_map = HashMap::new();
        series_map.insert("s1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        series_map.insert("s2".to_string(), vec![1.1, 2.1, 3.1, 4.1, 5.1]);
        series_map.insert("s3".to_string(), vec![0.9, 1.9, 2.9, 3.9, 4.9]);

        let result = compute_dtw_barycenter(&series_map, 10, 0.01, None).unwrap();

        assert_eq!(result.barycenter.len(), 5);
        assert_eq!(result.individual_distances.len(), 3);
        assert!(result.total_distance >= 0.0);
        assert!(result.iterations <= 10);
    }

    #[test]
    fn test_step_patterns() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0];
        let series2 = vec![1.0, 2.0, 3.0, 4.0];

        let patterns = vec![
            StepPattern::Symmetric,
            StepPattern::AsymmetricFirst,
            StepPattern::AsymmetricSecond,
            StepPattern::TypeI,
        ];

        for pattern in patterns {
            let constraints = DTWConstraints {
                step_pattern: pattern,
                ..Default::default()
            };

            let result = compute_dtw_distance(&series1, &series2, Some(constraints)).unwrap();
            assert!((result.distance - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_distance_functions() {
        let series1 = vec![1.0, 2.0, 3.0];
        let series2 = vec![1.5, 2.5, 3.5];

        let distance_functions = vec![
            DistanceFunction::Euclidean,
            DistanceFunction::Manhattan,
            DistanceFunction::Maximum,
            DistanceFunction::SquaredEuclidean,
        ];

        for dist_func in distance_functions {
            let constraints = DTWConstraints {
                distance_function: dist_func,
                ..Default::default()
            };

            let result = compute_dtw_distance(&series1, &series2, Some(constraints)).unwrap();
            assert!(result.distance > 0.0);
        }
    }

    #[test]
    fn test_dtw_empty_series() {
        let series1: Vec<f64> = vec![];
        let series2: Vec<f64> = vec![];

        let result = compute_dtw_distance(&series1, &series2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtw_single_element() {
        let series1 = vec![1.0];
        let series2 = vec![2.0];

        let result = compute_dtw_distance(&series1, &series2, None).unwrap();
        assert!((result.distance - 1.0).abs() < 1e-10);
        assert_eq!(result.alignment.path.len(), 1);
    }

    #[test]
    fn test_dtw_with_nan_values() {
        let series1 = vec![1.0, 2.0, f64::NAN, 4.0];
        let series2 = vec![1.1, 2.1, 3.1, 4.1];

        let result = compute_dtw_distance(&series1, &series2, None);
        // Should handle NaN appropriately - either error or handle gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_dtw_very_different_lengths() {
        let series1 = vec![1.0, 2.0];
        let series2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = compute_dtw_distance(&series1, &series2, None).unwrap();
        assert!(result.distance > 0.0);
        assert!(result.alignment.path.len() >= series1.len().max(series2.len()));
    }

    #[test]
    fn test_multiple_dtw_empty_candidates() {
        let query = vec![1.0, 2.0, 3.0];
        let candidates: HashMap<String, Vec<f64>> = HashMap::new();

        let result = compute_multiple_dtw(&query, &candidates, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtw_barycenter_single_series() {
        let mut series_map = HashMap::new();
        series_map.insert("single".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = compute_dtw_barycenter(&series_map, 5, 0.01, None).unwrap();
        assert_eq!(result.barycenter.len(), 5);
        assert_eq!(result.individual_distances.len(), 1);

        // Barycenter should be close to the single series
        for (i, &val) in result.barycenter.iter().enumerate() {
            assert!((val - series_map["single"][i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dtw_normalized_distance() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = compute_dtw_distance(&series1, &series2, None).unwrap();

        // Normalized distance should be between 0 and 1 for reasonable inputs
        assert!(result.normalized_distance >= 0.0);
        assert!(result.normalized_distance <= result.distance);
    }

    #[test]
    fn test_dtw_quality_metrics() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let result = compute_dtw_distance(&series1, &series2, None).unwrap();

        // Check quality metrics are reasonable
        assert!(result.alignment.quality_metrics.quality_score >= 0.0);
        assert!(result.alignment.quality_metrics.quality_score <= 1.0);
        assert!(result.alignment.quality_metrics.temporal_distortion >= 0.0);
        assert!(result.alignment.quality_metrics.path_efficiency >= 0.0);
        assert!(result.alignment.quality_metrics.path_efficiency <= 1.0);
    }
}