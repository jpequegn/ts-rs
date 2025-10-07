//! Clustering functionality for time series embeddings

use crate::ml::{MLResult, MLError};
use crate::timeseries::TimeSeries;

/// Clustering algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusteringAlgorithm {
    KMeans {
        k: usize,
        max_iterations: usize,
    },
    DBSCAN {
        eps: f64,
        min_samples: usize,
    },
}

/// Time series clusterer
pub struct TimeSeriesClusterer {
    pub algorithm: ClusteringAlgorithm,
}

impl TimeSeriesClusterer {
    /// Create new clusterer
    pub fn new(algorithm: ClusteringAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Fit and predict clusters
    pub fn fit_predict(&self, _time_series: &[TimeSeries]) -> MLResult<Vec<usize>> {
        Err(MLError::invalid_input(
            "Clustering not yet implemented - planned for future release",
        ))
    }
}
