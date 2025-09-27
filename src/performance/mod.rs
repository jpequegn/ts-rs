//! # Performance Optimization Module
//!
//! Comprehensive performance optimizations for Chronos time series analysis.
//! Provides memory management, parallel processing, caching, and progress tracking.

use std::sync::Arc;
use crate::config::PerformanceConfig;
use crate::{Result, TimeSeries};

pub mod memory;
pub mod parallel;
pub mod cache;
pub mod database;
pub mod progress;

pub use memory::{MemoryManager, StreamingProcessor, LazyDataLoader};
pub use parallel::{ParallelProcessor, ParallelConfig, TaskProcessor};
pub use cache::{CacheManager, CacheConfig, AnalysisCache};
pub use database::{DatabaseManager, DatabaseConfig, TimeSeriesDb};
pub use progress::{ProgressTracker, ProgressConfig, ProgressBar};

/// Performance optimization coordinator
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Memory management
    pub memory_manager: Arc<MemoryManager>,

    /// Parallel processing
    pub parallel_processor: Arc<ParallelProcessor>,

    /// Caching system
    pub cache_manager: Arc<CacheManager>,

    /// Database manager
    pub database_manager: Option<Arc<DatabaseManager>>,

    /// Progress tracking
    pub progress_tracker: Arc<ProgressTracker>,

    /// Configuration
    config: PerformanceConfig,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new(config: PerformanceConfig) -> Result<Self> {
        let memory_manager = Arc::new(MemoryManager::new(&config)?);
        let parallel_processor = Arc::new(ParallelProcessor::new(&config)?);
        let cache_manager = Arc::new(CacheManager::new(&config)?);
        let progress_tracker = Arc::new(ProgressTracker::new(&config)?);

        // Database manager is optional
        let database_manager = if config.enable_database {
            Some(Arc::new(DatabaseManager::new(&config)?))
        } else {
            None
        };

        Ok(Self {
            memory_manager,
            parallel_processor,
            cache_manager,
            database_manager,
            progress_tracker,
            config,
        })
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory_manager.get_stats()
    }

    /// Clear all caches
    pub fn clear_caches(&self) -> Result<()> {
        self.cache_manager.clear_all()
    }

    /// Optimize time series for analysis
    pub fn optimize_timeseries(&self, ts: &mut TimeSeries) -> Result<()> {
        // Apply memory optimizations
        self.memory_manager.optimize_timeseries(ts)?;

        // Prepare for parallel processing if enabled
        if self.config.enable_parallel {
            self.parallel_processor.prepare_timeseries(ts)?;
        }

        Ok(())
    }

    /// Check if operation should be cached
    pub fn should_cache(&self, operation: &str, data_size: usize) -> bool {
        self.cache_manager.should_cache(operation, data_size)
    }

    /// Get optimal chunk size for data processing
    pub fn get_chunk_size(&self, data_size: usize) -> usize {
        self.memory_manager.get_optimal_chunk_size(data_size)
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub used_memory_mb: f64,
    pub available_memory_mb: f64,
    pub cache_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub memory_efficiency: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub processing_time_ms: u64,
    pub memory_usage: MemoryStats,
    pub parallel_efficiency: f64,
    pub cache_hit_rate: f64,
    pub data_throughput_mb_s: f64,
}

/// Performance optimization error types
#[derive(Debug, thiserror::Error)]
pub enum PerformanceError {
    #[error("Memory limit exceeded: {0}MB")]
    MemoryLimitExceeded(usize),

    #[error("Parallel processing error: {0}")]
    ParallelProcessingError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Progress tracking error: {0}")]
    ProgressError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

