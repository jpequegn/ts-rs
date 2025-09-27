//! # Memory Management Module
//!
//! Provides memory-efficient data processing, streaming, and lazy loading capabilities.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::path::Path;
use memmap2::MmapOptions;
use crate::config::PerformanceConfig;
use crate::{Result, TimeSeries, TimeSeriesError};
use super::{MemoryStats, PerformanceError};

/// Memory manager for efficient data handling
#[derive(Debug)]
pub struct MemoryManager {
    config: PerformanceConfig,
    used_memory: Arc<AtomicUsize>,
    peak_memory: Arc<AtomicUsize>,
    cache_memory: Arc<AtomicUsize>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(config: &PerformanceConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            used_memory: Arc::new(AtomicUsize::new(0)),
            peak_memory: Arc::new(AtomicUsize::new(0)),
            cache_memory: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let used_mb = self.used_memory.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0;
        let peak_mb = self.peak_memory.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0;
        let cache_mb = self.cache_memory.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0;

        let available_mb = self.config.memory_limit_mb
            .map(|limit| limit as f64 - used_mb)
            .unwrap_or(f64::MAX);

        let efficiency = if peak_mb > 0.0 { used_mb / peak_mb } else { 1.0 };

        MemoryStats {
            used_memory_mb: used_mb,
            available_memory_mb: available_mb,
            cache_memory_mb: cache_mb,
            peak_memory_mb: peak_mb,
            memory_efficiency: efficiency,
        }
    }

    /// Check if we're within memory limits
    pub fn check_memory_limit(&self, additional_bytes: usize) -> Result<()> {
        if let Some(limit_mb) = self.config.memory_limit_mb {
            let current_mb = self.used_memory.load(Ordering::Relaxed) / 1024 / 1024;
            let additional_mb = additional_bytes / 1024 / 1024;

            if current_mb + additional_mb > limit_mb {
                return Err(PerformanceError::MemoryLimitExceeded(current_mb + additional_mb).into());
            }
        }
        Ok(())
    }

    /// Allocate memory and track usage
    pub fn allocate(&self, bytes: usize) -> Result<()> {
        self.check_memory_limit(bytes)?;

        let current = self.used_memory.fetch_add(bytes, Ordering::Relaxed);
        let new_total = current + bytes;

        // Update peak memory if necessary
        let mut peak = self.peak_memory.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_memory.compare_exchange_weak(peak, new_total, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }

        Ok(())
    }

    /// Deallocate memory and update tracking
    pub fn deallocate(&self, bytes: usize) {
        self.used_memory.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Get optimal chunk size based on available memory
    pub fn get_optimal_chunk_size(&self, _data_size: usize) -> usize {
        let available_memory = self.get_stats().available_memory_mb as usize * 1024 * 1024;
        let configured_chunk = self.config.chunk_size;

        // Use smaller of configured chunk size or what fits in available memory
        std::cmp::min(configured_chunk, available_memory / 4) // Use 1/4 of available memory
    }

    /// Optimize time series memory layout
    pub fn optimize_timeseries(&self, ts: &mut TimeSeries) -> Result<()> {
        // Implement memory optimizations like:
        // - Compact data representation
        // - Remove unnecessary allocations
        // - Optimize data structures

        // For now, just ensure we have enough memory
        let estimated_size = ts.len() * std::mem::size_of::<f64>() * 2; // timestamps + values
        self.check_memory_limit(estimated_size)?;

        Ok(())
    }
}

/// Streaming data processor for large files
#[derive(Debug)]
pub struct StreamingProcessor {
    chunk_size: usize,
    memory_manager: Arc<MemoryManager>,
}

impl StreamingProcessor {
    /// Create a new streaming processor
    pub fn new(memory_manager: Arc<MemoryManager>, chunk_size: usize) -> Self {
        Self {
            chunk_size,
            memory_manager,
        }
    }

    /// Process file in chunks using memory mapping
    pub fn process_file<F, R>(&self, file_path: &Path, processor: F) -> Result<Vec<R>>
    where
        F: Fn(&[u8]) -> Result<R> + Send + Sync,
        R: Send,
    {
        let file = std::fs::File::open(file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let mut results = Vec::new();
        let total_size = mmap.len();

        // Process file in chunks
        for chunk_start in (0..total_size).step_by(self.chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + self.chunk_size, total_size);
            let chunk = &mmap[chunk_start..chunk_end];

            // Check memory before processing chunk
            self.memory_manager.check_memory_limit(chunk.len())?;

            let result = processor(chunk)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Stream process large dataset with callback
    pub fn stream_process<T, F>(&self, data: Vec<T>, mut processor: F) -> Result<()>
    where
        T: Send,
        F: FnMut(&[T]) -> Result<()> + Send,
    {
        let item_size = std::mem::size_of::<T>();
        let chunk_size = self.memory_manager.get_optimal_chunk_size(data.len() * item_size);
        let items_per_chunk = chunk_size / item_size.max(1);

        for chunk in data.chunks(items_per_chunk) {
            // Check memory before processing
            self.memory_manager.check_memory_limit(chunk.len() * item_size)?;

            processor(chunk)?;
        }

        Ok(())
    }
}

/// Lazy data loader for on-demand loading
#[derive(Debug)]
pub struct LazyDataLoader {
    memory_manager: Arc<MemoryManager>,
    cache_size_limit: usize,
}

impl LazyDataLoader {
    /// Create a new lazy data loader
    pub fn new(memory_manager: Arc<MemoryManager>, cache_size_limit: usize) -> Self {
        Self {
            memory_manager,
            cache_size_limit,
        }
    }

    /// Load data lazily with caching
    pub fn load_lazy<T, F>(&self, _key: &str, loader: F) -> Result<T>
    where
        T: Clone + Send + Sync,
        F: FnOnce() -> Result<T>,
    {
        // For now, just load directly
        // In a full implementation, this would:
        // 1. Check if data is already loaded in cache
        // 2. If not, load using the loader function
        // 3. Cache the result if within limits
        // 4. Return the data

        loader()
    }

    /// Preload data in background
    pub fn preload<T, F>(&self, key: String, loader: F) -> Result<()>
    where
        T: Clone + Send + Sync + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        // Spawn background task to preload data
        std::thread::spawn(move || {
            if let Err(e) = loader() {
                eprintln!("Background preload failed for {}: {}", key, e);
            }
        });

        Ok(())
    }
}

/// Memory-efficient data structure for time series
#[derive(Debug)]
pub struct CompactTimeSeries {
    /// Compressed timestamp deltas
    timestamp_deltas: Vec<u32>,

    /// Compressed values using quantization
    quantized_values: Vec<u16>,

    /// Value range for dequantization
    value_min: f64,
    value_max: f64,

    /// Base timestamp
    base_timestamp: i64,
}

impl CompactTimeSeries {
    /// Create a compact representation from regular time series
    pub fn from_timeseries(ts: &TimeSeries) -> Result<Self> {
        if ts.is_empty() {
            return Err(TimeSeriesError::invalid_input("Empty time series").into());
        }

        let timestamps = ts.timestamps.clone();
        let values = ts.values.clone();

        // Calculate timestamp deltas
        let base_timestamp = timestamps[0].timestamp();
        let mut timestamp_deltas = Vec::with_capacity(timestamps.len());

        for (i, timestamp) in timestamps.iter().enumerate() {
            let delta = if i == 0 {
                0
            } else {
                (timestamp.timestamp() - timestamps[i-1].timestamp()) as u32
            };
            timestamp_deltas.push(delta);
        }

        // Quantize values
        let value_min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let value_max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let value_range = value_max - value_min;

        let quantized_values: Vec<u16> = values.iter()
            .map(|&v| {
                if value_range == 0.0 {
                    0
                } else {
                    ((v - value_min) / value_range * 65535.0) as u16
                }
            })
            .collect();

        Ok(Self {
            timestamp_deltas,
            quantized_values,
            value_min,
            value_max,
            base_timestamp,
        })
    }

    /// Get memory footprint reduction compared to original
    pub fn compression_ratio(&self, original_size: usize) -> f64 {
        let compressed_size = self.timestamp_deltas.len() * 4 + self.quantized_values.len() * 2 + 32;
        1.0 - (compressed_size as f64 / original_size as f64)
    }

    /// Decompress back to regular time series
    pub fn to_timeseries(&self) -> Result<TimeSeries> {
        let mut timestamps = Vec::with_capacity(self.timestamp_deltas.len());
        let mut values = Vec::with_capacity(self.quantized_values.len());

        let mut current_timestamp = self.base_timestamp;
        let value_range = self.value_max - self.value_min;

        for (i, &delta) in self.timestamp_deltas.iter().enumerate() {
            if i > 0 {
                current_timestamp += delta as i64;
            }

            let timestamp = chrono::DateTime::from_timestamp(current_timestamp, 0)
                .ok_or_else(|| TimeSeriesError::invalid_timestamp("Invalid timestamp"))?;
            timestamps.push(timestamp);

            let quantized = self.quantized_values[i];
            let value = if value_range == 0.0 {
                self.value_min
            } else {
                self.value_min + (quantized as f64 / 65535.0) * value_range
            };
            values.push(value);
        }

        TimeSeries::new("compressed_series".to_string(), timestamps, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PerformanceConfig;

    #[test]
    fn test_memory_manager_creation() {
        let config = PerformanceConfig::default();
        let manager = MemoryManager::new(&config).unwrap();
        let stats = manager.get_stats();
        assert_eq!(stats.used_memory_mb, 0.0);
    }

    #[test]
    fn test_memory_allocation_tracking() {
        let config = PerformanceConfig::default();
        let manager = MemoryManager::new(&config).unwrap();

        manager.allocate(1024 * 1024).unwrap(); // 1MB
        let stats = manager.get_stats();
        assert!(stats.used_memory_mb >= 1.0);

        manager.deallocate(1024 * 1024);
        let stats = manager.get_stats();
        assert_eq!(stats.used_memory_mb, 0.0);
    }

    #[test]
    fn test_memory_limit() {
        let mut config = PerformanceConfig::default();
        config.memory_limit_mb = Some(1); // 1MB limit
        let manager = MemoryManager::new(&config).unwrap();

        // Should succeed
        assert!(manager.allocate(512 * 1024).is_ok());

        // Should fail due to limit
        assert!(manager.allocate(1024 * 1024).is_err());
    }
}