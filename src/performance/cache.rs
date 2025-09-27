//! # Caching System Module
//!
//! Provides intelligent caching for analysis results with automatic invalidation.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use std::path::PathBuf;
use dashmap::DashMap;
use lz4::block::{compress, decompress};
use serde::{Deserialize, Serialize};
use crate::config::PerformanceConfig;
use crate::Result;
use super::PerformanceError;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_size_mb: usize,
    pub ttl_seconds: u64,
    pub enable_compression: bool,
    pub cache_directory: Option<PathBuf>,
    pub enable_persistent: bool,
}

impl From<&PerformanceConfig> for CacheConfig {
    fn from(config: &PerformanceConfig) -> Self {
        Self {
            max_size_mb: config.max_cache_size_mb,
            ttl_seconds: 3600, // 1 hour default TTL
            enable_compression: true,
            cache_directory: config.cache_directory.clone(),
            enable_persistent: true,
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    created_at: SystemTime,
    last_accessed: SystemTime,
    access_count: u64,
    size_bytes: usize,
    compressed: bool,
}

impl CacheEntry {
    fn new(data: Vec<u8>, compressed: bool) -> Self {
        let now = SystemTime::now();
        let size_bytes = data.len();

        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            compressed,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed().unwrap_or(Duration::MAX) > ttl
    }

    fn access(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }
}

/// Thread-safe cache manager
#[derive(Debug)]
pub struct CacheManager {
    config: CacheConfig,
    memory_cache: Arc<DashMap<String, CacheEntry>>,
    size_tracker: Arc<RwLock<usize>>,
    hit_count: Arc<RwLock<u64>>,
    miss_count: Arc<RwLock<u64>>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(config: &PerformanceConfig) -> Result<Self> {
        let cache_config = CacheConfig::from(config);

        // Ensure cache directory exists if persistent caching is enabled
        if let Some(ref cache_dir) = cache_config.cache_directory {
            if cache_config.enable_persistent {
                std::fs::create_dir_all(cache_dir)?;
            }
        }

        Ok(Self {
            config: cache_config,
            memory_cache: Arc::new(DashMap::new()),
            size_tracker: Arc::new(RwLock::new(0)),
            hit_count: Arc::new(RwLock::new(0)),
            miss_count: Arc::new(RwLock::new(0)),
        })
    }

    /// Check if an operation should be cached
    pub fn should_cache(&self, operation: &str, data_size: usize) -> bool {
        // Cache expensive operations or large datasets
        let expensive_operations = ["correlation_matrix", "fft", "arima_fit", "anomaly_detection"];

        expensive_operations.contains(&operation) || data_size > 1000
    }

    /// Get cached result
    pub fn get<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Try memory cache first
        if let Some(mut entry) = self.memory_cache.get_mut(key) {
            // Check if expired
            if entry.is_expired(Duration::from_secs(self.config.ttl_seconds)) {
                drop(entry);
                self.memory_cache.remove(key);
                self.increment_miss();
                return None;
            }

            entry.access();
            let data = if entry.compressed {
                match decompress(&entry.data, None) {
                    Ok(decompressed) => decompressed,
                    Err(_) => {
                        self.increment_miss();
                        return None;
                    }
                }
            } else {
                entry.data.clone()
            };

            match bincode::deserialize(&data) {
                Ok(result) => {
                    self.increment_hit();
                    Some(result)
                }
                Err(_) => {
                    self.increment_miss();
                    None
                }
            }
        } else {
            // Try persistent cache if enabled
            if self.config.enable_persistent {
                if let Some(result) = self.get_from_disk(key) {
                    self.increment_hit();
                    return Some(result);
                }
            }

            self.increment_miss();
            None
        }
    }

    /// Store result in cache
    pub fn set<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        let serialized = bincode::serialize(value)
            .map_err(|e| PerformanceError::CacheError(e.to_string()))?;

        let (data, compressed) = if self.config.enable_compression && serialized.len() > 1024 {
            match compress(&serialized, None, false) {
                Ok(compressed_data) => (compressed_data, true),
                Err(_) => (serialized, false),
            }
        } else {
            (serialized, false)
        };

        let entry = CacheEntry::new(data, compressed);
        let entry_size = entry.size_bytes;

        // Check if we need to evict entries to make space
        self.ensure_space(entry_size)?;

        // Add to memory cache
        self.memory_cache.insert(key.to_string(), entry);

        // Update size tracker
        {
            let mut size = self.size_tracker.write().unwrap();
            *size += entry_size;
        }

        // Save to disk if persistent caching is enabled
        if self.config.enable_persistent {
            self.save_to_disk(key, value)?;
        }

        Ok(())
    }

    /// Remove entry from cache
    pub fn remove(&self, key: &str) -> bool {
        if let Some((_, entry)) = self.memory_cache.remove(key) {
            let mut size = self.size_tracker.write().unwrap();
            *size = size.saturating_sub(entry.size_bytes);
            true
        } else {
            false
        }
    }

    /// Clear all cache entries
    pub fn clear_all(&self) -> Result<()> {
        self.memory_cache.clear();
        {
            let mut size = self.size_tracker.write().unwrap();
            *size = 0;
        }

        // Clear persistent cache if enabled
        if self.config.enable_persistent {
            if let Some(ref cache_dir) = self.config.cache_directory {
                if cache_dir.exists() {
                    std::fs::remove_dir_all(cache_dir)?;
                    std::fs::create_dir_all(cache_dir)?;
                }
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let size_mb = *self.size_tracker.read().unwrap() as f64 / 1024.0 / 1024.0;
        let hit_count = *self.hit_count.read().unwrap();
        let miss_count = *self.miss_count.read().unwrap();
        let total_requests = hit_count + miss_count;
        let hit_rate = if total_requests > 0 {
            hit_count as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStats {
            memory_usage_mb: size_mb,
            max_size_mb: self.config.max_size_mb as f64,
            entry_count: self.memory_cache.len(),
            hit_count,
            miss_count,
            hit_rate,
        }
    }

    /// Ensure we have enough space for new entry
    fn ensure_space(&self, needed_bytes: usize) -> Result<()> {
        let max_bytes = self.config.max_size_mb * 1024 * 1024;
        let current_size = *self.size_tracker.read().unwrap();

        if current_size + needed_bytes > max_bytes {
            // Evict least recently used entries
            self.evict_lru(needed_bytes)?;
        }

        Ok(())
    }

    /// Evict least recently used entries
    fn evict_lru(&self, needed_bytes: usize) -> Result<()> {
        let mut entries_to_remove = Vec::new();
        let mut freed_bytes = 0;

        // Collect entries sorted by last access time
        let mut entries: Vec<_> = self.memory_cache.iter()
            .map(|entry| (entry.key().clone(), entry.value().last_accessed, entry.value().size_bytes))
            .collect();

        entries.sort_by_key(|(_, last_accessed, _)| *last_accessed);

        // Remove oldest entries until we have enough space
        for (key, _, size) in entries {
            entries_to_remove.push(key);
            freed_bytes += size;

            if freed_bytes >= needed_bytes {
                break;
            }
        }

        // Remove the selected entries
        for key in entries_to_remove {
            self.remove(&key);
        }

        Ok(())
    }

    /// Save to persistent cache
    fn save_to_disk<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        if let Some(ref cache_dir) = self.config.cache_directory {
            let file_path = cache_dir.join(format!("{}.cache", key));
            let serialized = bincode::serialize(value)
                .map_err(|e| PerformanceError::CacheError(e.to_string()))?;

            std::fs::write(file_path, serialized)?;
        }
        Ok(())
    }

    /// Load from persistent cache
    fn get_from_disk<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        if let Some(ref cache_dir) = self.config.cache_directory {
            let file_path = cache_dir.join(format!("{}.cache", key));

            if let Ok(data) = std::fs::read(file_path) {
                if let Ok(result) = bincode::deserialize(&data) {
                    return Some(result);
                }
            }
        }
        None
    }

    fn increment_hit(&self) {
        let mut hits = self.hit_count.write().unwrap();
        *hits += 1;
    }

    fn increment_miss(&self) {
        let mut misses = self.miss_count.write().unwrap();
        *misses += 1;
    }
}

/// Analysis-specific cache for common operations
#[derive(Debug)]
pub struct AnalysisCache {
    cache_manager: Arc<CacheManager>,
}

impl AnalysisCache {
    /// Create new analysis cache
    pub fn new(cache_manager: Arc<CacheManager>) -> Self {
        Self { cache_manager }
    }

    /// Cache statistical analysis results
    pub fn cache_statistics(&self, data_hash: u64, results: &HashMap<String, f64>) -> Result<()> {
        let key = format!("stats_{}", data_hash);
        self.cache_manager.set(&key, results)
    }

    /// Get cached statistical analysis results
    pub fn get_statistics(&self, data_hash: u64) -> Option<HashMap<String, f64>> {
        let key = format!("stats_{}", data_hash);
        self.cache_manager.get(&key)
    }

    /// Cache correlation matrix
    pub fn cache_correlation_matrix(&self, data_hash: u64, matrix: &Vec<Vec<f64>>) -> Result<()> {
        let key = format!("corr_matrix_{}", data_hash);
        self.cache_manager.set(&key, matrix)
    }

    /// Get cached correlation matrix
    pub fn get_correlation_matrix(&self, data_hash: u64) -> Option<Vec<Vec<f64>>> {
        let key = format!("corr_matrix_{}", data_hash);
        self.cache_manager.get(&key)
    }

    /// Cache forecast results
    pub fn cache_forecast(&self, model_hash: u64, horizon: usize, forecast: &Vec<f64>) -> Result<()> {
        let key = format!("forecast_{}_{}", model_hash, horizon);
        self.cache_manager.set(&key, forecast)
    }

    /// Get cached forecast results
    pub fn get_forecast(&self, model_hash: u64, horizon: usize) -> Option<Vec<f64>> {
        let key = format!("forecast_{}_{}", model_hash, horizon);
        self.cache_manager.get(&key)
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub memory_usage_mb: f64,
    pub max_size_mb: f64,
    pub entry_count: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PerformanceConfig;

    #[test]
    fn test_cache_manager_creation() {
        let config = PerformanceConfig::default();
        let cache = CacheManager::new(&config).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 0);
    }

    #[test]
    fn test_cache_set_get() {
        let config = PerformanceConfig::default();
        let cache = CacheManager::new(&config).unwrap();

        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        cache.set("test_key", &test_data).unwrap();

        let retrieved: Option<Vec<f64>> = cache.get("test_key");
        assert_eq!(retrieved, Some(test_data));
    }

    #[test]
    fn test_cache_miss() {
        let config = PerformanceConfig::default();
        let cache = CacheManager::new(&config).unwrap();

        let retrieved: Option<Vec<f64>> = cache.get("nonexistent_key");
        assert_eq!(retrieved, None);

        let stats = cache.stats();
        assert_eq!(stats.miss_count, 1);
    }

    #[test]
    fn test_cache_removal() {
        let config = PerformanceConfig::default();
        let cache = CacheManager::new(&config).unwrap();

        let test_data = vec![1.0, 2.0, 3.0];
        cache.set("test_key", &test_data).unwrap();

        assert!(cache.remove("test_key"));
        let retrieved: Option<Vec<f64>> = cache.get("test_key");
        assert_eq!(retrieved, None);
    }

    #[test]
    fn test_should_cache() {
        let config = PerformanceConfig::default();
        let cache = CacheManager::new(&config).unwrap();

        assert!(cache.should_cache("correlation_matrix", 100));
        assert!(cache.should_cache("simple_op", 2000));
        assert!(!cache.should_cache("simple_op", 100));
    }
}