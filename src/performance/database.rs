//! # Database Integration Module
//!
//! Provides SQLite support and time series database integration for large datasets.

use std::path::PathBuf;
use rusqlite::{Connection, params, OptionalExtension};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::config::PerformanceConfig;
use crate::{Result, TimeSeries, TimeSeriesError};
use super::PerformanceError;

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub sqlite_path: Option<PathBuf>,
    pub enable_wal_mode: bool,
    pub enable_foreign_keys: bool,
    pub cache_size_kb: usize,
    pub busy_timeout_ms: u32,
    pub sync_mode: SyncMode,
}

impl From<&PerformanceConfig> for DatabaseConfig {
    fn from(config: &PerformanceConfig) -> Self {
        Self {
            sqlite_path: config.cache_directory.as_ref().map(|d| d.join("chronos.db")),
            enable_wal_mode: true,
            enable_foreign_keys: true,
            cache_size_kb: 64 * 1024, // 64MB cache
            busy_timeout_ms: 5000,
            sync_mode: SyncMode::Normal,
        }
    }
}

/// SQLite synchronization modes
#[derive(Debug, Clone)]
pub enum SyncMode {
    Off,
    Normal,
    Full,
}

impl SyncMode {
    fn as_str(&self) -> &'static str {
        match self {
            SyncMode::Off => "OFF",
            SyncMode::Normal => "NORMAL",
            SyncMode::Full => "FULL",
        }
    }
}

/// Database manager for time series data
#[derive(Debug)]
pub struct DatabaseManager {
    config: DatabaseConfig,
    connection: Connection,
}

impl DatabaseManager {
    /// Create a new database manager
    pub fn new(config: &PerformanceConfig) -> Result<Self> {
        let db_config = DatabaseConfig::from(config);

        let connection = if let Some(ref path) = db_config.sqlite_path {
            // Ensure parent directory exists
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Connection::open(path)?
        } else {
            Connection::open_in_memory()?
        };

        let mut manager = Self {
            config: db_config,
            connection,
        };

        manager.initialize_database()?;
        Ok(manager)
    }

    /// Initialize database schema and settings
    fn initialize_database(&mut self) -> Result<()> {
        // Set SQLite optimization settings
        self.connection.execute(&format!("PRAGMA cache_size = -{}", self.config.cache_size_kb), [])?;
        self.connection.execute(&format!("PRAGMA busy_timeout = {}", self.config.busy_timeout_ms), [])?;
        self.connection.execute(&format!("PRAGMA synchronous = {}", self.config.sync_mode.as_str()), [])?;

        if self.config.enable_wal_mode {
            self.connection.execute("PRAGMA journal_mode = WAL", [])?;
        }

        if self.config.enable_foreign_keys {
            self.connection.execute("PRAGMA foreign_keys = ON", [])?;
        }

        // Create time series tables
        self.create_tables()?;

        Ok(())
    }

    /// Create database tables for time series data
    fn create_tables(&self) -> Result<()> {
        // Main time series metadata table
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS time_series_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                frequency TEXT,
                start_time INTEGER NOT NULL,
                end_time INTEGER NOT NULL,
                data_points INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;

        // Time series data table (optimized for time-based queries)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS time_series_data (
                series_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                value REAL NOT NULL,
                FOREIGN KEY (series_id) REFERENCES time_series_metadata (id),
                PRIMARY KEY (series_id, timestamp)
            )",
            [],
        )?;

        // Analysis results cache table
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL UNIQUE,
                analysis_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                result_data BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER
            )",
            [],
        )?;

        // Performance metrics table
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                memory_usage_mb REAL NOT NULL,
                data_size_mb REAL NOT NULL,
                success BOOLEAN NOT NULL,
                timestamp INTEGER NOT NULL
            )",
            [],
        )?;

        // Create indexes for optimal query performance
        self.create_indexes()?;

        Ok(())
    }

    /// Create database indexes for performance
    fn create_indexes(&self) -> Result<()> {
        let indexes = vec![
            "CREATE INDEX IF NOT EXISTS idx_series_name ON time_series_metadata (name)",
            "CREATE INDEX IF NOT EXISTS idx_series_time_range ON time_series_metadata (start_time, end_time)",
            "CREATE INDEX IF NOT EXISTS idx_data_timestamp ON time_series_data (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_data_series_time ON time_series_data (series_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_cache_key ON analysis_cache (cache_key)",
            "CREATE INDEX IF NOT EXISTS idx_cache_type ON analysis_cache (analysis_type)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON analysis_cache (expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_operation ON performance_metrics (operation)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics (timestamp)",
        ];

        for index_sql in indexes {
            self.connection.execute(index_sql, [])?;
        }

        Ok(())
    }

    /// Store time series in database
    pub fn store_timeseries(&self, name: &str, ts: &TimeSeries, description: Option<&str>) -> Result<i64> {
        let timestamps = ts.timestamps.clone();
        let values = ts.values.clone();

        if timestamps.is_empty() {
            return Err(TimeSeriesError::invalid_input("Empty time series").into());
        }

        let start_time = timestamps[0].timestamp();
        let end_time = timestamps[timestamps.len() - 1].timestamp();
        let now = Utc::now().timestamp();

        // Insert metadata
        let series_id = self.connection.execute(
            "INSERT INTO time_series_metadata
             (name, description, start_time, end_time, data_points, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![name, description, start_time, end_time, timestamps.len(), now, now],
        ).map_err(|e| PerformanceError::DatabaseError(e.to_string()))?;

        let series_id = self.connection.last_insert_rowid();

        // Insert data points in batches for better performance
        let batch_size = 1000;
        let mut stmt = self.connection.prepare(
            "INSERT INTO time_series_data (series_id, timestamp, value) VALUES (?1, ?2, ?3)"
        )?;

        for chunk in timestamps.chunks(batch_size).zip(values.chunks(batch_size)) {
            let tx = self.connection.unchecked_transaction()?;

            for (timestamp, value) in chunk.0.iter().zip(chunk.1.iter()) {
                stmt.execute(params![series_id, timestamp.timestamp(), value])?;
            }

            tx.commit()?;
        }

        Ok(series_id)
    }

    /// Load time series from database
    pub fn load_timeseries(&self, name: &str) -> Result<TimeSeries> {
        // Get series metadata
        let metadata: SeriesMetadata = self.connection.query_row(
            "SELECT id, start_time, end_time, data_points FROM time_series_metadata WHERE name = ?1",
            params![name],
            |row| {
                Ok(SeriesMetadata {
                    id: row.get(0)?,
                    start_time: row.get(1)?,
                    end_time: row.get(2)?,
                    data_points: row.get(3)?,
                })
            },
        ).map_err(|e| PerformanceError::DatabaseError(e.to_string()))?;

        // Load data points
        let mut stmt = self.connection.prepare(
            "SELECT timestamp, value FROM time_series_data
             WHERE series_id = ?1 ORDER BY timestamp"
        )?;

        let rows = stmt.query_map(params![metadata.id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })?;

        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        for row in rows {
            let (timestamp, value) = row?;
            let dt = DateTime::from_timestamp(timestamp, 0)
                .ok_or_else(|| TimeSeriesError::invalid_timestamp("Invalid timestamp"))?;
            timestamps.push(dt);
            values.push(value);
        }

        TimeSeries::new(name.to_string(), timestamps, values)
    }

    /// Query time series data within time range
    pub fn query_time_range(&self, name: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<TimeSeries> {
        // Get series ID
        let series_id: i64 = self.connection.query_row(
            "SELECT id FROM time_series_metadata WHERE name = ?1",
            params![name],
            |row| row.get(0),
        ).map_err(|e| PerformanceError::DatabaseError(e.to_string()))?;

        // Query data in range
        let mut stmt = self.connection.prepare(
            "SELECT timestamp, value FROM time_series_data
             WHERE series_id = ?1 AND timestamp BETWEEN ?2 AND ?3
             ORDER BY timestamp"
        )?;

        let rows = stmt.query_map(
            params![series_id, start.timestamp(), end.timestamp()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        )?;

        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        for row in rows {
            let (timestamp, value) = row?;
            let dt = DateTime::from_timestamp(timestamp, 0)
                .ok_or_else(|| TimeSeriesError::invalid_timestamp("Invalid timestamp"))?;
            timestamps.push(dt);
            values.push(value);
        }

        TimeSeries::new(name.to_string(), timestamps, values)
    }

    /// Get all available time series names
    pub fn list_series(&self) -> Result<Vec<String>> {
        let mut stmt = self.connection.prepare("SELECT name FROM time_series_metadata ORDER BY name")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        let mut names = Vec::new();
        for row in rows {
            names.push(row?);
        }

        Ok(names)
    }

    /// Delete time series
    pub fn delete_timeseries(&self, name: &str) -> Result<bool> {
        let series_id: Option<i64> = self.connection.query_row(
            "SELECT id FROM time_series_metadata WHERE name = ?1",
            params![name],
            |row| row.get(0),
        ).optional()?;

        if let Some(id) = series_id {
            // Delete data points
            self.connection.execute("DELETE FROM time_series_data WHERE series_id = ?1", params![id])?;

            // Delete metadata
            self.connection.execute("DELETE FROM time_series_metadata WHERE id = ?1", params![id])?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Store analysis results in cache
    pub fn cache_analysis_result(&self, key: &str, analysis_type: &str, parameters: &str, result: &[u8], ttl_hours: Option<u32>) -> Result<()> {
        let now = Utc::now().timestamp();
        let expires_at = ttl_hours.map(|hours| now + (hours as i64 * 3600));

        self.connection.execute(
            "INSERT OR REPLACE INTO analysis_cache
             (cache_key, analysis_type, parameters, result_data, created_at, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![key, analysis_type, parameters, result, now, expires_at],
        )?;

        Ok(())
    }

    /// Get cached analysis result
    pub fn get_cached_analysis(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let now = Utc::now().timestamp();

        let result = self.connection.query_row(
            "SELECT result_data FROM analysis_cache
             WHERE cache_key = ?1 AND (expires_at IS NULL OR expires_at > ?2)",
            params![key, now],
            |row| row.get::<_, Vec<u8>>(0),
        ).optional()?;

        Ok(result)
    }

    /// Record performance metrics
    pub fn record_performance(&self, operation: &str, execution_time_ms: u64, memory_usage_mb: f64, data_size_mb: f64, success: bool) -> Result<()> {
        let now = Utc::now().timestamp();

        self.connection.execute(
            "INSERT INTO performance_metrics
             (operation, execution_time_ms, memory_usage_mb, data_size_mb, success, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![operation, execution_time_ms as i64, memory_usage_mb, data_size_mb, success, now],
        )?;

        Ok(())
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self, operation: Option<&str>) -> Result<Vec<PerformanceRecord>> {
        let (sql, params): (String, Vec<rusqlite::types::Value>) = if let Some(op) = operation {
            (
                "SELECT operation, execution_time_ms, memory_usage_mb, data_size_mb, success, timestamp
                 FROM performance_metrics WHERE operation = ?1 ORDER BY timestamp DESC LIMIT 100".to_string(),
                vec![rusqlite::types::Value::Text(op.to_string())]
            )
        } else {
            (
                "SELECT operation, execution_time_ms, memory_usage_mb, data_size_mb, success, timestamp
                 FROM performance_metrics ORDER BY timestamp DESC LIMIT 100".to_string(),
                vec![]
            )
        };

        let mut stmt = self.connection.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| {
            Ok(PerformanceRecord {
                operation: row.get(0)?,
                execution_time_ms: row.get::<_, i64>(1)? as u64,
                memory_usage_mb: row.get(2)?,
                data_size_mb: row.get(3)?,
                success: row.get(4)?,
                timestamp: row.get(5)?,
            })
        })?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }

        Ok(records)
    }

    /// Vacuum database to optimize storage
    pub fn vacuum(&self) -> Result<()> {
        self.connection.execute("VACUUM", [])?;
        Ok(())
    }

    /// Clean up expired cache entries
    pub fn cleanup_cache(&self) -> Result<usize> {
        let now = Utc::now().timestamp();
        let deleted = self.connection.execute(
            "DELETE FROM analysis_cache WHERE expires_at IS NOT NULL AND expires_at <= ?1",
            params![now],
        )?;
        Ok(deleted)
    }
}

/// Time series database interface
pub trait TimeSeriesDb {
    fn store(&self, name: &str, ts: &TimeSeries) -> Result<()>;
    fn load(&self, name: &str) -> Result<TimeSeries>;
    fn query_range(&self, name: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<TimeSeries>;
    fn delete(&self, name: &str) -> Result<bool>;
    fn list(&self) -> Result<Vec<String>>;
}

impl TimeSeriesDb for DatabaseManager {
    fn store(&self, name: &str, ts: &TimeSeries) -> Result<()> {
        self.store_timeseries(name, ts, None)?;
        Ok(())
    }

    fn load(&self, name: &str) -> Result<TimeSeries> {
        self.load_timeseries(name)
    }

    fn query_range(&self, name: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<TimeSeries> {
        self.query_time_range(name, start, end)
    }

    fn delete(&self, name: &str) -> Result<bool> {
        self.delete_timeseries(name)
    }

    fn list(&self) -> Result<Vec<String>> {
        self.list_series()
    }
}

/// Time series metadata
#[derive(Debug)]
struct SeriesMetadata {
    id: i64,
    start_time: i64,
    end_time: i64,
    data_points: usize,
}

/// Performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub operation: String,
    pub execution_time_ms: u64,
    pub memory_usage_mb: f64,
    pub data_size_mb: f64,
    pub success: bool,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PerformanceConfig;
    use chrono::{Duration, Utc};

    fn create_test_timeseries() -> TimeSeries {
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..100)
            .map(|i| now + Duration::seconds(i))
            .collect();
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();

        TimeSeries::new(timestamps, values).unwrap()
    }

    #[test]
    fn test_database_creation() {
        let config = PerformanceConfig::default();
        let db = DatabaseManager::new(&config).unwrap();

        // Test that tables were created
        let table_count: i64 = db.connection.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'",
            [],
            |row| row.get(0)
        ).unwrap();

        assert!(table_count >= 4); // Should have our 4 main tables
    }

    #[test]
    fn test_store_and_load_timeseries() {
        let config = PerformanceConfig::default();
        let db = DatabaseManager::new(&config).unwrap();
        let ts = create_test_timeseries();

        // Store time series
        let series_id = db.store_timeseries("test_series", &ts, Some("Test description")).unwrap();
        assert!(series_id > 0);

        // Load time series
        let loaded_ts = db.load_timeseries("test_series").unwrap();
        assert_eq!(loaded_ts.len(), ts.len());
    }

    #[test]
    fn test_list_series() {
        let config = PerformanceConfig::default();
        let db = DatabaseManager::new(&config).unwrap();
        let ts = create_test_timeseries();

        db.store_timeseries("series_1", &ts, None).unwrap();
        db.store_timeseries("series_2", &ts, None).unwrap();

        let series_list = db.list_series().unwrap();
        assert!(series_list.contains(&"series_1".to_string()));
        assert!(series_list.contains(&"series_2".to_string()));
    }

    #[test]
    fn test_performance_recording() {
        let config = PerformanceConfig::default();
        let db = DatabaseManager::new(&config).unwrap();

        db.record_performance("test_operation", 1000, 10.5, 5.2, true).unwrap();

        let stats = db.get_performance_stats(Some("test_operation")).unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].operation, "test_operation");
        assert_eq!(stats[0].execution_time_ms, 1000);
    }
}