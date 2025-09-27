//! Data import functionality for various file formats
//!
//! This module provides comprehensive data import capabilities for time series data,
//! supporting CSV, JSON, and future Parquet formats with robust preprocessing.

use std::collections::HashMap;
use std::path::Path;
use chrono::{DateTime, Utc, NaiveDateTime};
// use serde_json::Value; // TODO: Use when implementing JSON import
// use regex::Regex; // TODO: Use for advanced pattern matching

use crate::{TimeSeries, Frequency, MissingValuePolicy, Result, TimeSeriesError};

/// Import configuration for data loading
#[derive(Debug, Clone)]
pub struct ImportConfig {
    /// Name for the resulting time series
    pub name: String,

    /// Missing value policy to apply
    pub missing_value_policy: MissingValuePolicy,

    /// Expected frequency (if known)
    pub frequency: Option<Frequency>,

    /// CSV-specific settings
    pub csv_config: CsvConfig,

    /// JSON-specific settings
    pub json_config: JsonConfig,
}

/// CSV import configuration
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Delimiter character
    pub delimiter: char,

    /// Whether the first row contains headers
    pub has_headers: bool,

    /// Timestamp column name or index
    pub timestamp_column: TimestampColumn,

    /// Value columns to import
    pub value_columns: Vec<String>,

    /// Custom column mappings
    pub column_mappings: HashMap<String, String>,

    /// Skip rows at the beginning
    pub skip_rows: usize,

    /// Maximum rows to read (None for all)
    pub max_rows: Option<usize>,
}

/// JSON import configuration
#[derive(Debug, Clone)]
pub struct JsonConfig {
    /// Path to timestamp field in nested JSON
    pub timestamp_path: Vec<String>,

    /// Paths to value fields
    pub value_paths: Vec<Vec<String>>,

    /// Expected JSON structure format
    pub format: JsonFormat,
}

/// Supported JSON formats
#[derive(Debug, Clone)]
pub enum JsonFormat {
    /// Array of objects: [{"timestamp": "...", "value": ...}, ...]
    ArrayOfObjects,

    /// Single object with arrays: {"timestamps": [...], "values": [...]}
    ObjectWithArrays,

    /// Nested structure with configurable paths
    Nested,
}

/// Timestamp column specification
#[derive(Debug, Clone)]
pub enum TimestampColumn {
    /// Column name
    Name(String),

    /// Column index (0-based)
    Index(usize),

    /// Auto-detect timestamp column
    AutoDetect,

    /// Combine date and time columns
    Combined {
        date_column: String,
        time_column: String,
    },

    /// Unix timestamp column
    Unix(String),
}

/// Import result with metadata
#[derive(Debug)]
pub struct ImportResult {
    /// The imported time series
    pub timeseries: TimeSeries,

    /// Import statistics
    pub stats: ImportStats,

    /// Detected configuration (for auto-detection)
    pub detected_config: Option<ImportConfig>,
}

/// Import statistics and metadata
#[derive(Debug)]
pub struct ImportStats {
    /// Total rows processed
    pub rows_processed: usize,

    /// Rows skipped due to errors
    pub rows_skipped: usize,

    /// Missing values encountered
    pub missing_values: usize,

    /// Parsing errors encountered
    pub parsing_errors: Vec<String>,

    /// Detected file format details
    pub format_info: HashMap<String, String>,

    /// Import duration
    pub duration: std::time::Duration,
}

impl Default for ImportConfig {
    fn default() -> Self {
        ImportConfig {
            name: "imported_data".to_string(),
            missing_value_policy: MissingValuePolicy::NaN,
            frequency: None,
            csv_config: CsvConfig::default(),
            json_config: JsonConfig::default(),
        }
    }
}

impl Default for CsvConfig {
    fn default() -> Self {
        CsvConfig {
            delimiter: ',',
            has_headers: true,
            timestamp_column: TimestampColumn::AutoDetect,
            value_columns: Vec::new(),
            column_mappings: HashMap::new(),
            skip_rows: 0,
            max_rows: None,
        }
    }
}

impl Default for JsonConfig {
    fn default() -> Self {
        JsonConfig {
            timestamp_path: vec!["timestamp".to_string()],
            value_paths: vec![vec!["value".to_string()]],
            format: JsonFormat::ArrayOfObjects,
        }
    }
}

impl ImportStats {
    pub fn new() -> Self {
        ImportStats {
            rows_processed: 0,
            rows_skipped: 0,
            missing_values: 0,
            parsing_errors: Vec::new(),
            format_info: HashMap::new(),
            duration: std::time::Duration::default(),
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.rows_processed == 0 {
            0.0
        } else {
            (self.rows_processed - self.rows_skipped) as f64 / self.rows_processed as f64
        }
    }
}

/// Main import interface - auto-detects file format
pub fn import_from_file<P: AsRef<Path>>(path: P, config: Option<ImportConfig>) -> Result<ImportResult> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| TimeSeriesError::validation("Unable to determine file type"))?;

    let config = config.unwrap_or_default();

    match extension.to_lowercase().as_str() {
        "csv" | "tsv" | "txt" => import_csv(path, config),
        "json" | "jsonl" => import_json(path, config),
        "parquet" => import_parquet(path, config),
        _ => Err(Box::new(TimeSeriesError::validation(
            format!("Unsupported file format: {}", extension)
        ))),
    }
}

/// Import from CSV file with comprehensive auto-detection
pub fn import_csv<P: AsRef<Path>>(path: P, mut config: ImportConfig) -> Result<ImportResult> {
    use std::fs::File;
    use std::io::BufReader;

    let start_time = std::time::Instant::now();
    let mut stats = ImportStats::new();

    let file = File::open(path.as_ref())
        .map_err(|e| TimeSeriesError::validation(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);

    // Auto-detect CSV format if needed
    if matches!(config.csv_config.timestamp_column, TimestampColumn::AutoDetect) {
        config = detect_csv_format(path.as_ref(), config)?;
    }

    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(config.csv_config.delimiter as u8)
        .has_headers(config.csv_config.has_headers)
        .from_reader(reader);

    let headers = if config.csv_config.has_headers {
        csv_reader.headers()?.clone()
    } else {
        // Generate default column names
        let first_record = csv_reader.records().next();
        match first_record {
            Some(record) => {
                let record = record?;
                csv::StringRecord::from(
                    (0..record.len())
                        .map(|i| format!("column_{}", i))
                        .collect::<Vec<_>>()
                )
            }
            None => return Err(Box::new(TimeSeriesError::validation("Empty CSV file"))),
        }
    };

    // Determine timestamp column index
    let timestamp_idx = match &config.csv_config.timestamp_column {
        TimestampColumn::Name(name) => {
            headers.iter().position(|h| h == name)
                .ok_or_else(|| TimeSeriesError::validation(format!("Timestamp column '{}' not found", name)))?
        }
        TimestampColumn::Index(idx) => *idx,
        TimestampColumn::AutoDetect => {
            detect_timestamp_column(&headers)?
        }
        TimestampColumn::Combined { .. } => {
            return Err(Box::new(TimeSeriesError::validation("Combined date/time columns not yet implemented")));
        }
        TimestampColumn::Unix(_) => {
            return Err(Box::new(TimeSeriesError::validation("Unix timestamp columns not yet implemented")));
        }
    };

    // Determine value column indices
    let value_indices = if config.csv_config.value_columns.is_empty() {
        // Auto-detect: use all numeric columns except timestamp
        (0..headers.len())
            .filter(|&i| i != timestamp_idx)
            .collect::<Vec<_>>()
    } else {
        let mut indices = Vec::new();
        for name in &config.csv_config.value_columns {
            match headers.iter().position(|h| h == name) {
                Some(idx) => indices.push(idx),
                None => return Err(Box::new(TimeSeriesError::validation(format!("Value column '{}' not found", name)))),
            }
        }
        indices
    };

    // Parse CSV data
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut metadata = HashMap::new();

    for (row_num, result) in csv_reader.records().enumerate() {
        match result {
            Ok(record) => {
                stats.rows_processed += 1;

                // Skip rows if configured
                if row_num < config.csv_config.skip_rows {
                    continue;
                }

                // Check max rows limit
                if let Some(max_rows) = config.csv_config.max_rows {
                    if timestamps.len() >= max_rows {
                        break;
                    }
                }

                // Parse timestamp
                if let Some(timestamp_str) = record.get(timestamp_idx) {
                    match parse_timestamp(timestamp_str) {
                        Ok(timestamp) => {
                            timestamps.push(timestamp);

                            // Parse values (for now, just take the first value column)
                            if let Some(value_idx) = value_indices.first() {
                                if let Some(value_str) = record.get(*value_idx) {
                                    match parse_value(value_str, &config.missing_value_policy) {
                                        Ok(value) => values.push(value),
                                        Err(_) => {
                                            stats.missing_values += 1;
                                            values.push(f64::NAN);
                                        }
                                    }
                                } else {
                                    stats.missing_values += 1;
                                    values.push(f64::NAN);
                                }
                            }
                        }
                        Err(e) => {
                            stats.rows_skipped += 1;
                            stats.parsing_errors.push(format!("Row {}: {}", row_num + 1, e));
                        }
                    }
                }
            }
            Err(e) => {
                stats.rows_skipped += 1;
                stats.parsing_errors.push(format!("Row {}: {}", row_num + 1, e));
            }
        }
    }

    // Add metadata
    metadata.insert("source_file".to_string(),
                   path.as_ref().to_string_lossy().to_string());
    metadata.insert("import_format".to_string(), "csv".to_string());
    metadata.insert("delimiter".to_string(), config.csv_config.delimiter.to_string());

    // Create time series
    let timeseries = TimeSeries::with_options(
        config.name,
        timestamps,
        values,
        metadata,
        config.frequency,
        config.missing_value_policy,
    )?;

    stats.duration = start_time.elapsed();

    // Add format info to stats
    stats.format_info.insert("delimiter".to_string(), config.csv_config.delimiter.to_string());
    stats.format_info.insert("has_headers".to_string(), config.csv_config.has_headers.to_string());
    stats.format_info.insert("timestamp_column".to_string(), timestamp_idx.to_string());

    Ok(ImportResult {
        timeseries,
        stats,
        detected_config: None, // TODO: Add detected config
    })
}

/// Import from JSON file
pub fn import_json<P: AsRef<Path>>(_path: P, _config: ImportConfig) -> Result<ImportResult> {
    // TODO: Implement JSON import
    Err(Box::new(TimeSeriesError::validation("JSON import not yet implemented")))
}

/// Import from Parquet file (requires Polars)
pub fn import_parquet<P: AsRef<Path>>(_path: P, _config: ImportConfig) -> Result<ImportResult> {
    // TODO: Implement Parquet import when Polars is available
    Err(Box::new(TimeSeriesError::validation("Parquet import not yet implemented (requires Polars)")))
}

/// Auto-detect CSV format and configuration
fn detect_csv_format<P: AsRef<Path>>(path: P, mut config: ImportConfig) -> Result<ImportConfig> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines = reader.lines();

    // Read first few lines for detection
    let sample_lines: Vec<String> = lines
        .take(10)
        .collect::<std::io::Result<Vec<_>>>()?;

    if sample_lines.is_empty() {
        return Err(Box::new(TimeSeriesError::validation("Empty file")));
    }

    // Detect delimiter
    let delimiters = [',', ';', '\t', '|'];
    let mut delimiter_scores = HashMap::new();

    for &delimiter in &delimiters {
        let mut score = 0;
        for line in &sample_lines {
            score += line.matches(delimiter).count();
        }
        delimiter_scores.insert(delimiter, score);
    }

    let detected_delimiter = *delimiter_scores
        .iter()
        .max_by_key(|(_, &score)| score)
        .map(|(delimiter, _)| delimiter)
        .unwrap_or(&',');

    config.csv_config.delimiter = detected_delimiter;

    // Detect if headers are present by checking if first row contains non-numeric data
    if let Some(first_line) = sample_lines.first() {
        let fields: Vec<&str> = first_line.split(detected_delimiter).collect();
        let has_headers = fields.iter().any(|field| {
            field.trim().parse::<f64>().is_err() && !field.trim().is_empty()
        });
        config.csv_config.has_headers = has_headers;
    }

    Ok(config)
}

/// Detect which column contains timestamps
fn detect_timestamp_column(headers: &csv::StringRecord) -> Result<usize> {
    // Common timestamp column names
    let timestamp_patterns = [
        "timestamp", "time", "datetime", "date", "created_at", "updated_at",
        "ts", "dt", "when", "occurred_at", "recorded_at", "unix_time",
    ];

    // Check for exact matches first
    for (i, header) in headers.iter().enumerate() {
        let header_lower = header.to_lowercase();
        for pattern in &timestamp_patterns {
            if header_lower == *pattern || header_lower.contains(pattern) {
                return Ok(i);
            }
        }
    }

    // If no match found, assume first column
    Ok(0)
}

/// Parse timestamp string into DateTime<Utc>
fn parse_timestamp(timestamp_str: &str) -> Result<DateTime<Utc>> {
    let timestamp_str = timestamp_str.trim();

    // Try various timestamp formats
    let formats = [
        // ISO 8601 formats
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%.fZ",

        // Date only
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",

        // With timezone
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
    ];

    // Try parsing as Unix timestamp first
    if let Ok(unix_timestamp) = timestamp_str.parse::<i64>() {
        // Check if it's seconds or milliseconds
        if unix_timestamp > 1_000_000_000_000 {
            // Milliseconds
            if let Some(dt) = DateTime::from_timestamp_millis(unix_timestamp) {
                return Ok(dt);
            }
        } else {
            // Seconds
            if let Some(dt) = DateTime::from_timestamp(unix_timestamp, 0) {
                return Ok(dt);
            }
        }
    }

    // Try parsing with various formats
    for format in &formats {
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(timestamp_str, format) {
            return Ok(DateTime::from_naive_utc_and_offset(naive_dt, Utc));
        }
    }

    Err(Box::new(TimeSeriesError::invalid_timestamp(
        format!("Unable to parse timestamp: {}", timestamp_str)
    )))
}

/// Parse value string into f64
fn parse_value(value_str: &str, missing_policy: &MissingValuePolicy) -> Result<f64> {
    let value_str = value_str.trim();

    // Handle empty/missing values
    if value_str.is_empty() || value_str.eq_ignore_ascii_case("null")
        || value_str.eq_ignore_ascii_case("na") || value_str.eq_ignore_ascii_case("nan") {
        return match missing_policy {
            MissingValuePolicy::NaN => Ok(f64::NAN),
            MissingValuePolicy::Error => Err(Box::new(TimeSeriesError::missing_data("Missing value encountered"))),
            MissingValuePolicy::Default(default_str) => {
                default_str.parse::<f64>()
                    .map_err(|_| TimeSeriesError::validation("Invalid default value"))
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
            _ => Ok(f64::NAN), // Other policies handled during post-processing
        };
    }

    // Try to parse as number
    value_str.parse::<f64>()
        .map_err(|_| Box::new(TimeSeriesError::validation(
            format!("Unable to parse value: {}", value_str)
        )) as Box<dyn std::error::Error + Send + Sync>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use chrono::Datelike;

    #[test]
    fn test_parse_timestamp_iso() {
        let timestamp = parse_timestamp("2024-01-01 12:00:00").unwrap();
        assert_eq!(timestamp.year(), 2024);
        assert_eq!(timestamp.month(), 1);
        assert_eq!(timestamp.day(), 1);
    }

    #[test]
    fn test_parse_timestamp_unix() {
        let timestamp = parse_timestamp("1704110400").unwrap(); // 2024-01-01 12:00:00 UTC
        assert_eq!(timestamp.year(), 2024);
    }

    #[test]
    fn test_parse_value_normal() {
        let value = parse_value("123.45", &MissingValuePolicy::NaN).unwrap();
        assert_eq!(value, 123.45);
    }

    #[test]
    fn test_parse_value_missing_nan() {
        let value = parse_value("", &MissingValuePolicy::NaN).unwrap();
        assert!(value.is_nan());
    }

    #[test]
    fn test_parse_value_missing_error() {
        let result = parse_value("", &MissingValuePolicy::Error);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_timestamp_column() {
        let headers = csv::StringRecord::from(vec!["id", "timestamp", "value", "name"]);
        let timestamp_idx = detect_timestamp_column(&headers).unwrap();
        assert_eq!(timestamp_idx, 1);
    }

    #[test]
    fn test_csv_import_basic() -> Result<()> {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,value")?;
        writeln!(temp_file, "2024-01-01 00:00:00,100.5")?;
        writeln!(temp_file, "2024-01-01 01:00:00,102.3")?;
        writeln!(temp_file, "2024-01-01 02:00:00,101.8")?;

        let config = ImportConfig::default();
        let result = import_csv(temp_file.path(), config)?;

        assert_eq!(result.timeseries.len(), 3);
        assert_eq!(result.timeseries.name, "imported_data");
        assert_eq!(result.stats.rows_processed, 3);
        assert_eq!(result.stats.rows_skipped, 0);

        Ok(())
    }
}