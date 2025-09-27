//! # Data Source Plugin Interface
//!
//! Defines the interface for data source plugins that can import data from various sources.

use super::{PluginError, PluginResult, PluginContext};
use crate::TimeSeries;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data source plugin trait
pub trait DataSourcePlugin: Send + Sync {
    /// Get supported data formats
    fn supported_formats(&self) -> Vec<String>;

    /// Get supported connection types
    fn supported_connections(&self) -> Vec<ConnectionType>;

    /// Validate connection configuration
    fn validate_connection(&self, config: &DataSourceConfig) -> PluginResult<()>;

    /// Test connection to data source
    fn test_connection(&self, config: &DataSourceConfig) -> PluginResult<ConnectionInfo>;

    /// Import data from the data source
    fn import_data(
        &self,
        config: &DataSourceConfig,
        context: &PluginContext,
    ) -> PluginResult<DataSourceResult>;

    /// Get schema information from the data source
    fn get_schema(&self, config: &DataSourceConfig) -> PluginResult<DataSchema>;

    /// Get preview of data (limited rows)
    fn get_preview(
        &self,
        config: &DataSourceConfig,
        limit: usize,
    ) -> PluginResult<DataPreview>;

    /// Support for streaming import
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Start streaming import (if supported)
    fn start_streaming(
        &self,
        _config: &DataSourceConfig,
        _context: &PluginContext,
    ) -> PluginResult<Box<dyn DataStream>> {
        Err(PluginError::ExecutionError("Streaming not supported".to_string()))
    }
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Data source type
    pub source_type: ConnectionType,
    /// Connection parameters
    pub connection: ConnectionConfig,
    /// Import parameters
    pub import: ImportConfig,
    /// Authentication parameters
    pub auth: Option<AuthConfig>,
    /// Additional options
    pub options: HashMap<String, serde_json::Value>,
}

/// Connection type supported by data sources
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectionType {
    /// File-based sources
    File,
    /// HTTP/REST API
    Http,
    /// Database connections
    Database,
    /// Message queues
    MessageQueue,
    /// Cloud storage
    CloudStorage,
    /// Real-time streams
    Stream,
    /// Custom connection type
    Custom(String),
}

/// Connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Connection URL or path
    pub url: String,
    /// Connection timeout in seconds
    pub timeout: Option<u64>,
    /// Retry configuration
    pub retry: Option<RetryConfig>,
    /// Connection pool settings
    pub pool: Option<PoolConfig>,
    /// Additional headers for HTTP connections
    pub headers: Option<HashMap<String, String>>,
    /// Query parameters
    pub params: Option<HashMap<String, String>>,
}

/// Import configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Data format
    pub format: String,
    /// Time column specification
    pub time_column: Option<ColumnSpec>,
    /// Value columns specification
    pub value_columns: Vec<ColumnSpec>,
    /// Filtering options
    pub filter: Option<FilterConfig>,
    /// Transformation options
    pub transform: Option<TransformConfig>,
    /// Batch size for large imports
    pub batch_size: Option<usize>,
}

/// Column specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSpec {
    /// Column name or index
    pub name: String,
    /// Column data type
    pub data_type: Option<DataType>,
    /// Column format (for dates, numbers, etc.)
    pub format: Option<String>,
    /// Whether this column is required
    pub required: bool,
    /// Default value if missing
    pub default: Option<serde_json::Value>,
}

/// Data type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Date,
    Time,
    Json,
    Binary,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// Credentials
    pub credentials: HashMap<String, String>,
    /// Token configuration
    pub token: Option<TokenConfig>,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Basic,
    Bearer,
    ApiKey,
    OAuth2,
    Custom(String),
}

/// Token configuration for OAuth and similar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    /// Token endpoint URL
    pub endpoint: String,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Scopes
    pub scopes: Vec<String>,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial delay between retries in milliseconds
    pub initial_delay: u64,
    /// Maximum delay between retries in milliseconds
    pub max_delay: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: 1000,
            max_delay: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Minimum pool size
    pub min_size: Option<u32>,
    /// Maximum pool size
    pub max_size: u32,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Idle timeout in seconds
    pub idle_timeout: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_size: Some(1),
            max_size: 10,
            connection_timeout: 30,
            idle_timeout: 600,
        }
    }
}

/// Filter configuration for data import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Date range filter
    pub date_range: Option<DateRangeFilter>,
    /// Value filters
    pub value_filters: Vec<ValueFilter>,
    /// Row limit
    pub limit: Option<usize>,
    /// Skip rows
    pub skip: Option<usize>,
}

/// Date range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRangeFilter {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}

/// Value filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFilter {
    pub column: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// Filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    In,
    NotIn,
    IsNull,
    IsNotNull,
}

/// Transform configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    /// Column transformations
    pub columns: Vec<ColumnTransform>,
    /// Data type conversions
    pub conversions: Vec<TypeConversion>,
    /// Value mappings
    pub mappings: Vec<ValueMapping>,
}

/// Column transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnTransform {
    pub source: String,
    pub target: String,
    pub operation: TransformOperation,
}

/// Transform operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOperation {
    Rename,
    Cast(DataType),
    Expression(String),
    Aggregate(AggregateFunction),
}

/// Aggregate functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateFunction {
    Sum,
    Average,
    Count,
    Min,
    Max,
    First,
    Last,
}

/// Type conversion specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConversion {
    pub column: String,
    pub from_type: DataType,
    pub to_type: DataType,
    pub format: Option<String>,
}

/// Value mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueMapping {
    pub column: String,
    pub mappings: HashMap<String, String>,
    pub default_value: Option<String>,
}

/// Connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    /// Whether connection is successful
    pub connected: bool,
    /// Connection latency in milliseconds
    pub latency: Option<u64>,
    /// Server/source information
    pub server_info: Option<String>,
    /// Available tables/datasets
    pub available_datasets: Vec<String>,
    /// Connection error if any
    pub error: Option<String>,
}

/// Data schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Schema name
    pub name: String,
    /// Column definitions
    pub columns: Vec<ColumnDefinition>,
    /// Primary key columns
    pub primary_key: Vec<String>,
    /// Indexes
    pub indexes: Vec<IndexDefinition>,
    /// Estimated row count
    pub estimated_rows: Option<u64>,
    /// Data size estimate
    pub estimated_size: Option<u64>,
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDefinition {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub unique: bool,
    pub description: Option<String>,
    pub sample_values: Vec<String>,
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
    pub index_type: String,
}

/// Data preview result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreview {
    /// Schema information
    pub schema: DataSchema,
    /// Sample rows (limited)
    pub rows: Vec<HashMap<String, serde_json::Value>>,
    /// Total row count (if known)
    pub total_rows: Option<u64>,
}

/// Data source import result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceResult {
    /// Imported time series data
    pub timeseries: Vec<TimeSeries>,
    /// Import statistics
    pub stats: ImportStats,
    /// Metadata about the import
    pub metadata: ImportMetadata,
    /// Warnings encountered during import
    pub warnings: Vec<String>,
}

/// Import statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportStats {
    /// Number of rows processed
    pub rows_processed: u64,
    /// Number of rows imported successfully
    pub rows_imported: u64,
    /// Number of rows skipped
    pub rows_skipped: u64,
    /// Number of errors encountered
    pub errors: u64,
    /// Import duration
    pub duration: std::time::Duration,
    /// Data size processed
    pub bytes_processed: u64,
}

/// Import metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportMetadata {
    /// Source identifier
    pub source_id: String,
    /// Import timestamp
    pub import_time: chrono::DateTime<chrono::Utc>,
    /// Data range imported
    pub data_range: Option<DateRangeFilter>,
    /// Column mappings used
    pub column_mappings: HashMap<String, String>,
    /// Transformations applied
    pub transformations: Vec<String>,
}

/// Streaming data interface
pub trait DataStream: Send + Sync {
    /// Get next batch of data
    fn next_batch(&mut self) -> PluginResult<Option<Vec<HashMap<String, serde_json::Value>>>>;

    /// Check if stream has more data
    fn has_more(&self) -> bool;

    /// Close the stream
    fn close(&mut self) -> PluginResult<()>;

    /// Get stream statistics
    fn stats(&self) -> StreamStats;
}

/// Streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStats {
    pub batches_processed: u64,
    pub rows_processed: u64,
    pub bytes_processed: u64,
    pub errors: u64,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub last_batch_time: Option<chrono::DateTime<chrono::Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_source_config_serialization() {
        let config = DataSourceConfig {
            source_type: ConnectionType::File,
            connection: ConnectionConfig {
                url: "file://test.csv".to_string(),
                timeout: Some(30),
                retry: Some(RetryConfig::default()),
                pool: None,
                headers: None,
                params: None,
            },
            import: ImportConfig {
                format: "csv".to_string(),
                time_column: Some(ColumnSpec {
                    name: "timestamp".to_string(),
                    data_type: Some(DataType::DateTime),
                    format: Some("%Y-%m-%d %H:%M:%S".to_string()),
                    required: true,
                    default: None,
                }),
                value_columns: vec![ColumnSpec {
                    name: "value".to_string(),
                    data_type: Some(DataType::Float),
                    format: None,
                    required: true,
                    default: None,
                }],
                filter: None,
                transform: None,
                batch_size: Some(1000),
            },
            auth: None,
            options: HashMap::new(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DataSourceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.source_type, ConnectionType::File);
    }

    #[test]
    fn test_connection_types() {
        assert_eq!(ConnectionType::File, ConnectionType::File);
        assert_ne!(ConnectionType::File, ConnectionType::Http);

        let custom = ConnectionType::Custom("mongodb".to_string());
        match custom {
            ConnectionType::Custom(name) => assert_eq!(name, "mongodb"),
            _ => panic!("Expected custom connection type"),
        }
    }

    #[test]
    fn test_filter_config() {
        let filter = FilterConfig {
            date_range: Some(DateRangeFilter {
                start: Some(chrono::Utc::now()),
                end: None,
            }),
            value_filters: vec![ValueFilter {
                column: "status".to_string(),
                operator: FilterOperator::Equal,
                value: serde_json::Value::String("active".to_string()),
            }],
            limit: Some(1000),
            skip: None,
        };

        assert!(filter.date_range.is_some());
        assert_eq!(filter.value_filters.len(), 1);
        assert_eq!(filter.limit, Some(1000));
    }
}