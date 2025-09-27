//! # Plugin System for Chronos
//!
//! This module provides a comprehensive plugin architecture that allows users to extend
//! Chronos with custom data sources, analysis methods, and visualization capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

pub mod registry;
pub mod loader;
pub mod data_source;
pub mod analysis;
pub mod visualization;
pub mod management;

// Re-export commonly used types
pub use registry::{PluginRegistry, PluginInfo, PluginMetadata};
pub use loader::{PluginLoader, PluginLoadError};
pub use data_source::{DataSourcePlugin, DataSourceConfig, DataSourceResult};
pub use analysis::{AnalysisPlugin, AnalysisConfig, AnalysisResult as PluginAnalysisResult};
pub use visualization::{VisualizationPlugin, VisualizationConfig, VisualizationResult};
pub use management::{PluginManager, PluginInstallConfig, PluginUpdateConfig};

/// Plugin system errors
#[derive(Debug, Error)]
pub enum PluginError {
    #[error("Plugin not found: {0}")]
    NotFound(String),

    #[error("Plugin load error: {0}")]
    LoadError(String),

    #[error("Plugin configuration error: {0}")]
    ConfigError(String),

    #[error("Plugin execution error: {0}")]
    ExecutionError(String),

    #[error("Plugin validation error: {0}")]
    ValidationError(String),

    #[error("Plugin dependency error: {0}")]
    DependencyError(String),

    #[error("Plugin version incompatible: required {required}, found {found}")]
    VersionIncompatible { required: String, found: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Plugin types supported by the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginType {
    /// Data source plugins for importing data
    DataSource,
    /// Analysis plugins for custom statistical methods
    Analysis,
    /// Visualization plugins for custom plot types
    Visualization,
}

/// Plugin API version
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ApiVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Check if this version is compatible with the required version
    pub fn is_compatible(&self, required: &ApiVersion) -> bool {
        // Major version must match, minor/patch can be higher
        self.major == required.major &&
        (self.minor > required.minor ||
         (self.minor == required.minor && self.patch >= required.patch))
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Current plugin API version
pub const CURRENT_API_VERSION: ApiVersion = ApiVersion::new(0, 1, 0);

/// Plugin capability flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    /// Whether the plugin supports streaming data
    pub streaming: bool,
    /// Whether the plugin supports real-time processing
    pub real_time: bool,
    /// Whether the plugin requires external dependencies
    pub external_deps: bool,
    /// Whether the plugin supports configuration
    pub configurable: bool,
    /// Supported data formats
    pub data_formats: Vec<String>,
    /// Maximum data size the plugin can handle (in bytes)
    pub max_data_size: Option<usize>,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            streaming: false,
            real_time: false,
            external_deps: false,
            configurable: true,
            data_formats: vec!["csv".to_string(), "json".to_string()],
            max_data_size: None,
        }
    }
}

/// Base trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Get plugin type
    fn plugin_type(&self) -> PluginType;

    /// Get plugin capabilities
    fn capabilities(&self) -> &PluginCapabilities;

    /// Initialize the plugin with configuration
    fn initialize(&mut self, config: &serde_json::Value) -> Result<(), PluginError>;

    /// Validate plugin configuration
    fn validate_config(&self, config: &serde_json::Value) -> Result<(), PluginError>;

    /// Get plugin status and health
    fn status(&self) -> PluginStatus;

    /// Cleanup plugin resources
    fn cleanup(&mut self) -> Result<(), PluginError>;
}

/// Plugin status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStatus {
    pub is_healthy: bool,
    pub is_initialized: bool,
    pub last_error: Option<String>,
    pub memory_usage: Option<usize>,
    pub execution_count: u64,
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for PluginStatus {
    fn default() -> Self {
        Self {
            is_healthy: true,
            is_initialized: false,
            last_error: None,
            memory_usage: None,
            execution_count: 0,
            last_execution: None,
        }
    }
}

/// Plugin configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin identifier
    pub id: String,
    /// Plugin type
    pub plugin_type: PluginType,
    /// Plugin-specific configuration
    pub config: serde_json::Value,
    /// Whether the plugin is enabled
    pub enabled: bool,
    /// Plugin priority (higher number = higher priority)
    pub priority: i32,
    /// Plugin tags for categorization
    pub tags: Vec<String>,
}

/// Result type for plugin operations
pub type PluginResult<T> = Result<T, PluginError>;

/// Plugin execution context
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Request ID for tracking
    pub request_id: String,
    /// User-provided parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Working directory for temporary files
    pub work_dir: PathBuf,
    /// Maximum execution time
    pub timeout: Option<std::time::Duration>,
}

impl PluginContext {
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            parameters: HashMap::new(),
            work_dir: std::env::temp_dir(),
            timeout: Some(std::time::Duration::from_secs(300)), // 5 minutes default
        }
    }

    pub fn with_parameter(mut self, key: String, value: serde_json::Value) -> Self {
        self.parameters.insert(key, value);
        self
    }

    pub fn with_work_dir(mut self, work_dir: PathBuf) -> Self {
        self.work_dir = work_dir;
        self
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

#[cfg(test)]
pub mod tests;