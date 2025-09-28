//! # Plugin Loader
//!
//! Handles dynamic loading of plugins from various sources.

use super::{Plugin, PluginError, PluginResult, PluginMetadata, PluginType};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Plugin load errors
#[derive(Debug, Error)]
pub enum PluginLoadError {
    #[error("Dynamic library loading not supported on this platform")]
    UnsupportedPlatform,

    #[error("Plugin library not found: {0}")]
    LibraryNotFound(String),

    #[error("Plugin symbol not found: {0}")]
    SymbolNotFound(String),

    #[error("Plugin initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Plugin validation failed: {0}")]
    ValidationFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<PluginLoadError> for PluginError {
    fn from(error: PluginLoadError) -> Self {
        PluginError::LoadError(error.to_string())
    }
}

/// Plugin loader responsible for loading plugins from various sources
pub struct PluginLoader {
    /// Plugin search paths
    search_paths: Vec<std::path::PathBuf>,
    /// Loader configuration
    config: LoaderConfig,
}

/// Plugin loader configuration
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Whether to enable dynamic library loading
    pub enable_dynamic_loading: bool,
    /// Plugin load timeout
    pub load_timeout: std::time::Duration,
    /// Whether to validate plugins after loading
    pub validate_after_load: bool,
    /// Whether to allow unsigned plugins
    pub allow_unsigned: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_loading: false, // Disabled by default for security
            load_timeout: std::time::Duration::from_secs(30),
            validate_after_load: true,
            allow_unsigned: false,
        }
    }
}

impl PluginLoader {
    /// Create a new plugin loader
    pub fn new() -> Self {
        Self::with_config(LoaderConfig::default())
    }

    /// Create a new plugin loader with custom configuration
    pub fn with_config(config: LoaderConfig) -> Self {
        Self {
            search_paths: Vec::new(),
            config,
        }
    }

    /// Add a search path for plugins
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Get the number of search paths
    pub fn search_path_count(&self) -> usize {
        self.search_paths.len()
    }

    /// Get the configuration
    pub fn get_config(&self) -> &LoaderConfig {
        &self.config
    }

    /// Load a plugin from a given path
    pub fn load_plugin(&self, plugin_path: &Path, metadata: &PluginMetadata) -> PluginResult<Arc<dyn Plugin>> {
        // For now, we'll implement a simple factory-based approach
        // In a full implementation, this would handle dynamic library loading

        if plugin_path.is_dir() {
            // Look for a plugin entry point
            self.load_source_plugin(plugin_path, metadata)
        } else if self.is_dynamic_library(plugin_path) {
            self.load_dynamic_plugin(plugin_path, metadata)
        } else {
            Err(PluginError::LoadError(format!(
                "Unsupported plugin format: {:?}",
                plugin_path
            )))
        }
    }

    /// Check if a path is a dynamic library
    fn is_dynamic_library(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
            matches!(extension, "so" | "dylib" | "dll")
        } else {
            false
        }
    }

    /// Load a source-based plugin (Rust source or script)
    fn load_source_plugin(&self, plugin_path: &Path, metadata: &PluginMetadata) -> PluginResult<Arc<dyn Plugin>> {
        // For demonstration, create a mock plugin based on type
        // In a real implementation, this would compile and load Rust source

        match self.infer_plugin_type_from_path(plugin_path) {
            PluginType::DataSource => {
                Ok(Arc::new(MockDataSourcePlugin::new(metadata.clone())))
            }
            PluginType::Analysis => {
                Ok(Arc::new(MockAnalysisPlugin::new(metadata.clone())))
            }
            PluginType::Visualization => {
                Ok(Arc::new(MockVisualizationPlugin::new(metadata.clone())))
            }
        }
    }

    /// Load a dynamic library plugin
    fn load_dynamic_plugin(&self, _plugin_path: &Path, _metadata: &PluginMetadata) -> PluginResult<Arc<dyn Plugin>> {
        if !self.config.enable_dynamic_loading {
            return Err(PluginLoadError::UnsupportedPlatform.into());
        }

        // Dynamic library loading would be implemented here
        // This is a complex feature that requires careful security considerations
        Err(PluginLoadError::UnsupportedPlatform.into())
    }

    /// Infer plugin type from the plugin path contents
    fn infer_plugin_type_from_path(&self, plugin_path: &Path) -> PluginType {
        // Look for specific files or patterns to determine plugin type
        if plugin_path.join("data_source.rs").exists() ||
           plugin_path.join("connector.rs").exists() {
            PluginType::DataSource
        } else if plugin_path.join("analysis.rs").exists() ||
                  plugin_path.join("algorithm.rs").exists() {
            PluginType::Analysis
        } else if plugin_path.join("visualization.rs").exists() ||
                  plugin_path.join("plot.rs").exists() {
            PluginType::Visualization
        } else {
            // Default to analysis
            PluginType::Analysis
        }
    }

    /// Validate a loaded plugin
    pub fn validate_plugin(&self, plugin: &dyn Plugin) -> PluginResult<()> {
        if !self.config.validate_after_load {
            return Ok(());
        }

        // Check if plugin metadata is valid
        let metadata = plugin.metadata();
        if metadata.id.is_empty() {
            return Err(PluginError::ValidationError("Plugin ID cannot be empty".to_string()));
        }

        if metadata.name.is_empty() {
            return Err(PluginError::ValidationError("Plugin name cannot be empty".to_string()));
        }

        // Check if plugin is healthy
        let status = plugin.status();
        if !status.is_healthy {
            return Err(PluginError::ValidationError(
                status.last_error.unwrap_or_else(|| "Plugin is not healthy".to_string())
            ));
        }

        Ok(())
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

// Mock plugin implementations for demonstration

use super::{PluginCapabilities, PluginStatus};

/// Mock data source plugin
pub struct MockDataSourcePlugin {
    metadata: PluginMetadata,
    capabilities: PluginCapabilities,
    status: PluginStatus,
}

impl MockDataSourcePlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            capabilities: PluginCapabilities {
                streaming: true,
                real_time: false,
                external_deps: false,
                configurable: true,
                data_formats: vec!["csv".to_string(), "json".to_string(), "parquet".to_string()],
                max_data_size: Some(100 * 1024 * 1024), // 100MB
            },
            status: PluginStatus::default(),
        }
    }
}

impl Plugin for MockDataSourcePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::DataSource
    }

    fn capabilities(&self) -> &PluginCapabilities {
        &self.capabilities
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> Result<(), PluginError> {
        self.status.is_initialized = true;
        Ok(())
    }

    fn validate_config(&self, _config: &serde_json::Value) -> Result<(), PluginError> {
        Ok(())
    }

    fn status(&self) -> PluginStatus {
        self.status.clone()
    }

    fn cleanup(&mut self) -> Result<(), PluginError> {
        self.status.is_initialized = false;
        Ok(())
    }
}

/// Mock analysis plugin
pub struct MockAnalysisPlugin {
    metadata: PluginMetadata,
    capabilities: PluginCapabilities,
    status: PluginStatus,
}

impl MockAnalysisPlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            capabilities: PluginCapabilities {
                streaming: false,
                real_time: true,
                external_deps: false,
                configurable: true,
                data_formats: vec!["csv".to_string(), "json".to_string()],
                max_data_size: Some(50 * 1024 * 1024), // 50MB
            },
            status: PluginStatus::default(),
        }
    }
}

impl Plugin for MockAnalysisPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Analysis
    }

    fn capabilities(&self) -> &PluginCapabilities {
        &self.capabilities
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> Result<(), PluginError> {
        self.status.is_initialized = true;
        Ok(())
    }

    fn validate_config(&self, _config: &serde_json::Value) -> Result<(), PluginError> {
        Ok(())
    }

    fn status(&self) -> PluginStatus {
        self.status.clone()
    }

    fn cleanup(&mut self) -> Result<(), PluginError> {
        self.status.is_initialized = false;
        Ok(())
    }
}

/// Mock visualization plugin
pub struct MockVisualizationPlugin {
    metadata: PluginMetadata,
    capabilities: PluginCapabilities,
    status: PluginStatus,
}

impl MockVisualizationPlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            capabilities: PluginCapabilities {
                streaming: false,
                real_time: false,
                external_deps: true,
                configurable: true,
                data_formats: vec!["csv".to_string(), "json".to_string()],
                max_data_size: Some(200 * 1024 * 1024), // 200MB
            },
            status: PluginStatus::default(),
        }
    }
}

impl Plugin for MockVisualizationPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Visualization
    }

    fn capabilities(&self) -> &PluginCapabilities {
        &self.capabilities
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> Result<(), PluginError> {
        self.status.is_initialized = true;
        Ok(())
    }

    fn validate_config(&self, _config: &serde_json::Value) -> Result<(), PluginError> {
        Ok(())
    }

    fn status(&self) -> PluginStatus {
        self.status.clone()
    }

    fn cleanup(&mut self) -> Result<(), PluginError> {
        self.status.is_initialized = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::CURRENT_API_VERSION;

    fn create_test_metadata() -> PluginMetadata {
        PluginMetadata {
            id: "test-plugin".to_string(),
            name: "Test Plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "A test plugin".to_string(),
            author: "Test Author".to_string(),
            homepage: None,
            repository: None,
            license: "MIT".to_string(),
            api_version: CURRENT_API_VERSION,
            dependencies: vec![],
            keywords: vec!["test".to_string()],
            config_schema: None,
        }
    }

    #[test]
    fn test_loader_creation() {
        let loader = PluginLoader::new();
        assert_eq!(loader.search_paths.len(), 0);
        assert!(!loader.config.enable_dynamic_loading);
    }

    #[test]
    fn test_search_path_management() {
        let mut loader = PluginLoader::new();
        loader.add_search_path("/test/path");
        assert_eq!(loader.search_paths.len(), 1);
        assert_eq!(loader.search_paths[0], std::path::PathBuf::from("/test/path"));
    }

    #[test]
    fn test_plugin_type_inference() {
        let loader = PluginLoader::new();

        // Test default case
        let temp_dir = tempfile::TempDir::new().unwrap();
        let plugin_type = loader.infer_plugin_type_from_path(temp_dir.path());
        assert_eq!(plugin_type, PluginType::Analysis);
    }

    #[test]
    fn test_plugin_validation() {
        let loader = PluginLoader::new();
        let metadata = create_test_metadata();
        let mut plugin = MockAnalysisPlugin::new(metadata);
        plugin.initialize(&serde_json::Value::Null).unwrap();

        assert!(loader.validate_plugin(&plugin).is_ok());
    }

    #[test]
    fn test_plugin_validation_failure() {
        let loader = PluginLoader::new();
        let mut metadata = create_test_metadata();
        metadata.id = "".to_string(); // Invalid empty ID
        let plugin = MockAnalysisPlugin::new(metadata);

        assert!(loader.validate_plugin(&plugin).is_err());
    }
}