//! # Plugin Registry
//!
//! Handles plugin discovery, registration, and lifecycle management.

use super::{Plugin, PluginError, PluginResult, PluginType, ApiVersion, CURRENT_API_VERSION};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Plugin metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin identifier (must be unique)
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin homepage URL
    pub homepage: Option<String>,
    /// Plugin repository URL
    pub repository: Option<String>,
    /// Plugin license
    pub license: String,
    /// Required API version
    pub api_version: ApiVersion,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
    /// Plugin keywords/tags
    pub keywords: Vec<String>,
    /// Plugin configuration schema (JSON Schema)
    pub config_schema: Option<serde_json::Value>,
}

/// Plugin dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Dependency name
    pub name: String,
    /// Required version range
    pub version: String,
    /// Whether this dependency is optional
    pub optional: bool,
}

/// Plugin information stored in the registry
#[derive(Clone)]
pub struct PluginInfo {
    /// Plugin metadata
    pub metadata: PluginMetadata,
    /// Plugin type
    pub plugin_type: PluginType,
    /// Path to plugin library or source
    pub path: PathBuf,
    /// Plugin load status
    pub status: PluginLoadStatus,
    /// Plugin instance (if loaded)
    pub instance: Option<Arc<dyn Plugin>>,
    /// Plugin configuration
    pub config: Option<serde_json::Value>,
}

impl std::fmt::Debug for PluginInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginInfo")
            .field("metadata", &self.metadata)
            .field("plugin_type", &self.plugin_type)
            .field("path", &self.path)
            .field("status", &self.status)
            .field("instance", &self.instance.as_ref().map(|_| "Plugin instance"))
            .field("config", &self.config)
            .finish()
    }
}

/// Plugin load status
#[derive(Debug, Clone, PartialEq)]
pub enum PluginLoadStatus {
    /// Plugin is discovered but not loaded
    Discovered,
    /// Plugin is loaded and ready
    Loaded,
    /// Plugin failed to load
    Failed(String),
    /// Plugin is disabled
    Disabled,
}

/// Plugin registry for managing discovered and loaded plugins
pub struct PluginRegistry {
    /// Map of plugin ID to plugin info
    plugins: RwLock<HashMap<String, PluginInfo>>,
    /// Plugin search paths
    search_paths: RwLock<Vec<PathBuf>>,
    /// Registry configuration
    config: RegistryConfig,
}

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Whether to auto-discover plugins on startup
    pub auto_discovery: bool,
    /// Whether to auto-load discovered plugins
    pub auto_load: bool,
    /// Maximum number of plugins to load
    pub max_plugins: Option<usize>,
    /// Plugin load timeout
    pub load_timeout: std::time::Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            auto_load: false,
            max_plugins: Some(100),
            load_timeout: std::time::Duration::from_secs(30),
        }
    }
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Create a new plugin registry with custom configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        let registry = Self {
            plugins: RwLock::new(HashMap::new()),
            search_paths: RwLock::new(Vec::new()),
            config,
        };

        // Add default search paths
        registry.add_default_search_paths();

        registry
    }

    /// Add default plugin search paths
    fn add_default_search_paths(&self) {
        let mut search_paths = self.search_paths.write().unwrap();

        // Add user plugin directory
        if let Some(home_dir) = dirs::home_dir() {
            search_paths.push(home_dir.join(".config/chronos/plugins"));
        }

        // Add system plugin directory
        search_paths.push(PathBuf::from("/usr/local/share/chronos/plugins"));

        // Add local plugin directory
        if let Ok(current_dir) = std::env::current_dir() {
            search_paths.push(current_dir.join("plugins"));
        }
    }

    /// Add a search path for plugins
    pub fn add_search_path<P: AsRef<Path>>(&self, path: P) -> PluginResult<()> {
        let path = path.as_ref().to_path_buf();
        let mut search_paths = self.search_paths.write().unwrap();

        if !search_paths.contains(&path) {
            search_paths.push(path);
        }

        Ok(())
    }

    /// Remove a search path
    pub fn remove_search_path<P: AsRef<Path>>(&self, path: P) -> PluginResult<()> {
        let path = path.as_ref().to_path_buf();
        let mut search_paths = self.search_paths.write().unwrap();
        search_paths.retain(|p| p != &path);
        Ok(())
    }

    /// Get all search paths
    pub fn get_search_paths(&self) -> Vec<PathBuf> {
        self.search_paths.read().unwrap().clone()
    }

    /// Discover plugins in search paths
    pub fn discover_plugins(&self) -> PluginResult<usize> {
        let search_paths = self.get_search_paths();
        let mut discovered_count = 0;

        for search_path in search_paths {
            if !search_path.exists() {
                continue;
            }

            discovered_count += self.discover_plugins_in_path(&search_path)?;
        }

        Ok(discovered_count)
    }

    /// Discover plugins in a specific path
    fn discover_plugins_in_path(&self, path: &Path) -> PluginResult<usize> {
        let mut discovered_count = 0;

        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();

                if entry_path.is_dir() {
                    // Look for plugin.toml or plugin.json files
                    let metadata_file = entry_path.join("plugin.toml")
                        .exists()
                        .then(|| entry_path.join("plugin.toml"))
                        .or_else(|| {
                            entry_path.join("plugin.json")
                                .exists()
                                .then(|| entry_path.join("plugin.json"))
                        });

                    if let Some(metadata_file) = metadata_file {
                        match self.load_plugin_metadata(&metadata_file) {
                            Ok(metadata) => {
                                self.register_discovered_plugin(metadata, entry_path)?;
                                discovered_count += 1;
                            }
                            Err(e) => {
                                eprintln!("Failed to load plugin metadata from {:?}: {}", metadata_file, e);
                            }
                        }
                    }
                } else if entry_path.extension().and_then(|s| s.to_str()) == Some("so") ||
                         entry_path.extension().and_then(|s| s.to_str()) == Some("dylib") ||
                         entry_path.extension().and_then(|s| s.to_str()) == Some("dll") {
                    // Dynamic library plugin (future enhancement)
                    // For now, skip these
                }
            }
        }

        Ok(discovered_count)
    }

    /// Load plugin metadata from file
    fn load_plugin_metadata(&self, path: &Path) -> PluginResult<PluginMetadata> {
        let content = std::fs::read_to_string(path)?;

        let metadata = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::from_str(&content)
                .map_err(|e| PluginError::ConfigError(format!("Invalid TOML: {}", e)))?
        } else {
            serde_json::from_str(&content)
                .map_err(|e| PluginError::ConfigError(format!("Invalid JSON: {}", e)))?
        };

        Ok(metadata)
    }

    /// Register a discovered plugin
    pub fn register_discovered_plugin(&self, metadata: PluginMetadata, path: PathBuf) -> PluginResult<()> {
        // Validate API version compatibility
        if !metadata.api_version.is_compatible(&CURRENT_API_VERSION) {
            return Err(PluginError::VersionIncompatible {
                required: CURRENT_API_VERSION.to_string(),
                found: metadata.api_version.to_string(),
            });
        }

        let plugin_type = self.infer_plugin_type(&metadata)?;

        let plugin_info = PluginInfo {
            metadata: metadata.clone(),
            plugin_type,
            path,
            status: PluginLoadStatus::Discovered,
            instance: None,
            config: None,
        };

        let mut plugins = self.plugins.write().unwrap();
        plugins.insert(metadata.id.clone(), plugin_info);

        Ok(())
    }

    /// Infer plugin type from metadata
    fn infer_plugin_type(&self, metadata: &PluginMetadata) -> PluginResult<PluginType> {
        // Check keywords for plugin type hints
        for keyword in &metadata.keywords {
            match keyword.as_str() {
                "data-source" | "import" | "connector" => return Ok(PluginType::DataSource),
                "analysis" | "statistics" | "algorithm" => return Ok(PluginType::Analysis),
                "visualization" | "plot" | "chart" => return Ok(PluginType::Visualization),
                _ => {}
            }
        }

        // Default to analysis plugin
        Ok(PluginType::Analysis)
    }

    /// Get plugin by ID
    pub fn get_plugin(&self, id: &str) -> Option<PluginInfo> {
        self.plugins.read().unwrap().get(id).cloned()
    }

    /// List all plugins
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        self.plugins.read().unwrap().values().cloned().collect()
    }

    /// List plugins by type
    pub fn list_plugins_by_type(&self, plugin_type: PluginType) -> Vec<PluginInfo> {
        self.plugins
            .read()
            .unwrap()
            .values()
            .filter(|info| info.plugin_type == plugin_type)
            .cloned()
            .collect()
    }

    /// List loaded plugins
    pub fn list_loaded_plugins(&self) -> Vec<PluginInfo> {
        self.plugins
            .read()
            .unwrap()
            .values()
            .filter(|info| matches!(info.status, PluginLoadStatus::Loaded))
            .cloned()
            .collect()
    }

    /// Enable plugin
    pub fn enable_plugin(&self, id: &str) -> PluginResult<()> {
        let mut plugins = self.plugins.write().unwrap();
        if let Some(plugin_info) = plugins.get_mut(id) {
            if matches!(plugin_info.status, PluginLoadStatus::Disabled) {
                plugin_info.status = PluginLoadStatus::Discovered;
            }
            Ok(())
        } else {
            Err(PluginError::NotFound(id.to_string()))
        }
    }

    /// Disable plugin
    pub fn disable_plugin(&self, id: &str) -> PluginResult<()> {
        let mut plugins = self.plugins.write().unwrap();
        if let Some(plugin_info) = plugins.get_mut(id) {
            plugin_info.status = PluginLoadStatus::Disabled;
            plugin_info.instance = None;
            Ok(())
        } else {
            Err(PluginError::NotFound(id.to_string()))
        }
    }

    /// Remove plugin from registry
    pub fn remove_plugin(&self, id: &str) -> PluginResult<()> {
        let mut plugins = self.plugins.write().unwrap();
        if plugins.remove(id).is_some() {
            Ok(())
        } else {
            Err(PluginError::NotFound(id.to_string()))
        }
    }

    /// Get registry statistics
    pub fn get_stats(&self) -> RegistryStats {
        let plugins = self.plugins.read().unwrap();
        let total = plugins.len();
        let loaded = plugins.values().filter(|p| matches!(p.status, PluginLoadStatus::Loaded)).count();
        let disabled = plugins.values().filter(|p| matches!(p.status, PluginLoadStatus::Disabled)).count();
        let failed = plugins.values().filter(|p| matches!(p.status, PluginLoadStatus::Failed(_))).count();

        let mut by_type = HashMap::new();
        for plugin in plugins.values() {
            *by_type.entry(plugin.plugin_type.clone()).or_insert(0) += 1;
        }

        RegistryStats {
            total,
            loaded,
            disabled,
            failed,
            by_type,
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize)]
pub struct RegistryStats {
    pub total: usize,
    pub loaded: usize,
    pub disabled: usize,
    pub failed: usize,
    pub by_type: HashMap<PluginType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_registry_creation() {
        let registry = PluginRegistry::new();
        let stats = registry.get_stats();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_search_path_management() {
        let registry = PluginRegistry::new();
        let test_path = PathBuf::from("/test/path");

        registry.add_search_path(&test_path).unwrap();
        let paths = registry.get_search_paths();
        assert!(paths.contains(&test_path));

        registry.remove_search_path(&test_path).unwrap();
        let paths = registry.get_search_paths();
        assert!(!paths.contains(&test_path));
    }

    #[test]
    fn test_plugin_type_inference() {
        let registry = PluginRegistry::new();

        let mut metadata = PluginMetadata {
            id: "test".to_string(),
            name: "Test Plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Test Author".to_string(),
            homepage: None,
            repository: None,
            license: "MIT".to_string(),
            api_version: CURRENT_API_VERSION,
            dependencies: vec![],
            keywords: vec!["data-source".to_string()],
            config_schema: None,
        };

        assert_eq!(registry.infer_plugin_type(&metadata).unwrap(), PluginType::DataSource);

        metadata.keywords = vec!["visualization".to_string()];
        assert_eq!(registry.infer_plugin_type(&metadata).unwrap(), PluginType::Visualization);

        metadata.keywords = vec!["analysis".to_string()];
        assert_eq!(registry.infer_plugin_type(&metadata).unwrap(), PluginType::Analysis);

        metadata.keywords = vec![];
        assert_eq!(registry.infer_plugin_type(&metadata).unwrap(), PluginType::Analysis);
    }
}