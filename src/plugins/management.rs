//! # Plugin Management System
//!
//! Handles plugin installation, updates, configuration, and lifecycle management.

use super::{
    PluginError, PluginResult, PluginRegistry, PluginInfo, PluginMetadata,
    PluginLoader
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

/// Plugin manager for handling installation, updates, and configuration
pub struct PluginManager {
    /// Plugin registry
    registry: PluginRegistry,
    /// Plugin loader
    loader: PluginLoader,
    /// Management configuration
    config: ManagementConfig,
    /// Plugin repositories
    repositories: Vec<PluginRepository>,
}

/// Plugin management configuration
#[derive(Debug, Clone)]
pub struct ManagementConfig {
    /// Plugin installation directory
    pub install_dir: PathBuf,
    /// Cache directory for downloads
    pub cache_dir: PathBuf,
    /// Configuration directory
    pub config_dir: PathBuf,
    /// Maximum download size in bytes
    pub max_download_size: usize,
    /// Download timeout
    pub download_timeout: std::time::Duration,
    /// Whether to verify signatures
    pub verify_signatures: bool,
    /// Whether to auto-update plugins
    pub auto_update: bool,
    /// Backup directory for plugin updates
    pub backup_dir: Option<PathBuf>,
}

impl Default for ManagementConfig {
    fn default() -> Self {
        let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let base_dir = home_dir.join(".config").join("chronos");

        Self {
            install_dir: base_dir.join("plugins"),
            cache_dir: base_dir.join("cache"),
            config_dir: base_dir.join("config"),
            max_download_size: 100 * 1024 * 1024, // 100MB
            download_timeout: std::time::Duration::from_secs(300), // 5 minutes
            verify_signatures: true,
            auto_update: false,
            backup_dir: Some(base_dir.join("backups")),
        }
    }
}

/// Plugin repository specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRepository {
    /// Repository name
    pub name: String,
    /// Repository URL
    pub url: String,
    /// Repository type
    pub repo_type: RepositoryType,
    /// Authentication configuration
    pub auth: Option<RepositoryAuth>,
    /// Whether repository is enabled
    pub enabled: bool,
    /// Repository priority (higher = higher priority)
    pub priority: i32,
}

/// Repository types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepositoryType {
    /// Git repository
    Git,
    /// HTTP/HTTPS repository
    Http,
    /// Local file system
    Local,
    /// Package registry (npm-like)
    Registry,
}

/// Repository authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryAuth {
    /// Authentication type
    pub auth_type: AuthType,
    /// Credentials
    pub credentials: HashMap<String, String>,
}

/// Authentication types for repositories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Token,
    BasicAuth,
    SSH,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> PluginResult<Self> {
        Self::with_config(ManagementConfig::default())
    }

    /// Create a new plugin manager with custom configuration
    pub fn with_config(config: ManagementConfig) -> PluginResult<Self> {
        // Create directories if they don't exist
        fs::create_dir_all(&config.install_dir)?;
        fs::create_dir_all(&config.cache_dir)?;
        fs::create_dir_all(&config.config_dir)?;

        if let Some(backup_dir) = &config.backup_dir {
            fs::create_dir_all(backup_dir)?;
        }

        let registry = PluginRegistry::new();
        let loader = PluginLoader::new();

        Ok(Self {
            registry,
            loader,
            config,
            repositories: Vec::new(),
        })
    }

    /// Initialize plugin manager (discover existing plugins)
    pub fn initialize(&self) -> PluginResult<usize> {
        self.registry.add_search_path(&self.config.install_dir)?;
        self.registry.discover_plugins()
    }

    /// Add a plugin repository
    pub fn add_repository(&mut self, repository: PluginRepository) -> PluginResult<()> {
        self.repositories.push(repository);
        self.repositories.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }

    /// Remove a plugin repository
    pub fn remove_repository(&mut self, name: &str) -> PluginResult<()> {
        self.repositories.retain(|repo| repo.name != name);
        Ok(())
    }

    /// List available plugins from all repositories
    pub fn list_available_plugins(&self) -> PluginResult<Vec<AvailablePlugin>> {
        let mut available_plugins = Vec::new();

        for repository in &self.repositories {
            if !repository.enabled {
                continue;
            }

            match self.query_repository(repository) {
                Ok(plugins) => available_plugins.extend(plugins),
                Err(e) => {
                    eprintln!("Failed to query repository {}: {}", repository.name, e);
                }
            }
        }

        Ok(available_plugins)
    }

    /// Query a specific repository for available plugins
    fn query_repository(&self, repository: &PluginRepository) -> PluginResult<Vec<AvailablePlugin>> {
        match repository.repo_type {
            RepositoryType::Local => self.query_local_repository(repository),
            RepositoryType::Http => self.query_http_repository(repository),
            RepositoryType::Git => self.query_git_repository(repository),
            RepositoryType::Registry => self.query_registry_repository(repository),
        }
    }

    /// Query local file system repository
    fn query_local_repository(&self, repository: &PluginRepository) -> PluginResult<Vec<AvailablePlugin>> {
        let repo_path = PathBuf::from(&repository.url);
        let mut plugins = Vec::new();

        if repo_path.exists() && repo_path.is_dir() {
            for entry in fs::read_dir(&repo_path)? {
                let entry = entry?;
                let plugin_path = entry.path();

                if plugin_path.is_dir() {
                    if let Ok(metadata) = self.load_plugin_metadata_from_path(&plugin_path) {
                        plugins.push(AvailablePlugin {
                            metadata,
                            repository: repository.name.clone(),
                            download_url: plugin_path.to_string_lossy().to_string(),
                            checksum: None,
                            signature: None,
                            size: self.calculate_directory_size(&plugin_path).ok(),
                        });
                    }
                }
            }
        }

        Ok(plugins)
    }

    /// Query HTTP repository (placeholder implementation)
    fn query_http_repository(&self, _repository: &PluginRepository) -> PluginResult<Vec<AvailablePlugin>> {
        // In a real implementation, this would make HTTP requests to query plugins
        // For now, return empty list
        Ok(Vec::new())
    }

    /// Query Git repository (placeholder implementation)
    fn query_git_repository(&self, _repository: &PluginRepository) -> PluginResult<Vec<AvailablePlugin>> {
        // In a real implementation, this would clone/fetch Git repositories
        // For now, return empty list
        Ok(Vec::new())
    }

    /// Query package registry (placeholder implementation)
    fn query_registry_repository(&self, _repository: &PluginRepository) -> PluginResult<Vec<AvailablePlugin>> {
        // In a real implementation, this would query package registries like npm
        // For now, return empty list
        Ok(Vec::new())
    }

    /// Install a plugin
    pub fn install_plugin(&self, config: &PluginInstallConfig) -> PluginResult<InstallResult> {
        let start_time = std::time::Instant::now();

        // Check if plugin is already installed
        if let Some(existing) = self.registry.get_plugin(&config.plugin_id) {
            if !config.force {
                return Err(PluginError::ConfigError(format!(
                    "Plugin {} is already installed. Use --force to reinstall.",
                    config.plugin_id
                )));
            }

            // Backup existing plugin if backup is enabled
            if self.config.backup_dir.is_some() {
                self.backup_plugin(&existing)?;
            }
        }

        // Find plugin in repositories
        let available_plugin = self.find_plugin_in_repositories(&config.plugin_id, &config.version)?;

        // Download plugin
        let download_path = self.download_plugin(&available_plugin)?;

        // Validate plugin
        self.validate_plugin_package(&download_path, &available_plugin)?;

        // Install plugin
        let install_path = self.extract_and_install_plugin(&download_path, &config.plugin_id)?;

        // Load and register plugin
        let metadata = self.load_plugin_metadata_from_path(&install_path)?;
        self.registry.register_discovered_plugin(metadata.clone(), install_path.clone())?;

        // Configure plugin if configuration provided
        if let Some(plugin_config) = &config.config {
            self.configure_plugin(&config.plugin_id, plugin_config)?;
        }

        let duration = start_time.elapsed();

        let size = self.calculate_directory_size(&install_path).unwrap_or(0);

        Ok(InstallResult {
            plugin_id: config.plugin_id.clone(),
            version: metadata.version,
            install_path,
            duration,
            size,
            warnings: Vec::new(),
        })
    }

    /// Uninstall a plugin
    pub fn uninstall_plugin(&self, plugin_id: &str, config: &PluginUninstallConfig) -> PluginResult<UninstallResult> {
        let start_time = std::time::Instant::now();

        // Check if plugin exists
        let plugin_info = self.registry.get_plugin(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        // Backup plugin if requested
        if config.backup {
            self.backup_plugin(&plugin_info)?;
        }

        // Remove plugin files
        let plugin_size = self.calculate_directory_size(&plugin_info.path).unwrap_or(0);

        if config.remove_data {
            // Remove plugin data directory
            let data_dir = self.get_plugin_data_dir(plugin_id);
            if data_dir.exists() {
                fs::remove_dir_all(&data_dir)?;
            }
        }

        // Remove plugin installation
        fs::remove_dir_all(&plugin_info.path)?;

        // Remove from registry
        self.registry.remove_plugin(plugin_id)?;

        let duration = start_time.elapsed();

        Ok(UninstallResult {
            plugin_id: plugin_id.to_string(),
            removed_size: plugin_size,
            duration,
            backup_path: if config.backup {
                Some(self.get_plugin_backup_path(plugin_id))
            } else {
                None
            },
        })
    }

    /// Update a plugin
    pub fn update_plugin(&self, config: &PluginUpdateConfig) -> PluginResult<UpdateResult> {
        let start_time = std::time::Instant::now();

        // Get current plugin info
        let current_plugin = self.registry.get_plugin(&config.plugin_id)
            .ok_or_else(|| PluginError::NotFound(config.plugin_id.to_string()))?;

        // Find latest version in repositories
        let available_plugin = self.find_plugin_in_repositories(&config.plugin_id, &config.target_version)?;

        // Check if update is needed
        let current_version = &current_plugin.metadata.version;
        let target_version = &available_plugin.metadata.version;

        if current_version == target_version && !config.force {
            return Ok(UpdateResult {
                plugin_id: config.plugin_id.clone(),
                old_version: current_version.clone(),
                new_version: target_version.clone(),
                updated: false,
                duration: start_time.elapsed(),
                warnings: vec!["Plugin is already up to date".to_string()],
            });
        }

        // Backup current plugin
        if self.config.backup_dir.is_some() {
            self.backup_plugin(&current_plugin)?;
        }

        // Download new version
        let download_path = self.download_plugin(&available_plugin)?;

        // Validate plugin
        self.validate_plugin_package(&download_path, &available_plugin)?;

        // Remove old version
        fs::remove_dir_all(&current_plugin.path)?;

        // Install new version
        let install_path = self.extract_and_install_plugin(&download_path, &config.plugin_id)?;

        // Load and register new plugin
        let metadata = self.load_plugin_metadata_from_path(&install_path)?;
        self.registry.remove_plugin(&config.plugin_id)?;
        self.registry.register_discovered_plugin(metadata, install_path)?;

        let duration = start_time.elapsed();

        Ok(UpdateResult {
            plugin_id: config.plugin_id.clone(),
            old_version: current_version.clone(),
            new_version: target_version.clone(),
            updated: true,
            duration,
            warnings: Vec::new(),
        })
    }

    /// Configure a plugin
    pub fn configure_plugin(&self, plugin_id: &str, config: &serde_json::Value) -> PluginResult<()> {
        // Get plugin info
        let plugin_info = self.registry.get_plugin(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        // Validate configuration against plugin's schema
        if let Some(schema) = &plugin_info.metadata.config_schema {
            self.validate_config_against_schema(config, schema)?;
        }

        // Save configuration
        let config_path = self.get_plugin_config_path(plugin_id);
        let config_content = serde_json::to_string_pretty(config)?;
        fs::write(config_path, config_content)?;

        Ok(())
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, plugin_id: &str) -> PluginResult<serde_json::Value> {
        let config_path = self.get_plugin_config_path(plugin_id);

        if config_path.exists() {
            let config_content = fs::read_to_string(config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_content)?;
            Ok(config)
        } else {
            Ok(serde_json::Value::Object(serde_json::Map::new()))
        }
    }

    /// List installed plugins
    pub fn list_installed_plugins(&self) -> Vec<PluginInfo> {
        self.registry.list_plugins()
    }

    /// Get the number of repositories
    pub fn repository_count(&self) -> usize {
        self.repositories.len()
    }

    /// Get repository by name
    pub fn get_repository(&self, name: &str) -> Option<&PluginRepository> {
        self.repositories.iter().find(|repo| repo.name == name)
    }

    /// Get plugin status
    pub fn get_plugin_status(&self, plugin_id: &str) -> PluginResult<PluginStatusInfo> {
        let plugin_info = self.registry.get_plugin(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        let config_exists = self.get_plugin_config_path(plugin_id).exists();
        let data_dir = self.get_plugin_data_dir(plugin_id);
        let data_size = if data_dir.exists() {
            self.calculate_directory_size(&data_dir).unwrap_or(0)
        } else {
            0
        };

        let install_size = self.calculate_directory_size(&plugin_info.path).unwrap_or(0);

        Ok(PluginStatusInfo {
            plugin_info: plugin_info.clone(),
            config_exists,
            data_size,
            install_size,
            last_used: None, // Would be tracked in real implementation
            health_status: HealthStatus::Unknown, // Would be determined by health checks
        })
    }

    /// Check for plugin updates
    pub fn check_updates(&self) -> PluginResult<Vec<UpdateInfo>> {
        let installed_plugins = self.list_installed_plugins();
        let mut updates = Vec::new();

        for plugin in installed_plugins {
            if let Ok(available_plugin) = self.find_latest_plugin_version(&plugin.metadata.id) {
                if available_plugin.metadata.version != plugin.metadata.version {
                    updates.push(UpdateInfo {
                        plugin_id: plugin.metadata.id.clone(),
                        current_version: plugin.metadata.version.clone(),
                        available_version: available_plugin.metadata.version.clone(),
                        update_priority: self.calculate_update_priority(&plugin.metadata, &available_plugin.metadata),
                        breaking_changes: self.check_breaking_changes(&plugin.metadata, &available_plugin.metadata),
                    });
                }
            }
        }

        Ok(updates)
    }

    // Helper methods

    fn find_plugin_in_repositories(&self, plugin_id: &str, version: &Option<String>) -> PluginResult<AvailablePlugin> {
        let available_plugins = self.list_available_plugins()?;

        let candidates: Vec<_> = available_plugins
            .into_iter()
            .filter(|p| p.metadata.id == plugin_id)
            .filter(|p| {
                if let Some(required_version) = version {
                    &p.metadata.version == required_version
                } else {
                    true
                }
            })
            .collect();

        if candidates.is_empty() {
            return Err(PluginError::NotFound(format!(
                "Plugin {} not found in any repository",
                plugin_id
            )));
        }

        // Return the latest version if no specific version requested
        let selected = if version.is_none() {
            candidates.into_iter().max_by(|a, b| a.metadata.version.cmp(&b.metadata.version))
        } else {
            candidates.into_iter().next()
        };

        selected.ok_or_else(|| PluginError::NotFound(format!(
            "No suitable version of plugin {} found",
            plugin_id
        )))
    }

    fn find_latest_plugin_version(&self, plugin_id: &str) -> PluginResult<AvailablePlugin> {
        self.find_plugin_in_repositories(plugin_id, &None)
    }

    fn download_plugin(&self, plugin: &AvailablePlugin) -> PluginResult<PathBuf> {
        let filename = format!("{}-{}.zip", plugin.metadata.id, plugin.metadata.version);
        let download_path = self.config.cache_dir.join(filename);

        // For local repositories, just copy the directory
        if plugin.download_url.starts_with('/') || plugin.download_url.starts_with("file://") {
            let source_path = PathBuf::from(plugin.download_url.trim_start_matches("file://"));
            self.copy_directory(&source_path, &download_path.parent().unwrap().join(&plugin.metadata.id))?;
            return Ok(download_path.parent().unwrap().join(&plugin.metadata.id));
        }

        // For remote URLs, this would implement actual downloading
        // For now, return an error
        Err(PluginError::ExecutionError("Remote download not implemented".to_string()))
    }

    fn validate_plugin_package(&self, _path: &Path, _plugin: &AvailablePlugin) -> PluginResult<()> {
        // Validate checksums, signatures, etc.
        // For now, just return Ok
        Ok(())
    }

    fn extract_and_install_plugin(&self, source_path: &Path, plugin_id: &str) -> PluginResult<PathBuf> {
        let install_path = self.config.install_dir.join(plugin_id);

        if install_path.exists() {
            fs::remove_dir_all(&install_path)?;
        }

        self.copy_directory(source_path, &install_path)?;
        Ok(install_path)
    }

    fn backup_plugin(&self, plugin: &PluginInfo) -> PluginResult<()> {
        if let Some(backup_dir) = &self.config.backup_dir {
            let backup_path = backup_dir.join(format!(
                "{}-{}-{}.backup",
                plugin.metadata.id,
                plugin.metadata.version,
                chrono::Utc::now().format("%Y%m%d_%H%M%S")
            ));
            self.copy_directory(&plugin.path, &backup_path)?;
        }
        Ok(())
    }

    fn copy_directory(&self, source: &Path, dest: &Path) -> PluginResult<()> {
        if source.is_dir() {
            fs::create_dir_all(dest)?;
            for entry in fs::read_dir(source)? {
                let entry = entry?;
                let source_path = entry.path();
                let dest_path = dest.join(entry.file_name());

                if source_path.is_dir() {
                    self.copy_directory(&source_path, &dest_path)?;
                } else {
                    fs::copy(source_path, dest_path)?;
                }
            }
        } else {
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, dest)?;
        }
        Ok(())
    }

    fn calculate_directory_size(&self, path: &Path) -> PluginResult<u64> {
        let mut size = 0;

        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();

                if entry_path.is_dir() {
                    size += self.calculate_directory_size(&entry_path)?;
                } else {
                    size += entry.metadata()?.len();
                }
            }
        } else {
            size = path.metadata()?.len();
        }

        Ok(size)
    }

    fn load_plugin_metadata_from_path(&self, path: &Path) -> PluginResult<PluginMetadata> {
        let metadata_file = path.join("plugin.toml");
        if metadata_file.exists() {
            let content = fs::read_to_string(metadata_file)?;
            let metadata: PluginMetadata = toml::from_str(&content)
                .map_err(|e| PluginError::ConfigError(format!("Invalid TOML: {}", e)))?;
            return Ok(metadata);
        }

        let metadata_file = path.join("plugin.json");
        if metadata_file.exists() {
            let content = fs::read_to_string(metadata_file)?;
            let metadata: PluginMetadata = serde_json::from_str(&content)
                .map_err(|e| PluginError::ConfigError(format!("Invalid JSON: {}", e)))?;
            return Ok(metadata);
        }

        Err(PluginError::ConfigError("No plugin metadata file found".to_string()))
    }

    fn validate_config_against_schema(&self, _config: &serde_json::Value, _schema: &serde_json::Value) -> PluginResult<()> {
        // JSON Schema validation would be implemented here
        // For now, just return Ok
        Ok(())
    }

    fn get_plugin_config_path(&self, plugin_id: &str) -> PathBuf {
        self.config.config_dir.join(format!("{}.json", plugin_id))
    }

    fn get_plugin_data_dir(&self, plugin_id: &str) -> PathBuf {
        self.config.config_dir.join("data").join(plugin_id)
    }

    fn get_plugin_backup_path(&self, plugin_id: &str) -> PathBuf {
        self.config.backup_dir.as_ref()
            .unwrap_or(&self.config.config_dir)
            .join(format!("{}.backup", plugin_id))
    }

    fn calculate_update_priority(&self, _current: &PluginMetadata, _available: &PluginMetadata) -> UpdatePriority {
        // Would analyze version differences, security updates, etc.
        UpdatePriority::Normal
    }

    fn check_breaking_changes(&self, _current: &PluginMetadata, _available: &PluginMetadata) -> bool {
        // Would check API version compatibility, etc.
        false
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default plugin manager")
    }
}

// Configuration structs

/// Plugin installation configuration
#[derive(Debug, Clone)]
pub struct PluginInstallConfig {
    pub plugin_id: String,
    pub version: Option<String>,
    pub repository: Option<String>,
    pub force: bool,
    pub config: Option<serde_json::Value>,
}

/// Plugin uninstall configuration
#[derive(Debug, Clone)]
pub struct PluginUninstallConfig {
    pub backup: bool,
    pub remove_data: bool,
}

impl Default for PluginUninstallConfig {
    fn default() -> Self {
        Self {
            backup: true,
            remove_data: false,
        }
    }
}

/// Plugin update configuration
#[derive(Debug, Clone)]
pub struct PluginUpdateConfig {
    pub plugin_id: String,
    pub target_version: Option<String>,
    pub force: bool,
}

// Result structs

/// Available plugin information
#[derive(Debug, Clone)]
pub struct AvailablePlugin {
    pub metadata: PluginMetadata,
    pub repository: String,
    pub download_url: String,
    pub checksum: Option<String>,
    pub signature: Option<String>,
    pub size: Option<u64>,
}

/// Installation result
#[derive(Debug, Clone)]
pub struct InstallResult {
    pub plugin_id: String,
    pub version: String,
    pub install_path: PathBuf,
    pub duration: std::time::Duration,
    pub size: u64,
    pub warnings: Vec<String>,
}

/// Uninstallation result
#[derive(Debug, Clone)]
pub struct UninstallResult {
    pub plugin_id: String,
    pub removed_size: u64,
    pub duration: std::time::Duration,
    pub backup_path: Option<PathBuf>,
}

/// Update result
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub plugin_id: String,
    pub old_version: String,
    pub new_version: String,
    pub updated: bool,
    pub duration: std::time::Duration,
    pub warnings: Vec<String>,
}

/// Plugin status information
#[derive(Debug, Clone)]
pub struct PluginStatusInfo {
    pub plugin_info: PluginInfo,
    pub config_exists: bool,
    pub data_size: u64,
    pub install_size: u64,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub health_status: HealthStatus,
}

/// Plugin health status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Error,
    Unknown,
}

/// Update information
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub plugin_id: String,
    pub current_version: String,
    pub available_version: String,
    pub update_priority: UpdatePriority,
    pub breaking_changes: bool,
}

/// Update priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Low,
    Normal,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_plugin_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = ManagementConfig {
            install_dir: temp_dir.path().join("plugins"),
            cache_dir: temp_dir.path().join("cache"),
            config_dir: temp_dir.path().join("config"),
            ..ManagementConfig::default()
        };

        let manager = PluginManager::with_config(config).unwrap();
        assert_eq!(manager.repositories.len(), 0);
    }

    #[test]
    fn test_repository_management() {
        let mut manager = PluginManager::new().unwrap();

        let repo = PluginRepository {
            name: "test-repo".to_string(),
            url: "https://example.com/plugins".to_string(),
            repo_type: RepositoryType::Http,
            auth: None,
            enabled: true,
            priority: 100,
        };

        manager.add_repository(repo).unwrap();
        assert_eq!(manager.repositories.len(), 1);

        manager.remove_repository("test-repo").unwrap();
        assert_eq!(manager.repositories.len(), 0);
    }

    #[test]
    fn test_plugin_install_config() {
        let config = PluginInstallConfig {
            plugin_id: "test-plugin".to_string(),
            version: Some("1.0.0".to_string()),
            repository: None,
            force: false,
            config: None,
        };

        assert_eq!(config.plugin_id, "test-plugin");
        assert_eq!(config.version, Some("1.0.0".to_string()));
        assert!(!config.force);
    }

    #[test]
    fn test_update_priority_ordering() {
        assert!(UpdatePriority::Critical > UpdatePriority::High);
        assert!(UpdatePriority::High > UpdatePriority::Normal);
        assert!(UpdatePriority::Normal > UpdatePriority::Low);
    }
}