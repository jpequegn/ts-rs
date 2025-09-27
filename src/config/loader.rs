//! # Configuration Loader
//!
//! This module handles loading configurations from multiple sources and formats.

use super::*;
use figment::{Figment, providers::{Format, Toml, Yaml, Json, Env}};
use std::path::{Path, PathBuf};
use std::fs;

/// Configuration loader that handles multiple sources and formats
pub struct ConfigLoader {
    /// Search paths for configuration files
    search_paths: Vec<PathBuf>,

    /// Environment variable prefix
    env_prefix: String,

    /// Enable/disable specific sources
    sources: ConfigSources,
}

/// Configuration sources control
#[derive(Debug, Clone)]
pub struct ConfigSources {
    pub defaults: bool,
    pub global_config: bool,
    pub user_config: bool,
    pub local_config: bool,
    pub environment: bool,
    pub explicit_file: bool,
}

impl Default for ConfigSources {
    fn default() -> Self {
        Self {
            defaults: true,
            global_config: true,
            user_config: true,
            local_config: true,
            environment: true,
            explicit_file: true,
        }
    }
}

impl ConfigLoader {
    /// Create a new configuration loader
    pub fn new() -> Self {
        Self {
            search_paths: Self::default_search_paths(),
            env_prefix: "CHRONOS".to_string(),
            sources: ConfigSources::default(),
        }
    }

    /// Create a loader with custom search paths
    pub fn with_search_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.search_paths = paths;
        self
    }

    /// Set the environment variable prefix
    pub fn with_env_prefix(mut self, prefix: String) -> Self {
        self.env_prefix = prefix;
        self
    }

    /// Configure which sources to use
    pub fn with_sources(mut self, sources: ConfigSources) -> Self {
        self.sources = sources;
        self
    }

    /// Load configuration from all available sources
    pub fn load(&self) -> Result<Config> {
        let mut figment = Figment::new();

        // Start with defaults if enabled
        if self.sources.defaults {
            figment = figment.merge(("default", Config::default()));
        }

        // Add configuration files in priority order (lowest to highest)
        if self.sources.global_config {
            if let Some(global_path) = self.find_global_config() {
                figment = self.add_config_file(figment, &global_path)?;
            }
        }

        if self.sources.user_config {
            if let Some(user_path) = self.find_user_config() {
                figment = self.add_config_file(figment, &user_path)?;
            }
        }

        if self.sources.local_config {
            if let Some(local_path) = self.find_local_config() {
                figment = self.add_config_file(figment, &local_path)?;
            }
        }

        // Environment variables have high priority
        if self.sources.environment {
            figment = figment.merge(Env::prefixed(&format!("{}_", self.env_prefix)));
        }

        // Extract and validate configuration
        let mut config: Config = figment.extract()
            .map_err(|e| ConfigError::InvalidFormat(e.to_string()))?;

        // Update metadata
        config.metadata.sources = self.get_active_sources();
        config.metadata.last_modified = Some(chrono::Utc::now());

        // Validate the configuration
        self.validate_config(&config)?;

        Ok(config)
    }

    /// Load configuration with an explicit file
    pub fn load_with_file<P: AsRef<Path>>(&self, config_path: P) -> Result<Config> {
        let mut figment = Figment::new();

        // Start with defaults
        if self.sources.defaults {
            figment = figment.merge(("default", Config::default()));
        }

        // Load the explicit file
        if self.sources.explicit_file {
            figment = self.add_config_file(figment, config_path.as_ref())?;
        }

        // Environment variables can still override
        if self.sources.environment {
            figment = figment.merge(Env::prefixed(&format!("{}_", self.env_prefix)));
        }

        let mut config: Config = figment.extract()
            .map_err(|e| ConfigError::InvalidFormat(e.to_string()))?;

        // Update metadata
        config.metadata.sources = vec![
            "defaults".to_string(),
            config_path.as_ref().display().to_string(),
            "environment".to_string(),
        ];
        config.metadata.last_modified = Some(chrono::Utc::now());

        self.validate_config(&config)?;

        Ok(config)
    }

    /// Load a specific profile
    pub fn load_profile(&self, profile_name: &str) -> Result<Config> {
        let mut config = self.load()?;

        // Apply profile-specific overrides
        if let Some(profile) = config.profiles.definitions.get(profile_name) {
            // Merge profile overrides into the main config
            config.merge(profile.overrides.clone());
            config.metadata.active_profile = profile_name.to_string();
            config.metadata.sources.push(format!("profile:{}", profile_name));
        } else {
            return Err(ConfigError::ProfileNotFound(profile_name.to_string()));
        }

        Ok(config)
    }

    /// Get default search paths for configuration files
    fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // System-wide configuration
        paths.push(PathBuf::from("/etc/chronos"));

        // User configuration directory
        if let Some(home) = dirs::home_dir() {
            paths.push(home.join(".config").join("chronos"));
            paths.push(home.join(".chronos"));
        }

        // Current directory
        paths.push(PathBuf::from("."));

        paths
    }

    /// Find global configuration file
    fn find_global_config(&self) -> Option<PathBuf> {
        let global_paths = vec![
            PathBuf::from("/etc/chronos/config.toml"),
            PathBuf::from("/etc/chronos/config.yaml"),
            PathBuf::from("/etc/chronos/config.yml"),
            PathBuf::from("/etc/chronos/config.json"),
        ];

        global_paths.into_iter().find(|p| p.exists())
    }

    /// Find user configuration file
    fn find_user_config(&self) -> Option<PathBuf> {
        if let Some(home) = dirs::home_dir() {
            let user_paths = vec![
                home.join(".config/chronos/config.toml"),
                home.join(".config/chronos/config.yaml"),
                home.join(".config/chronos/config.yml"),
                home.join(".config/chronos/config.json"),
                home.join(".chronos/config.toml"),
                home.join(".chronos/config.yaml"),
                home.join(".chronos/config.yml"),
                home.join(".chronos/config.json"),
                home.join(".chronos.toml"),
                home.join(".chronos.yaml"),
                home.join(".chronos.yml"),
                home.join(".chronos.json"),
            ];

            return user_paths.into_iter().find(|p| p.exists());
        }

        None
    }

    /// Find local configuration file
    fn find_local_config(&self) -> Option<PathBuf> {
        let local_paths = vec![
            PathBuf::from("chronos.toml"),
            PathBuf::from("chronos.yaml"),
            PathBuf::from("chronos.yml"),
            PathBuf::from("chronos.json"),
            PathBuf::from(".chronos.toml"),
            PathBuf::from(".chronos.yaml"),
            PathBuf::from(".chronos.yml"),
            PathBuf::from(".chronos.json"),
        ];

        local_paths.into_iter().find(|p| p.exists())
    }

    /// Add a configuration file to the figment based on its extension
    fn add_config_file(&self, figment: Figment, path: &Path) -> Result<Figment> {
        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()));
        }

        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "toml" => Ok(figment.merge(Toml::file(path))),
            "yaml" | "yml" => Ok(figment.merge(Yaml::file(path))),
            "json" => Ok(figment.merge(Json::file(path))),
            _ => {
                // Try to detect format from content
                let content = fs::read_to_string(path)?;
                if content.trim_start().starts_with('{') {
                    Ok(figment.merge(Json::string(&content)))
                } else if content.contains("---") || content.contains(":") {
                    Ok(figment.merge(Yaml::string(&content)))
                } else {
                    Ok(figment.merge(Toml::string(&content)))
                }
            }
        }
    }

    /// Get list of active configuration sources
    fn get_active_sources(&self) -> Vec<String> {
        let mut sources = Vec::new();

        if self.sources.defaults {
            sources.push("defaults".to_string());
        }

        if self.sources.global_config {
            if let Some(path) = self.find_global_config() {
                sources.push(format!("global:{}", path.display()));
            }
        }

        if self.sources.user_config {
            if let Some(path) = self.find_user_config() {
                sources.push(format!("user:{}", path.display()));
            }
        }

        if self.sources.local_config {
            if let Some(path) = self.find_local_config() {
                sources.push(format!("local:{}", path.display()));
            }
        }

        if self.sources.environment {
            sources.push("environment".to_string());
        }

        sources
    }

    /// Validate the loaded configuration
    fn validate_config(&self, config: &Config) -> Result<()> {
        // Validate version
        if config.metadata.version.is_empty() {
            return Err(ConfigError::ValidationFailed("Version cannot be empty".to_string()));
        }

        // Validate profile exists
        if !config.profiles.available.contains(&config.metadata.active_profile) {
            return Err(ConfigError::ValidationFailed(
                format!("Active profile '{}' is not available", config.metadata.active_profile)
            ));
        }

        // Validate numeric ranges
        if config.analysis.statistics.confidence_level <= 0.0 || config.analysis.statistics.confidence_level >= 1.0 {
            return Err(ConfigError::ValidationFailed(
                "Confidence level must be between 0 and 1".to_string()
            ));
        }

        if config.analysis.statistics.significance_level <= 0.0 || config.analysis.statistics.significance_level >= 1.0 {
            return Err(ConfigError::ValidationFailed(
                "Significance level must be between 0 and 1".to_string()
            ));
        }

        // Validate paths
        if let Some(ref cache_dir) = config.performance.cache_directory {
            if cache_dir.exists() && !cache_dir.is_dir() {
                return Err(ConfigError::ValidationFailed(
                    format!("Cache directory path is not a directory: {}", cache_dir.display())
                ));
            }
        }

        if let Some(ref output_dir) = config.output.default_directory {
            if output_dir.exists() && !output_dir.is_dir() {
                return Err(ConfigError::ValidationFailed(
                    format!("Output directory path is not a directory: {}", output_dir.display())
                ));
            }
        }

        // Validate color format
        for color in &config.visualization.colors.palette {
            if !color.starts_with('#') || color.len() != 7 {
                return Err(ConfigError::ValidationFailed(
                    format!("Invalid color format: {}", color)
                ));
            }
        }

        Ok(())
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration management utilities
pub struct ConfigManager;

impl ConfigManager {
    /// Create a default configuration file at the specified path
    pub fn create_default_config<P: AsRef<Path>>(path: P, format: ConfigFormat) -> Result<()> {
        let config = Config::default();
        Self::save_config(&config, path, format)
    }

    /// Save configuration to a file
    pub fn save_config<P: AsRef<Path>>(config: &Config, path: P, format: ConfigFormat) -> Result<()> {
        let content = match format {
            ConfigFormat::Toml => toml::to_string_pretty(config)
                .map_err(|e| ConfigError::SerializationError(e.to_string()))?,
            ConfigFormat::Yaml => serde_yaml::to_string(config)
                .map_err(|e| ConfigError::SerializationError(e.to_string()))?,
            ConfigFormat::Json => serde_json::to_string_pretty(config)
                .map_err(|e| ConfigError::SerializationError(e.to_string()))?,
        };

        fs::write(path, content)?;
        Ok(())
    }

    /// Get recommended configuration path for the current user
    pub fn get_user_config_path() -> Option<PathBuf> {
        if let Some(home) = dirs::home_dir() {
            let config_dir = home.join(".config").join("chronos");
            if !config_dir.exists() {
                let _ = fs::create_dir_all(&config_dir);
            }
            Some(config_dir.join("config.toml"))
        } else {
            None
        }
    }

    /// List available profiles in a configuration
    pub fn list_profiles(config: &Config) -> Vec<&str> {
        config.profiles.available.iter().map(|s| s.as_str()).collect()
    }

    /// Get profile description
    pub fn get_profile_description<'a>(config: &'a Config, profile_name: &str) -> Option<&'a str> {
        config.profiles.definitions.get(profile_name)
            .map(|p| p.description.as_str())
    }
}

/// Configuration file formats
#[derive(Debug, Clone, Copy)]
pub enum ConfigFormat {
    Toml,
    Yaml,
    Json,
}

impl ConfigFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            ConfigFormat::Toml => "toml",
            ConfigFormat::Yaml => "yaml",
            ConfigFormat::Json => "json",
        }
    }

    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "toml" => Some(ConfigFormat::Toml),
            "yaml" | "yml" => Some(ConfigFormat::Yaml),
            "json" => Some(ConfigFormat::Json),
            _ => None,
        }
    }
}