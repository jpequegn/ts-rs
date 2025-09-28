//! # Plugin System Tests
//!
//! Comprehensive tests for the plugin system functionality.

#[cfg(test)]
mod tests {
    use super::super::*;
    use tempfile::TempDir;
    use std::fs;

    /// Create a test plugin metadata
    fn create_test_plugin_metadata(id: &str) -> PluginMetadata {
        PluginMetadata {
            id: id.to_string(),
            name: format!("Test Plugin {}", id),
            version: "1.0.0".to_string(),
            description: "A test plugin for unit testing".to_string(),
            author: "Test Author".to_string(),
            homepage: Some("https://example.com".to_string()),
            repository: Some("https://github.com/example/plugin".to_string()),
            license: "MIT".to_string(),
            api_version: CURRENT_API_VERSION,
            dependencies: vec![],
            keywords: vec!["test".to_string(), "analysis".to_string()],
            config_schema: None,
        }
    }

    /// Create a test plugin directory with metadata
    fn create_test_plugin_dir(temp_dir: &TempDir, plugin_id: &str) -> std::io::Result<std::path::PathBuf> {
        let plugin_dir = temp_dir.path().join(plugin_id);
        fs::create_dir_all(&plugin_dir)?;

        let metadata = create_test_plugin_metadata(plugin_id);
        let metadata_content = toml::to_string(&metadata).unwrap();
        fs::write(plugin_dir.join("plugin.toml"), metadata_content)?;

        Ok(plugin_dir)
    }

    #[test]
    fn test_plugin_registry_creation() {
        let registry = PluginRegistry::new();
        let stats = registry.get_stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.loaded, 0);
        assert_eq!(stats.disabled, 0);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_plugin_registry_search_path_management() {
        let registry = PluginRegistry::new();
        let test_path = std::path::PathBuf::from("/test/path");

        // Add search path
        registry.add_search_path(&test_path).unwrap();
        let paths = registry.get_search_paths();
        assert!(paths.contains(&test_path));

        // Remove search path
        registry.remove_search_path(&test_path).unwrap();
        let paths = registry.get_search_paths();
        assert!(!paths.contains(&test_path));
    }

    #[test]
    fn test_plugin_discovery() {
        let temp_dir = TempDir::new().unwrap();
        let registry = PluginRegistry::new();

        // Create test plugin directory
        let plugin_dir = create_test_plugin_dir(&temp_dir, "test-plugin").unwrap();

        // Add search path
        registry.add_search_path(temp_dir.path()).unwrap();

        // Discover plugins
        let discovered_count = registry.discover_plugins().unwrap();
        assert_eq!(discovered_count, 1);

        // Check plugin was registered
        let plugin = registry.get_plugin("test-plugin");
        assert!(plugin.is_some());

        let plugin = plugin.unwrap();
        assert_eq!(plugin.metadata.id, "test-plugin");
        assert_eq!(plugin.metadata.name, "Test Plugin test-plugin");
        assert_eq!(plugin.plugin_type, PluginType::Analysis); // Default inferred type
    }

    #[test]
    fn test_plugin_type_inference() {
        let temp_dir = TempDir::new().unwrap();
        let registry = PluginRegistry::new();

        // Create plugin with data-source keyword
        let mut metadata = create_test_plugin_metadata("data-source-plugin");
        metadata.keywords = vec!["data-source".to_string()];
        let metadata_content = toml::to_string(&metadata).unwrap();

        let plugin_dir = temp_dir.path().join("data-source-plugin");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(plugin_dir.join("plugin.toml"), metadata_content).unwrap();

        registry.add_search_path(temp_dir.path()).unwrap();
        registry.discover_plugins().unwrap();

        let plugin = registry.get_plugin("data-source-plugin").unwrap();
        assert_eq!(plugin.plugin_type, PluginType::DataSource);
    }

    #[test]
    fn test_plugin_enable_disable() {
        let temp_dir = TempDir::new().unwrap();
        let registry = PluginRegistry::new();

        create_test_plugin_dir(&temp_dir, "test-plugin").unwrap();
        registry.add_search_path(temp_dir.path()).unwrap();
        registry.discover_plugins().unwrap();

        // Initially discovered
        let plugin = registry.get_plugin("test-plugin").unwrap();
        assert_eq!(plugin.status, registry::PluginLoadStatus::Discovered);

        // Disable plugin
        registry.disable_plugin("test-plugin").unwrap();
        let plugin = registry.get_plugin("test-plugin").unwrap();
        assert_eq!(plugin.status, registry::PluginLoadStatus::Disabled);

        // Enable plugin
        registry.enable_plugin("test-plugin").unwrap();
        let plugin = registry.get_plugin("test-plugin").unwrap();
        assert_eq!(plugin.status, registry::PluginLoadStatus::Discovered);
    }

    #[test]
    fn test_plugin_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = management::ManagementConfig {
            install_dir: temp_dir.path().join("plugins"),
            cache_dir: temp_dir.path().join("cache"),
            config_dir: temp_dir.path().join("config"),
            backup_dir: Some(temp_dir.path().join("backups")),
            ..management::ManagementConfig::default()
        };

        let manager = PluginManager::with_config(config).unwrap();

        // Check directories were created
        assert!(temp_dir.path().join("plugins").exists());
        assert!(temp_dir.path().join("cache").exists());
        assert!(temp_dir.path().join("config").exists());
        assert!(temp_dir.path().join("backups").exists());
    }

    #[test]
    fn test_plugin_repository_management() {
        let mut manager = PluginManager::new().unwrap();

        let repo = management::PluginRepository {
            name: "test-repo".to_string(),
            url: "https://example.com/plugins".to_string(),
            repo_type: management::RepositoryType::Http,
            auth: None,
            enabled: true,
            priority: 100,
        };

        // Add repository
        manager.add_repository(repo.clone()).unwrap();
        assert_eq!(manager.repository_count(), 1);
        assert_eq!(manager.get_repository("test-repo").unwrap().name, "test-repo");

        // Remove repository
        manager.remove_repository("test-repo").unwrap();
        assert_eq!(manager.repository_count(), 0);
    }

    #[test]
    fn test_plugin_loader_creation() {
        let loader = PluginLoader::new();
        assert_eq!(loader.search_path_count(), 0);
        assert!(!loader.get_config().enable_dynamic_loading);
    }

    #[test]
    fn test_plugin_loader_validation() {
        let loader = PluginLoader::new();
        let metadata = create_test_plugin_metadata("test-plugin");
        let plugin = loader::MockAnalysisPlugin::new(metadata);

        // Initialize plugin
        let mut plugin = plugin;
        plugin.initialize(&serde_json::Value::Null).unwrap();

        // Validate plugin
        assert!(loader.validate_plugin(&plugin).is_ok());
    }

    #[test]
    fn test_plugin_loader_validation_failure() {
        let loader = PluginLoader::new();
        let mut metadata = create_test_plugin_metadata("test-plugin");
        metadata.id = "".to_string(); // Invalid empty ID
        let plugin = loader::MockAnalysisPlugin::new(metadata);

        // Validation should fail
        assert!(loader.validate_plugin(&plugin).is_err());
    }

    #[test]
    fn test_api_version_compatibility() {
        let current = ApiVersion::new(1, 2, 3);

        // Compatible versions
        let compatible_higher_patch = ApiVersion::new(1, 2, 4);
        let compatible_higher_minor = ApiVersion::new(1, 3, 0);
        let compatible_same = ApiVersion::new(1, 2, 3);

        assert!(compatible_higher_patch.is_compatible(&current));
        assert!(compatible_higher_minor.is_compatible(&current));
        assert!(compatible_same.is_compatible(&current));

        // Incompatible versions
        let incompatible_lower_patch = ApiVersion::new(1, 2, 2);
        let incompatible_lower_minor = ApiVersion::new(1, 1, 5);
        let incompatible_major = ApiVersion::new(2, 2, 3);

        assert!(!incompatible_lower_patch.is_compatible(&current));
        assert!(!incompatible_lower_minor.is_compatible(&current));
        assert!(!incompatible_major.is_compatible(&current));
    }

    #[test]
    fn test_plugin_context_creation() {
        let ctx = PluginContext::new("test-123".to_string())
            .with_parameter("key1".to_string(), serde_json::Value::String("value1".to_string()))
            .with_timeout(std::time::Duration::from_secs(60));

        assert_eq!(ctx.request_id, "test-123");
        assert_eq!(ctx.parameters.len(), 1);
        assert!(ctx.parameters.contains_key("key1"));
        assert_eq!(ctx.timeout, Some(std::time::Duration::from_secs(60)));
    }

    #[test]
    fn test_data_source_config_serialization() {
        let config = data_source::DataSourceConfig {
            source_type: data_source::ConnectionType::File,
            connection: data_source::ConnectionConfig {
                url: "file://test.csv".to_string(),
                timeout: Some(30),
                retry: Some(data_source::RetryConfig::default()),
                pool: None,
                headers: None,
                params: None,
            },
            import: data_source::ImportConfig {
                format: "csv".to_string(),
                time_column: Some(data_source::ColumnSpec {
                    name: "timestamp".to_string(),
                    data_type: Some(data_source::DataType::DateTime),
                    format: Some("%Y-%m-%d %H:%M:%S".to_string()),
                    required: true,
                    default: None,
                }),
                value_columns: vec![data_source::ColumnSpec {
                    name: "value".to_string(),
                    data_type: Some(data_source::DataType::Float),
                    format: None,
                    required: true,
                    default: None,
                }],
                filter: None,
                transform: None,
                batch_size: Some(1000),
            },
            auth: None,
            options: std::collections::HashMap::new(),
        };

        // Test serialization/deserialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: data_source::DataSourceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.source_type, data_source::ConnectionType::File);
        assert_eq!(deserialized.connection.url, "file://test.csv");
    }

    #[test]
    fn test_analysis_config_serialization() {
        let config = analysis::AnalysisConfig {
            method: "linear_regression".to_string(),
            parameters: {
                let mut params = std::collections::HashMap::new();
                params.insert("alpha".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.05).unwrap()));
                params
            },
            input: analysis::InputConfig {
                columns: vec!["value".to_string()],
                preprocessing: None,
                filter: None,
                window: None,
            },
            output: analysis::OutputConfig {
                format: analysis::OutputFormat::Structured,
                include_confidence: true,
                confidence_level: Some(0.95),
                include_diagnostics: true,
                include_intermediate: false,
                precision: Some(4),
            },
            performance: None,
            validation: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: analysis::AnalysisConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.method, "linear_regression");
        assert_eq!(deserialized.input.columns.len(), 1);
    }

    #[test]
    fn test_visualization_config_serialization() {
        let config = visualization::VisualizationConfig {
            plot_type: "line_plot".to_string(),
            title: Some("Test Plot".to_string()),
            style: visualization::StyleConfig {
                theme: Some("dark".to_string()),
                colors: None,
                fonts: None,
                lines: None,
                markers: None,
                background: None,
                custom_style: None,
            },
            data: visualization::DataConfig {
                series: vec![visualization::SeriesConfig {
                    name: "Series 1".to_string(),
                    column: "value".to_string(),
                    series_type: "line".to_string(),
                    style: None,
                    y_axis: visualization::AxisSelection::Primary,
                    visible: true,
                }],
                x_axis: visualization::AxisConfig {
                    title: Some("Time".to_string()),
                    scale: visualization::AxisScale::Time,
                    range: None,
                    ticks: None,
                    grid: None,
                    format: None,
                },
                y_axis: visualization::AxisConfig {
                    title: Some("Value".to_string()),
                    scale: visualization::AxisScale::Linear,
                    range: None,
                    ticks: None,
                    grid: None,
                    format: None,
                },
                y2_axis: None,
                filter: None,
                aggregation: None,
            },
            layout: visualization::LayoutConfig {
                dimensions: Some(visualization::Dimensions {
                    width: 800.0,
                    height: 600.0,
                    aspect_ratio: None,
                }),
                margins: None,
                legend: None,
                subplots: None,
                annotations: vec![],
            },
            interactive: None,
            export: visualization::ExportConfig {
                format: visualization::ExportFormat::PNG,
                dpi: Some(300),
                quality: None,
                transparent: false,
                include_metadata: true,
            },
            performance: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: visualization::VisualizationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.plot_type, "line_plot");
        assert_eq!(deserialized.data.series.len(), 1);
    }

    #[test]
    fn test_plugin_registry_stats() {
        let temp_dir = TempDir::new().unwrap();
        let registry = PluginRegistry::new();

        // Create multiple test plugins
        create_test_plugin_dir(&temp_dir, "plugin1").unwrap();
        create_test_plugin_dir(&temp_dir, "plugin2").unwrap();
        create_test_plugin_dir(&temp_dir, "plugin3").unwrap();

        registry.add_search_path(temp_dir.path()).unwrap();
        let discovered_count = registry.discover_plugins().unwrap();
        assert_eq!(discovered_count, 3);

        // Disable one plugin
        registry.disable_plugin("plugin2").unwrap();

        // Check stats
        let stats = registry.get_stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.loaded, 0);
        assert_eq!(stats.disabled, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.by_type.get(&PluginType::Analysis), Some(&3));
    }

    #[test]
    fn test_plugin_list_operations() {
        let temp_dir = TempDir::new().unwrap();
        let registry = PluginRegistry::new();

        // Create plugins with different types
        let mut metadata1 = create_test_plugin_metadata("analysis-plugin");
        metadata1.keywords = vec!["analysis".to_string()];

        let mut metadata2 = create_test_plugin_metadata("datasource-plugin");
        metadata2.keywords = vec!["data-source".to_string()];

        // Save plugins
        let plugin_dir1 = temp_dir.path().join("analysis-plugin");
        fs::create_dir_all(&plugin_dir1).unwrap();
        fs::write(plugin_dir1.join("plugin.toml"), toml::to_string(&metadata1).unwrap()).unwrap();

        let plugin_dir2 = temp_dir.path().join("datasource-plugin");
        fs::create_dir_all(&plugin_dir2).unwrap();
        fs::write(plugin_dir2.join("plugin.toml"), toml::to_string(&metadata2).unwrap()).unwrap();

        registry.add_search_path(temp_dir.path()).unwrap();
        registry.discover_plugins().unwrap();

        // Test list all plugins
        let all_plugins = registry.list_plugins();
        assert_eq!(all_plugins.len(), 2);

        // Test list plugins by type
        let analysis_plugins = registry.list_plugins_by_type(PluginType::Analysis);
        assert_eq!(analysis_plugins.len(), 1);
        assert_eq!(analysis_plugins[0].metadata.id, "analysis-plugin");

        let datasource_plugins = registry.list_plugins_by_type(PluginType::DataSource);
        assert_eq!(datasource_plugins.len(), 1);
        assert_eq!(datasource_plugins[0].metadata.id, "datasource-plugin");

        // Test list loaded plugins (should be empty)
        let loaded_plugins = registry.list_loaded_plugins();
        assert_eq!(loaded_plugins.len(), 0);
    }
}