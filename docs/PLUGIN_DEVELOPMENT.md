# Chronos Plugin Development Guide

This guide explains how to develop plugins for the Chronos time series analysis tool. The plugin system allows you to extend Chronos with custom data sources, analysis methods, and visualization capabilities.

## Plugin System Overview

Chronos supports three types of plugins:

1. **Data Source Plugins**: Import data from custom sources (databases, APIs, files)
2. **Analysis Plugins**: Implement custom statistical methods and algorithms
3. **Visualization Plugins**: Create custom plot types and export formats

## Plugin Architecture

### Plugin Metadata

Every plugin must include a `plugin.toml` or `plugin.json` file with metadata:

```toml
id = "my-awesome-plugin"
name = "My Awesome Plugin"
version = "1.0.0"
description = "A plugin that does awesome things with time series data"
author = "Your Name <your.email@example.com>"
license = "MIT"
homepage = "https://github.com/yourname/my-awesome-plugin"
repository = "https://github.com/yourname/my-awesome-plugin"

[api_version]
major = 0
minor = 1
patch = 0

keywords = ["analysis", "statistics", "forecasting"]
dependencies = []

# Optional: JSON Schema for plugin configuration
[config_schema]
# ... JSON Schema definition
```

### Plugin Directory Structure

```
my-awesome-plugin/
├── plugin.toml              # Plugin metadata
├── README.md               # Plugin documentation
├── src/                   # Plugin source code
│   ├── lib.rs            # Main plugin implementation
│   ├── config.rs         # Configuration handling
│   └── analysis.rs       # Analysis implementation
├── examples/             # Usage examples
│   └── basic_usage.rs
├── tests/               # Plugin tests
│   └── integration_tests.rs
└── docs/               # Additional documentation
    └── api.md
```

## Data Source Plugin Development

Data source plugins implement the `DataSourcePlugin` trait to import data from external sources.

### Basic Implementation

```rust
use chronos::plugins::{
    DataSourcePlugin, DataSourceConfig, DataSourceResult, PluginResult,
    ConnectionType, PluginContext
};

pub struct MyDataSourcePlugin {
    // Plugin state
}

impl DataSourcePlugin for MyDataSourcePlugin {
    fn supported_formats(&self) -> Vec<String> {
        vec!["custom_format".to_string()]
    }

    fn supported_connections(&self) -> Vec<ConnectionType> {
        vec![ConnectionType::Http, ConnectionType::Database]
    }

    fn validate_connection(&self, config: &DataSourceConfig) -> PluginResult<()> {
        // Validate connection parameters
        if config.connection.url.is_empty() {
            return Err(PluginError::ConfigError("URL is required".to_string()));
        }
        Ok(())
    }

    fn import_data(
        &self,
        config: &DataSourceConfig,
        context: &PluginContext,
    ) -> PluginResult<DataSourceResult> {
        // Implement data import logic
        // Return imported time series data
        todo!("Implement data import")
    }

    // ... implement other required methods
}
```

### Configuration Schema

Define a JSON Schema for your plugin's configuration:

```json
{
  "type": "object",
  "properties": {
    "api_endpoint": {
      "type": "string",
      "description": "API endpoint URL"
    },
    "api_key": {
      "type": "string",
      "description": "API authentication key"
    },
    "timeout": {
      "type": "integer",
      "minimum": 1,
      "maximum": 300,
      "default": 30,
      "description": "Request timeout in seconds"
    }
  },
  "required": ["api_endpoint", "api_key"]
}
```

## Analysis Plugin Development

Analysis plugins implement custom statistical methods and algorithms.

### Basic Implementation

```rust
use chronos::plugins::{
    AnalysisPlugin, AnalysisConfig, AnalysisResult, PluginResult,
    AnalysisMethod, AnalysisCategory, PluginContext
};
use chronos::TimeSeries;

pub struct MyAnalysisPlugin {
    // Plugin state
}

impl AnalysisPlugin for MyAnalysisPlugin {
    fn supported_methods(&self) -> Vec<AnalysisMethod> {
        vec![
            AnalysisMethod {
                id: "custom_regression".to_string(),
                name: "Custom Regression".to_string(),
                description: "A custom regression algorithm".to_string(),
                category: AnalysisCategory::Statistical,
                version: "1.0.0".to_string(),
                experimental: false,
            }
        ]
    }

    fn analyze(
        &self,
        data: &[TimeSeries],
        config: &AnalysisConfig,
        context: &PluginContext,
    ) -> PluginResult<AnalysisResult> {
        match config.method.as_str() {
            "custom_regression" => self.perform_custom_regression(data, config),
            _ => Err(PluginError::ExecutionError(
                format!("Unsupported method: {}", config.method)
            ))
        }
    }

    // ... implement other required methods
}

impl MyAnalysisPlugin {
    fn perform_custom_regression(
        &self,
        data: &[TimeSeries],
        config: &AnalysisConfig,
    ) -> PluginResult<AnalysisResult> {
        // Implement your custom analysis algorithm
        todo!("Implement custom regression")
    }
}
```

### Parameters and Configuration

Define parameters for your analysis methods:

```rust
fn get_required_parameters(&self, method: &str) -> Vec<ParameterDefinition> {
    match method {
        "custom_regression" => vec![
            ParameterDefinition {
                name: "degree".to_string(),
                description: "Polynomial degree".to_string(),
                param_type: ParameterType::Integer,
                default: Some(serde_json::Value::Number(serde_json::Number::from(2))),
                constraints: Some(ParameterConstraints::Range { min: 1.0, max: 10.0 }),
            }
        ],
        _ => vec![]
    }
}
```

## Visualization Plugin Development

Visualization plugins create custom plot types and export formats.

### Basic Implementation

```rust
use chronos::plugins::{
    VisualizationPlugin, VisualizationConfig, VisualizationResult, PluginResult,
    PlotType, ExportFormat, PluginContext
};
use chronos::TimeSeries;

pub struct MyVisualizationPlugin {
    // Plugin state
}

impl VisualizationPlugin for MyVisualizationPlugin {
    fn supported_plot_types(&self) -> Vec<PlotType> {
        vec![
            PlotType {
                id: "custom_heatmap".to_string(),
                name: "Custom Heatmap".to_string(),
                description: "A specialized heatmap visualization".to_string(),
                category: PlotCategory::Statistical,
                data_requirements: DataRequirements {
                    min_points: 4,
                    max_points: Some(10000),
                    required_columns: vec!["x".to_string(), "y".to_string(), "value".to_string()],
                    optional_columns: vec!["weight".to_string()],
                    supported_types: vec![DataType::Numeric],
                    requires_temporal: false,
                },
                multi_series: true,
                supports_3d: false,
            }
        ]
    }

    fn create_visualization(
        &self,
        data: &[TimeSeries],
        config: &VisualizationConfig,
        context: &PluginContext,
    ) -> PluginResult<VisualizationResult> {
        match config.plot_type.as_str() {
            "custom_heatmap" => self.create_heatmap(data, config),
            _ => Err(PluginError::ExecutionError(
                format!("Unsupported plot type: {}", config.plot_type)
            ))
        }
    }

    // ... implement other required methods
}
```

## Plugin Registration

### Implementing the Plugin Trait

All plugins must implement the base `Plugin` trait:

```rust
use chronos::plugins::{Plugin, PluginType, PluginMetadata, PluginCapabilities, PluginError};

impl Plugin for MyAnalysisPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Analysis
    }

    fn capabilities(&self) -> &PluginCapabilities {
        &self.capabilities
    }

    fn initialize(&mut self, config: &serde_json::Value) -> Result<(), PluginError> {
        // Initialize plugin with configuration
        Ok(())
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), PluginError> {
        // Validate configuration
        Ok(())
    }

    fn status(&self) -> PluginStatus {
        // Return plugin health status
        PluginStatus::default()
    }

    fn cleanup(&mut self) -> Result<(), PluginError> {
        // Clean up resources
        Ok(())
    }
}
```

## Testing Your Plugin

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chronos::TimeSeries;

    #[test]
    fn test_plugin_initialization() {
        let mut plugin = MyAnalysisPlugin::new();
        let config = serde_json::json!({
            "param1": "value1"
        });

        assert!(plugin.initialize(&config).is_ok());
        assert!(plugin.status().is_healthy);
    }

    #[test]
    fn test_analysis_method() {
        let plugin = MyAnalysisPlugin::new();
        let data = vec![create_test_time_series()];
        let config = AnalysisConfig {
            method: "custom_regression".to_string(),
            // ... other config
        };
        let context = PluginContext::new("test".to_string());

        let result = plugin.analyze(&data, &config, &context);
        assert!(result.is_ok());
    }
}
```

### Integration Tests

```rust
#[test]
fn test_plugin_with_chronos() {
    let registry = PluginRegistry::new();
    let loader = PluginLoader::new();

    // Test plugin discovery and loading
    // Test plugin execution with real data
}
```

## Plugin Distribution

### Packaging

Create a plugin package for distribution:

```bash
# Build your plugin
cargo build --release

# Create plugin package
chronos plugin dev package ./my-plugin --output my-plugin-1.0.0.zip
```

### Plugin Repository

Submit your plugin to the Chronos plugin repository:

1. Create a pull request to the plugin repository
2. Include plugin metadata and documentation
3. Ensure all tests pass
4. Follow the contribution guidelines

## Best Practices

### Error Handling

- Use proper error types and messages
- Provide helpful error context
- Handle edge cases gracefully
- Validate inputs thoroughly

### Performance

- Optimize for large datasets
- Use streaming when possible
- Implement proper memory management
- Consider parallel processing

### Security

- Validate all inputs
- Sanitize user-provided data
- Use secure communication protocols
- Follow security best practices

### Documentation

- Provide clear API documentation
- Include usage examples
- Document configuration options
- Explain algorithm details

### Compatibility

- Follow semantic versioning
- Test with multiple Chronos versions
- Handle API changes gracefully
- Provide migration guides

## Examples

### Simple Analysis Plugin

```rust
// A simple moving average plugin
pub struct MovingAveragePlugin;

impl AnalysisPlugin for MovingAveragePlugin {
    fn analyze(
        &self,
        data: &[TimeSeries],
        config: &AnalysisConfig,
    ) -> PluginResult<AnalysisResult> {
        let window_size: usize = config.parameters
            .get("window_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let series = &data[0];
        let moving_avg = calculate_moving_average(&series.values, window_size);

        Ok(AnalysisResult {
            method: "moving_average".to_string(),
            results: serde_json::json!({
                "moving_average": moving_avg,
                "window_size": window_size
            }),
            // ... other fields
        })
    }
}
```

### Database Data Source Plugin

```rust
// A PostgreSQL data source plugin
pub struct PostgreSQLPlugin {
    connection_pool: Option<Pool>,
}

impl DataSourcePlugin for PostgreSQLPlugin {
    fn import_data(
        &self,
        config: &DataSourceConfig,
        context: &PluginContext,
    ) -> PluginResult<DataSourceResult> {
        let connection_string = &config.connection.url;
        let query = config.options.get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PluginError::ConfigError("Query is required".to_string()))?;

        // Connect to database and execute query
        let results = self.execute_query(connection_string, query)?;

        // Convert to TimeSeries format
        let timeseries = self.convert_to_timeseries(results)?;

        Ok(DataSourceResult {
            timeseries,
            // ... other fields
        })
    }
}
```

## Getting Help

- Check the [API documentation](api.md)
- Visit the [GitHub repository](https://github.com/chronos/plugins)
- Join the [community forum](https://forum.chronos.dev)
- Report issues on [GitHub Issues](https://github.com/chronos/chronos/issues)

## Contributing

We welcome contributions to the plugin ecosystem! Please read our [Contributing Guide](CONTRIBUTING.md) for details on how to submit plugins and improvements.