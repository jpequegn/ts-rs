//! # Default Configuration Values
//!
//! This module provides sensible default configurations for all components.

use super::*;
use std::collections::HashMap;

impl Default for Config {
    fn default() -> Self {
        Self {
            metadata: ConfigMetadata::default(),
            analysis: AnalysisConfig::default(),
            visualization: VisualizationConfig::default(),
            output: OutputConfig::default(),
            performance: PerformanceConfig::default(),
            profiles: ProfilesConfig::default(),
        }
    }
}

impl Default for ConfigMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            active_profile: "general".to_string(),
            sources: vec!["defaults".to_string()],
            last_modified: Some(chrono::Utc::now()),
            description: Some("Default Chronos configuration".to_string()),
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            statistics: StatisticsConfig::default(),
            trend: TrendConfig::default(),
            seasonality: SeasonalityConfig::default(),
            anomaly: AnomalyConfig::default(),
            forecasting: ForecastingConfig::default(),
            correlation: CorrelationConfig::default(),
        }
    }
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            significance_level: 0.05,
            auto_normality_test: true,
            auto_stationarity_test: true,
            max_autocorrelation_lags: 40,
            auto_changepoint_detection: false,
        }
    }
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            default_method: "decompose".to_string(),
            default_seasonal_period: None,
            smoothing_alpha: 0.3,
            auto_significance_test: true,
            generate_plots: false,
        }
    }
}

impl Default for SeasonalityConfig {
    fn default() -> Self {
        Self {
            default_method: "detect".to_string(),
            max_period: 365,
            min_period: 2,
            detect_multiple: true,
            detect_calendar_effects: false,
            default_adjustment_method: "stl".to_string(),
        }
    }
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            default_methods: vec![
                "zscore".to_string(),
                "iqr".to_string(),
                "isolation_forest".to_string(),
            ],
            zscore_threshold: 3.0,
            modified_zscore_threshold: 3.5,
            iqr_factor: 1.5,
            contamination: 0.1,
            scoring_method: "weighted".to_string(),
            min_severity: "medium".to_string(),
            max_anomalies: 100,
        }
    }
}

impl Default for ForecastingConfig {
    fn default() -> Self {
        Self {
            default_method: "auto_arima".to_string(),
            default_horizon: 12,
            confidence_level: 0.95,
            include_intervals: true,
            auto_evaluation: false,
            cv_folds: 5,
            arima: ArimaConfig::default(),
            exponential_smoothing: ExponentialSmoothingConfig::default(),
        }
    }
}

impl Default for ArimaConfig {
    fn default() -> Self {
        Self {
            default_p: 1,
            default_d: 1,
            default_q: 1,
            max_order: 5,
            use_aic: true,
        }
    }
}

impl Default for ExponentialSmoothingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            seasonal_type: "additive".to_string(),
        }
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            default_types: vec!["pearson".to_string(), "spearman".to_string()],
            rolling_window: None,
            enable_cross_correlation: false,
            max_lag: 20,
            significance_level: 0.05,
            enable_granger_causality: false,
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            default_theme: "professional".to_string(),
            default_dimensions: (800, 600),
            default_dpi: 100,
            colors: ColorConfig::default(),
            fonts: FontConfig::default(),
            interactive_by_default: false,
            show_grid: true,
            export_formats: vec!["png".to_string(), "html".to_string()],
        }
    }
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self {
            palette: vec![
                "#1f77b4".to_string(), // Blue
                "#ff7f0e".to_string(), // Orange
                "#2ca02c".to_string(), // Green
                "#d62728".to_string(), // Red
                "#9467bd".to_string(), // Purple
                "#8c564b".to_string(), // Brown
                "#e377c2".to_string(), // Pink
                "#7f7f7f".to_string(), // Gray
                "#bcbd22".to_string(), // Olive
                "#17becf".to_string(), // Cyan
            ],
            background: "#ffffff".to_string(),
            text: "#333333".to_string(),
            grid: "#e0e0e0".to_string(),
            anomaly: "#ff4444".to_string(),
            trend: "#2196f3".to_string(),
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Arial".to_string(),
            size: 12,
            title_size: 16,
            label_size: 10,
            legend_size: 10,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            default_format: "text".to_string(),
            default_directory: None,
            timestamp_filenames: false,
            compression_level: 6,
            include_metadata: true,
            numeric_precision: 6,
            auto_open: false,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            num_threads: None, // Use system default
            memory_limit_mb: None,
            chunk_size: 10000,
            enable_caching: true,
            cache_directory: None, // Use system temp directory
            max_cache_size_mb: 1024, // 1GB
            progress_threshold: 1000, // Show progress for operations with >1000 items
            enable_database: false, // Disabled by default
        }
    }
}

impl Default for ProfilesConfig {
    fn default() -> Self {
        let mut definitions = HashMap::new();

        // Add built-in profiles
        definitions.insert("general".to_string(), create_general_profile());
        definitions.insert("finance".to_string(), create_finance_profile());
        definitions.insert("iot".to_string(), create_iot_profile());
        definitions.insert("weather".to_string(), create_weather_profile());

        Self {
            available: vec![
                "general".to_string(),
                "finance".to_string(),
                "iot".to_string(),
                "weather".to_string(),
            ],
            definitions,
            auto_switch: false,
            detection_rules: ProfileDetectionRules::default(),
        }
    }
}

impl Default for ProfileDetectionRules {
    fn default() -> Self {
        Self {
            enabled: false,
            confidence_threshold: 0.8,
            min_data_points: 100,
        }
    }
}

/// Create a base config without profiles to avoid circular dependency
fn create_base_config() -> Config {
    Config {
        metadata: ConfigMetadata::default(),
        analysis: AnalysisConfig::default(),
        visualization: VisualizationConfig::default(),
        output: OutputConfig::default(),
        performance: PerformanceConfig::default(),
        profiles: ProfilesConfig {
            available: vec![],
            definitions: HashMap::new(),
            auto_switch: false,
            detection_rules: ProfileDetectionRules::default(),
        },
    }
}

/// Create general-purpose profile
fn create_general_profile() -> ProfileDefinition {
    ProfileDefinition {
        name: "general".to_string(),
        description: "General-purpose configuration suitable for most time series analysis tasks".to_string(),
        overrides: create_base_config(),
        data_characteristics: DataCharacteristics {
            frequency: None,
            seasonality_patterns: vec![],
            value_ranges: None,
            volatility_level: None,
            domain_keywords: vec!["general".to_string(), "default".to_string()],
        },
    }
}

/// Create finance-specific profile
fn create_finance_profile() -> ProfileDefinition {
    let mut overrides = create_base_config();

    // Finance-specific analysis settings
    overrides.analysis.anomaly.zscore_threshold = 2.5; // More sensitive to outliers
    overrides.analysis.anomaly.contamination = 0.05; // Lower contamination rate
    overrides.analysis.forecasting.default_method = "arima".to_string();
    overrides.analysis.forecasting.include_intervals = true;
    overrides.analysis.trend.auto_significance_test = true;
    overrides.analysis.seasonality.detect_calendar_effects = true;

    // Finance-specific visualization
    overrides.visualization.colors.palette = vec![
        "#2E8B57".to_string(), // Green for gains
        "#DC143C".to_string(), // Red for losses
        "#4169E1".to_string(), // Blue for neutral
        "#FF8C00".to_string(), // Orange for warning
        "#9932CC".to_string(), // Purple for special
    ];

    // High precision for financial data
    overrides.output.numeric_precision = 8;

    ProfileDefinition {
        name: "finance".to_string(),
        description: "Optimized for financial time series analysis with high precision and volatility detection".to_string(),
        overrides,
        data_characteristics: DataCharacteristics {
            frequency: Some("daily".to_string()),
            seasonality_patterns: vec![5, 22, 252], // Weekly, monthly, yearly
            value_ranges: None,
            volatility_level: Some("high".to_string()),
            domain_keywords: vec![
                "price".to_string(),
                "stock".to_string(),
                "return".to_string(),
                "volatility".to_string(),
                "finance".to_string(),
                "trading".to_string(),
            ],
        },
    }
}

/// Create IoT-specific profile
fn create_iot_profile() -> ProfileDefinition {
    let mut overrides = create_base_config();

    // IoT-specific analysis settings
    overrides.analysis.anomaly.default_methods = vec![
        "isolation_forest".to_string(),
        "lof".to_string(),
        "dbscan".to_string(),
    ];
    overrides.analysis.anomaly.contamination = 0.15; // Higher contamination expected
    overrides.analysis.seasonality.max_period = 1440; // Daily patterns (minutes)
    overrides.analysis.seasonality.detect_multiple = true;

    // Performance optimizations for high-frequency data
    overrides.performance.chunk_size = 50000;
    overrides.performance.enable_parallel = true;
    overrides.performance.progress_threshold = 10000;

    // Compact output for sensor data
    overrides.output.numeric_precision = 3;
    overrides.output.timestamp_filenames = true;

    ProfileDefinition {
        name: "iot".to_string(),
        description: "Optimized for IoT sensor data with high-frequency sampling and multiple anomaly types".to_string(),
        overrides,
        data_characteristics: DataCharacteristics {
            frequency: Some("minutely".to_string()),
            seasonality_patterns: vec![60, 1440, 10080], // Hourly, daily, weekly (in minutes)
            value_ranges: None,
            volatility_level: Some("medium".to_string()),
            domain_keywords: vec![
                "sensor".to_string(),
                "temperature".to_string(),
                "humidity".to_string(),
                "pressure".to_string(),
                "iot".to_string(),
                "device".to_string(),
            ],
        },
    }
}

/// Create weather-specific profile
fn create_weather_profile() -> ProfileDefinition {
    let mut overrides = create_base_config();

    // Weather-specific analysis settings
    overrides.analysis.seasonality.max_period = 365;
    overrides.analysis.seasonality.detect_calendar_effects = true;
    overrides.analysis.seasonality.detect_multiple = true;
    overrides.analysis.trend.default_seasonal_period = Some(365);
    overrides.analysis.forecasting.default_method = "holt_winters".to_string();

    // Weather-specific anomaly detection
    overrides.analysis.anomaly.zscore_threshold = 3.5; // Weather can be naturally extreme
    overrides.analysis.anomaly.iqr_factor = 2.0;

    // Weather-appropriate colors
    overrides.visualization.colors.palette = vec![
        "#87CEEB".to_string(), // Sky blue
        "#32CD32".to_string(), // Lime green
        "#FFD700".to_string(), // Gold (sunny)
        "#8B4513".to_string(), // Saddle brown
        "#4682B4".to_string(), // Steel blue
    ];

    ProfileDefinition {
        name: "weather".to_string(),
        description: "Optimized for weather and climate data with strong seasonal patterns".to_string(),
        overrides,
        data_characteristics: DataCharacteristics {
            frequency: Some("daily".to_string()),
            seasonality_patterns: vec![7, 30, 365], // Weekly, monthly, yearly
            value_ranges: None,
            volatility_level: Some("medium".to_string()),
            domain_keywords: vec![
                "temperature".to_string(),
                "weather".to_string(),
                "climate".to_string(),
                "precipitation".to_string(),
                "humidity".to_string(),
                "wind".to_string(),
            ],
        },
    }
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            frequency: None,
            seasonality_patterns: vec![],
            value_ranges: None,
            volatility_level: None,
            domain_keywords: vec![],
        }
    }
}

impl Default for ProfileDefinition {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: "Default profile definition".to_string(),
            overrides: create_base_config(),
            data_characteristics: DataCharacteristics::default(),
        }
    }
}