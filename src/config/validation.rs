//! # Configuration Validation
//!
//! This module provides comprehensive validation for configuration settings.

use super::*;
use std::collections::HashSet;

/// Configuration validator
pub struct ConfigValidator {
    /// Validation rules and constraints
    rules: ValidationRules,

    /// Known good values and ranges
    constraints: ValidationConstraints,
}

/// Validation rules configuration
#[derive(Debug, Clone)]
pub struct ValidationRules {
    /// Validate numeric ranges
    pub numeric_ranges: bool,

    /// Validate file paths
    pub file_paths: bool,

    /// Validate color formats
    pub color_formats: bool,

    /// Validate enum values
    pub enum_values: bool,

    /// Validate profile definitions
    pub profile_definitions: bool,

    /// Strict mode (fail on warnings)
    pub strict_mode: bool,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            numeric_ranges: true,
            file_paths: true,
            color_formats: true,
            enum_values: true,
            profile_definitions: true,
            strict_mode: false,
        }
    }
}

/// Validation constraints and acceptable values
#[derive(Debug, Clone)]
pub struct ValidationConstraints {
    /// Valid forecasting methods
    pub forecasting_methods: HashSet<String>,

    /// Valid anomaly detection methods
    pub anomaly_methods: HashSet<String>,

    /// Valid seasonal adjustment methods
    pub seasonal_adjustment_methods: HashSet<String>,

    /// Valid correlation types
    pub correlation_types: HashSet<String>,

    /// Valid output formats
    pub output_formats: HashSet<String>,

    /// Valid themes
    pub themes: HashSet<String>,

    /// Valid scoring methods
    pub scoring_methods: HashSet<String>,

    /// Valid severity levels
    pub severity_levels: HashSet<String>,
}

impl Default for ValidationConstraints {
    fn default() -> Self {
        Self {
            forecasting_methods: [
                "sma", "exponential_smoothing", "linear_trend", "seasonal_naive",
                "holt_winters", "arima", "sarima", "auto_arima", "ets", "theta", "prophet"
            ].iter().map(|s| s.to_string()).collect(),

            anomaly_methods: [
                "zscore", "modified_zscore", "iqr", "grubbs", "isolation_forest",
                "lof", "dbscan", "seasonal", "trend", "level_shift", "volatility", "contextual"
            ].iter().map(|s| s.to_string()).collect(),

            seasonal_adjustment_methods: [
                "stl", "x13", "moving_average"
            ].iter().map(|s| s.to_string()).collect(),

            correlation_types: [
                "pearson", "spearman", "kendall"
            ].iter().map(|s| s.to_string()).collect(),

            output_formats: [
                "text", "json", "csv", "markdown", "html", "pdf"
            ].iter().map(|s| s.to_string()).collect(),

            themes: [
                "professional", "dark", "light", "colorful", "minimal"
            ].iter().map(|s| s.to_string()).collect(),

            scoring_methods: [
                "maximum", "weighted", "ensemble"
            ].iter().map(|s| s.to_string()).collect(),

            severity_levels: [
                "low", "medium", "high", "critical"
            ].iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,

    /// Validation errors (critical issues)
    pub errors: Vec<ValidationError>,

    /// Validation warnings (non-critical issues)
    pub warnings: Vec<ValidationWarning>,

    /// Suggestions for improvement
    pub suggestions: Vec<ValidationSuggestion>,
}

/// Validation error details
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error category
    pub category: ErrorCategory,

    /// Configuration path where error occurred
    pub path: String,

    /// Error message
    pub message: String,

    /// Suggested fix
    pub suggested_fix: Option<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.path, self.message)?;
        if let Some(ref fix) = self.suggested_fix {
            write!(f, " (Suggested: {})", fix)?;
        }
        Ok(())
    }
}

/// Validation warning details
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning category
    pub category: WarningCategory,

    /// Configuration path
    pub path: String,

    /// Warning message
    pub message: String,

    /// Recommendation
    pub recommendation: Option<String>,
}

impl std::fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.path, self.message)?;
        if let Some(ref rec) = self.recommendation {
            write!(f, " (Recommendation: {})", rec)?;
        }
        Ok(())
    }
}

/// Validation suggestion
#[derive(Debug, Clone)]
pub struct ValidationSuggestion {
    /// Suggestion category
    pub category: SuggestionCategory,

    /// Configuration path
    pub path: String,

    /// Suggestion message
    pub message: String,

    /// Potential benefit
    pub benefit: String,
}

impl std::fmt::Display for ValidationSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} (Benefit: {})", self.path, self.message, self.benefit)
    }
}

/// Error categories
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    InvalidRange,
    InvalidFormat,
    MissingRequired,
    InvalidEnum,
    FileSystemError,
    ProfileError,
    DependencyError,
}

/// Warning categories
#[derive(Debug, Clone, PartialEq)]
pub enum WarningCategory {
    PerformanceImpact,
    SecurityConcern,
    Deprecated,
    Suboptimal,
    CompatibilityIssue,
}

/// Suggestion categories
#[derive(Debug, Clone, PartialEq)]
pub enum SuggestionCategory {
    Optimization,
    BestPractice,
    FeatureRecommendation,
    ProfileRecommendation,
}

impl ConfigValidator {
    /// Create a new validator with default rules
    pub fn new() -> Self {
        Self {
            rules: ValidationRules::default(),
            constraints: ValidationConstraints::default(),
        }
    }

    /// Create validator with custom rules
    pub fn with_rules(rules: ValidationRules) -> Self {
        Self {
            rules,
            constraints: ValidationConstraints::default(),
        }
    }

    /// Validate a complete configuration
    pub fn validate(&self, config: &Config) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        // Validate metadata
        self.validate_metadata(&config.metadata, &mut result);

        // Validate analysis configuration
        self.validate_analysis(&config.analysis, &mut result);

        // Validate visualization configuration
        self.validate_visualization(&config.visualization, &mut result);

        // Validate output configuration
        self.validate_output(&config.output, &mut result);

        // Validate performance configuration
        self.validate_performance(&config.performance, &mut result);

        // Validate profiles configuration
        self.validate_profiles(&config.profiles, &mut result);

        // Check overall consistency
        self.validate_consistency(config, &mut result);

        // Determine final validity
        result.is_valid = result.errors.is_empty() && (!self.rules.strict_mode || result.warnings.is_empty());

        result
    }

    /// Validate metadata section
    fn validate_metadata(&self, metadata: &ConfigMetadata, result: &mut ValidationResult) {
        // Validate version format
        if metadata.version.is_empty() {
            result.errors.push(ValidationError {
                category: ErrorCategory::MissingRequired,
                path: "metadata.version".to_string(),
                message: "Configuration version cannot be empty".to_string(),
                suggested_fix: Some("Set version to '1.0.0'".to_string()),
            });
        } else if !self.is_valid_version(&metadata.version) {
            result.warnings.push(ValidationWarning {
                category: WarningCategory::Suboptimal,
                path: "metadata.version".to_string(),
                message: "Version format doesn't follow semantic versioning".to_string(),
                recommendation: Some("Use semantic versioning format (e.g., '1.0.0')".to_string()),
            });
        }
    }

    /// Validate analysis configuration
    fn validate_analysis(&self, analysis: &AnalysisConfig, result: &mut ValidationResult) {
        // Validate statistics config
        self.validate_statistics_config(&analysis.statistics, result);

        // Validate forecasting config
        self.validate_forecasting_config(&analysis.forecasting, result);

        // Validate anomaly config
        self.validate_anomaly_config(&analysis.anomaly, result);

        // Validate correlation config
        self.validate_correlation_config(&analysis.correlation, result);
    }

    /// Validate statistics configuration
    fn validate_statistics_config(&self, stats: &StatisticsConfig, result: &mut ValidationResult) {
        if self.rules.numeric_ranges {
            // Validate confidence level
            if stats.confidence_level <= 0.0 || stats.confidence_level >= 1.0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "analysis.statistics.confidence_level".to_string(),
                    message: "Confidence level must be between 0 and 1".to_string(),
                    suggested_fix: Some("Set to 0.95 for 95% confidence".to_string()),
                });
            }

            // Validate significance level
            if stats.significance_level <= 0.0 || stats.significance_level >= 1.0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "analysis.statistics.significance_level".to_string(),
                    message: "Significance level must be between 0 and 1".to_string(),
                    suggested_fix: Some("Set to 0.05 for 5% significance".to_string()),
                });
            }

            // Check for common statistical relationships
            if stats.confidence_level + stats.significance_level != 1.0 {
                result.suggestions.push(ValidationSuggestion {
                    category: SuggestionCategory::BestPractice,
                    path: "analysis.statistics".to_string(),
                    message: "Confidence level and significance level should sum to 1.0".to_string(),
                    benefit: "Ensures statistical consistency".to_string(),
                });
            }
        }
    }

    /// Validate forecasting configuration
    fn validate_forecasting_config(&self, forecasting: &ForecastingConfig, result: &mut ValidationResult) {
        if self.rules.enum_values {
            // Validate forecasting method
            if !self.constraints.forecasting_methods.contains(&forecasting.default_method) {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidEnum,
                    path: "analysis.forecasting.default_method".to_string(),
                    message: format!("Unknown forecasting method: {}", forecasting.default_method),
                    suggested_fix: Some("Use one of: arima, holt_winters, exponential_smoothing".to_string()),
                });
            }
        }

        if self.rules.numeric_ranges {
            // Validate horizon
            if forecasting.default_horizon == 0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "analysis.forecasting.default_horizon".to_string(),
                    message: "Forecast horizon must be greater than 0".to_string(),
                    suggested_fix: Some("Set to 12 for monthly forecasts".to_string()),
                });
            }

            // Validate confidence level
            if forecasting.confidence_level <= 0.0 || forecasting.confidence_level >= 1.0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "analysis.forecasting.confidence_level".to_string(),
                    message: "Confidence level must be between 0 and 1".to_string(),
                    suggested_fix: Some("Set to 0.95 for 95% confidence intervals".to_string()),
                });
            }
        }
    }

    /// Validate anomaly detection configuration
    fn validate_anomaly_config(&self, anomaly: &AnomalyConfig, result: &mut ValidationResult) {
        if self.rules.enum_values {
            // Validate anomaly detection methods
            for method in &anomaly.default_methods {
                if !self.constraints.anomaly_methods.contains(method) {
                    result.errors.push(ValidationError {
                        category: ErrorCategory::InvalidEnum,
                        path: "analysis.anomaly.default_methods".to_string(),
                        message: format!("Unknown anomaly detection method: {}", method),
                        suggested_fix: Some("Use one of: zscore, iqr, isolation_forest".to_string()),
                    });
                }
            }

            // Validate scoring method
            if !self.constraints.scoring_methods.contains(&anomaly.scoring_method) {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidEnum,
                    path: "analysis.anomaly.scoring_method".to_string(),
                    message: format!("Unknown scoring method: {}", anomaly.scoring_method),
                    suggested_fix: Some("Use one of: maximum, weighted, ensemble".to_string()),
                });
            }

            // Validate severity level
            if !self.constraints.severity_levels.contains(&anomaly.min_severity) {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidEnum,
                    path: "analysis.anomaly.min_severity".to_string(),
                    message: format!("Unknown severity level: {}", anomaly.min_severity),
                    suggested_fix: Some("Use one of: low, medium, high, critical".to_string()),
                });
            }
        }

        if self.rules.numeric_ranges {
            // Validate thresholds
            if anomaly.zscore_threshold <= 0.0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "analysis.anomaly.zscore_threshold".to_string(),
                    message: "Z-score threshold must be positive".to_string(),
                    suggested_fix: Some("Set to 3.0 for standard anomaly detection".to_string()),
                });
            }

            if anomaly.contamination <= 0.0 || anomaly.contamination >= 1.0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "analysis.anomaly.contamination".to_string(),
                    message: "Contamination rate must be between 0 and 1".to_string(),
                    suggested_fix: Some("Set to 0.1 for 10% expected anomalies".to_string()),
                });
            }
        }
    }

    /// Validate correlation configuration
    fn validate_correlation_config(&self, correlation: &CorrelationConfig, result: &mut ValidationResult) {
        if self.rules.enum_values {
            // Validate correlation types
            for corr_type in &correlation.default_types {
                if !self.constraints.correlation_types.contains(corr_type) {
                    result.errors.push(ValidationError {
                        category: ErrorCategory::InvalidEnum,
                        path: "analysis.correlation.default_types".to_string(),
                        message: format!("Unknown correlation type: {}", corr_type),
                        suggested_fix: Some("Use one of: pearson, spearman, kendall".to_string()),
                    });
                }
            }
        }
    }

    /// Validate visualization configuration
    fn validate_visualization(&self, viz: &VisualizationConfig, result: &mut ValidationResult) {
        if self.rules.color_formats {
            // Validate color palette
            for (i, color) in viz.colors.palette.iter().enumerate() {
                if !self.is_valid_color(color) {
                    result.errors.push(ValidationError {
                        category: ErrorCategory::InvalidFormat,
                        path: format!("visualization.colors.palette[{}]", i),
                        message: format!("Invalid color format: {}", color),
                        suggested_fix: Some("Use hex format like '#1f77b4'".to_string()),
                    });
                }
            }

            // Validate individual colors
            let color_fields = [
                ("background", &viz.colors.background),
                ("text", &viz.colors.text),
                ("grid", &viz.colors.grid),
                ("anomaly", &viz.colors.anomaly),
                ("trend", &viz.colors.trend),
            ];

            for (field, color) in color_fields {
                if !self.is_valid_color(color) {
                    result.errors.push(ValidationError {
                        category: ErrorCategory::InvalidFormat,
                        path: format!("visualization.colors.{}", field),
                        message: format!("Invalid color format: {}", color),
                        suggested_fix: Some("Use hex format like '#ffffff'".to_string()),
                    });
                }
            }
        }

        if self.rules.enum_values {
            // Validate theme
            if !self.constraints.themes.contains(&viz.default_theme) {
                result.warnings.push(ValidationWarning {
                    category: WarningCategory::Suboptimal,
                    path: "visualization.default_theme".to_string(),
                    message: format!("Unknown theme: {}", viz.default_theme),
                    recommendation: Some("Use one of: professional, dark, light, colorful, minimal".to_string()),
                });
            }
        }
    }

    /// Validate output configuration
    fn validate_output(&self, output: &OutputConfig, result: &mut ValidationResult) {
        if self.rules.enum_values {
            // Validate output format
            if !self.constraints.output_formats.contains(&output.default_format) {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidEnum,
                    path: "output.default_format".to_string(),
                    message: format!("Unknown output format: {}", output.default_format),
                    suggested_fix: Some("Use one of: text, json, csv, markdown, html, pdf".to_string()),
                });
            }
        }

        if self.rules.file_paths {
            // Validate output directory
            if let Some(ref dir) = output.default_directory {
                if dir.exists() && !dir.is_dir() {
                    result.errors.push(ValidationError {
                        category: ErrorCategory::FileSystemError,
                        path: "output.default_directory".to_string(),
                        message: "Output directory path exists but is not a directory".to_string(),
                        suggested_fix: Some("Use a valid directory path".to_string()),
                    });
                }
            }
        }
    }

    /// Validate performance configuration
    fn validate_performance(&self, perf: &PerformanceConfig, result: &mut ValidationResult) {
        if let Some(threads) = perf.num_threads {
            if threads == 0 {
                result.errors.push(ValidationError {
                    category: ErrorCategory::InvalidRange,
                    path: "performance.num_threads".to_string(),
                    message: "Number of threads must be greater than 0".to_string(),
                    suggested_fix: Some("Set to None for automatic detection".to_string()),
                });
            }

            let max_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1);

            if threads > max_threads * 2 {
                result.warnings.push(ValidationWarning {
                    category: WarningCategory::PerformanceImpact,
                    path: "performance.num_threads".to_string(),
                    message: format!("Thread count ({}) is much higher than available cores ({})", threads, max_threads),
                    recommendation: Some(format!("Consider using {} or fewer threads", max_threads)),
                });
            }
        }

        if perf.chunk_size == 0 {
            result.errors.push(ValidationError {
                category: ErrorCategory::InvalidRange,
                path: "performance.chunk_size".to_string(),
                message: "Chunk size must be greater than 0".to_string(),
                suggested_fix: Some("Set to 10000 for reasonable performance".to_string()),
            });
        }
    }

    /// Validate profiles configuration
    fn validate_profiles(&self, profiles: &ProfilesConfig, result: &mut ValidationResult) {
        // Check that all available profiles have definitions
        for profile_name in &profiles.available {
            if !profiles.definitions.contains_key(profile_name) {
                result.errors.push(ValidationError {
                    category: ErrorCategory::ProfileError,
                    path: format!("profiles.available[{}]", profile_name),
                    message: format!("Profile '{}' is listed as available but has no definition", profile_name),
                    suggested_fix: Some("Add profile definition or remove from available list".to_string()),
                });
            }
        }

        // Check that all defined profiles are in available list
        for profile_name in profiles.definitions.keys() {
            if !profiles.available.contains(profile_name) {
                result.warnings.push(ValidationWarning {
                    category: WarningCategory::Suboptimal,
                    path: format!("profiles.definitions[{}]", profile_name),
                    message: format!("Profile '{}' is defined but not in available list", profile_name),
                    recommendation: Some("Add to available list to make it accessible".to_string()),
                });
            }
        }
    }

    /// Validate overall configuration consistency
    fn validate_consistency(&self, config: &Config, result: &mut ValidationResult) {
        // Check that active profile exists
        if !config.profiles.available.contains(&config.metadata.active_profile) {
            result.errors.push(ValidationError {
                category: ErrorCategory::ProfileError,
                path: "metadata.active_profile".to_string(),
                message: format!("Active profile '{}' is not available", config.metadata.active_profile),
                suggested_fix: Some("Set to 'general' or another available profile".to_string()),
            });
        }

        // Performance consistency checks
        if config.performance.enable_parallel && config.performance.num_threads == Some(1) {
            result.warnings.push(ValidationWarning {
                category: WarningCategory::Suboptimal,
                path: "performance".to_string(),
                message: "Parallel processing is enabled but only 1 thread is configured".to_string(),
                recommendation: Some("Set num_threads to None for automatic detection or increase thread count".to_string()),
            });
        }
    }

    /// Check if a version string follows semantic versioning
    fn is_valid_version(&self, version: &str) -> bool {
        let parts: Vec<&str> = version.split('.').collect();
        parts.len() == 3 && parts.iter().all(|part| part.parse::<u32>().is_ok())
    }

    /// Check if a color string is in valid hex format
    fn is_valid_color(&self, color: &str) -> bool {
        if !color.starts_with('#') || color.len() != 7 {
            return false;
        }

        color[1..].chars().all(|c| c.is_ascii_hexdigit())
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for validation
impl ValidationResult {
    /// Check if validation passed without errors
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get all issues (errors + warnings)
    pub fn all_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        for error in &self.errors {
            issues.push(format!("ERROR [{}]: {}", error.path, error.message));
        }

        for warning in &self.warnings {
            issues.push(format!("WARNING [{}]: {}", warning.path, warning.message));
        }

        issues
    }

    /// Get formatted report
    pub fn report(&self) -> String {
        let mut report = String::new();

        if self.is_valid {
            report.push_str("✅ Configuration validation passed\n");
        } else {
            report.push_str("❌ Configuration validation failed\n");
        }

        if !self.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for error in &self.errors {
                report.push_str(&format!("  ❌ [{}] {}\n", error.path, error.message));
                if let Some(ref fix) = error.suggested_fix {
                    report.push_str(&format!("     💡 {}\n", fix));
                }
            }
        }

        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                report.push_str(&format!("  ⚠️  [{}] {}\n", warning.path, warning.message));
                if let Some(ref rec) = warning.recommendation {
                    report.push_str(&format!("     💡 {}\n", rec));
                }
            }
        }

        if !self.suggestions.is_empty() {
            report.push_str("\nSuggestions:\n");
            for suggestion in &self.suggestions {
                report.push_str(&format!("  💡 [{}] {}\n", suggestion.path, suggestion.message));
                report.push_str(&format!("     ✨ Benefit: {}\n", suggestion.benefit));
            }
        }

        report
    }
}