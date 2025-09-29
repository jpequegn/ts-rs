//! # Chronos Time Series Analysis Library
//!
//! Core data structures and functionality for time series analysis,
//! providing robust, efficient data handling and analysis capabilities.

pub mod types;
pub mod timeseries;
pub mod analysis;
pub mod validation;
pub mod import;
pub mod preprocessing;
pub mod stats;
pub mod trend;
pub mod seasonality;
pub mod reporting;
pub mod cli;
pub mod anomaly;
pub mod forecasting;
pub mod correlation;
pub mod plotting;
pub mod config;
pub mod performance;
pub mod plugins;
pub mod quality;

// Re-export commonly used types
pub use types::{Frequency, MissingValuePolicy};
pub use timeseries::TimeSeries;
pub use analysis::{AnalysisResult, TrendAnalysis, SeasonalAnalysis, AnomalyDetection};
pub use import::{ImportConfig, ImportResult, ImportStats, import_from_file, import_csv, import_json};
pub use preprocessing::{PreprocessingConfig, PreprocessingResult, preprocess_timeseries};
pub use stats::{
    StatisticalAnalysisResult, AnalysisMetadata,
    DescriptiveStats, compute_descriptive_stats,
    DistributionAnalysis, compute_distribution_analysis,
    TimeSeriesStats, compute_autocorrelation, compute_partial_autocorrelation, compute_cross_correlation,
    StationarityTest, test_stationarity,
    ChangePoint, detect_changepoints,
    ExportFormat, export_stats_results, analyze_timeseries
};
pub use trend::{
    // Main analysis functions
    analyze_comprehensive,

    // Decomposition
    DecompositionMethod, DecompositionResult, perform_decomposition,

    // Detection
    TrendTest, TrendTestResult, detect_trend, test_trend_significance,

    // Detrending
    DetrendingMethod, DetrendingResult, perform_detrending,
    linear_detrend, difference_detrend, moving_average_detrend, hp_filter_detrend,

    // Analysis and classification
    TrendAnalysisConfig, analyze_trend_comprehensive, classify_trend_direction, compute_trend_strength,

    // Visualization
    TrendPlotData, DecompositionPlotData, generate_trend_plot_data, generate_decomposition_plot_data,

    // Main result types
    ComprehensiveTrendAnalysis, TrendSummary
};
pub use seasonality::{
    // Main analysis functions
    analyze_comprehensive_seasonality,

    // Detection
    SeasonalityDetectionResult, SeasonalityMethod, detect_seasonality,
    detect_multiple_seasonalities, analyze_fourier_spectrum, compute_periodogram,
    FourierAnalysis, PeriodogramAnalysis, AutocorrelationAnalysis,

    // Pattern analysis
    SeasonalPatternAnalysis, SeasonalStrength, PatternConsistency,
    analyze_seasonal_patterns, compute_seasonal_strength, analyze_pattern_consistency,

    // Adjustment
    SeasonalAdjustmentMethod, SeasonalAdjustmentResult, perform_seasonal_adjustment,
    apply_x13_adjustment, apply_moving_average_adjustment, apply_stl_adjustment,

    // Calendar effects
    CalendarEffects, HolidayImpact, TradingDayEffects, LeapYearAdjustment,
    detect_calendar_effects, detect_holiday_impacts, analyze_trading_day_effects,

    // Advanced features
    AdvancedSeasonalityAnalysis, MultipleSeasonalPeriods, EvolvingSeasonality,
    SeasonalBreaks, detect_evolving_seasonality, find_seasonal_breaks,
    analyze_multiple_seasonal_periods, comprehensive_seasonality_analysis,

    // Main result types and configuration
    ComprehensiveSeasonalityAnalysis, SeasonalPeriod, SeasonalityAnalysisConfig
};
pub use anomaly::{
    // Main detection functions
    detect_anomalies, detect_anomalies_single_method,

    // Configuration
    AnomalyDetectionConfig, AnomalyMethod, ThresholdConfig, ContextualConfig,
    ScoringConfig, StreamingConfig, ScoringMethod, SeverityThresholds,

    // Streaming detection
    streaming::{StreamingAnomalyDetector, StreamingMethod, AdaptiveConfig, LearningConfig},

    // Utilities
    utils::{StatUtils, DataUtils, ThresholdUtils, AnomalyUtils, DistanceUtils}
};
pub use forecasting::{
    // Main forecasting functions
    forecast_timeseries, forecast_with_intervals, evaluate_forecast_model,

    // Configuration and result types
    ForecastConfig, ForecastResult, ForecastMethod, ModelEvaluation,
    EvaluationConfig, EvaluationMetric, FeatureConfig,

    // Method-specific types
    SeasonalType, ETSComponent, GrowthType, SeasonalityMode, EnsembleCombination,
    LagConfig, RollingConfig, CalendarConfig, RollingStatistic,

    // Feature engineering
    features::{EnhancedTimeSeries, create_enhanced_timeseries}
};
pub use correlation::{
    // Main analysis functions
    analyze_correlations, AnalysisConfig,

    // Basic correlation types and functions
    CorrelationType, CorrelationMatrix, compute_correlation_matrix,

    // Rolling correlation analysis
    RollingCorrelation, compute_rolling_correlation,

    // Cross-correlation analysis
    CrossCorrelationAnalysis, LeadLagResult,

    // Granger causality testing
    GrangerCausalityResult, VARModel, test_granger_causality,

    // Cointegration analysis
    CointegrationResult, test_cointegration,

    // Dynamic Time Warping
    DTWResult, compute_dtw_distance,

    // Principal Component Analysis
    PCAResult, compute_pca,

    // Main result type
    CorrelationAnalysisResult
};
pub use plotting::{
    // Core plotting types
    PlotConfig, PlotType, ExportFormat as PlotExportFormat, Theme, PlotResult, PlotData, PlotPoint, PlotSeries,

    // Main plotting functions
    plot, render_plot,

    // Time series plotting
    create_line_plot, create_scatter_plot, create_multiple_series_plot, create_subplot_layout,

    // Statistical plotting
    create_histogram, create_box_plot, create_violin_plot, create_qq_plot,
    create_acf_plot, create_pacf_plot, create_density_plot,

    // Correlation plotting
    create_correlation_heatmap, create_scatter_matrix, create_correlation_plot,

    // Decomposition and forecast plotting
    create_decomposition_plot, create_seasonal_plot, create_trend_plot,
    create_forecast_plot, create_anomaly_plot,

    // Styling and themes
    apply_theme, customize_styling,

    // Export functionality
    export_to_file, export_to_html as plot_export_to_html, export_to_png, export_to_svg, export_to_pdf as plot_export_to_pdf,
    ExportOptions
};

/// Result type used throughout the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Error types for the time series library
#[derive(Debug, thiserror::Error)]
pub enum TimeSeriesError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Data inconsistency: {0}")]
    DataInconsistency(String),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error("Missing data: {0}")]
    MissingData(String),

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl TimeSeriesError {
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    pub fn data_inconsistency(msg: impl Into<String>) -> Self {
        Self::DataInconsistency(msg.into())
    }

    pub fn invalid_timestamp(msg: impl Into<String>) -> Self {
        Self::InvalidTimestamp(msg.into())
    }

    pub fn missing_data(msg: impl Into<String>) -> Self {
        Self::MissingData(msg.into())
    }

    pub fn analysis(msg: impl Into<String>) -> Self {
        Self::Analysis(msg.into())
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}

// Re-export reporting module
pub use reporting::{
    // Core types
    ReportConfig, ReportTemplate, ReportExportFormat, ReportSections,
    ReportMetadata, AdvancedReportConfig, ReportResult, ReportContent,
    AnalysisData, Insight, InsightCategory, InsightImportance,

    // Main functions
    generate_comprehensive_report, generate_executive_summary,
    generate_technical_report, generate_data_quality_report, generate_forecasting_report,

    // Batch processing
    BatchReportConfig, process_batch_reports, generate_comparison_report,

    // Export functions
    export_report, export_to_html as report_export_to_html, export_to_markdown, export_to_pdf as report_export_to_pdf, export_to_json,
};

// Re-export configuration module
pub use config::{
    // Core configuration types
    Config, ConfigMetadata, AnalysisConfig as ConfigAnalysisConfig, VisualizationConfig, OutputConfig, PerformanceConfig,
    ProfilesConfig, ProfileDefinition, DataCharacteristics, ProfileDetectionRules,

    // Configuration management
    ConfigLoader, ConfigManager, ConfigFormat, ConfigError, Result as ConfigResult,

    // Validation
    validation::{ConfigValidator, ValidationResult},
};

// Re-export performance module
pub use performance::{
    // Core performance types
    PerformanceOptimizer, PerformanceMetrics, MemoryStats, PerformanceError,

    // Memory management
    memory::{MemoryManager, StreamingProcessor, LazyDataLoader, CompactTimeSeries},

    // Parallel processing
    parallel::{ParallelProcessor, ParallelConfig, TaskProcessor, SeriesStatistics, ThreadInfo},

    // Caching
    cache::{CacheManager, CacheConfig, AnalysisCache, CacheStats},

    // Database integration
    database::{DatabaseManager, DatabaseConfig, TimeSeriesDb, PerformanceRecord},

    // Progress tracking
    progress::{ProgressTracker, ProgressConfig, ProgressBar, ProgressSpinner, ProgressAware, OperationStatus},
};

// Re-export plugin module
pub use plugins::{
    // Core plugin types
    Plugin, PluginError, PluginResult, PluginType, PluginContext, PluginConfig,
    PluginCapabilities, PluginStatus, ApiVersion,

    // Plugin registry and management
    PluginRegistry, PluginInfo, PluginMetadata, PluginLoader,
    PluginManager, PluginInstallConfig, PluginUpdateConfig,

    // Data source plugins
    data_source::{
        DataSourcePlugin, DataSourceConfig, DataSourceResult,
        ConnectionType, ImportConfig as PluginImportConfig, AuthConfig, DataSchema
    },

    // Analysis plugins
    analysis::{
        AnalysisPlugin, AnalysisConfig as PluginAnalysisConfig, AnalysisResult as PluginAnalysisResult,
        AnalysisMethod, AnalysisCategory, AnalysisCapabilities
    },

    // Visualization plugins
    visualization::{
        VisualizationPlugin, VisualizationConfig as PluginVisualizationConfig, VisualizationResult,
        PlotType as PluginPlotType, PlotCategory, ExportFormat as VisualizationExportFormat
    },

    // Management types
    management::{
        AvailablePlugin, InstallResult, UpdateResult, UninstallResult,
        PluginRepository, RepositoryType as PluginRepositoryType
    },
};

// Re-export quality module
pub use quality::{
    // Core types
    QualityAssessment, QualityIssue, QualityMetrics, OutlierMethod, TimeRange,

    // Configuration
    QualityConfig, OutlierDetectionConfig, ConsistencyConfig,
    TimelinessConfig, RangeConfig,

    // Error handling
    QualityError, QualityResult,
};