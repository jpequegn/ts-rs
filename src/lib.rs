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
    export_report, export_to_html, export_to_markdown, export_to_pdf, export_to_json,
};