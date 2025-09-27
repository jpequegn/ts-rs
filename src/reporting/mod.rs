//! # Comprehensive Reporting Module
//!
//! Provides automated report generation capabilities for time series analysis,
//! including multiple report templates, export formats, and intelligent insights.

pub mod types;
pub mod engine;
pub mod templates;
pub mod insights;
pub mod export;
pub mod batch;

// Re-export main types and functions
pub use types::{
    ReportConfig, ReportTemplate, ReportExportFormat, ReportSections,
    ReportMetadata, AdvancedReportConfig, ReportStyling,
    ReportResult, ReportContent, AnalysisData, DataSummary,
    Insight, InsightCategory, InsightImportance,
    ReportExportInfo, BatchReportConfig, BatchOptions
};

pub use engine::{ReportEngine, generate_report};

pub use templates::{
    ExecutiveTemplate, TechnicalTemplate, DataQualityTemplate, ForecastingTemplate,
    render_template
};

pub use insights::{
    InsightEngine, generate_insights, generate_recommendations
};

pub use export::{
    export_report, export_to_markdown, export_to_html, export_to_pdf, export_to_json
};

pub use batch::{
    process_batch_reports, generate_comparison_report
};

use crate::{TimeSeries, Result};
use std::collections::HashMap;

/// Generate a comprehensive report from time series data
///
/// This is the main entry point for report generation. It performs all necessary
/// analyses and generates a report according to the provided configuration.
///
/// # Arguments
///
/// * `data` - Time series data as a HashMap of column names to values
/// * `timestamps` - Optional timestamps for the data
/// * `config` - Report configuration specifying template, format, and options
///
/// # Example
///
/// ```rust
/// use chronos::reporting::{generate_comprehensive_report, ReportConfig, ReportTemplate, ReportExportFormat};
/// use std::collections::HashMap;
///
/// let mut data = HashMap::new();
/// data.insert("value".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
///
/// let config = ReportConfig {
///     template: ReportTemplate::Executive,
///     export_format: ReportExportFormat::HTML,
///     ..Default::default()
/// };
///
/// let report = generate_comprehensive_report(&data, None, config)?;
/// ```
pub fn generate_comprehensive_report(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[chrono::DateTime<chrono::Utc>]>,
    config: ReportConfig,
) -> Result<ReportResult> {
    let engine = ReportEngine::new(config);
    engine.generate_report(data, timestamps)
}

/// Generate a quick executive summary report
///
/// Convenience function for generating a basic executive summary report
/// with default settings.
pub fn generate_executive_summary(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[chrono::DateTime<chrono::Utc>]>,
    output_path: Option<String>,
) -> Result<ReportResult> {
    let config = ReportConfig {
        template: ReportTemplate::Executive,
        export_format: ReportExportFormat::HTML,
        output_path,
        ..Default::default()
    };

    generate_comprehensive_report(data, timestamps, config)
}

/// Generate a technical analysis report
///
/// Convenience function for generating a detailed technical analysis report
/// with all sections enabled.
pub fn generate_technical_report(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[chrono::DateTime<chrono::Utc>]>,
    output_path: Option<String>,
) -> Result<ReportResult> {
    let mut sections = ReportSections::default();
    sections.forecasting = true;
    sections.correlation_analysis = true;

    let mut advanced = AdvancedReportConfig::default();
    advanced.include_technical_appendix = true;

    let config = ReportConfig {
        template: ReportTemplate::Technical,
        export_format: ReportExportFormat::HTML,
        sections,
        output_path,
        advanced,
        ..Default::default()
    };

    generate_comprehensive_report(data, timestamps, config)
}

/// Generate a data quality assessment report
///
/// Convenience function for generating a focused data quality report.
pub fn generate_data_quality_report(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[chrono::DateTime<chrono::Utc>]>,
    output_path: Option<String>,
) -> Result<ReportResult> {
    let mut sections = ReportSections::default();
    sections.trend_analysis = false;
    sections.seasonality_analysis = false;
    sections.anomaly_detection = true;  // Relevant for data quality
    sections.forecasting = false;
    sections.correlation_analysis = false;

    let config = ReportConfig {
        template: ReportTemplate::DataQuality,
        export_format: ReportExportFormat::HTML,
        sections,
        output_path,
        ..Default::default()
    };

    generate_comprehensive_report(data, timestamps, config)
}

/// Generate a forecasting-focused report
///
/// Convenience function for generating a report focused on forecasting analysis.
pub fn generate_forecasting_report(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[chrono::DateTime<chrono::Utc>]>,
    output_path: Option<String>,
    _forecast_config: Option<String>, // Placeholder until forecasting module is available
) -> Result<ReportResult> {
    let mut sections = ReportSections::default();
    sections.forecasting = true;
    sections.trend_analysis = true;  // Important for forecasting
    sections.seasonality_analysis = true;  // Important for forecasting

    let mut advanced = AdvancedReportConfig::default();
    advanced.include_confidence_intervals = true;

    let config = ReportConfig {
        template: ReportTemplate::Forecasting,
        export_format: ReportExportFormat::HTML,
        sections,
        output_path,
        advanced,
        ..Default::default()
    };

    generate_comprehensive_report(data, timestamps, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_data() -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();
        data.insert("value".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0]);
        data
    }

    #[test]
    fn test_default_report_config() {
        let config = ReportConfig::default();
        assert_eq!(config.template, ReportTemplate::Executive);
        assert_eq!(config.export_format, ReportExportFormat::HTML);
        assert!(config.sections.executive_summary);
        assert!(config.sections.data_quality);
    }

    #[test]
    fn test_report_template_display() {
        assert_eq!(ReportTemplate::Executive.to_string(), "Executive Summary");
        assert_eq!(ReportTemplate::Technical.to_string(), "Technical Analysis");
        assert_eq!(ReportTemplate::DataQuality.to_string(), "Data Quality Assessment");
        assert_eq!(ReportTemplate::Forecasting.to_string(), "Forecasting Analysis");
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(ReportExportFormat::HTML.to_string(), "HTML");
        assert_eq!(ReportExportFormat::Markdown.to_string(), "Markdown");
        assert_eq!(ReportExportFormat::PDF.to_string(), "PDF");
        assert_eq!(ReportExportFormat::JSON.to_string(), "JSON");
    }

    #[test]
    fn test_insight_categories() {
        let insight = Insight {
            category: InsightCategory::Trend,
            title: "Test".to_string(),
            description: "Test insight".to_string(),
            confidence: 0.8,
            importance: InsightImportance::High,
            evidence: vec!["Evidence 1".to_string()],
            recommendations: vec!["Recommendation 1".to_string()],
        };

        assert_eq!(insight.category, InsightCategory::Trend);
        assert_eq!(insight.importance, InsightImportance::High);
    }
}