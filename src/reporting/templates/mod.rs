//! # Report Templates Module
//!
//! Provides various report templates for different audiences and use cases.

pub mod executive;
pub mod technical;
pub mod data_quality;
pub mod forecasting;

use crate::reporting::types::{AnalysisData, Insight, ReportConfig, ReportContent};
use crate::Result;

// Re-export template structs and functions
pub use executive::ExecutiveTemplate;
pub use technical::TechnicalTemplate;
pub use data_quality::DataQualityTemplate;
pub use forecasting::ForecastingTemplate;

// Re-export template render functions
pub use executive::render_template as render_executive_template;
pub use technical::render_template as render_technical_template;
pub use data_quality::render_template as render_data_quality_template;
pub use forecasting::render_template as render_forecasting_template;

/// Trait for report template implementations
pub trait ReportTemplateImpl {
    /// Render the template with the given data and configuration
    fn render(
        analysis_data: &AnalysisData,
        insights: &[Insight],
        config: &ReportConfig,
    ) -> Result<ReportContent>;

    /// Get the template name
    fn name() -> &'static str;

    /// Get the template description
    fn description() -> &'static str;
}

// Template structs are defined in their respective modules and re-exported above

/// Render a template based on the template type
pub fn render_template(
    template_type: &crate::reporting::types::ReportTemplate,
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    match template_type {
        crate::reporting::types::ReportTemplate::Executive => {
            executive::render_template(analysis_data, insights, config)
        },
        crate::reporting::types::ReportTemplate::Technical => {
            technical::render_template(analysis_data, insights, config)
        },
        crate::reporting::types::ReportTemplate::DataQuality => {
            data_quality::render_template(analysis_data, insights, config)
        },
        crate::reporting::types::ReportTemplate::Forecasting => {
            forecasting::render_template(analysis_data, insights, config)
        },
        crate::reporting::types::ReportTemplate::Custom { name: _name, sections: _sections } => {
            // For custom templates, use technical template as base
            // In a full implementation, this would support custom section rendering
            technical::render_template(analysis_data, insights, config)
        },
    }
}

/// Common formatting utilities for templates
pub mod formatting {
    use chrono::{DateTime, Utc};

    /// Format a float with appropriate precision
    pub fn format_float(value: f64, decimals: usize) -> String {
        format!("{:.1$}", value, decimals)
    }

    /// Format a percentage
    pub fn format_percentage(value: f64) -> String {
        format!("{:.1}%", value * 100.0)
    }

    /// Format a timestamp
    pub fn format_timestamp(timestamp: &DateTime<Utc>) -> String {
        timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string()
    }

    /// Format a date range
    pub fn format_date_range(start: &DateTime<Utc>, end: &DateTime<Utc>) -> String {
        format!("{} to {}",
            start.format("%Y-%m-%d"),
            end.format("%Y-%m-%d")
        )
    }

    /// Format a large number with thousand separators
    pub fn format_number(value: usize) -> String {
        let s = value.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    /// Create a confidence indicator
    pub fn confidence_indicator(confidence: f64) -> &'static str {
        match confidence {
            c if c >= 0.9 => "🔥 Very High",
            c if c >= 0.7 => "✅ High",
            c if c >= 0.5 => "⚠️ Medium",
            c if c >= 0.3 => "❓ Low",
            _ => "❌ Very Low",
        }
    }

    /// Create an importance indicator
    pub fn importance_indicator(importance: &crate::reporting::types::InsightImportance) -> &'static str {
        match importance {
            crate::reporting::types::InsightImportance::Critical => "🚨 Critical",
            crate::reporting::types::InsightImportance::High => "🔴 High",
            crate::reporting::types::InsightImportance::Medium => "🟡 Medium",
            crate::reporting::types::InsightImportance::Low => "🟢 Low",
        }
    }

    /// Generate a quality score indicator
    pub fn quality_score_indicator(score: f64) -> &'static str {
        match score {
            s if s >= 0.9 => "🌟 Excellent",
            s if s >= 0.8 => "✅ Good",
            s if s >= 0.7 => "⚠️ Fair",
            s if s >= 0.5 => "❗ Poor",
            _ => "❌ Critical",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::types::{ReportConfig, AnalysisData, DataSummary};
    use std::collections::HashMap;

    fn create_test_analysis_data() -> AnalysisData {
        AnalysisData {
            statistics: None,
            trend: None,
            seasonality: None,
            anomalies: None,
            forecasting: None,
            correlation: None,
            data_quality: None,
            plots: Vec::new(),
            data_summary: DataSummary {
                n_points: 100,
                date_range: None,
                frequency: Some("1D".to_string()),
                columns: vec!["value".to_string()],
                missing_percentage: 0.0,
            },
        }
    }

    #[test]
    fn test_template_names() {
        assert_eq!(ExecutiveTemplate::name(), "Executive Summary");
        assert_eq!(TechnicalTemplate::name(), "Technical Analysis");
        assert_eq!(DataQualityTemplate::name(), "Data Quality Assessment");
        assert_eq!(ForecastingTemplate::name(), "Forecasting Analysis");
    }

    #[test]
    fn test_formatting_utilities() {
        use formatting::*;

        assert_eq!(format_float(3.14159, 2), "3.14");
        assert_eq!(format_percentage(0.25), "25.0%");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(confidence_indicator(0.95), "🔥 Very High");
        assert_eq!(confidence_indicator(0.3), "❓ Low");
    }
}