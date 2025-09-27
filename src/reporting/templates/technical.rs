//! # Technical Analysis Report Template
//!
//! Provides detailed technical analysis with comprehensive statistical information.

use crate::reporting::types::{AnalysisData, Insight, ReportConfig, ReportContent};
use crate::reporting::templates::{ReportTemplate, formatting::*};
use crate::Result;

/// Technical analysis template implementation
pub struct TechnicalTemplate;

impl ReportTemplate for TechnicalTemplate {
    fn render(
        analysis_data: &AnalysisData,
        insights: &[Insight],
        config: &ReportConfig,
    ) -> Result<ReportContent> {
        // For now, create a basic technical report
        // In a full implementation, this would include detailed statistical analysis,
        // mathematical formulations, confidence intervals, test results, etc.

        let content = format!(
            "# Technical Analysis Report\n\n\
            Generated: {}\n\n\
            ## Statistical Summary\n\n\
            Data Points: {}\n\
            Columns: {}\n\
            Missing Data: {:.1}%\n\n\
            ## Detailed Analysis\n\n\
            [Technical analysis details would be included here]\n\n\
            ## Insights\n\n\
            {} insights generated\n",
            format_timestamp(&config.metadata.generated_at),
            format_number(analysis_data.data_summary.n_points),
            analysis_data.data_summary.columns.join(", "),
            analysis_data.data_summary.missing_percentage,
            insights.len()
        );

        Ok(ReportContent::Markdown(content))
    }

    fn name() -> &'static str {
        "Technical Analysis"
    }

    fn description() -> &'static str {
        "Detailed technical analysis with comprehensive statistical information"
    }
}

/// Public function to render the technical template
pub fn render_template(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    TechnicalTemplate::render(analysis_data, insights, config)
}