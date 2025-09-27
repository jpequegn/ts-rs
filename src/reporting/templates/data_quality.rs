//! # Data Quality Assessment Report Template
//!
//! Provides focused data quality assessment and recommendations.

use crate::reporting::types::{AnalysisData, Insight, ReportConfig, ReportContent, InsightCategory};
use crate::reporting::templates::{ReportTemplate, formatting::*};
use crate::Result;

/// Data quality assessment template implementation
pub struct DataQualityTemplate;

impl ReportTemplate for DataQualityTemplate {
    fn render(
        analysis_data: &AnalysisData,
        insights: &[Insight],
        config: &ReportConfig,
    ) -> Result<ReportContent> {
        let mut content = String::new();

        // Header
        content.push_str(&format!("# Data Quality Assessment Report\n\n"));
        content.push_str(&format!("**Generated:** {}\n", format_timestamp(&config.metadata.generated_at)));
        content.push_str(&format!("**Data Source:** {}\n\n", config.metadata.data_source));

        // Overall Quality Score
        if let Some(ref quality) = analysis_data.data_quality {
            let quality_score = quality.quality_score(analysis_data.data_summary.n_points);
            content.push_str("## Overall Quality Assessment\n\n");
            content.push_str(&format!("**Quality Score:** {} ({:.1}%)\n\n",
                quality_score_indicator(quality_score),
                quality_score * 100.0
            ));

            // Detailed Quality Metrics
            content.push_str("## Quality Metrics\n\n");
            content.push_str(&format!("- **Total Data Points:** {}\n", format_number(analysis_data.data_summary.n_points)));
            content.push_str(&format!("- **Missing Values:** {}\n", format_number(quality.nan_count)));
            content.push_str(&format!("- **Infinite Values:** {}\n", format_number(quality.infinite_count)));
            content.push_str(&format!("- **Duplicate Timestamps:** {}\n", format_number(quality.duplicate_timestamps)));
            content.push_str(&format!("- **Potential Outliers:** {}\n", format_number(quality.potential_outliers)));
            content.push_str("\n");
        }

        // Data Quality Insights
        let quality_insights: Vec<_> = insights.iter()
            .filter(|i| i.category == InsightCategory::DataQuality)
            .collect();

        if !quality_insights.is_empty() {
            content.push_str("## Data Quality Issues\n\n");
            for insight in quality_insights {
                content.push_str(&format!("### {} {}\n\n",
                    importance_indicator(&insight.importance),
                    insight.title
                ));
                content.push_str(&format!("{}\n\n", insight.description));

                if !insight.evidence.is_empty() {
                    content.push_str("**Evidence:**\n");
                    for evidence in &insight.evidence {
                        content.push_str(&format!("- {}\n", evidence));
                    }
                    content.push_str("\n");
                }

                if !insight.recommendations.is_empty() {
                    content.push_str("**Recommendations:**\n");
                    for rec in &insight.recommendations {
                        content.push_str(&format!("- {}\n", rec));
                    }
                    content.push_str("\n");
                }
            }
        }

        Ok(ReportContent::Markdown(content))
    }

    fn name() -> &'static str {
        "Data Quality Assessment"
    }

    fn description() -> &'static str {
        "Focused data quality assessment and recommendations"
    }
}

/// Public function to render the data quality template
pub fn render_template(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    DataQualityTemplate::render(analysis_data, insights, config)
}