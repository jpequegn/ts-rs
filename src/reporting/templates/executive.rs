//! # Executive Summary Report Template
//!
//! Provides a high-level executive summary focused on key insights and business impact.

use crate::reporting::types::{AnalysisData, Insight, ReportConfig, ReportContent, ReportExportFormat, InsightImportance};
use crate::reporting::templates::{ReportTemplate, formatting::*};
use crate::Result;

/// Executive summary template implementation
pub struct ExecutiveTemplate;

impl ReportTemplate for ExecutiveTemplate {
    fn render(
        analysis_data: &AnalysisData,
        insights: &[Insight],
        config: &ReportConfig,
    ) -> Result<ReportContent> {
        match config.export_format {
            ReportExportFormat::Markdown => render_markdown(analysis_data, insights, config),
            ReportExportFormat::HTML => render_html(analysis_data, insights, config),
            ReportExportFormat::JSON => render_json(analysis_data, insights, config),
            ReportExportFormat::Text => render_text(analysis_data, insights, config),
            ReportExportFormat::PDF => {
                // For PDF, render HTML first then convert
                render_html(analysis_data, insights, config)
            }
        }
    }

    fn name() -> &'static str {
        "Executive Summary"
    }

    fn description() -> &'static str {
        "High-level overview focusing on key insights and business impact"
    }
}

/// Render the executive template in Markdown format
fn render_markdown(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    let mut content = String::new();

    // Header
    content.push_str(&format!("# {}\n\n", config.metadata.title));
    content.push_str(&format!("**Generated:** {}\n", format_timestamp(&config.metadata.generated_at)));
    if let Some(ref author) = config.metadata.author {
        content.push_str(&format!("**Author:** {}\n", author));
    }
    content.push_str(&format!("**Data Source:** {}\n\n", config.metadata.data_source));

    // Executive Summary
    content.push_str("## Executive Summary\n\n");
    content.push_str(&generate_executive_summary(analysis_data, insights));

    // Key Findings
    content.push_str("\n## Key Findings\n\n");
    content.push_str(&generate_key_findings(analysis_data, insights));

    // Data Overview
    content.push_str("\n## Data Overview\n\n");
    content.push_str(&generate_data_overview(analysis_data));

    // Critical Insights
    let critical_insights: Vec<_> = insights.iter()
        .filter(|i| i.importance == InsightImportance::Critical || i.importance == InsightImportance::High)
        .collect();

    if !critical_insights.is_empty() {
        content.push_str("\n## Critical Insights\n\n");
        for insight in critical_insights {
            content.push_str(&format!("### {} {}\n\n",
                importance_indicator(&insight.importance),
                insight.title
            ));
            content.push_str(&format!("{}\n\n", insight.description));

            if !insight.recommendations.is_empty() {
                content.push_str("**Recommended Actions:**\n");
                for rec in &insight.recommendations {
                    content.push_str(&format!("- {}\n", rec));
                }
                content.push_str("\n");
            }
        }
    }

    // Quality Assessment
    if let Some(ref quality) = analysis_data.data_quality {
        content.push_str("\n## Data Quality Assessment\n\n");
        let quality_score = quality.quality_score(analysis_data.data_summary.n_points);
        content.push_str(&format!("**Overall Quality:** {} ({:.1}%)\n\n",
            quality_score_indicator(quality_score),
            quality_score * 100.0
        ));

        if quality.nan_count > 0 || quality.infinite_count > 0 {
            content.push_str("**Data Issues Identified:**\n");
            if quality.nan_count > 0 {
                content.push_str(&format!("- {} missing values detected\n", format_number(quality.nan_count)));
            }
            if quality.infinite_count > 0 {
                content.push_str(&format!("- {} infinite values detected\n", format_number(quality.infinite_count)));
            }
            content.push_str("\n");
        }
    }

    // Next Steps
    content.push_str("\n## Recommended Next Steps\n\n");
    content.push_str(&generate_next_steps(analysis_data, insights));

    Ok(ReportContent::Markdown(content))
}

/// Render the executive template in HTML format
fn render_html(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    let mut html = String::new();

    // HTML header with CSS
    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="{}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .metadata {{ margin-top: 15px; opacity: 0.9; }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .insight {{
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            background-color: #f8f9ff;
        }}
        .insight.critical {{ border-color: #dc3545; background-color: #fff5f5; }}
        .insight.high {{ border-color: #fd7e14; background-color: #fff8f1; }}
        .metric {{
            display: inline-block;
            padding: 10px 15px;
            margin: 5px;
            background-color: #e9ecef;
            border-radius: 5px;
            font-weight: bold;
        }}
        .quality-excellent {{ color: #28a745; }}
        .quality-good {{ color: #20c997; }}
        .quality-fair {{ color: #ffc107; }}
        .quality-poor {{ color: #fd7e14; }}
        .quality-critical {{ color: #dc3545; }}
        .recommendations {{
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .recommendations ul {{ margin: 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{}</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {}</p>
"#,
        config.advanced.language,
        config.metadata.title,
        config.metadata.title,
        format_timestamp(&config.metadata.generated_at)
    ));

    if let Some(ref author) = config.metadata.author {
        html.push_str(&format!("            <p><strong>Author:</strong> {}</p>\n", author));
    }
    html.push_str(&format!("            <p><strong>Data Source:</strong> {}</p>\n", config.metadata.data_source));
    html.push_str("        </div>\n    </div>\n");

    // Executive Summary
    html.push_str("    <div class=\"section\">\n");
    html.push_str("        <h2>📊 Executive Summary</h2>\n");
    html.push_str(&format!("        <p>{}</p>\n", generate_executive_summary(analysis_data, insights)));
    html.push_str("    </div>\n");

    // Key Metrics
    html.push_str("    <div class=\"section\">\n");
    html.push_str("        <h2>📈 Key Metrics</h2>\n");
    html.push_str(&generate_key_metrics_html(analysis_data));
    html.push_str("    </div>\n");

    // Critical Insights
    let critical_insights: Vec<_> = insights.iter()
        .filter(|i| i.importance == InsightImportance::Critical || i.importance == InsightImportance::High)
        .collect();

    if !critical_insights.is_empty() {
        html.push_str("    <div class=\"section\">\n");
        html.push_str("        <h2>🚨 Critical Insights</h2>\n");

        for insight in critical_insights {
            let class = match insight.importance {
                InsightImportance::Critical => "critical",
                InsightImportance::High => "high",
                _ => "",
            };

            html.push_str(&format!("        <div class=\"insight {}\">\n", class));
            html.push_str(&format!("            <h3>{} {}</h3>\n",
                importance_indicator(&insight.importance),
                insight.title
            ));
            html.push_str(&format!("            <p>{}</p>\n", insight.description));

            if !insight.recommendations.is_empty() {
                html.push_str("            <div class=\"recommendations\">\n");
                html.push_str("                <strong>Recommended Actions:</strong>\n");
                html.push_str("                <ul>\n");
                for rec in &insight.recommendations {
                    html.push_str(&format!("                    <li>{}</li>\n", rec));
                }
                html.push_str("                </ul>\n");
                html.push_str("            </div>\n");
            }
            html.push_str("        </div>\n");
        }
        html.push_str("    </div>\n");
    }

    // Data Quality
    if let Some(ref quality) = analysis_data.data_quality {
        html.push_str("    <div class=\"section\">\n");
        html.push_str("        <h2>🔍 Data Quality Assessment</h2>\n");

        let quality_score = quality.quality_score(analysis_data.data_summary.n_points);
        let quality_class = match quality_score {
            s if s >= 0.9 => "quality-excellent",
            s if s >= 0.8 => "quality-good",
            s if s >= 0.7 => "quality-fair",
            s if s >= 0.5 => "quality-poor",
            _ => "quality-critical",
        };

        html.push_str(&format!("        <p class=\"{}\"><strong>Overall Quality:</strong> {} ({:.1}%)</p>\n",
            quality_class,
            quality_score_indicator(quality_score),
            quality_score * 100.0
        ));

        if quality.nan_count > 0 || quality.infinite_count > 0 {
            html.push_str("        <h4>Issues Identified:</h4>\n");
            html.push_str("        <ul>\n");
            if quality.nan_count > 0 {
                html.push_str(&format!("            <li>{} missing values detected</li>\n", format_number(quality.nan_count)));
            }
            if quality.infinite_count > 0 {
                html.push_str(&format!("            <li>{} infinite values detected</li>\n", format_number(quality.infinite_count)));
            }
            html.push_str("        </ul>\n");
        }
        html.push_str("    </div>\n");
    }

    // Next Steps
    html.push_str("    <div class=\"section\">\n");
    html.push_str("        <h2>🎯 Recommended Next Steps</h2>\n");
    html.push_str(&format!("        {}\n", generate_next_steps_html(analysis_data, insights)));
    html.push_str("    </div>\n");

    html.push_str("</body>\n</html>");

    Ok(ReportContent::HTML(html))
}

/// Render the executive template in JSON format
fn render_json(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    let json_data = serde_json::json!({
        "report_type": "executive_summary",
        "metadata": config.metadata,
        "executive_summary": generate_executive_summary(analysis_data, insights),
        "key_findings": generate_key_findings(analysis_data, insights),
        "data_overview": {
            "total_points": analysis_data.data_summary.n_points,
            "date_range": analysis_data.data_summary.date_range,
            "frequency": analysis_data.data_summary.frequency,
            "columns": analysis_data.data_summary.columns,
            "missing_percentage": analysis_data.data_summary.missing_percentage
        },
        "critical_insights": insights.iter()
            .filter(|i| i.importance == InsightImportance::Critical || i.importance == InsightImportance::High)
            .collect::<Vec<_>>(),
        "data_quality": analysis_data.data_quality.as_ref().map(|q| {
            serde_json::json!({
                "quality_score": q.quality_score(analysis_data.data_summary.n_points),
                "nan_count": q.nan_count,
                "infinite_count": q.infinite_count,
                "duplicate_timestamps": q.duplicate_timestamps,
                "potential_outliers": q.potential_outliers
            })
        }),
        "generated_at": config.metadata.generated_at
    });

    Ok(ReportContent::JSON(json_data))
}

/// Render the executive template in plain text format
fn render_text(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    let mut content = String::new();

    // Header
    content.push_str(&format!("{}\n", config.metadata.title));
    content.push_str(&format!("{}\n\n", "=".repeat(config.metadata.title.len())));
    content.push_str(&format!("Generated: {}\n", format_timestamp(&config.metadata.generated_at)));
    if let Some(ref author) = config.metadata.author {
        content.push_str(&format!("Author: {}\n", author));
    }
    content.push_str(&format!("Data Source: {}\n\n", config.metadata.data_source));

    // Executive Summary
    content.push_str("EXECUTIVE SUMMARY\n");
    content.push_str("-----------------\n\n");
    content.push_str(&generate_executive_summary(analysis_data, insights));

    // Key Findings
    content.push_str("\n\nKEY FINDINGS\n");
    content.push_str("------------\n\n");
    content.push_str(&generate_key_findings(analysis_data, insights));

    // Data Overview
    content.push_str("\n\nDATA OVERVIEW\n");
    content.push_str("-------------\n\n");
    content.push_str(&generate_data_overview(analysis_data));

    // Critical Insights
    let critical_insights: Vec<_> = insights.iter()
        .filter(|i| i.importance == InsightImportance::Critical || i.importance == InsightImportance::High)
        .collect();

    if !critical_insights.is_empty() {
        content.push_str("\n\nCRITICAL INSIGHTS\n");
        content.push_str("-----------------\n\n");

        for insight in critical_insights {
            content.push_str(&format!("{} {}\n",
                importance_indicator(&insight.importance),
                insight.title
            ));
            content.push_str(&format!("{}\n\n", insight.description));

            if !insight.recommendations.is_empty() {
                content.push_str("Recommended Actions:\n");
                for rec in &insight.recommendations {
                    content.push_str(&format!("  - {}\n", rec));
                }
                content.push_str("\n");
            }
        }
    }

    Ok(ReportContent::Text(content))
}

/// Generate executive summary text
fn generate_executive_summary(analysis_data: &AnalysisData, insights: &[Insight]) -> String {
    let mut summary = String::new();

    summary.push_str(&format!(
        "This report analyzes {} data points across {} time series column(s). ",
        format_number(analysis_data.data_summary.n_points),
        analysis_data.data_summary.columns.len()
    ));

    if let Some(ref date_range) = analysis_data.data_summary.date_range {
        summary.push_str(&format!(
            "The analysis covers data from {} to {}. ",
            date_range.0.format("%Y-%m-%d"),
            date_range.1.format("%Y-%m-%d")
        ));
    }

    if analysis_data.data_summary.missing_percentage > 0.0 {
        summary.push_str(&format!(
            "Data quality analysis reveals {:.1}% missing values. ",
            analysis_data.data_summary.missing_percentage
        ));
    }

    let high_importance_insights = insights.iter()
        .filter(|i| i.importance == InsightImportance::Critical || i.importance == InsightImportance::High)
        .count();

    if high_importance_insights > 0 {
        summary.push_str(&format!(
            "{} critical insights were identified that require immediate attention.",
            high_importance_insights
        ));
    } else {
        summary.push_str("The analysis indicates stable data patterns with no critical issues identified.");
    }

    summary
}

/// Generate key findings text
fn generate_key_findings(analysis_data: &AnalysisData, insights: &[Insight]) -> String {
    let mut findings = String::new();

    // Data volume finding
    findings.push_str(&format!("• Dataset contains {} observations",
        format_number(analysis_data.data_summary.n_points)));

    if let Some(ref freq) = analysis_data.data_summary.frequency {
        findings.push_str(&format!(" sampled at {} intervals", freq));
    }
    findings.push_str("\n");

    // Data quality finding
    if let Some(ref quality) = analysis_data.data_quality {
        let quality_score = quality.quality_score(analysis_data.data_summary.n_points);
        findings.push_str(&format!("• Data quality rated as {} ({:.1}%)\n",
            quality_score_indicator(quality_score),
            quality_score * 100.0
        ));
    }

    // Statistical insights
    if let Some(ref stats) = analysis_data.statistics {
        if let Some(ref desc) = stats.descriptive {
            findings.push_str(&format!("• Mean value: {:.2}, Standard deviation: {:.2}\n",
                desc.mean, desc.std_dev));
        }
    }

    // Top insights
    let top_insights: Vec<_> = insights.iter().take(3).collect();
    if !top_insights.is_empty() {
        findings.push_str("\nKey insights identified:\n");
        for insight in top_insights {
            findings.push_str(&format!("• {}\n", insight.title));
        }
    }

    findings
}

/// Generate data overview text
fn generate_data_overview(analysis_data: &AnalysisData) -> String {
    let mut overview = String::new();

    overview.push_str(&format!("**Total Data Points:** {}\n",
        format_number(analysis_data.data_summary.n_points)));

    overview.push_str(&format!("**Columns Analyzed:** {}\n",
        analysis_data.data_summary.columns.join(", ")));

    if let Some(ref freq) = analysis_data.data_summary.frequency {
        overview.push_str(&format!("**Sampling Frequency:** {}\n", freq));
    }

    if let Some(ref date_range) = analysis_data.data_summary.date_range {
        overview.push_str(&format!("**Time Period:** {}\n",
            format_date_range(&date_range.0, &date_range.1)));
    }

    overview.push_str(&format!("**Missing Data:** {:.1}%\n",
        analysis_data.data_summary.missing_percentage));

    overview
}

/// Generate next steps recommendations
fn generate_next_steps(analysis_data: &AnalysisData, insights: &[Insight]) -> String {
    let mut steps = String::new();

    // Collect all recommendations from insights
    let mut all_recommendations = Vec::new();
    for insight in insights {
        all_recommendations.extend(insight.recommendations.iter());
    }

    if all_recommendations.is_empty() {
        steps.push_str("1. Continue monitoring data quality and patterns\n");
        steps.push_str("2. Consider implementing automated alerts for anomalies\n");
        steps.push_str("3. Schedule regular analysis updates\n");
    } else {
        for (i, rec) in all_recommendations.iter().take(5).enumerate() {
            steps.push_str(&format!("{}. {}\n", i + 1, rec));
        }
    }

    // Add data quality recommendations if needed
    if let Some(ref quality) = analysis_data.data_quality {
        if quality.nan_count > 0 || quality.infinite_count > 0 {
            if !all_recommendations.is_empty() {
                steps.push_str(&format!("{}. Address data quality issues identified in the assessment\n",
                    all_recommendations.len() + 1));
            }
        }
    }

    steps
}

/// Generate key metrics HTML
fn generate_key_metrics_html(analysis_data: &AnalysisData) -> String {
    let mut html = String::new();

    html.push_str(&format!("        <div class=\"metric\">📊 {} Data Points</div>\n",
        format_number(analysis_data.data_summary.n_points)));

    html.push_str(&format!("        <div class=\"metric\">📈 {} Columns</div>\n",
        analysis_data.data_summary.columns.len()));

    if let Some(ref freq) = analysis_data.data_summary.frequency {
        html.push_str(&format!("        <div class=\"metric\">⏱️ {} Frequency</div>\n", freq));
    }

    html.push_str(&format!("        <div class=\"metric\">🎯 {:.1}% Complete</div>\n",
        100.0 - analysis_data.data_summary.missing_percentage));

    html
}

/// Generate next steps HTML
fn generate_next_steps_html(analysis_data: &AnalysisData, insights: &[Insight]) -> String {
    let steps = generate_next_steps(analysis_data, insights);
    let lines: Vec<&str> = steps.lines().collect();

    let mut html = String::new();
    html.push_str("        <ol>\n");
    for line in lines {
        if !line.trim().is_empty() {
            // Remove the number prefix from the line
            let content = line.split(". ").skip(1).collect::<Vec<_>>().join(". ");
            html.push_str(&format!("            <li>{}</li>\n", content));
        }
    }
    html.push_str("        </ol>\n");

    html
}

/// Public function to render the executive template
pub fn render_template(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    ExecutiveTemplate::render(analysis_data, insights, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::types::{ReportConfig, AnalysisData, DataSummary, InsightCategory};

    fn create_test_data() -> (AnalysisData, Vec<Insight>, ReportConfig) {
        let analysis_data = AnalysisData {
            statistics: None,
            trend: None,
            seasonality: None,
            anomalies: None,
            forecasting: None,
            correlation: None,
            data_quality: None,
            plots: Vec::new(),
            data_summary: DataSummary {
                n_points: 1000,
                date_range: None,
                frequency: Some("1D".to_string()),
                columns: vec!["value".to_string()],
                missing_percentage: 2.5,
            },
        };

        let insights = vec![
            Insight {
                category: InsightCategory::Trend,
                title: "Strong upward trend detected".to_string(),
                description: "Data shows consistent growth over the analysis period".to_string(),
                confidence: 0.9,
                importance: InsightImportance::High,
                evidence: vec!["Positive slope coefficient".to_string()],
                recommendations: vec!["Continue monitoring trend".to_string()],
            }
        ];

        let config = ReportConfig::default();

        (analysis_data, insights, config)
    }

    #[test]
    fn test_executive_template_markdown() {
        let (analysis_data, insights, config) = create_test_data();
        let result = render_markdown(&analysis_data, &insights, &config);
        assert!(result.is_ok());

        if let Ok(ReportContent::Markdown(content)) = result {
            assert!(content.contains("Executive Summary"));
            assert!(content.contains("1,000"));
            assert!(content.contains("Strong upward trend"));
        }
    }

    #[test]
    fn test_executive_template_html() {
        let (analysis_data, insights, config) = create_test_data();
        let result = render_html(&analysis_data, &insights, &config);
        assert!(result.is_ok());

        if let Ok(ReportContent::HTML(content)) = result {
            assert!(content.contains("<html"));
            assert!(content.contains("Executive Summary"));
            assert!(content.contains("1,000"));
        }
    }

    #[test]
    fn test_executive_summary_generation() {
        let (analysis_data, insights, _) = create_test_data();
        let summary = generate_executive_summary(&analysis_data, &insights);

        assert!(summary.contains("1,000 data points"));
        assert!(summary.contains("1 critical insights"));
    }

    #[test]
    fn test_template_trait_implementation() {
        assert_eq!(ExecutiveTemplate::name(), "Executive Summary");
        assert!(ExecutiveTemplate::description().contains("High-level"));
    }
}