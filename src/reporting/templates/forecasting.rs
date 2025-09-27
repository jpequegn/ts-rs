//! # Forecasting Analysis Report Template
//!
//! Provides forecasting analysis with predictions and model evaluation.

use crate::reporting::types::{AnalysisData, Insight, ReportConfig, ReportContent, InsightCategory};
use crate::reporting::templates::{ReportTemplate, formatting::*};
use crate::Result;

/// Forecasting analysis template implementation
pub struct ForecastingTemplate;

impl ReportTemplate for ForecastingTemplate {
    fn render(
        analysis_data: &AnalysisData,
        insights: &[Insight],
        config: &ReportConfig,
    ) -> Result<ReportContent> {
        let mut content = String::new();

        // Header
        content.push_str(&format!("# Forecasting Analysis Report\n\n"));
        content.push_str(&format!("**Generated:** {}\n", format_timestamp(&config.metadata.generated_at)));
        content.push_str(&format!("**Data Source:** {}\n\n", config.metadata.data_source));

        // Data Overview
        content.push_str("## Data Overview\n\n");
        content.push_str(&format!("- **Data Points:** {}\n", format_number(analysis_data.data_summary.n_points)));
        if let Some(ref date_range) = analysis_data.data_summary.date_range {
            content.push_str(&format!("- **Time Period:** {}\n",
                format_date_range(&date_range.0, &date_range.1)));
        }
        if let Some(ref freq) = analysis_data.data_summary.frequency {
            content.push_str(&format!("- **Frequency:** {}\n", freq));
        }
        content.push_str("\n");

        // Forecast Results
        if let Some(ref forecast) = analysis_data.forecasting {
            content.push_str("## Forecast Results\n\n");
            content.push_str("[Forecast results would be displayed here]\n\n");
        } else {
            content.push_str("## Forecast Status\n\n");
            content.push_str("No forecasting analysis was performed for this dataset.\n\n");
        }

        // Trend Analysis (important for forecasting)
        if let Some(ref trend) = analysis_data.trend {
            content.push_str("## Trend Analysis\n\n");
            content.push_str(&format!("**Trend Direction:** {:?}\n", trend.trend_summary.direction));
            content.push_str(&format!("**Trend Strength:** {:.1}%\n", trend.trend_summary.strength * 100.0));
            if let Some(growth_rate) = trend.trend_summary.growth_rate {
                content.push_str(&format!("**Annual Growth Rate:** {:.1}%\n", growth_rate * 100.0));
            }
            content.push_str("\n");
        }

        // Forecasting Insights
        let forecast_insights: Vec<_> = insights.iter()
            .filter(|i| i.category == InsightCategory::Forecasting)
            .collect();

        if !forecast_insights.is_empty() {
            content.push_str("## Forecasting Insights\n\n");
            for insight in forecast_insights {
                content.push_str(&format!("### {} {}\n\n",
                    importance_indicator(&insight.importance),
                    insight.title
                ));
                content.push_str(&format!("{}\n\n", insight.description));

                if !insight.recommendations.is_empty() {
                    content.push_str("**Recommendations:**\n");
                    for rec in &insight.recommendations {
                        content.push_str(&format!("- {}\n", rec));
                    }
                    content.push_str("\n");
                }
            }
        }

        // Model Recommendations
        content.push_str("## Model Recommendations\n\n");
        content.push_str("Based on the data characteristics:\n\n");

        if let Some(ref trend) = analysis_data.trend {
            if trend.trend_summary.strength > 0.5 {
                content.push_str("- Strong trend detected: Consider trend-aware models (Holt-Winters, ARIMA with trend)\n");
            }
        }

        if let Some(ref seasonality) = analysis_data.seasonality {
            content.push_str("- Seasonality analysis available: Review for seasonal patterns\n");
        }

        content.push_str("- Evaluate model performance using cross-validation\n");
        content.push_str("- Monitor forecast accuracy and retrain models as needed\n");

        Ok(ReportContent::Markdown(content))
    }

    fn name() -> &'static str {
        "Forecasting Analysis"
    }

    fn description() -> &'static str {
        "Forecasting analysis with predictions and model evaluation"
    }
}

/// Public function to render the forecasting template
pub fn render_template(
    analysis_data: &AnalysisData,
    insights: &[Insight],
    config: &ReportConfig,
) -> Result<ReportContent> {
    ForecastingTemplate::render(analysis_data, insights, config)
}