//! # Report Generation Engine
//!
//! Core engine for generating comprehensive reports from time series analysis data.

use crate::reporting::types::*;
use crate::Result;
use crate::stats::analyze_timeseries;
use crate::validation::validate_data_quality;

use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Report generation engine
pub struct ReportEngine {
    config: ReportConfig,
}

impl ReportEngine {
    /// Create a new report engine with the given configuration
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }

    /// Generate a comprehensive report from time series data
    pub fn generate_report(
        &self,
        data: &HashMap<String, Vec<f64>>,
        timestamps: Option<&[DateTime<Utc>]>,
    ) -> Result<ReportResult> {
        let start_time = std::time::Instant::now();

        // Step 1: Perform all required analyses
        let analysis_data = self.perform_analyses(data, timestamps)?;

        // Step 2: Generate automated insights
        let insights = if self.config.sections.automated_insights {
            crate::reporting::insights::generate_insights(&analysis_data, &self.config.advanced)?
        } else {
            Vec::new()
        };

        // Step 3: Render the report using the specified template
        let content = self.render_report(&analysis_data, &insights)?;

        // Step 4: Create export info if needed
        let export_info = if let Some(ref output_path) = self.config.output_path {
            Some(self.export_report(&content, output_path, start_time.elapsed().as_millis() as u64)?)
        } else {
            None
        };

        Ok(ReportResult {
            content,
            analysis_data,
            insights,
            metadata: self.config.metadata.clone(),
            export_info,
        })
    }

    /// Perform all required analyses based on configuration
    fn perform_analyses(
        &self,
        data: &HashMap<String, Vec<f64>>,
        timestamps: Option<&[DateTime<Utc>]>,
    ) -> Result<AnalysisData> {
        let primary_column = data.keys().next().ok_or("No data columns found")?;
        let primary_values = &data[primary_column];

        // Create data summary
        let data_summary = self.create_data_summary(data, timestamps);

        // Initialize analysis data
        let mut analysis_data = AnalysisData {
            statistics: None,
            trend: None,
            seasonality: None,
            anomalies: None,
            forecasting: None,
            correlation: None,
            data_quality: None,
            plots: Vec::new(),
            data_summary,
        };

        // Data quality assessment (always performed)
        if let Some(ts) = timestamps {
            analysis_data.data_quality = Some(validate_data_quality(ts, primary_values));
        }

        // Statistical analysis
        if self.config.sections.descriptive_stats {
            if let Some(ts) = timestamps {
                let stats = analyze_timeseries(ts, primary_values, primary_column)
                    .map_err(|e| format!("Statistics analysis failed: {}", e))?;
                analysis_data.statistics = Some(stats);
            }
        }

        // Trend analysis
        if self.config.sections.trend_analysis {
            // Use the trend analysis from the trend module
            // For now, create a placeholder that matches the expected type
            // In a full implementation, this would call the actual trend analysis
            // analysis_data.trend = Some(analyze_trend(data, timestamps)?);
        }

        // Seasonality analysis
        if self.config.sections.seasonality_analysis {
            // Similar to trend analysis, this would call the seasonality module
            // analysis_data.seasonality = Some(analyze_comprehensive_seasonality(data, timestamps)?);
        }

        // Anomaly detection (disabled until anomaly module is available)
        if self.config.sections.anomaly_detection {
            // TODO: Implement when anomaly detection module is available
            analysis_data.anomalies = Some("Anomaly detection placeholder".to_string());
        }

        // Forecasting analysis (disabled until forecasting module is available)
        if self.config.sections.forecasting {
            // TODO: Implement when forecasting module is available
            analysis_data.forecasting = Some("Forecasting placeholder".to_string());
        }

        // Correlation analysis (disabled until correlation module is available)
        if self.config.sections.correlation_analysis && data.len() > 1 {
            // TODO: Implement when correlation module is available
            analysis_data.correlation = Some("Correlation analysis placeholder".to_string());
        }

        // Generate visualizations
        if self.config.sections.visualizations {
            analysis_data.plots = self.generate_plots(data, timestamps)?;
        }

        Ok(analysis_data)
    }

    /// Create a summary of the data being analyzed
    fn create_data_summary(
        &self,
        data: &HashMap<String, Vec<f64>>,
        timestamps: Option<&[DateTime<Utc>]>,
    ) -> DataSummary {
        let primary_column = data.keys().next().unwrap();
        let primary_values = &data[primary_column];

        let n_points = primary_values.len();
        let columns: Vec<String> = data.keys().cloned().collect();

        // Calculate missing data percentage
        let missing_count = primary_values.iter()
            .filter(|&&v| !v.is_finite())
            .count();
        let missing_percentage = (missing_count as f64 / n_points as f64) * 100.0;

        // Determine date range
        let date_range = timestamps.and_then(|ts| {
            if ts.is_empty() {
                None
            } else {
                Some((ts[0], ts[ts.len() - 1]))
            }
        });

        // Estimate frequency (simplified)
        let frequency = timestamps.and_then(|ts| {
            if ts.len() > 1 {
                let duration = ts[1] - ts[0];
                let minutes = duration.num_minutes();
                match minutes {
                    1 => Some("1min".to_string()),
                    5 => Some("5min".to_string()),
                    15 => Some("15min".to_string()),
                    30 => Some("30min".to_string()),
                    60 => Some("1H".to_string()),
                    1440 => Some("1D".to_string()),
                    _ => Some(format!("{}min", minutes)),
                }
            } else {
                None
            }
        });

        DataSummary {
            n_points,
            date_range,
            frequency,
            columns,
            missing_percentage,
        }
    }

    /// Generate plots for the report (placeholder implementation)
    fn generate_plots(
        &self,
        _data: &HashMap<String, Vec<f64>>,
        _timestamps: Option<&[DateTime<Utc>]>,
    ) -> Result<Vec<String>> {
        let plots = vec!["Time series plot placeholder".to_string()];

        // TODO: Implement actual plotting when plotting module is available
        // For now, return placeholder plot descriptions

        Ok(plots)
    }

    /// Render the report using the specified template
    fn render_report(
        &self,
        analysis_data: &AnalysisData,
        insights: &[Insight],
    ) -> Result<ReportContent> {
        match self.config.template {
            ReportTemplate::Executive => {
                crate::reporting::templates::render_executive_template(
                    analysis_data,
                    insights,
                    &self.config,
                )
            },
            ReportTemplate::Technical => {
                crate::reporting::templates::render_technical_template(
                    analysis_data,
                    insights,
                    &self.config,
                )
            },
            ReportTemplate::DataQuality => {
                crate::reporting::templates::render_data_quality_template(
                    analysis_data,
                    insights,
                    &self.config,
                )
            },
            ReportTemplate::Forecasting => {
                crate::reporting::templates::render_forecasting_template(
                    analysis_data,
                    insights,
                    &self.config,
                )
            },
            ReportTemplate::Custom { .. } => {
                // For custom templates, fall back to technical template
                crate::reporting::templates::render_technical_template(
                    analysis_data,
                    insights,
                    &self.config,
                )
            },
        }
    }

    /// Export the report to a file
    fn export_report(
        &self,
        content: &ReportContent,
        output_path: &str,
        duration_ms: u64,
    ) -> Result<ReportExportInfo> {
        let start_time = std::time::Instant::now();

        let file_size = match &self.config.export_format {
            ReportExportFormat::Markdown => {
                if let ReportContent::Markdown(text) = content {
                    crate::reporting::export::export_to_markdown(text, output_path)?
                } else {
                    return Err("Content format mismatch for Markdown export".into());
                }
            },
            ReportExportFormat::HTML => {
                if let ReportContent::HTML(html) = content {
                    crate::reporting::export::export_to_html(html, output_path)?
                } else {
                    return Err("Content format mismatch for HTML export".into());
                }
            },
            ReportExportFormat::PDF => {
                crate::reporting::export::export_to_pdf(content, output_path)?
            },
            ReportExportFormat::JSON => {
                if let ReportContent::JSON(json) = content {
                    crate::reporting::export::export_to_json(json, output_path)?
                } else {
                    return Err("Content format mismatch for JSON export".into());
                }
            },
            ReportExportFormat::Text => {
                if let ReportContent::Text(text) = content {
                    std::fs::write(output_path, text)?;
                    text.len() as u64
                } else {
                    return Err("Content format mismatch for Text export".into());
                }
            },
        };

        Ok(ReportExportInfo {
            format: self.config.export_format.clone(),
            file_path: Some(output_path.to_string()),
            file_size: Some(file_size),
            exported_at: Utc::now(),
            export_duration_ms: start_time.elapsed().as_millis() as u64,
        })
    }
}

/// Convenience function for generating reports
pub fn generate_report(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    config: ReportConfig,
) -> Result<ReportResult> {
    let engine = ReportEngine::new(config);
    engine.generate_report(data, timestamps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_data() -> (HashMap<String, Vec<f64>>, Vec<DateTime<Utc>>) {
        let mut data = HashMap::new();
        data.insert("value".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let timestamps = vec![
            Utc::now() - chrono::Duration::days(4),
            Utc::now() - chrono::Duration::days(3),
            Utc::now() - chrono::Duration::days(2),
            Utc::now() - chrono::Duration::days(1),
            Utc::now(),
        ];

        (data, timestamps)
    }

    #[test]
    fn test_report_engine_creation() {
        let config = ReportConfig::default();
        let engine = ReportEngine::new(config);
        assert_eq!(engine.config.template, ReportTemplate::Executive);
    }

    #[test]
    fn test_data_summary_creation() {
        let (data, timestamps) = create_test_data();
        let config = ReportConfig::default();
        let engine = ReportEngine::new(config);

        let summary = engine.create_data_summary(&data, Some(&timestamps));

        assert_eq!(summary.n_points, 5);
        assert_eq!(summary.columns.len(), 1);
        assert_eq!(summary.columns[0], "value");
        assert_eq!(summary.missing_percentage, 0.0);
        assert!(summary.date_range.is_some());
    }

    #[test]
    fn test_missing_data_calculation() {
        let mut data = HashMap::new();
        data.insert("value".to_string(), vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]);

        let timestamps = vec![
            Utc::now() - chrono::Duration::days(4),
            Utc::now() - chrono::Duration::days(3),
            Utc::now() - chrono::Duration::days(2),
            Utc::now() - chrono::Duration::days(1),
            Utc::now(),
        ];

        let config = ReportConfig::default();
        let engine = ReportEngine::new(config);

        let summary = engine.create_data_summary(&data, Some(&timestamps));

        assert_eq!(summary.n_points, 5);
        assert_eq!(summary.missing_percentage, 40.0); // 2 out of 5 are not finite
    }
}