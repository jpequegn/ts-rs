//! # Batch Reporting Functions
//!
//! Functions for processing multiple reports and generating comparison analyses.

use crate::reporting::types::{
    BatchReportConfig, BatchOptions, ReportResult, ReportConfig, AnalysisData,
    Insight, InsightCategory, InsightImportance, ReportContent, ReportMetadata
};
use crate::reporting::engine::ReportEngine;
use crate::Result;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use rayon::prelude::*;

/// Process multiple reports in batch
pub fn process_batch_reports(
    datasets: &[(&HashMap<String, Vec<f64>>, Option<&[DateTime<Utc>]>, String)], // (data, timestamps, name)
    config: BatchReportConfig,
) -> Result<Vec<ReportResult>> {
    let mut reports = Vec::new();

    // Determine if we should process in parallel
    let use_parallel = config.batch_options.max_parallel > 1 && datasets.len() > 1;

    if use_parallel {
        // Process in parallel using rayon
        let parallel_results: Result<Vec<_>> = datasets
            .par_iter()
            .enumerate()
            .map(|(idx, (data, timestamps, name))| {
                let mut report_config = if idx < config.reports.len() {
                    config.reports[idx].clone()
                } else {
                    config.reports[0].clone() // Use first config as template
                };

                // Update metadata with dataset-specific information
                report_config.metadata.title = format!("{} - {}", report_config.metadata.title, name);
                report_config.metadata.data_source = name.clone();

                // Generate output path if not specified
                if report_config.output_path.is_none() {
                    let filename = generate_batch_filename(idx, name, &report_config);
                    let output_path = std::path::Path::new(&config.output_directory)
                        .join(filename);
                    report_config.output_path = Some(output_path.to_string_lossy().to_string());
                }

                let engine = ReportEngine::new(report_config);
                engine.generate_report(data, *timestamps)
            })
            .collect();

        reports = parallel_results?;
    } else {
        // Process sequentially
        for (idx, (data, timestamps, name)) in datasets.iter().enumerate() {
            let result = if config.batch_options.continue_on_error {
                match process_single_report(data, *timestamps, name, idx, &config) {
                    Ok(report) => report,
                    Err(e) => {
                        eprintln!("Warning: Failed to process dataset '{}': {}", name, e);
                        continue;
                    }
                }
            } else {
                process_single_report(data, *timestamps, name, idx, &config)?
            };

            reports.push(result);
        }
    }

    // Generate summary report if requested
    if config.batch_options.generate_summary {
        let summary_report = generate_summary_report(&reports, &config)?;
        reports.push(summary_report);
    }

    Ok(reports)
}

/// Process a single report in a batch
fn process_single_report(
    data: &HashMap<String, Vec<f64>>,
    timestamps: Option<&[DateTime<Utc>]>,
    name: &str,
    index: usize,
    config: &BatchReportConfig,
) -> Result<ReportResult> {
    let mut report_config = if index < config.reports.len() {
        config.reports[index].clone()
    } else {
        config.reports[0].clone() // Use first config as template
    };

    // Update metadata
    report_config.metadata.title = format!("{} - {}", report_config.metadata.title, name);
    report_config.metadata.data_source = name.to_string();

    // Generate output path if not specified
    if report_config.output_path.is_none() {
        let filename = generate_batch_filename(index, name, &report_config);
        let output_path = std::path::Path::new(&config.output_directory)
            .join(filename);
        report_config.output_path = Some(output_path.to_string_lossy().to_string());
    }

    let engine = ReportEngine::new(report_config);
    engine.generate_report(data, timestamps)
}

/// Generate a comparison report across multiple datasets
pub fn generate_comparison_report(
    reports: &[ReportResult],
    config: &BatchReportConfig,
) -> Result<ReportResult> {
    let mut content = String::new();

    // Header
    content.push_str("# Batch Analysis Comparison Report\n\n");
    content.push_str(&format!("**Generated:** {}\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    content.push_str(&format!("**Number of Datasets:** {}\n\n", reports.len()));

    // Summary Statistics
    content.push_str("## Summary Statistics\n\n");
    content.push_str("| Dataset | Data Points | Missing % | Quality Score |\n");
    content.push_str("|---------|-------------|-----------|---------------|\n");

    for report in reports {
        let quality_score = if let Some(ref quality) = report.analysis_data.data_quality {
            format!("{:.1}%", quality.quality_score(report.analysis_data.data_summary.n_points) * 100.0)
        } else {
            "N/A".to_string()
        };

        content.push_str(&format!("| {} | {} | {:.1}% | {} |\n",
            report.metadata.data_source,
            format_number(report.analysis_data.data_summary.n_points),
            report.analysis_data.data_summary.missing_percentage,
            quality_score
        ));
    }
    content.push_str("\n");

    // Insights Comparison
    content.push_str("## Insights Summary\n\n");
    let insight_summary = summarize_insights_across_reports(reports);

    for (category, count) in insight_summary {
        content.push_str(&format!("- **{:?}:** {} insights across datasets\n", category, count));
    }
    content.push_str("\n");

    // Common Issues
    let common_issues = find_common_issues(reports);
    if !common_issues.is_empty() {
        content.push_str("## Common Issues\n\n");
        for issue in common_issues {
            content.push_str(&format!("- {}\n", issue));
        }
        content.push_str("\n");
    }

    // Recommendations
    content.push_str("## Recommendations\n\n");
    let recommendations = generate_batch_recommendations(reports);
    for (i, rec) in recommendations.iter().enumerate() {
        content.push_str(&format!("{}. {}\n", i + 1, rec));
    }

    // Create metadata for the comparison report
    let metadata = ReportMetadata {
        title: "Batch Analysis Comparison".to_string(),
        author: None,
        data_source: format!("{} datasets", reports.len()),
        analysis_period: None,
        generated_at: Utc::now(),
        version: "1.0".to_string(),
        custom_fields: HashMap::new(),
    };

    // Create a summary analysis data
    let summary_analysis = create_summary_analysis_data(reports);

    Ok(ReportResult {
        content: ReportContent::Markdown(content),
        analysis_data: summary_analysis,
        insights: Vec::new(), // Could aggregate insights here
        metadata,
        export_info: None,
    })
}

/// Generate filename for batch reports
fn generate_batch_filename(index: usize, name: &str, config: &ReportConfig) -> String {
    let sanitized_name = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>();

    let extension = match config.export_format {
        crate::reporting::types::ReportExportFormat::Markdown => "md",
        crate::reporting::types::ReportExportFormat::HTML => "html",
        crate::reporting::types::ReportExportFormat::PDF => "pdf",
        crate::reporting::types::ReportExportFormat::JSON => "json",
        crate::reporting::types::ReportExportFormat::Text => "txt",
    };

    format!("{}_{:03}.{}", sanitized_name, index + 1, extension)
}

/// Summarize insights across multiple reports
fn summarize_insights_across_reports(reports: &[ReportResult]) -> HashMap<InsightCategory, usize> {
    let mut summary = HashMap::new();

    for report in reports {
        for insight in &report.insights {
            *summary.entry(insight.category.clone()).or_insert(0) += 1;
        }
    }

    summary
}

/// Find common issues across reports
fn find_common_issues(reports: &[ReportResult]) -> Vec<String> {
    let mut issues = Vec::new();

    // Check for data quality issues
    let mut missing_data_count = 0;
    let mut infinite_values_count = 0;
    let mut duplicate_timestamps_count = 0;

    for report in reports {
        if let Some(ref quality) = report.analysis_data.data_quality {
            if quality.nan_count > 0 {
                missing_data_count += 1;
            }
            if quality.infinite_count > 0 {
                infinite_values_count += 1;
            }
            if quality.duplicate_timestamps > 0 {
                duplicate_timestamps_count += 1;
            }
        }
    }

    let total_reports = reports.len();
    if missing_data_count > total_reports / 2 {
        issues.push(format!("Missing data issues found in {} out of {} datasets", missing_data_count, total_reports));
    }
    if infinite_values_count > 0 {
        issues.push(format!("Infinite values found in {} out of {} datasets", infinite_values_count, total_reports));
    }
    if duplicate_timestamps_count > 0 {
        issues.push(format!("Duplicate timestamps found in {} out of {} datasets", duplicate_timestamps_count, total_reports));
    }

    // Check for common insight patterns
    let insight_counts = summarize_insights_across_reports(reports);
    for (category, count) in insight_counts {
        if count > total_reports / 2 {
            issues.push(format!("{:?} issues identified in majority of datasets", category));
        }
    }

    issues
}

/// Generate recommendations for batch processing
fn generate_batch_recommendations(reports: &[ReportResult]) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Analyze data quality across all reports
    let total_reports = reports.len();
    let mut low_quality_count = 0;

    for report in reports {
        if let Some(ref quality) = report.analysis_data.data_quality {
            let quality_score = quality.quality_score(report.analysis_data.data_summary.n_points);
            if quality_score < 0.7 {
                low_quality_count += 1;
            }
        }
    }

    if low_quality_count > total_reports / 3 {
        recommendations.push("Implement standardized data quality checks across all data sources".to_string());
        recommendations.push("Consider data preprocessing pipeline to improve overall quality".to_string());
    }

    // Check for inconsistent data patterns
    let data_point_sizes: Vec<usize> = reports.iter()
        .map(|r| r.analysis_data.data_summary.n_points)
        .collect();

    if data_point_sizes.len() > 1 {
        let min_size = *data_point_sizes.iter().min().unwrap();
        let max_size = *data_point_sizes.iter().max().unwrap();

        if max_size > min_size * 10 {
            recommendations.push("Consider normalizing dataset sizes for more consistent analysis".to_string());
        }
    }

    // General batch processing recommendations
    recommendations.push("Schedule regular automated analysis runs for continuous monitoring".to_string());
    recommendations.push("Implement alerting for datasets with critical quality issues".to_string());
    recommendations.push("Consider consolidating datasets with similar characteristics".to_string());

    recommendations
}

/// Create summary analysis data from multiple reports
fn create_summary_analysis_data(reports: &[ReportResult]) -> AnalysisData {
    use crate::reporting::types::DataSummary;

    let total_points: usize = reports.iter()
        .map(|r| r.analysis_data.data_summary.n_points)
        .sum();

    let avg_missing_percentage: f64 = reports.iter()
        .map(|r| r.analysis_data.data_summary.missing_percentage)
        .sum::<f64>() / reports.len() as f64;

    let all_columns: Vec<String> = reports.iter()
        .flat_map(|r| r.analysis_data.data_summary.columns.iter())
        .cloned()
        .collect();

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
            n_points: total_points,
            date_range: None, // Could be computed from all reports
            frequency: None,
            columns: all_columns,
            missing_percentage: avg_missing_percentage,
        },
    }
}

/// Generate summary report for batch processing
pub fn generate_summary_report(
    reports: &[ReportResult],
    config: &BatchReportConfig,
) -> Result<ReportResult> {
    generate_comparison_report(reports, config)
}

/// Helper function to format numbers with thousand separators
fn format_number(value: usize) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::types::{ReportConfig, ReportTemplate, ReportExportFormat, ReportSections};
    use std::collections::HashMap;

    fn create_test_dataset() -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();
        data.insert("value".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data
    }

    fn create_test_batch_config() -> BatchReportConfig {
        BatchReportConfig {
            reports: vec![ReportConfig::default()],
            output_directory: "/tmp/test_reports".to_string(),
            batch_options: BatchOptions::default(),
        }
    }

    #[test]
    fn test_batch_filename_generation() {
        let config = ReportConfig::default();
        let filename = generate_batch_filename(0, "test dataset", &config);
        assert_eq!(filename, "test_dataset_001.html");
    }

    #[test]
    fn test_insight_summarization() {
        let reports = vec![]; // Empty for testing
        let summary = summarize_insights_across_reports(&reports);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_common_issues_detection() {
        let reports = vec![]; // Empty for testing
        let issues = find_common_issues(&reports);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_batch_recommendations() {
        let reports = vec![]; // Empty for testing
        let recommendations = generate_batch_recommendations(&reports);
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("automated analysis")));
    }

    #[test]
    fn test_summary_analysis_data() {
        let reports = vec![]; // Empty for testing
        let summary = create_summary_analysis_data(&reports);
        assert_eq!(summary.data_summary.n_points, 0);
    }
}