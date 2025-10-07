//! # Automated Insights Generation
//!
//! Generates intelligent insights and recommendations from time series analysis data.

use crate::reporting::types::{
    AnalysisData, Insight, InsightCategory, InsightImportance, AdvancedReportConfig
};
use crate::Result;

/// Insight generation engine
pub struct InsightEngine {
    config: AdvancedReportConfig,
}

impl InsightEngine {
    /// Create a new insight engine with configuration
    pub fn new(config: AdvancedReportConfig) -> Self {
        Self { config }
    }

    /// Generate insights from analysis data
    pub fn generate_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        // Data quality insights
        insights.extend(self.generate_data_quality_insights(analysis_data)?);

        // Statistical insights
        insights.extend(self.generate_statistical_insights(analysis_data)?);

        // Trend insights
        insights.extend(self.generate_trend_insights(analysis_data)?);

        // Seasonality insights
        insights.extend(self.generate_seasonality_insights(analysis_data)?);

        // Anomaly insights
        insights.extend(self.generate_anomaly_insights(analysis_data)?);

        // Forecasting insights
        insights.extend(self.generate_forecasting_insights(analysis_data)?);

        // Sort by importance and confidence
        insights.sort_by(|a, b| {
            let importance_order = |i: &InsightImportance| match i {
                InsightImportance::Critical => 4,
                InsightImportance::High => 3,
                InsightImportance::Medium => 2,
                InsightImportance::Low => 1,
            };

            importance_order(&b.importance).cmp(&importance_order(&a.importance))
                .then_with(|| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Limit to max insights if configured
        if insights.len() > self.config.max_insights {
            insights.truncate(self.config.max_insights);
        }

        Ok(insights)
    }

    /// Generate data quality insights
    fn generate_data_quality_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        if let Some(ref quality) = analysis_data.data_quality {
            let quality_score = quality.quality_score(analysis_data.data_summary.n_points);

            // Overall quality assessment
            let (importance, title, description, recommendations) = match quality_score {
                s if s >= 0.9 => (
                    InsightImportance::Low,
                    "Excellent Data Quality".to_string(),
                    format!("Data quality is excellent with a score of {:.1}%. No major issues detected.", s * 100.0),
                    vec!["Continue current data collection practices".to_string()],
                ),
                s if s >= 0.7 => (
                    InsightImportance::Medium,
                    "Good Data Quality with Minor Issues".to_string(),
                    format!("Data quality is good with a score of {:.1}%. Some minor issues were identified.", s * 100.0),
                    vec!["Review data collection processes for improvement opportunities".to_string()],
                ),
                s if s >= 0.5 => (
                    InsightImportance::High,
                    "Fair Data Quality Requires Attention".to_string(),
                    format!("Data quality is fair with a score of {:.1}%. Several issues need attention.", s * 100.0),
                    vec![
                        "Implement data validation checks".to_string(),
                        "Review data collection processes".to_string(),
                        "Consider data cleaning procedures".to_string(),
                    ],
                ),
                _ => (
                    InsightImportance::Critical,
                    "Poor Data Quality Detected".to_string(),
                    format!("Data quality is poor with a score of {:.1}%. Immediate action required.", quality_score * 100.0),
                    vec![
                        "Halt analysis until data quality issues are resolved".to_string(),
                        "Implement comprehensive data validation".to_string(),
                        "Review entire data pipeline".to_string(),
                    ],
                ),
            };

            insights.push(Insight {
                category: InsightCategory::DataQuality,
                title,
                description,
                confidence: 0.95,
                importance,
                evidence: vec![format!("Quality score: {:.1}%", quality_score * 100.0)],
                recommendations,
            });

            // Missing values insight
            if quality.nan_count > 0 {
                let missing_percentage = (quality.nan_count as f64 / analysis_data.data_summary.n_points as f64) * 100.0;
                let importance = if missing_percentage > 20.0 {
                    InsightImportance::Critical
                } else if missing_percentage > 10.0 {
                    InsightImportance::High
                } else if missing_percentage > 5.0 {
                    InsightImportance::Medium
                } else {
                    InsightImportance::Low
                };

                insights.push(Insight {
                    category: InsightCategory::DataQuality,
                    title: "Missing Values Detected".to_string(),
                    description: format!("{} missing values found ({:.1}% of total data). This may impact analysis accuracy.",
                        quality.nan_count, missing_percentage),
                    confidence: 1.0,
                    importance,
                    evidence: vec![
                        format!("{} NaN values", quality.nan_count),
                        format!("{:.1}% missing data", missing_percentage),
                    ],
                    recommendations: vec![
                        "Investigate the source of missing values".to_string(),
                        "Consider imputation techniques for missing data".to_string(),
                        "Implement data validation at collection point".to_string(),
                    ],
                });
            }

            // Infinite values insight
            if quality.infinite_count > 0 {
                insights.push(Insight {
                    category: InsightCategory::DataQuality,
                    title: "Infinite Values Detected".to_string(),
                    description: format!("{} infinite values found. This indicates potential data collection or calculation errors.",
                        quality.infinite_count),
                    confidence: 1.0,
                    importance: InsightImportance::High,
                    evidence: vec![format!("{} infinite values", quality.infinite_count)],
                    recommendations: vec![
                        "Investigate calculation processes that may produce infinite values".to_string(),
                        "Implement bounds checking in data collection".to_string(),
                        "Remove or correct infinite values before analysis".to_string(),
                    ],
                });
            }

            // Duplicate timestamps insight
            if quality.duplicate_timestamps > 0 {
                insights.push(Insight {
                    category: InsightCategory::DataQuality,
                    title: "Duplicate Timestamps Found".to_string(),
                    description: format!("{} duplicate timestamps detected. This may indicate data collection issues or system errors.",
                        quality.duplicate_timestamps),
                    confidence: 1.0,
                    importance: InsightImportance::Medium,
                    evidence: vec![format!("{} duplicate timestamps", quality.duplicate_timestamps)],
                    recommendations: vec![
                        "Review data collection system for timestamp generation".to_string(),
                        "Implement deduplication logic".to_string(),
                        "Check for system clock issues".to_string(),
                    ],
                });
            }

            // Outliers insight
            if quality.potential_outliers > 0 {
                let outlier_percentage = (quality.potential_outliers as f64 / analysis_data.data_summary.n_points as f64) * 100.0;

                insights.push(Insight {
                    category: InsightCategory::DataQuality,
                    title: "Potential Outliers Identified".to_string(),
                    description: format!("{} potential outliers detected ({:.1}% of data). These may represent genuine extreme values or data errors.",
                        quality.potential_outliers, outlier_percentage),
                    confidence: 0.8,
                    importance: if outlier_percentage > 10.0 { InsightImportance::High } else { InsightImportance::Medium },
                    evidence: vec![
                        format!("{} outliers", quality.potential_outliers),
                        format!("{:.1}% of total data", outlier_percentage),
                    ],
                    recommendations: vec![
                        "Manually review identified outliers".to_string(),
                        "Determine if outliers represent valid extreme events".to_string(),
                        "Consider robust statistical methods if outliers are valid".to_string(),
                    ],
                });
            }
        }

        Ok(insights)
    }

    /// Generate statistical insights
    fn generate_statistical_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        if let Some(ref stats) = analysis_data.statistics {
            if let Some(ref desc) = stats.descriptive {
                // Variability insight
                let cv = desc.std_dev / desc.mean.abs(); // Coefficient of variation
                if cv > 1.0 {
                    insights.push(Insight {
                        category: InsightCategory::Statistical,
                        title: "High Data Variability".to_string(),
                        description: format!("The data shows high variability with a coefficient of variation of {:.2}. This indicates significant fluctuations around the mean.",
                            cv),
                        confidence: 0.9,
                        importance: InsightImportance::Medium,
                        evidence: vec![
                            format!("Coefficient of variation: {:.2}", cv),
                            format!("Standard deviation: {:.2}", desc.std_dev),
                            format!("Mean: {:.2}", desc.mean),
                        ],
                        recommendations: vec![
                            "Consider investigating sources of variability".to_string(),
                            "Evaluate if high variability is expected for this data type".to_string(),
                            "Consider using robust statistical methods".to_string(),
                        ],
                    });
                } else if cv < 0.1 {
                    insights.push(Insight {
                        category: InsightCategory::Statistical,
                        title: "Low Data Variability".to_string(),
                        description: format!("The data shows low variability with a coefficient of variation of {:.2}. This indicates stable, consistent values.",
                            cv),
                        confidence: 0.9,
                        importance: InsightImportance::Low,
                        evidence: vec![
                            format!("Coefficient of variation: {:.2}", cv),
                            format!("Standard deviation: {:.2}", desc.std_dev),
                        ],
                        recommendations: vec![
                            "Verify that low variability is expected".to_string(),
                            "Check for data collection issues if more variation is expected".to_string(),
                        ],
                    });
                }

                // Skewness insight
                if let Some(ref dist) = stats.distribution {
                    if dist.skewness.abs() > 1.0 {
                        let direction = if dist.skewness > 0.0 { "right" } else { "left" };
                        insights.push(Insight {
                            category: InsightCategory::Statistical,
                            title: format!("Significant {} Skewness Detected", direction.to_uppercase()),
                            description: format!("The data distribution is significantly skewed to the {} (skewness = {:.2}). This may affect analysis assumptions.",
                                direction, dist.skewness),
                            confidence: 0.85,
                            importance: InsightImportance::Medium,
                            evidence: vec![format!("Skewness coefficient: {:.2}", dist.skewness)],
                            recommendations: vec![
                                "Consider data transformation (e.g., log transformation)".to_string(),
                                "Use non-parametric statistical methods".to_string(),
                                "Investigate causes of skewness".to_string(),
                            ],
                        });
                    }
                }
            }

            // Stationarity insights
            if !stats.stationarity_tests.is_empty() {
                let stationary_tests = stats.stationarity_tests.values()
                    .filter(|test| test.is_stationary)
                    .count();
                let total_tests = stats.stationarity_tests.len();

                if stationary_tests == 0 {
                    insights.push(Insight {
                        category: InsightCategory::Statistical,
                        title: "Non-Stationary Time Series".to_string(),
                        description: "All stationarity tests indicate the time series is non-stationary. This suggests the presence of trends or changing variance.".to_string(),
                        confidence: 0.9,
                        importance: InsightImportance::High,
                        evidence: stats.stationarity_tests.iter()
                            .map(|(name, test)| format!("{}: p-value = {:.4}", name, test.p_value))
                            .collect(),
                        recommendations: vec![
                            "Apply differencing to make the series stationary".to_string(),
                            "Use methods appropriate for non-stationary data".to_string(),
                            "Consider detrending techniques".to_string(),
                        ],
                    });
                } else if stationary_tests == total_tests {
                    insights.push(Insight {
                        category: InsightCategory::Statistical,
                        title: "Stationary Time Series".to_string(),
                        description: "All stationarity tests confirm the time series is stationary. This is ideal for many time series analysis methods.".to_string(),
                        confidence: 0.9,
                        importance: InsightImportance::Low,
                        evidence: stats.stationarity_tests.iter()
                            .map(|(name, test)| format!("{}: p-value = {:.4}", name, test.p_value))
                            .collect(),
                        recommendations: vec![
                            "Proceed with standard time series analysis methods".to_string(),
                            "Consider ARIMA or similar models".to_string(),
                        ],
                    });
                }
            }
        }

        Ok(insights)
    }

    /// Generate trend insights
    fn generate_trend_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        if let Some(ref trend) = analysis_data.trend {
            let strength = trend.trend_summary.strength;
            let direction = &trend.trend_summary.direction;

            if strength > 0.7 {
                insights.push(Insight {
                    category: InsightCategory::Trend,
                    title: format!("Strong {:?} Trend Detected", direction),
                    description: format!("A strong {:?} trend is present with {:.1}% strength. This indicates a clear directional pattern in the data.",
                        direction, strength * 100.0),
                    confidence: 0.9,
                    importance: InsightImportance::High,
                    evidence: vec![
                        format!("Trend strength: {:.1}%", strength * 100.0),
                        format!("Direction: {:?}", direction),
                    ],
                    recommendations: vec![
                        "Monitor trend continuation for forecasting".to_string(),
                        "Investigate factors driving the trend".to_string(),
                        "Consider trend-aware forecasting models".to_string(),
                    ],
                });
            } else if strength > 0.3 {
                insights.push(Insight {
                    category: InsightCategory::Trend,
                    title: format!("Moderate {:?} Trend Present", direction),
                    description: format!("A moderate {:?} trend is present with {:.1}% strength.",
                        direction, strength * 100.0),
                    confidence: 0.7,
                    importance: InsightImportance::Medium,
                    evidence: vec![
                        format!("Trend strength: {:.1}%", strength * 100.0),
                        format!("Direction: {:?}", direction),
                    ],
                    recommendations: vec![
                        "Monitor for trend strengthening or weakening".to_string(),
                        "Consider external factors that may influence trend".to_string(),
                    ],
                });
            }

            // Growth rate insight
            if let Some(growth_rate) = trend.trend_summary.growth_rate {
                if growth_rate.abs() > 0.1 {
                    let growth_type = if growth_rate > 0.0 { "growth" } else { "decline" };
                    insights.push(Insight {
                        category: InsightCategory::Trend,
                        title: format!("Significant Annual {}", growth_type.to_uppercase()),
                        description: format!("The data shows an annual {} rate of {:.1}%.",
                            growth_type, growth_rate * 100.0),
                        confidence: 0.8,
                        importance: InsightImportance::High,
                        evidence: vec![format!("Annual growth rate: {:.1}%", growth_rate * 100.0)],
                        recommendations: vec![
                            format!("Plan for continued {}", growth_type),
                            "Update forecasting models with growth rate".to_string(),
                            "Monitor for changes in growth trajectory".to_string(),
                        ],
                    });
                }
            }
        }

        Ok(insights)
    }

    /// Generate seasonality insights
    fn generate_seasonality_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        if let Some(ref _seasonality) = analysis_data.seasonality {
            // Add seasonality insights based on the seasonality analysis
            // This would be implemented based on the actual seasonality analysis structure
            insights.push(Insight {
                category: InsightCategory::Seasonality,
                title: "Seasonality Analysis Available".to_string(),
                description: "Seasonality analysis has been performed. Check the detailed results for seasonal patterns.".to_string(),
                confidence: 0.5,
                importance: InsightImportance::Low,
                evidence: vec!["Seasonality analysis completed".to_string()],
                recommendations: vec!["Review seasonal patterns in detailed analysis".to_string()],
            });
        }

        Ok(insights)
    }

    /// Generate anomaly insights
    fn generate_anomaly_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        if let Some(ref anomalies) = analysis_data.anomalies {
            if !anomalies.is_empty() {
                let anomaly_count = anomalies.len();
                let anomaly_percentage = (anomaly_count as f64 / analysis_data.data_summary.n_points as f64) * 100.0;

                let importance = if anomaly_percentage > 10.0 {
                    InsightImportance::Critical
                } else if anomaly_percentage > 5.0 {
                    InsightImportance::High
                } else if anomaly_percentage > 1.0 {
                    InsightImportance::Medium
                } else {
                    InsightImportance::Low
                };

                insights.push(Insight {
                    category: InsightCategory::Anomaly,
                    title: "Anomalies Detected".to_string(),
                    description: format!("{} anomalies detected ({:.1}% of data). These represent unusual patterns that deviate from normal behavior.",
                        anomaly_count, anomaly_percentage),
                    confidence: 0.8,
                    importance,
                    evidence: vec![
                        format!("{} anomalies found", anomaly_count),
                        format!("{:.1}% of total data", anomaly_percentage),
                    ],
                    recommendations: vec![
                        "Investigate the root causes of anomalies".to_string(),
                        "Determine if anomalies represent real events or data errors".to_string(),
                        "Consider implementing real-time anomaly monitoring".to_string(),
                    ],
                });
            }
        }

        Ok(insights)
    }

    /// Generate forecasting insights
    fn generate_forecasting_insights(&self, analysis_data: &AnalysisData) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        if let Some(ref _forecast) = analysis_data.forecasting {
            // Add forecasting insights based on the forecast results
            // This would be implemented based on the actual forecasting result structure
            insights.push(Insight {
                category: InsightCategory::Forecasting,
                title: "Forecast Available".to_string(),
                description: "Forecasting analysis has been completed. Review the forecast results for future projections.".to_string(),
                confidence: 0.6,
                importance: InsightImportance::Medium,
                evidence: vec!["Forecast model executed".to_string()],
                recommendations: vec![
                    "Review forecast accuracy metrics".to_string(),
                    "Consider model improvements if accuracy is low".to_string(),
                    "Use forecast for planning purposes".to_string(),
                ],
            });
        }

        Ok(insights)
    }
}

/// Generate insights from analysis data
pub fn generate_insights(
    analysis_data: &AnalysisData,
    config: &AdvancedReportConfig,
) -> Result<Vec<Insight>> {
    let engine = InsightEngine::new(config.clone());
    engine.generate_insights(analysis_data)
}

/// Generate recommendations from insights
pub fn generate_recommendations(insights: &[Insight]) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Collect all unique recommendations from insights
    for insight in insights {
        for rec in &insight.recommendations {
            if !recommendations.contains(rec) {
                recommendations.push(rec.clone());
            }
        }
    }

    // Prioritize recommendations based on insight importance
    let mut prioritized = Vec::new();

    // Add critical recommendations first
    for insight in insights.iter().filter(|i| i.importance == InsightImportance::Critical) {
        for rec in &insight.recommendations {
            if !prioritized.contains(rec) {
                prioritized.push(rec.clone());
            }
        }
    }

    // Add high importance recommendations
    for insight in insights.iter().filter(|i| i.importance == InsightImportance::High) {
        for rec in &insight.recommendations {
            if !prioritized.contains(rec) {
                prioritized.push(rec.clone());
            }
        }
    }

    // Add remaining recommendations
    for rec in recommendations {
        if !prioritized.contains(&rec) {
            prioritized.push(rec);
        }
    }

    prioritized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::types::{AnalysisData, DataSummary};
    use crate::validation::DataQualityReport;

    fn create_test_analysis_data() -> AnalysisData {
        let mut quality_report = DataQualityReport {
            nan_count: 10,
            infinite_count: 2,
            duplicate_timestamps: 1,
            potential_outliers: 5,
            gaps: Vec::new(),
        };

        AnalysisData {
            statistics: None,
            trend: None,
            seasonality: None,
            anomalies: None,
            forecasting: None,
            correlation: None,
            data_quality: Some(quality_report),
            plots: Vec::new(),
            data_summary: DataSummary {
                n_points: 1000,
                date_range: None,
                frequency: Some("1H".to_string()),
                columns: vec!["value".to_string()],
                missing_percentage: 1.0,
            },
        }
    }

    #[test]
    fn test_insight_engine_creation() {
        let config = AdvancedReportConfig::default();
        let engine = InsightEngine::new(config);
        assert_eq!(engine.config.max_insights, 10);
    }

    #[test]
    fn test_data_quality_insights() {
        let config = AdvancedReportConfig::default();
        let engine = InsightEngine::new(config);
        let analysis_data = create_test_analysis_data();

        let insights = engine.generate_data_quality_insights(&analysis_data).unwrap();

        assert!(!insights.is_empty());
        assert!(insights.iter().any(|i| i.category == InsightCategory::DataQuality));
        assert!(insights.iter().any(|i| i.title.contains("Missing Values")));
    }

    #[test]
    fn test_recommendations_generation() {
        let insights = vec![
            Insight {
                category: InsightCategory::DataQuality,
                title: "Test Insight".to_string(),
                description: "Test".to_string(),
                confidence: 0.8,
                importance: InsightImportance::High,
                evidence: vec![],
                recommendations: vec![
                    "Recommendation 1".to_string(),
                    "Recommendation 2".to_string(),
                ],
            },
        ];

        let recommendations = generate_recommendations(&insights);
        assert_eq!(recommendations.len(), 2);
        assert!(recommendations.contains(&"Recommendation 1".to_string()));
        assert!(recommendations.contains(&"Recommendation 2".to_string()));
    }

    #[test]
    fn test_insight_sorting() {
        let config = AdvancedReportConfig::default();
        let engine = InsightEngine::new(config);
        let analysis_data = create_test_analysis_data();

        let insights = engine.generate_insights(&analysis_data).unwrap();

        // Verify insights are sorted by importance
        for i in 1..insights.len() {
            let prev_importance = match insights[i-1].importance {
                InsightImportance::Critical => 4,
                InsightImportance::High => 3,
                InsightImportance::Medium => 2,
                InsightImportance::Low => 1,
            };
            let curr_importance = match insights[i].importance {
                InsightImportance::Critical => 4,
                InsightImportance::High => 3,
                InsightImportance::Medium => 2,
                InsightImportance::Low => 1,
            };
            assert!(prev_importance >= curr_importance);
        }
    }
}