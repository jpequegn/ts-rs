//! # Seasonality Detection and Analysis Module
//!
//! Provides comprehensive seasonality analysis including detection, pattern analysis,
//! seasonal adjustment, calendar effects, and advanced seasonality features for time series data.

pub mod detection;
pub mod patterns;
pub mod adjustment;
pub mod calendar;
pub mod advanced;

// Re-export commonly used types and functions
pub use detection::{
    SeasonalityDetectionResult, SeasonalityMethod, FourierAnalysis, PeriodogramAnalysis,
    AutocorrelationAnalysis, MultipleSeasonalityResult, detect_seasonality,
    detect_multiple_seasonalities, analyze_fourier_spectrum, compute_periodogram
};
pub use patterns::{
    SeasonalPatternAnalysis, SeasonalStrength, PatternConsistency, SeasonalTrendInteraction,
    CalendarEffectDetection, analyze_seasonal_patterns, compute_seasonal_strength,
    analyze_pattern_consistency, detect_seasonal_trend_interaction
};
pub use adjustment::{
    SeasonalAdjustmentMethod, SeasonalAdjustmentResult, X13ArimaAdjustment,
    MovingAverageAdjustment, STLAdjustment, perform_seasonal_adjustment,
    apply_x13_adjustment, apply_moving_average_adjustment, apply_stl_adjustment
};
pub use calendar::{
    CalendarEffects, HolidayImpact, TradingDayEffects, LeapYearAdjustment,
    detect_holiday_impacts, analyze_trading_day_effects, apply_leap_year_adjustment,
    detect_calendar_effects
};
pub use advanced::{
    AdvancedSeasonalityAnalysis, EvolvingSeasonality, SeasonalBreaks,
    MultipleSeasonalPeriods, detect_evolving_seasonality, find_seasonal_breaks,
    analyze_multiple_seasonal_periods, comprehensive_seasonality_analysis
};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Comprehensive seasonality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveSeasonalityAnalysis {
    /// Seasonality detection results
    pub detection: SeasonalityDetectionResult,

    /// Seasonal pattern analysis
    pub patterns: SeasonalPatternAnalysis,

    /// Seasonal adjustment results (if performed)
    pub adjustment: Option<SeasonalAdjustmentResult>,

    /// Calendar effects analysis
    pub calendar_effects: CalendarEffects,

    /// Advanced seasonality features
    pub advanced: AdvancedSeasonalityAnalysis,

    /// Analysis metadata
    pub metadata: SeasonalityAnalysisMetadata,
}

/// Detected seasonal period information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPeriod {
    /// Period length (in time units)
    pub period: f64,

    /// Seasonal strength (0.0 to 1.0)
    pub strength: f64,

    /// Phase shift
    pub phase: f64,

    /// Amplitude of seasonal component
    pub amplitude: f64,

    /// Confidence level of detection (0.0 to 1.0)
    pub confidence: f64,

    /// Period type classification
    pub period_type: SeasonalPeriodType,
}

/// Classification of seasonal period types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SeasonalPeriodType {
    /// Daily seasonality (24 hours, intraday patterns)
    Daily,
    /// Weekly seasonality (7 days)
    Weekly,
    /// Monthly seasonality (approximately 30 days)
    Monthly,
    /// Quarterly seasonality (approximately 90 days)
    Quarterly,
    /// Yearly seasonality (365 days)
    Yearly,
    /// Custom seasonal period
    Custom(f64),
}

impl Eq for SeasonalPeriodType {}

impl std::hash::Hash for SeasonalPeriodType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            SeasonalPeriodType::Daily => 0.hash(state),
            SeasonalPeriodType::Weekly => 1.hash(state),
            SeasonalPeriodType::Monthly => 2.hash(state),
            SeasonalPeriodType::Quarterly => 3.hash(state),
            SeasonalPeriodType::Yearly => 4.hash(state),
            // For Custom, we hash the integer bits to make f64 hashable
            SeasonalPeriodType::Custom(f) => (5, f.to_bits()).hash(state),
        }
    }
}

/// Configuration for seasonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysisConfig {
    /// Maximum period to consider for detection
    pub max_period: usize,

    /// Minimum period to consider for detection
    pub min_period: usize,

    /// Significance level for statistical tests
    pub alpha: f64,

    /// Methods to use for seasonality detection
    pub detection_methods: Vec<SeasonalityMethod>,

    /// Whether to perform seasonal adjustment
    pub perform_adjustment: bool,

    /// Seasonal adjustment method to use
    pub adjustment_method: SeasonalAdjustmentMethod,

    /// Whether to analyze calendar effects
    pub analyze_calendar_effects: bool,

    /// Whether to detect multiple seasonalities
    pub detect_multiple: bool,

    /// Whether to analyze evolving seasonality
    pub analyze_evolving: bool,

    /// Window size for rolling analysis
    pub rolling_window: Option<usize>,

    /// Whether to generate plot data
    pub generate_plot_data: bool,
}

impl Default for SeasonalityAnalysisConfig {
    fn default() -> Self {
        Self {
            max_period: 365,
            min_period: 2,
            alpha: 0.05,
            detection_methods: vec![
                SeasonalityMethod::Autocorrelation,
                SeasonalityMethod::Fourier,
                SeasonalityMethod::Periodogram,
            ],
            perform_adjustment: false,
            adjustment_method: SeasonalAdjustmentMethod::STL,
            analyze_calendar_effects: true,
            detect_multiple: true,
            analyze_evolving: false,
            rolling_window: None,
            generate_plot_data: false,
        }
    }
}

/// Metadata for seasonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysisMetadata {
    /// Number of data points analyzed
    pub n_observations: usize,

    /// Time span of the data
    pub time_span: Option<chrono::Duration>,

    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,

    /// Methods used in the analysis
    pub methods_used: Vec<String>,

    /// Analysis duration in milliseconds
    pub duration_ms: u64,

    /// Data quality indicators
    pub data_quality: DataQualityIndicators,

    /// Detected frequency characteristics
    pub frequency_info: FrequencyInfo,
}

/// Data quality indicators for seasonality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityIndicators {
    /// Percentage of missing values
    pub missing_percentage: f64,

    /// Number of gaps in the time series
    pub gap_count: usize,

    /// Regularity of time intervals
    pub regularity_score: f64,

    /// Presence of outliers that might affect seasonality
    pub outlier_count: usize,

    /// Data coverage across different periods
    pub period_coverage: HashMap<String, f64>,
}

/// Information about the frequency characteristics of the data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyInfo {
    /// Detected sampling frequency
    pub sampling_frequency: Option<f64>,

    /// Time between observations (in seconds)
    pub sampling_interval: Option<f64>,

    /// Whether the series has regular intervals
    pub is_regular: bool,

    /// Nyquist frequency (for Fourier analysis)
    pub nyquist_frequency: Option<f64>,
}

impl ComprehensiveSeasonalityAnalysis {
    /// Create a new empty seasonality analysis result
    pub fn new(n_observations: usize) -> Self {
        Self {
            detection: SeasonalityDetectionResult::default(),
            patterns: SeasonalPatternAnalysis::default(),
            adjustment: None,
            calendar_effects: CalendarEffects::default(),
            advanced: AdvancedSeasonalityAnalysis::default(),
            metadata: SeasonalityAnalysisMetadata {
                n_observations,
                time_span: None,
                timestamp: Utc::now(),
                methods_used: Vec::new(),
                duration_ms: 0,
                data_quality: DataQualityIndicators {
                    missing_percentage: 0.0,
                    gap_count: 0,
                    regularity_score: 1.0,
                    outlier_count: 0,
                    period_coverage: HashMap::new(),
                },
                frequency_info: FrequencyInfo {
                    sampling_frequency: None,
                    sampling_interval: None,
                    is_regular: true,
                    nyquist_frequency: None,
                },
            },
        }
    }

    /// Generate a comprehensive summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("Seasonality Analysis Results\n");
        report.push_str("============================\n\n");

        // Detected seasonal periods
        if !self.detection.seasonal_periods.is_empty() {
            report.push_str("🔍 Detected Seasonal Periods:\n");
            for (i, period) in self.detection.seasonal_periods.iter().enumerate().take(5) {
                let custom_desc;
                let period_type_desc = match &period.period_type {
                    SeasonalPeriodType::Daily => "Daily",
                    SeasonalPeriodType::Weekly => "Weekly",
                    SeasonalPeriodType::Monthly => "Monthly",
                    SeasonalPeriodType::Quarterly => "Quarterly",
                    SeasonalPeriodType::Yearly => "Yearly",
                    SeasonalPeriodType::Custom(p) => {
                        custom_desc = format!("Custom ({})", p);
                        &custom_desc
                    },
                };

                report.push_str(&format!(
                    "  {}. {} - Period: {:.1}, Strength: {:.2} ({:.0}% confidence)\n",
                    i + 1,
                    period_type_desc,
                    period.period,
                    period.strength,
                    period.confidence * 100.0
                ));
            }

            if self.detection.seasonal_periods.len() > 5 {
                report.push_str(&format!(
                    "  ... and {} more periods detected\n",
                    self.detection.seasonal_periods.len() - 5
                ));
            }
        } else {
            report.push_str("🔍 No significant seasonal periods detected\n");
        }

        // Overall seasonality strength
        report.push_str(&format!(
            "\n📊 Overall Seasonality Strength: {:.2} ({})\n",
            self.patterns.overall_strength.strength,
            strength_description(self.patterns.overall_strength.strength)
        ));

        // Pattern consistency
        if let Some(ref consistency) = self.patterns.consistency {
            report.push_str(&format!(
                "🔄 Pattern Consistency: {:.2} ({})\n",
                consistency.consistency_score,
                consistency_description(consistency.consistency_score)
            ));
        }

        // Calendar effects
        if self.calendar_effects.has_calendar_effects {
            report.push_str("\n📅 Calendar Effects Detected:\n");
            if self.calendar_effects.trading_day_effects.is_some() {
                report.push_str("  • Trading day effects present\n");
            }
            if !self.calendar_effects.holiday_impacts.is_empty() {
                report.push_str(&format!("  • {} holiday impacts detected\n",
                    self.calendar_effects.holiday_impacts.len()));
            }
        }

        // Data quality summary
        report.push_str(&format!(
            "\n📈 Data Quality:\n  • {} observations\n  • {:.1}% missing values\n  • {} gaps detected\n",
            self.metadata.n_observations,
            self.metadata.data_quality.missing_percentage,
            self.metadata.data_quality.gap_count
        ));

        // Analysis performance
        report.push_str(&format!(
            "\n⚡ Analysis completed in {} ms using {} methods\n",
            self.metadata.duration_ms,
            self.metadata.methods_used.len()
        ));

        report
    }
}

/// Perform comprehensive seasonality analysis on time series data
pub fn analyze_comprehensive_seasonality(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    config: Option<SeasonalityAnalysisConfig>,
) -> Result<ComprehensiveSeasonalityAnalysis, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let config = config.unwrap_or_default();

    // Filter out non-finite values
    let valid_data: Vec<(DateTime<Utc>, f64)> = timestamps
        .iter()
        .zip(values.iter())
        .filter(|(_, &v)| v.is_finite())
        .map(|(&t, &v)| (t, v))
        .collect();

    if valid_data.len() < config.min_period * 2 {
        return Err("Insufficient data for seasonality analysis".into());
    }

    let valid_timestamps: Vec<DateTime<Utc>> = valid_data.iter().map(|(t, _)| *t).collect();
    let valid_values: Vec<f64> = valid_data.iter().map(|(_, v)| *v).collect();

    let mut result = ComprehensiveSeasonalityAnalysis::new(valid_values.len());

    // Update metadata
    result.metadata.time_span = if valid_timestamps.len() > 1 {
        Some(valid_timestamps[valid_timestamps.len() - 1] - valid_timestamps[0])
    } else {
        None
    };

    result.metadata.data_quality.missing_percentage =
        (values.len() - valid_values.len()) as f64 / values.len() as f64 * 100.0;

    // Analyze frequency characteristics
    if valid_timestamps.len() > 1 {
        let intervals: Vec<i64> = valid_timestamps.windows(2)
            .map(|w| (w[1] - w[0]).num_seconds())
            .collect();

        if !intervals.is_empty() {
            let avg_interval = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
            result.metadata.frequency_info.sampling_interval = Some(avg_interval);
            result.metadata.frequency_info.sampling_frequency = Some(1.0 / avg_interval);
            result.metadata.frequency_info.nyquist_frequency = Some(1.0 / (2.0 * avg_interval));

            // Check regularity
            let variance = intervals.iter()
                .map(|&x| (x as f64 - avg_interval).powi(2))
                .sum::<f64>() / intervals.len() as f64;
            result.metadata.frequency_info.is_regular = variance < avg_interval * avg_interval * 0.01;
        }
    }

    // 1. Seasonality Detection
    result.detection = detect_seasonality(&valid_values, &config)?;
    result.metadata.methods_used.extend(config.detection_methods.iter().map(|m| format!("{:?}", m)));

    // 2. Seasonal Pattern Analysis
    result.patterns = analyze_seasonal_patterns(&valid_values, &result.detection.seasonal_periods)?;
    result.metadata.methods_used.push("pattern_analysis".to_string());

    // 3. Seasonal Adjustment (if requested)
    if config.perform_adjustment {
        result.adjustment = Some(perform_seasonal_adjustment(
            &valid_values,
            config.adjustment_method.clone(),
            &result.detection.seasonal_periods
        )?);
        result.metadata.methods_used.push(format!("{:?}", config.adjustment_method));
    }

    // 4. Calendar Effects Analysis (if requested)
    if config.analyze_calendar_effects {
        result.calendar_effects = detect_calendar_effects(&valid_timestamps, &valid_values)?;
        result.metadata.methods_used.push("calendar_effects".to_string());
    }

    // 5. Advanced Seasonality Analysis
    if config.analyze_evolving || config.detect_multiple {
        result.advanced = comprehensive_seasonality_analysis(
            &valid_timestamps,
            &valid_values,
            &config
        )?;
        result.metadata.methods_used.push("advanced_analysis".to_string());
    }

    result.metadata.duration_ms = start_time.elapsed().as_millis() as u64;

    Ok(result)
}

// Helper functions
fn strength_description(strength: f64) -> &'static str {
    if strength >= 0.8 {
        "Very Strong"
    } else if strength >= 0.6 {
        "Strong"
    } else if strength >= 0.4 {
        "Moderate"
    } else if strength >= 0.2 {
        "Weak"
    } else {
        "Very Weak"
    }
}

fn consistency_description(consistency: f64) -> &'static str {
    if consistency >= 0.9 {
        "Very Consistent"
    } else if consistency >= 0.75 {
        "Consistent"
    } else if consistency >= 0.5 {
        "Moderately Consistent"
    } else if consistency >= 0.25 {
        "Inconsistent"
    } else {
        "Very Inconsistent"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seasonality_analysis_creation() {
        let analysis = ComprehensiveSeasonalityAnalysis::new(100);
        assert_eq!(analysis.metadata.n_observations, 100);
        assert!(analysis.detection.seasonal_periods.is_empty());
    }

    #[test]
    fn test_seasonal_period_classification() {
        let period = SeasonalPeriod {
            period: 7.0,
            strength: 0.8,
            phase: 0.0,
            amplitude: 1.0,
            confidence: 0.95,
            period_type: SeasonalPeriodType::Weekly,
        };

        assert_eq!(period.period_type, SeasonalPeriodType::Weekly);
        assert_eq!(period.period, 7.0);
    }

    #[test]
    fn test_default_config() {
        let config = SeasonalityAnalysisConfig::default();
        assert_eq!(config.max_period, 365);
        assert_eq!(config.min_period, 2);
        assert_eq!(config.alpha, 0.05);
        assert!(config.detect_multiple);
    }

    #[test]
    fn test_strength_description() {
        assert_eq!(strength_description(0.9), "Very Strong");
        assert_eq!(strength_description(0.7), "Strong");
        assert_eq!(strength_description(0.5), "Moderate");
        assert_eq!(strength_description(0.3), "Weak");
        assert_eq!(strength_description(0.1), "Very Weak");
    }
}