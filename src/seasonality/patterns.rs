//! # Seasonal Pattern Analysis Module
//!
//! Provides analysis of seasonal patterns including strength measurement,
//! consistency analysis, trend interactions, and calendar effects detection.

use crate::seasonality::{SeasonalPeriod, SeasonalPeriodType};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive seasonal pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPatternAnalysis {
    /// Overall seasonal strength
    pub overall_strength: SeasonalStrength,

    /// Pattern consistency analysis
    pub consistency: Option<PatternConsistency>,

    /// Seasonal-trend interaction analysis
    pub trend_interaction: Option<SeasonalTrendInteraction>,

    /// Calendar effects detection
    pub calendar_effects: Option<CalendarEffectDetection>,

    /// Strength by period type
    pub strength_by_type: HashMap<SeasonalPeriodType, f64>,

    /// Pattern stability over time
    pub stability: PatternStability,

    /// Quality metrics
    pub quality_metrics: PatternQualityMetrics,
}

impl Default for SeasonalPatternAnalysis {
    fn default() -> Self {
        Self {
            overall_strength: SeasonalStrength::default(),
            consistency: None,
            trend_interaction: None,
            calendar_effects: None,
            strength_by_type: HashMap::new(),
            stability: PatternStability::default(),
            quality_metrics: PatternQualityMetrics::default(),
        }
    }
}

/// Seasonal strength measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalStrength {
    /// Overall strength score (0.0 to 1.0)
    pub strength: f64,

    /// Method used for calculation
    pub method: StrengthMethod,

    /// Confidence interval
    pub confidence_interval: (f64, f64),

    /// Statistical significance
    pub p_value: f64,

    /// Component strengths for different periods
    pub component_strengths: Vec<ComponentStrength>,
}

impl Default for SeasonalStrength {
    fn default() -> Self {
        Self {
            strength: 0.0,
            method: StrengthMethod::VarianceRatio,
            confidence_interval: (0.0, 0.0),
            p_value: 1.0,
            component_strengths: Vec::new(),
        }
    }
}

/// Methods for calculating seasonal strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrengthMethod {
    /// Ratio of seasonal variance to total variance
    VarianceRatio,
    /// X-13 seasonal strength measure
    X13Strength,
    /// STL-based seasonal strength
    STLStrength,
    /// Fourier-based strength
    FourierStrength,
    /// Combined multiple methods
    Combined,
}

/// Component strength for individual seasonal periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStrength {
    /// Seasonal period
    pub period: f64,

    /// Strength of this component
    pub strength: f64,

    /// Period type
    pub period_type: SeasonalPeriodType,

    /// Relative importance
    pub relative_importance: f64,
}

/// Pattern consistency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConsistency {
    /// Overall consistency score (0.0 to 1.0)
    pub consistency_score: f64,

    /// Consistency by period
    pub period_consistency: Vec<(f64, f64)>,

    /// Temporal stability of patterns
    pub temporal_stability: f64,

    /// Pattern evolution metrics
    pub evolution_metrics: PatternEvolution,

    /// Outlier periods identified
    pub outlier_periods: Vec<OutlierPeriod>,
}

/// Pattern evolution over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEvolution {
    /// Rate of change in seasonal patterns
    pub change_rate: f64,

    /// Periods with significant changes
    pub change_points: Vec<usize>,

    /// Evolution trend
    pub evolution_trend: EvolutionTrend,
}

/// Trend in pattern evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionTrend {
    /// Patterns are stable over time
    Stable,
    /// Patterns are strengthening
    Strengthening,
    /// Patterns are weakening
    Weakening,
    /// Patterns are shifting/changing
    Shifting,
    /// Irregular evolution
    Irregular,
}

/// Outlier period detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierPeriod {
    /// Time index of outlier
    pub time_index: usize,

    /// Expected seasonal value
    pub expected_value: f64,

    /// Observed value
    pub observed_value: f64,

    /// Deviation magnitude
    pub deviation: f64,

    /// Outlier type
    pub outlier_type: OutlierType,
}

/// Types of outliers in seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierType {
    /// Additive outlier (shift in level)
    Additive,
    /// Innovation outlier (temporary shock)
    Innovation,
    /// Seasonal outlier (break in seasonal pattern)
    Seasonal,
    /// Level shift outlier
    LevelShift,
}

/// Seasonal-trend interaction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalTrendInteraction {
    /// Interaction strength
    pub interaction_strength: f64,

    /// Type of interaction
    pub interaction_type: InteractionType,

    /// Interaction by seasonal period
    pub period_interactions: Vec<(f64, f64)>,

    /// Statistical significance
    pub p_value: f64,

    /// Trend modulation of seasonality
    pub trend_modulation: TrendModulation,
}

/// Types of seasonal-trend interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// No significant interaction
    Independent,
    /// Trend amplifies seasonal patterns
    Amplifying,
    /// Trend dampens seasonal patterns
    Dampening,
    /// Complex interaction pattern
    Complex,
}

/// How trend modulates seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendModulation {
    /// Does trend affect seasonal amplitude?
    pub amplitude_modulation: bool,

    /// Does trend affect seasonal phase?
    pub phase_modulation: bool,

    /// Modulation strength
    pub modulation_strength: f64,
}

/// Calendar effects detection in seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEffectDetection {
    /// Day-of-week effects
    pub day_of_week_effects: Option<DayOfWeekEffects>,

    /// Month-of-year effects
    pub month_of_year_effects: Option<MonthOfYearEffects>,

    /// Holiday proximity effects
    pub holiday_effects: Option<HolidayProximityEffects>,

    /// Overall calendar effect strength
    pub overall_calendar_strength: f64,
}

/// Day of week effects analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayOfWeekEffects {
    /// Effect size for each day (Monday=0, Sunday=6)
    pub daily_effects: [f64; 7],

    /// Statistical significance
    pub significance: f64,

    /// Most significant day
    pub most_significant_day: usize,

    /// Effect pattern type
    pub pattern_type: DayPatternType,
}

/// Pattern types for day-of-week effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DayPatternType {
    /// Weekday vs weekend pattern
    WeekdayWeekend,
    /// Monday effect
    Monday,
    /// Friday effect
    Friday,
    /// Custom pattern
    Custom,
}

/// Month of year effects analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthOfYearEffects {
    /// Effect size for each month (January=0, December=11)
    pub monthly_effects: [f64; 12],

    /// Statistical significance
    pub significance: f64,

    /// Seasonal quarters analysis
    pub quarterly_pattern: Option<QuarterlyPattern>,

    /// Peak and trough months
    pub peak_months: Vec<usize>,
    pub trough_months: Vec<usize>,
}

/// Quarterly pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarterlyPattern {
    /// Quarterly effects (Q1, Q2, Q3, Q4)
    pub quarterly_effects: [f64; 4],

    /// Quarterly pattern strength
    pub strength: f64,
}

/// Holiday proximity effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolidayProximityEffects {
    /// Effect strength before holidays
    pub pre_holiday_effect: f64,

    /// Effect strength after holidays
    pub post_holiday_effect: f64,

    /// Effect duration (days)
    pub effect_duration: usize,

    /// Holiday impact by type
    pub holiday_impacts: HashMap<String, f64>,
}

/// Pattern stability over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStability {
    /// Overall stability score
    pub stability_score: f64,

    /// Stability by period
    pub period_stability: Vec<(f64, f64)>,

    /// Change detection results
    pub change_detection: ChangeDetectionResults,

    /// Structural breaks
    pub structural_breaks: Vec<StructuralBreak>,
}

impl Default for PatternStability {
    fn default() -> Self {
        Self {
            stability_score: 1.0,
            period_stability: Vec::new(),
            change_detection: ChangeDetectionResults::default(),
            structural_breaks: Vec::new(),
        }
    }
}

/// Change detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeDetectionResults {
    /// Number of changes detected
    pub num_changes: usize,

    /// Change points (time indices)
    pub change_points: Vec<usize>,

    /// Change magnitudes
    pub change_magnitudes: Vec<f64>,

    /// Change types
    pub change_types: Vec<ChangeType>,
}

impl Default for ChangeDetectionResults {
    fn default() -> Self {
        Self {
            num_changes: 0,
            change_points: Vec::new(),
            change_magnitudes: Vec::new(),
            change_types: Vec::new(),
        }
    }
}

/// Types of changes in seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Change in seasonal amplitude
    AmplitudeChange,
    /// Change in seasonal phase
    PhaseChange,
    /// Change in seasonal period
    PeriodChange,
    /// New seasonal component appears
    ComponentAppearance,
    /// Seasonal component disappears
    ComponentDisappearance,
}

/// Structural breaks in seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralBreak {
    /// Time index of break
    pub break_point: usize,

    /// Break type
    pub break_type: BreakType,

    /// Break magnitude
    pub magnitude: f64,

    /// Statistical significance
    pub significance: f64,
}

/// Types of structural breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakType {
    /// Break in seasonal level
    LevelBreak,
    /// Break in seasonal trend
    TrendBreak,
    /// Break in seasonal pattern
    SeasonalBreak,
    /// Temporary intervention
    TemporaryBreak,
}

/// Quality metrics for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternQualityMetrics {
    /// R-squared for seasonal fit
    pub r_squared: f64,

    /// Mean absolute error of seasonal fit
    pub mae: f64,

    /// Root mean squared error
    pub rmse: f64,

    /// Information criteria
    pub aic: f64,
    pub bic: f64,

    /// Residual diagnostics
    pub residual_diagnostics: ResidualDiagnostics,
}

impl Default for PatternQualityMetrics {
    fn default() -> Self {
        Self {
            r_squared: 0.0,
            mae: 0.0,
            rmse: 0.0,
            aic: 0.0,
            bic: 0.0,
            residual_diagnostics: ResidualDiagnostics::default(),
        }
    }
}

/// Diagnostic tests for residuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualDiagnostics {
    /// Ljung-Box test for residual autocorrelation
    pub ljung_box_statistic: f64,
    pub ljung_box_p_value: f64,

    /// Normality test (Shapiro-Wilk approximation)
    pub normality_statistic: f64,
    pub normality_p_value: f64,

    /// Heteroscedasticity test
    pub heteroscedasticity_statistic: f64,
    pub heteroscedasticity_p_value: f64,
}

impl Default for ResidualDiagnostics {
    fn default() -> Self {
        Self {
            ljung_box_statistic: 0.0,
            ljung_box_p_value: 1.0,
            normality_statistic: 0.0,
            normality_p_value: 1.0,
            heteroscedasticity_statistic: 0.0,
            heteroscedasticity_p_value: 1.0,
        }
    }
}

/// Analyze seasonal patterns in the data
pub fn analyze_seasonal_patterns(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalPatternAnalysis, Box<dyn std::error::Error>> {
    let mut analysis = SeasonalPatternAnalysis::default();

    // 1. Compute overall seasonal strength
    analysis.overall_strength = compute_seasonal_strength(data, seasonal_periods)?;

    // 2. Analyze pattern consistency
    if !seasonal_periods.is_empty() {
        analysis.consistency = Some(analyze_pattern_consistency(data, seasonal_periods)?);
    }

    // 3. Analyze seasonal-trend interaction
    analysis.trend_interaction = Some(detect_seasonal_trend_interaction(data, seasonal_periods)?);

    // 4. Compute strength by period type
    analysis.strength_by_type = compute_strength_by_period_type(seasonal_periods);

    // 5. Analyze pattern stability
    analysis.stability = analyze_pattern_stability(data, seasonal_periods)?;

    // 6. Compute quality metrics
    analysis.quality_metrics = compute_pattern_quality_metrics(data, seasonal_periods)?;

    Ok(analysis)
}

/// Compute seasonal strength using multiple methods
pub fn compute_seasonal_strength(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalStrength, Box<dyn std::error::Error>> {
    if data.len() < 10 || seasonal_periods.is_empty() {
        return Ok(SeasonalStrength::default());
    }

    // Use variance ratio method as primary
    let strength = compute_variance_ratio_strength(data, seasonal_periods)?;

    // Compute component strengths
    let component_strengths = seasonal_periods.iter()
        .map(|period| ComponentStrength {
            period: period.period,
            strength: period.strength,
            period_type: period.period_type.clone(),
            relative_importance: period.strength / seasonal_periods.iter()
                .map(|p| p.strength)
                .sum::<f64>(),
        })
        .collect();

    // Simple confidence interval (would need bootstrap for more accurate estimate)
    let se = (strength * (1.0 - strength) / data.len() as f64).sqrt();
    let confidence_interval = (
        (strength - 1.96 * se).max(0.0),
        (strength + 1.96 * se).min(1.0)
    );

    // Approximate p-value using t-test assumption
    let t_stat = strength / se.max(0.001);
    let p_value = if t_stat > 2.0 { 0.05 } else { 0.2 };

    Ok(SeasonalStrength {
        strength,
        method: StrengthMethod::VarianceRatio,
        confidence_interval,
        p_value,
        component_strengths,
    })
}

/// Analyze pattern consistency over time
pub fn analyze_pattern_consistency(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<PatternConsistency, Box<dyn std::error::Error>> {
    if data.is_empty() || seasonal_periods.is_empty() {
        return Err("Insufficient data for consistency analysis".into());
    }

    // Analyze consistency for each period
    let mut period_consistency = Vec::new();
    let mut overall_consistency = 0.0;

    for period in seasonal_periods {
        let consistency = compute_period_consistency(data, period.period as usize)?;
        period_consistency.push((period.period, consistency));
        overall_consistency += consistency * period.strength;
    }

    // Normalize by total strength
    let total_strength: f64 = seasonal_periods.iter().map(|p| p.strength).sum();
    if total_strength > 0.0 {
        overall_consistency /= total_strength;
    }

    // Analyze temporal stability
    let temporal_stability = compute_temporal_stability(data, seasonal_periods)?;

    // Detect pattern evolution
    let evolution_metrics = analyze_pattern_evolution(data, seasonal_periods)?;

    // Detect outlier periods
    let outlier_periods = detect_outlier_periods(data, seasonal_periods)?;

    Ok(PatternConsistency {
        consistency_score: overall_consistency,
        period_consistency,
        temporal_stability,
        evolution_metrics,
        outlier_periods,
    })
}

/// Detect seasonal-trend interaction
pub fn detect_seasonal_trend_interaction(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<SeasonalTrendInteraction, Box<dyn std::error::Error>> {
    if data.len() < 20 {
        return Err("Insufficient data for trend interaction analysis".into());
    }

    // Extract trend using simple moving average
    let window = data.len() / 10;
    let mut trend = Vec::new();
    for i in 0..data.len() {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(data.len());
        let avg = data[start..end].iter().sum::<f64>() / (end - start) as f64;
        trend.push(avg);
    }

    // Detrend the data
    let detrended: Vec<f64> = data.iter().zip(trend.iter())
        .map(|(&x, &t)| x - t)
        .collect();

    // Compute seasonal components for detrended data
    let mut period_interactions = Vec::new();
    let mut total_interaction = 0.0;

    for period in seasonal_periods {
        let period_int = period.period as usize;
        if period_int >= data.len() / 2 {
            continue;
        }

        // Compute seasonal component
        let mut seasonal = vec![0.0; data.len()];
        for i in 0..data.len() {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / period.period;
            seasonal[i] = phase.sin() + 0.5 * (2.0 * phase).sin();
        }

        // Compute interaction between trend and seasonal
        let interaction = compute_trend_seasonal_correlation(&trend, &seasonal)?;
        period_interactions.push((period.period, interaction));
        total_interaction += interaction.abs() * period.strength;
    }

    // Normalize interaction strength
    let total_strength: f64 = seasonal_periods.iter().map(|p| p.strength).sum();
    if total_strength > 0.0 {
        total_interaction /= total_strength;
    }

    // Classify interaction type
    let interaction_type = if total_interaction.abs() < 0.1 {
        InteractionType::Independent
    } else if total_interaction > 0.0 {
        InteractionType::Amplifying
    } else if total_interaction < -0.3 {
        InteractionType::Dampening
    } else {
        InteractionType::Complex
    };

    // Analyze trend modulation
    let trend_modulation = analyze_trend_modulation(&trend, &detrended, seasonal_periods)?;

    // Approximate p-value
    let p_value = if total_interaction.abs() > 0.2 { 0.05 } else { 0.3 };

    Ok(SeasonalTrendInteraction {
        interaction_strength: total_interaction.abs(),
        interaction_type,
        period_interactions,
        p_value,
        trend_modulation,
    })
}

// Helper functions

fn compute_variance_ratio_strength(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<f64, Box<dyn std::error::Error>> {
    // Compute total variance
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let total_variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64;

    if total_variance == 0.0 {
        return Ok(0.0);
    }

    // Estimate seasonal variance as sum of component variances
    let mut seasonal_variance = 0.0;
    for period in seasonal_periods {
        // Simple seasonal component estimation
        let period_int = period.period as usize;
        if period_int < data.len() / 2 {
            let mut seasonal_sum = 0.0;
            let mut count = 0;

            for i in 0..data.len() {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / period.period;
                let seasonal_value = period.amplitude * phase.sin();
                seasonal_sum += seasonal_value * seasonal_value;
                count += 1;
            }

            if count > 0 {
                seasonal_variance += seasonal_sum / count as f64;
            }
        }
    }

    Ok((seasonal_variance / total_variance).min(1.0))
}

fn compute_period_consistency(data: &[f64], period: usize) -> Result<f64, Box<dyn std::error::Error>> {
    if period >= data.len() / 2 {
        return Ok(0.0);
    }

    // Split data into cycles and compute consistency
    let num_cycles = data.len() / period;
    if num_cycles < 2 {
        return Ok(0.0);
    }

    let mut cycles = Vec::new();
    for i in 0..num_cycles {
        let start = i * period;
        let end = ((i + 1) * period).min(data.len());
        if end - start == period {
            cycles.push(&data[start..end]);
        }
    }

    if cycles.len() < 2 {
        return Ok(0.0);
    }

    // Compute correlation between cycles
    let mut correlations = Vec::new();
    for i in 0..cycles.len() {
        for j in (i+1)..cycles.len() {
            let corr = compute_correlation(cycles[i], cycles[j])?;
            correlations.push(corr);
        }
    }

    // Return average correlation as consistency measure
    Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
}

fn compute_temporal_stability(
    data: &[f64],
    _seasonal_periods: &[SeasonalPeriod],
) -> Result<f64, Box<dyn std::error::Error>> {
    if data.len() < 20 {
        return Ok(1.0);
    }

    // Simple stability measure using rolling variance
    let window = data.len() / 5;
    let mut variances = Vec::new();

    for i in 0..(data.len() - window) {
        let segment = &data[i..i+window];
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        let variance = segment.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / segment.len() as f64;
        variances.push(variance);
    }

    if variances.is_empty() {
        return Ok(1.0);
    }

    // Stability as inverse of variance of variances
    let mean_var = variances.iter().sum::<f64>() / variances.len() as f64;
    let var_of_vars = variances.iter()
        .map(|&x| (x - mean_var).powi(2))
        .sum::<f64>() / variances.len() as f64;

    // Convert to 0-1 scale where 1 is most stable
    let stability = 1.0 / (1.0 + var_of_vars / mean_var.max(0.001));
    Ok(stability)
}

fn analyze_pattern_evolution(
    data: &[f64],
    _seasonal_periods: &[SeasonalPeriod],
) -> Result<PatternEvolution, Box<dyn std::error::Error>> {
    // Simple change detection using moving averages
    let window = data.len() / 10;
    let mut change_points = Vec::new();
    let mut change_magnitudes = Vec::new();

    if window < 5 {
        return Ok(PatternEvolution {
            change_rate: 0.0,
            change_points,
            evolution_trend: EvolutionTrend::Stable,
        });
    }

    let mut prev_mean = data[0..window].iter().sum::<f64>() / window as f64;

    for i in window..data.len()-window {
        let curr_mean = data[i-window/2..i+window/2].iter().sum::<f64>() / window as f64;
        let change = (curr_mean - prev_mean).abs();

        // Simple threshold for change detection
        let threshold = data.iter().map(|&x| x.abs()).sum::<f64>() / data.len() as f64 * 0.1;

        if change > threshold {
            change_points.push(i);
            change_magnitudes.push(change);
        }

        prev_mean = curr_mean;
    }

    let change_rate = change_points.len() as f64 / data.len() as f64;

    // Determine evolution trend (simplified)
    let evolution_trend = if change_rate < 0.01 {
        EvolutionTrend::Stable
    } else if change_rate > 0.05 {
        EvolutionTrend::Irregular
    } else {
        EvolutionTrend::Shifting
    };

    Ok(PatternEvolution {
        change_rate,
        change_points,
        evolution_trend,
    })
}

fn detect_outlier_periods(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<Vec<OutlierPeriod>, Box<dyn std::error::Error>> {
    let mut outliers = Vec::new();

    if data.is_empty() || seasonal_periods.is_empty() {
        return Ok(outliers);
    }

    // Use largest seasonal period for analysis
    let main_period = seasonal_periods.iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
        .unwrap();

    let period = main_period.period as usize;
    if period >= data.len() / 2 {
        return Ok(outliers);
    }

    // Compute expected seasonal values using simple approach
    let mut seasonal_means = vec![0.0; period];
    let mut counts = vec![0; period];

    for (i, &value) in data.iter().enumerate() {
        let seasonal_idx = i % period;
        seasonal_means[seasonal_idx] += value;
        counts[seasonal_idx] += 1;
    }

    // Compute averages
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_means[i] /= counts[i] as f64;
        }
    }

    // Identify outliers
    let overall_std = {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    };

    let threshold = 2.0 * overall_std; // 2-sigma threshold

    for (i, &value) in data.iter().enumerate() {
        let seasonal_idx = i % period;
        let expected = seasonal_means[seasonal_idx];
        let deviation = (value - expected).abs();

        if deviation > threshold {
            outliers.push(OutlierPeriod {
                time_index: i,
                expected_value: expected,
                observed_value: value,
                deviation,
                outlier_type: if deviation > 3.0 * threshold {
                    OutlierType::Innovation
                } else {
                    OutlierType::Additive
                },
            });
        }
    }

    Ok(outliers)
}

fn compute_strength_by_period_type(seasonal_periods: &[SeasonalPeriod]) -> HashMap<SeasonalPeriodType, f64> {
    let mut strength_by_type = HashMap::new();

    for period in seasonal_periods {
        let entry = strength_by_type.entry(period.period_type.clone()).or_insert(0.0);
        *entry = f64::max(*entry, period.strength);
    }

    strength_by_type
}

fn analyze_pattern_stability(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<PatternStability, Box<dyn std::error::Error>> {
    let stability_score = compute_temporal_stability(data, seasonal_periods)?;

    let mut period_stability = Vec::new();
    for period in seasonal_periods {
        let period_stab = compute_period_stability(data, period.period as usize)?;
        period_stability.push((period.period, period_stab));
    }

    // Simple change detection
    let change_detection = detect_pattern_changes(data)?;

    // Placeholder for structural breaks (would need more sophisticated analysis)
    let structural_breaks = Vec::new();

    Ok(PatternStability {
        stability_score,
        period_stability,
        change_detection,
        structural_breaks,
    })
}

fn compute_period_stability(data: &[f64], period: usize) -> Result<f64, Box<dyn std::error::Error>> {
    if period >= data.len() / 3 {
        return Ok(1.0);
    }

    // Compute stability as consistency of seasonal pattern over time
    compute_period_consistency(data, period)
}

fn detect_pattern_changes(data: &[f64]) -> Result<ChangeDetectionResults, Box<dyn std::error::Error>> {
    // Simple change detection using cumulative sum approach
    let mut change_points = Vec::new();
    let mut change_magnitudes = Vec::new();
    let mut change_types = Vec::new();

    if data.len() < 20 {
        return Ok(ChangeDetectionResults {
            num_changes: 0,
            change_points,
            change_magnitudes,
            change_types,
        });
    }

    let window = data.len() / 10;
    let mut cumsum = 0.0;
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    for (i, &value) in data.iter().enumerate() {
        cumsum += value - mean;

        // Simple threshold-based detection
        if i > window && cumsum.abs() > 3.0 * (data.len() as f64).sqrt() {
            change_points.push(i);
            change_magnitudes.push(cumsum.abs());
            change_types.push(ChangeType::AmplitudeChange);
            cumsum = 0.0; // Reset after detection
        }
    }

    Ok(ChangeDetectionResults {
        num_changes: change_points.len(),
        change_points,
        change_magnitudes,
        change_types,
    })
}

fn compute_pattern_quality_metrics(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<PatternQualityMetrics, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Ok(PatternQualityMetrics::default());
    }

    // Estimate seasonal fit
    let (fitted, residuals) = fit_seasonal_model(data, seasonal_periods)?;

    // Compute R-squared
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;
    let ss_tot = data.iter().map(|&x| (x - data_mean).powi(2)).sum::<f64>();
    let ss_res = residuals.iter().map(|&x| x.powi(2)).sum::<f64>();
    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    // Compute error metrics
    let mae = residuals.iter().map(|&x| x.abs()).sum::<f64>() / residuals.len() as f64;
    let rmse = (residuals.iter().map(|&x| x.powi(2)).sum::<f64>() / residuals.len() as f64).sqrt();

    // Information criteria (simplified)
    let n = data.len() as f64;
    let k = seasonal_periods.len() as f64 * 2.0; // Parameters per period (amplitude, phase)
    let log_likelihood = -0.5 * n * (2.0 * std::f64::consts::PI).ln() - 0.5 * n * (ss_res / n).ln() - 0.5 * n;
    let aic = -2.0 * log_likelihood + 2.0 * k;
    let bic = -2.0 * log_likelihood + k * n.ln();

    // Residual diagnostics
    let residual_diagnostics = compute_residual_diagnostics(&residuals)?;

    Ok(PatternQualityMetrics {
        r_squared,
        mae,
        rmse,
        aic,
        bic,
        residual_diagnostics,
    })
}

// Additional helper functions

fn compute_trend_seasonal_correlation(
    trend: &[f64],
    seasonal: &[f64],
) -> Result<f64, Box<dyn std::error::Error>> {
    compute_correlation(trend, seasonal)
}

fn compute_correlation(x: &[f64], y: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if x.len() != y.len() || x.is_empty() {
        return Err("Invalid data for correlation".into());
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let diff_x = xi - mean_x;
        let diff_y = yi - mean_y;

        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator > 0.0 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

fn analyze_trend_modulation(
    trend: &[f64],
    detrended: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<TrendModulation, Box<dyn std::error::Error>> {
    // Simple analysis of whether trend affects seasonal amplitude or phase
    let mut amplitude_modulation = false;
    let mut phase_modulation = false;
    let mut modulation_strength = 0.0;

    if !seasonal_periods.is_empty() && trend.len() == detrended.len() {
        // Check amplitude modulation by correlating trend with seasonal amplitude
        let window = trend.len() / 10;
        let mut amplitude_correlations = Vec::new();

        for i in window..(trend.len() - window) {
            let trend_segment = &trend[i-window/2..i+window/2];
            let seasonal_segment = &detrended[i-window/2..i+window/2];

            let trend_level = trend_segment.iter().sum::<f64>() / trend_segment.len() as f64;
            let seasonal_amplitude = seasonal_segment.iter().map(|&x| x.abs()).sum::<f64>() / seasonal_segment.len() as f64;

            amplitude_correlations.push((trend_level, seasonal_amplitude));
        }

        if !amplitude_correlations.is_empty() {
            let trend_levels: Vec<f64> = amplitude_correlations.iter().map(|(t, _)| *t).collect();
            let amplitudes: Vec<f64> = amplitude_correlations.iter().map(|(_, a)| *a).collect();

            let correlation = compute_correlation(&trend_levels, &amplitudes)?;
            if correlation.abs() > 0.3 {
                amplitude_modulation = true;
                modulation_strength = correlation.abs();
            }
        }
    }

    Ok(TrendModulation {
        amplitude_modulation,
        phase_modulation,
        modulation_strength,
    })
}

fn fit_seasonal_model(
    data: &[f64],
    seasonal_periods: &[SeasonalPeriod],
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let n = data.len();
    let mut fitted = vec![0.0; n];

    // Simple additive seasonal model
    for period in seasonal_periods {
        let period_len = period.period as usize;
        if period_len < n / 2 {
            for i in 0..n {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / period.period;
                fitted[i] += period.amplitude * period.strength * phase.sin();
            }
        }
    }

    // Adjust for overall level
    let data_mean = data.iter().sum::<f64>() / n as f64;
    let fitted_mean = fitted.iter().sum::<f64>() / n as f64;
    let adjustment = data_mean - fitted_mean;

    for value in fitted.iter_mut() {
        *value += adjustment;
    }

    // Compute residuals
    let residuals: Vec<f64> = data.iter().zip(fitted.iter())
        .map(|(&obs, &fit)| obs - fit)
        .collect();

    Ok((fitted, residuals))
}

fn compute_residual_diagnostics(residuals: &[f64]) -> Result<ResidualDiagnostics, Box<dyn std::error::Error>> {
    let n = residuals.len();
    if n < 5 {
        return Ok(ResidualDiagnostics::default());
    }

    // Ljung-Box test (simplified)
    let max_lag = (n / 4).min(10);
    let mut ljung_box_stat = 0.0;

    // Compute autocorrelations
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let variance = residuals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    for lag in 1..=max_lag {
        let mut autocorr = 0.0;
        for i in 0..(n - lag) {
            autocorr += (residuals[i] - mean) * (residuals[i + lag] - mean);
        }
        autocorr = autocorr / ((n - lag) as f64 * variance);

        ljung_box_stat += autocorr * autocorr / (n - lag) as f64;
    }
    ljung_box_stat *= n as f64 * (n + 2) as f64;

    let ljung_box_p_value = if ljung_box_stat > 15.0 { 0.05 } else { 0.2 };

    // Normality test (simplified Shapiro-Wilk approximation)
    let normality_stat = compute_normality_statistic(residuals)?;
    let normality_p_value = if normality_stat < 0.9 { 0.05 } else { 0.2 };

    // Heteroscedasticity test (simplified)
    let heteroscedasticity_stat = compute_heteroscedasticity_statistic(residuals)?;
    let heteroscedasticity_p_value = if heteroscedasticity_stat > 10.0 { 0.05 } else { 0.2 };

    Ok(ResidualDiagnostics {
        ljung_box_statistic: ljung_box_stat,
        ljung_box_p_value,
        normality_statistic: normality_stat,
        normality_p_value,
        heteroscedasticity_statistic: heteroscedasticity_stat,
        heteroscedasticity_p_value,
    })
}

fn compute_normality_statistic(data: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if data.len() < 3 {
        return Ok(1.0);
    }

    // Simple normality measure based on skewness and kurtosis
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Ok(1.0);
    }

    // Compute skewness and kurtosis
    let mut skewness = 0.0;
    let mut kurtosis = 0.0;

    for &x in data {
        let z = (x - mean) / std_dev;
        skewness += z.powi(3);
        kurtosis += z.powi(4);
    }

    skewness /= n;
    kurtosis = kurtosis / n - 3.0; // Excess kurtosis

    // Approximate normality statistic
    let normality_stat = 1.0 / (1.0 + skewness.abs() + kurtosis.abs());
    Ok(normality_stat)
}

fn compute_heteroscedasticity_statistic(residuals: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    let n = residuals.len();
    if n < 10 {
        return Ok(0.0);
    }

    // Simple Breusch-Pagan type test using squared residuals
    let squared_residuals: Vec<f64> = residuals.iter().map(|&x| x * x).collect();
    let mean_sq_resid = squared_residuals.iter().sum::<f64>() / n as f64;

    // Test against time trend (simple approach)
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_x_sq = 0.0;

    for (i, &sq_resid) in squared_residuals.iter().enumerate() {
        let x = i as f64;
        sum_xy += x * sq_resid;
        sum_x += x;
        sum_x_sq += x * x;
    }

    let slope = (n as f64 * sum_xy - sum_x * squared_residuals.iter().sum::<f64>()) /
                (n as f64 * sum_x_sq - sum_x * sum_x);

    // Simple test statistic
    let test_stat = (slope / mean_sq_resid).abs() * n as f64;
    Ok(test_stat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seasonality::SeasonalPeriodType;

    #[test]
    fn test_seasonal_strength_computation() {
        let data: Vec<f64> = (0..48).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        let periods = vec![
            SeasonalPeriod {
                period: 12.0,
                strength: 0.8,
                phase: 0.0,
                amplitude: 1.0,
                confidence: 0.95,
                period_type: SeasonalPeriodType::Monthly,
            }
        ];

        let strength = compute_seasonal_strength(&data, &periods).unwrap();
        assert!(strength.strength > 0.0);
        assert!(strength.strength <= 1.0);
    }

    #[test]
    fn test_pattern_consistency() {
        let data: Vec<f64> = (0..60).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() +
            0.1 * rand::random::<f64>()
        }).collect();

        let periods = vec![
            SeasonalPeriod {
                period: 12.0,
                strength: 0.7,
                phase: 0.0,
                amplitude: 1.0,
                confidence: 0.9,
                period_type: SeasonalPeriodType::Monthly,
            }
        ];

        let consistency = analyze_pattern_consistency(&data, &periods).unwrap();
        assert!(consistency.consistency_score >= 0.0);
        assert!(consistency.consistency_score <= 1.0);
    }

    #[test]
    fn test_correlation_computation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = compute_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    fn test_outlier_detection() {
        let mut data: Vec<f64> = (0..48).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        // Add outlier
        data[24] = 10.0;

        let periods = vec![
            SeasonalPeriod {
                period: 12.0,
                strength: 0.8,
                phase: 0.0,
                amplitude: 1.0,
                confidence: 0.95,
                period_type: SeasonalPeriodType::Monthly,
            }
        ];

        let outliers = detect_outlier_periods(&data, &periods).unwrap();
        assert!(!outliers.is_empty());
    }
}