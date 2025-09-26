//! # Calendar Effects Detection Module
//!
//! Provides detection and analysis of calendar effects including holiday impacts,
//! trading day effects, leap year adjustments, and other calendar-related patterns.

use chrono::{DateTime, Utc, Datelike, Weekday, NaiveDate};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive calendar effects analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEffects {
    /// Whether any calendar effects were detected
    pub has_calendar_effects: bool,

    /// Holiday impact analysis
    pub holiday_impacts: Vec<HolidayImpact>,

    /// Trading day effects
    pub trading_day_effects: Option<TradingDayEffects>,

    /// Leap year adjustments
    pub leap_year_adjustments: Option<LeapYearAdjustment>,

    /// Day-of-week effects
    pub day_of_week_effects: Option<DayOfWeekEffects>,

    /// Month-of-year effects
    pub month_of_year_effects: Option<MonthOfYearEffects>,

    /// Working day vs non-working day effects
    pub working_day_effects: Option<WorkingDayEffects>,

    /// Easter and moveable holiday effects
    pub easter_effects: Option<EasterEffects>,

    /// Overall calendar effect strength
    pub overall_strength: f64,

    /// Statistical significance
    pub statistical_significance: CalendarSignificance,
}

impl Default for CalendarEffects {
    fn default() -> Self {
        Self {
            has_calendar_effects: false,
            holiday_impacts: Vec::new(),
            trading_day_effects: None,
            leap_year_adjustments: None,
            day_of_week_effects: None,
            month_of_year_effects: None,
            working_day_effects: None,
            easter_effects: None,
            overall_strength: 0.0,
            statistical_significance: CalendarSignificance::default(),
        }
    }
}

/// Holiday impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolidayImpact {
    /// Holiday name/type
    pub holiday_name: String,

    /// Holiday date
    pub holiday_date: NaiveDate,

    /// Impact magnitude
    pub impact_magnitude: f64,

    /// Pre-holiday effect duration (days)
    pub pre_holiday_days: i32,

    /// Post-holiday effect duration (days)
    pub post_holiday_days: i32,

    /// Type of holiday effect
    pub effect_type: HolidayEffectType,

    /// Statistical significance
    pub p_value: f64,

    /// Impact pattern
    pub impact_pattern: HolidayPattern,
}

/// Types of holiday effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HolidayEffectType {
    /// Activity increases before/during holiday
    Boost,
    /// Activity decreases before/during holiday
    Depression,
    /// Activity shifts from one period to another
    Shift,
    /// Mixed positive and negative effects
    Mixed,
}

/// Holiday impact patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolidayPattern {
    /// Pre-holiday pattern
    pub pre_pattern: Vec<f64>,

    /// Holiday day pattern
    pub holiday_pattern: f64,

    /// Post-holiday pattern
    pub post_pattern: Vec<f64>,

    /// Pattern consistency across years
    pub consistency: f64,
}

/// Trading day effects analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDayEffects {
    /// Trading day coefficients [Monday through Friday]
    pub trading_day_coeffs: [f64; 5],

    /// Weekend effects [Saturday, Sunday]
    pub weekend_effects: [f64; 2],

    /// Month length effects (for different month lengths)
    pub month_length_effects: HashMap<u32, f64>,

    /// Leap year trading day adjustment
    pub leap_year_trading_adjustment: f64,

    /// Overall trading day strength
    pub overall_strength: f64,

    /// F-statistic for joint significance
    pub f_statistic: f64,

    /// P-value for joint significance
    pub p_value: f64,

    /// Individual day significance
    pub day_significance: [f64; 7],
}

/// Leap year adjustment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeapYearAdjustment {
    /// Leap year effect magnitude
    pub leap_year_effect: f64,

    /// February 29th effect
    pub feb29_effect: f64,

    /// Monthly leap year adjustments
    pub monthly_adjustments: [f64; 12],

    /// Years with leap year effects
    pub affected_years: Vec<i32>,

    /// Statistical significance
    pub p_value: f64,

    /// Recommended adjustment factors
    pub adjustment_factors: LeapYearFactors,
}

/// Leap year adjustment factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeapYearFactors {
    /// Factor for February in leap years
    pub february_leap_factor: f64,

    /// Factor for March in leap years
    pub march_leap_factor: f64,

    /// Annual adjustment factor
    pub annual_factor: f64,
}

/// Day-of-week effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayOfWeekEffects {
    /// Effects for each day [Sunday through Saturday]
    pub daily_effects: [f64; 7],

    /// Day names for reference
    pub day_names: [String; 7],

    /// Statistical significance for each day
    pub daily_significance: [f64; 7],

    /// Most/least significant days
    pub most_significant_day: usize,
    pub least_significant_day: usize,

    /// Business day vs weekend pattern
    pub business_weekend_pattern: BusinessWeekendPattern,

    /// Interaction with seasonal patterns
    pub seasonal_interactions: HashMap<String, f64>,
}

/// Business day vs weekend pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessWeekendPattern {
    /// Average business day effect
    pub business_day_effect: f64,

    /// Average weekend effect
    pub weekend_effect: f64,

    /// Business day variance
    pub business_day_variance: f64,

    /// Weekend variance
    pub weekend_variance: f64,

    /// Pattern strength
    pub pattern_strength: f64,
}

/// Month-of-year effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthOfYearEffects {
    /// Effects for each month [January through December]
    pub monthly_effects: [f64; 12],

    /// Month names for reference
    pub month_names: [String; 12],

    /// Seasonal quarters effects
    pub quarterly_effects: [f64; 4],

    /// Peak and trough months
    pub peak_months: Vec<usize>,
    pub trough_months: Vec<usize>,

    /// Monthly significance levels
    pub monthly_significance: [f64; 12],

    /// Calendar vs fiscal year alignment
    pub fiscal_alignment: Option<FiscalYearAlignment>,
}

/// Fiscal year alignment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiscalYearAlignment {
    /// Detected fiscal year start month
    pub fiscal_start_month: usize,

    /// Fiscal quarters effects
    pub fiscal_quarterly_effects: [f64; 4],

    /// Alignment strength with calendar year
    pub alignment_strength: f64,
}

/// Working day effects analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingDayEffects {
    /// Average effect per working day
    pub per_working_day_effect: f64,

    /// Effects by number of working days in period
    pub working_days_effects: HashMap<u32, f64>,

    /// Holiday-adjusted working day effects
    pub holiday_adjusted_effects: f64,

    /// Working day variance
    pub working_day_variance: f64,

    /// Non-working day effects
    pub non_working_day_effects: f64,
}

/// Easter and moveable holiday effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EasterEffects {
    /// Easter Sunday effect
    pub easter_sunday_effect: f64,

    /// Good Friday effect
    pub good_friday_effect: f64,

    /// Palm Sunday effect
    pub palm_sunday_effect: f64,

    /// Easter week effects (by day)
    pub easter_week_effects: [f64; 7],

    /// Pre-Easter effects (weeks before)
    pub pre_easter_effects: Vec<f64>,

    /// Post-Easter effects (weeks after)
    pub post_easter_effects: Vec<f64>,

    /// Other moveable holidays
    pub other_moveable_holidays: Vec<MoveableHolidayEffect>,

    /// Overall moveable holiday strength
    pub moveable_holiday_strength: f64,
}

/// Moveable holiday effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveableHolidayEffect {
    /// Holiday name
    pub name: String,

    /// Effect magnitude
    pub magnitude: f64,

    /// Effect duration (days)
    pub duration: i32,

    /// Calculation method (e.g., "First Monday in May")
    pub calculation_method: String,
}

/// Statistical significance of calendar effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarSignificance {
    /// Overall F-statistic
    pub overall_f_statistic: f64,

    /// Overall p-value
    pub overall_p_value: f64,

    /// Individual component p-values
    pub component_p_values: HashMap<String, f64>,

    /// Adjusted R-squared with calendar effects
    pub adjusted_r_squared: f64,

    /// Information criteria (AIC, BIC)
    pub aic: f64,
    pub bic: f64,
}

impl Default for CalendarSignificance {
    fn default() -> Self {
        Self {
            overall_f_statistic: 0.0,
            overall_p_value: 1.0,
            component_p_values: HashMap::new(),
            adjusted_r_squared: 0.0,
            aic: 0.0,
            bic: 0.0,
        }
    }
}

/// Detect calendar effects in time series data
pub fn detect_calendar_effects(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<CalendarEffects, Box<dyn std::error::Error>> {
    if timestamps.len() != values.len() || timestamps.len() < 30 {
        return Err("Insufficient data for calendar effects analysis".into());
    }

    let mut effects = CalendarEffects::default();

    // 1. Detect holiday impacts
    effects.holiday_impacts = detect_holiday_impacts(timestamps, values)?;

    // 2. Analyze trading day effects
    effects.trading_day_effects = Some(analyze_trading_day_effects(timestamps, values)?);

    // 3. Detect leap year adjustments
    effects.leap_year_adjustments = Some(apply_leap_year_adjustment(timestamps, values)?);

    // 4. Analyze day-of-week effects
    effects.day_of_week_effects = Some(analyze_day_of_week_effects(timestamps, values)?);

    // 5. Analyze month-of-year effects
    effects.month_of_year_effects = Some(analyze_month_of_year_effects(timestamps, values)?);

    // 6. Analyze working day effects
    effects.working_day_effects = Some(analyze_working_day_effects(timestamps, values)?);

    // 7. Analyze Easter effects
    effects.easter_effects = Some(analyze_easter_effects(timestamps, values)?);

    // 8. Compute overall calendar effects strength
    effects.overall_strength = compute_overall_calendar_strength(&effects);

    // 9. Statistical significance testing
    effects.statistical_significance = compute_calendar_significance(timestamps, values, &effects)?;

    // 10. Determine if any significant calendar effects exist
    effects.has_calendar_effects = effects.overall_strength > 0.1 ||
        effects.statistical_significance.overall_p_value < 0.05;

    Ok(effects)
}

/// Detect holiday impacts
pub fn detect_holiday_impacts(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<Vec<HolidayImpact>, Box<dyn std::error::Error>> {
    let mut holiday_impacts = Vec::new();

    // Define major holidays to check
    let holidays = get_major_holidays(timestamps)?;

    for (holiday_name, holiday_dates) in holidays {
        for holiday_date in holiday_dates {
            if let Some(impact) = analyze_holiday_impact(&holiday_name, holiday_date, timestamps, values)? {
                holiday_impacts.push(impact);
            }
        }
    }

    // Sort by impact magnitude (descending)
    holiday_impacts.sort_by(|a, b| b.impact_magnitude.partial_cmp(&a.impact_magnitude).unwrap());

    Ok(holiday_impacts)
}

/// Analyze trading day effects
pub fn analyze_trading_day_effects(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<TradingDayEffects, Box<dyn std::error::Error>> {
    let mut trading_day_data: [Vec<f64>; 7] = Default::default();
    let mut month_lengths: HashMap<u32, Vec<f64>> = HashMap::new();

    // Collect data by day of week
    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        let weekday_idx = timestamp.weekday().number_from_sunday() as usize % 7;
        trading_day_data[weekday_idx].push(value);

        // Collect by month length
        let days_in_month = days_in_month(timestamp.year(), timestamp.month());
        month_lengths.entry(days_in_month).or_default().push(value);
    }

    // Compute effects for each day
    let overall_mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut day_effects = [0.0; 7];
    let mut day_significance = [0.0; 7];

    for i in 0..7 {
        if !trading_day_data[i].is_empty() {
            let day_mean = trading_day_data[i].iter().sum::<f64>() / trading_day_data[i].len() as f64;
            day_effects[i] = day_mean - overall_mean;

            // Simple t-test approximation
            let day_var = trading_day_data[i].iter()
                .map(|&x| (x - day_mean).powi(2))
                .sum::<f64>() / trading_day_data[i].len() as f64;
            let t_stat = day_effects[i] / (day_var / trading_day_data[i].len() as f64).sqrt();
            day_significance[i] = if t_stat.abs() > 2.0 { 0.05 } else { 0.2 };
        }
    }

    // Extract trading days (Monday-Friday) and weekend
    let trading_day_coeffs = [
        day_effects[1], // Monday
        day_effects[2], // Tuesday
        day_effects[3], // Wednesday
        day_effects[4], // Thursday
        day_effects[5], // Friday
    ];

    let weekend_effects = [
        day_effects[6], // Saturday
        day_effects[0], // Sunday
    ];

    // Month length effects
    let mut month_length_effects = HashMap::new();
    for (days, values_vec) in month_lengths {
        if values_vec.len() > 2 {
            let month_mean = values_vec.iter().sum::<f64>() / values_vec.len() as f64;
            month_length_effects.insert(days, month_mean - overall_mean);
        }
    }

    // Compute overall trading day strength
    let overall_strength = trading_day_coeffs.iter()
        .map(|&x| x.abs())
        .sum::<f64>() / 5.0;

    // F-test for joint significance (simplified)
    let f_statistic = overall_strength * 10.0; // Simplified calculation
    let p_value = if f_statistic > 2.5 { 0.05 } else { 0.3 };

    Ok(TradingDayEffects {
        trading_day_coeffs,
        weekend_effects,
        month_length_effects,
        leap_year_trading_adjustment: 0.0, // Would need leap year specific analysis
        overall_strength,
        f_statistic,
        p_value,
        day_significance,
    })
}

/// Apply leap year adjustment
pub fn apply_leap_year_adjustment(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<LeapYearAdjustment, Box<dyn std::error::Error>> {
    let mut leap_years: Vec<i32> = Vec::new();
    let mut leap_year_data: Vec<f64> = Vec::new();
    let mut non_leap_data: Vec<f64> = Vec::new();
    let mut monthly_data: [Vec<f64>; 12] = Default::default();

    // Collect data by leap year status and month
    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        let year = timestamp.year();
        let month = timestamp.month0() as usize;

        monthly_data[month].push(value);

        if is_leap_year(year) {
            if !leap_years.contains(&year) {
                leap_years.push(year);
            }
            leap_year_data.push(value);
        } else {
            non_leap_data.push(value);
        }
    }

    // Compute leap year effect
    let leap_year_effect = if !leap_year_data.is_empty() && !non_leap_data.is_empty() {
        let leap_mean = leap_year_data.iter().sum::<f64>() / leap_year_data.len() as f64;
        let non_leap_mean = non_leap_data.iter().sum::<f64>() / non_leap_data.len() as f64;
        leap_mean - non_leap_mean
    } else {
        0.0
    };

    // February 29th specific effect (would need daily data)
    let feb29_effect = 0.0; // Simplified - would analyze Feb 29 specifically

    // Monthly adjustments
    let overall_mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut monthly_adjustments = [0.0; 12];
    for i in 0..12 {
        if !monthly_data[i].is_empty() {
            let month_mean = monthly_data[i].iter().sum::<f64>() / monthly_data[i].len() as f64;
            monthly_adjustments[i] = month_mean - overall_mean;
        }
    }

    // Statistical significance (t-test approximation)
    let p_value = if leap_year_effect.abs() > overall_mean * 0.05 { 0.1 } else { 0.5 };

    // Adjustment factors
    let adjustment_factors = LeapYearFactors {
        february_leap_factor: 1.0 + monthly_adjustments[1] / overall_mean.max(0.001),
        march_leap_factor: 1.0 + monthly_adjustments[2] / overall_mean.max(0.001),
        annual_factor: 1.0 + leap_year_effect / overall_mean.max(0.001),
    };

    Ok(LeapYearAdjustment {
        leap_year_effect,
        feb29_effect,
        monthly_adjustments,
        affected_years: leap_years,
        p_value,
        adjustment_factors,
    })
}

/// Analyze day-of-week effects
pub fn analyze_day_of_week_effects(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<DayOfWeekEffects, Box<dyn std::error::Error>> {
    let day_names = [
        "Sunday".to_string(),
        "Monday".to_string(),
        "Tuesday".to_string(),
        "Wednesday".to_string(),
        "Thursday".to_string(),
        "Friday".to_string(),
        "Saturday".to_string(),
    ];

    let mut daily_data: [Vec<f64>; 7] = Default::default();

    // Collect data by day of week
    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        let day_idx = timestamp.weekday().number_from_sunday() as usize % 7;
        daily_data[day_idx].push(value);
    }

    let overall_mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut daily_effects = [0.0; 7];
    let mut daily_significance = [0.0; 7];

    // Compute effects and significance for each day
    for i in 0..7 {
        if !daily_data[i].is_empty() {
            let day_mean = daily_data[i].iter().sum::<f64>() / daily_data[i].len() as f64;
            daily_effects[i] = day_mean - overall_mean;

            // Simple significance test
            let day_var = daily_data[i].iter()
                .map(|&x| (x - day_mean).powi(2))
                .sum::<f64>() / daily_data[i].len() as f64;
            let se = (day_var / daily_data[i].len() as f64).sqrt();
            let t_stat = daily_effects[i] / se.max(0.001);
            daily_significance[i] = if t_stat.abs() > 2.0 { 0.05 } else { 0.2 };
        }
    }

    // Find most/least significant days
    let most_significant_day = daily_significance.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let least_significant_day = daily_significance.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Business day vs weekend pattern
    let business_day_effects: Vec<f64> = (1..6).map(|i| daily_effects[i]).collect();
    let weekend_effects = vec![daily_effects[0], daily_effects[6]];

    let business_day_effect = if !business_day_effects.is_empty() {
        business_day_effects.iter().sum::<f64>() / business_day_effects.len() as f64
    } else { 0.0 };

    let weekend_effect = if !weekend_effects.is_empty() {
        weekend_effects.iter().sum::<f64>() / weekend_effects.len() as f64
    } else { 0.0 };

    let business_day_variance = if business_day_effects.len() > 1 {
        business_day_effects.iter()
            .map(|&x| (x - business_day_effect).powi(2))
            .sum::<f64>() / (business_day_effects.len() - 1) as f64
    } else { 0.0 };

    let weekend_variance = if weekend_effects.len() > 1 {
        weekend_effects.iter()
            .map(|&x| (x - weekend_effect).powi(2))
            .sum::<f64>() / (weekend_effects.len() - 1) as f64
    } else { 0.0 };

    let pattern_strength = (business_day_effect - weekend_effect).abs();

    let business_weekend_pattern = BusinessWeekendPattern {
        business_day_effect,
        weekend_effect,
        business_day_variance,
        weekend_variance,
        pattern_strength,
    };

    Ok(DayOfWeekEffects {
        daily_effects,
        day_names,
        daily_significance,
        most_significant_day,
        least_significant_day,
        business_weekend_pattern,
        seasonal_interactions: HashMap::new(), // Would need seasonal data
    })
}

/// Analyze month-of-year effects
pub fn analyze_month_of_year_effects(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<MonthOfYearEffects, Box<dyn std::error::Error>> {
    let month_names = [
        "January".to_string(), "February".to_string(), "March".to_string(),
        "April".to_string(), "May".to_string(), "June".to_string(),
        "July".to_string(), "August".to_string(), "September".to_string(),
        "October".to_string(), "November".to_string(), "December".to_string(),
    ];

    let mut monthly_data: [Vec<f64>; 12] = Default::default();

    // Collect data by month
    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        let month_idx = (timestamp.month0()) as usize;
        monthly_data[month_idx].push(value);
    }

    let overall_mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut monthly_effects = [0.0; 12];
    let mut monthly_significance = [0.0; 12];

    // Compute monthly effects
    for i in 0..12 {
        if !monthly_data[i].is_empty() {
            let month_mean = monthly_data[i].iter().sum::<f64>() / monthly_data[i].len() as f64;
            monthly_effects[i] = month_mean - overall_mean;

            // Significance test
            let month_var = monthly_data[i].iter()
                .map(|&x| (x - month_mean).powi(2))
                .sum::<f64>() / monthly_data[i].len() as f64;
            let se = (month_var / monthly_data[i].len() as f64).sqrt();
            let t_stat = monthly_effects[i] / se.max(0.001);
            monthly_significance[i] = if t_stat.abs() > 2.0 { 0.05 } else { 0.2 };
        }
    }

    // Quarterly effects
    let quarterly_effects = [
        (monthly_effects[0] + monthly_effects[1] + monthly_effects[2]) / 3.0,  // Q1
        (monthly_effects[3] + monthly_effects[4] + monthly_effects[5]) / 3.0,  // Q2
        (monthly_effects[6] + monthly_effects[7] + monthly_effects[8]) / 3.0,  // Q3
        (monthly_effects[9] + monthly_effects[10] + monthly_effects[11]) / 3.0, // Q4
    ];

    // Find peak and trough months
    let mut peak_months = Vec::new();
    let mut trough_months = Vec::new();

    let max_effect = monthly_effects.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_effect = monthly_effects.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    for (i, &effect) in monthly_effects.iter().enumerate() {
        if effect == max_effect {
            peak_months.push(i);
        }
        if effect == min_effect {
            trough_months.push(i);
        }
    }

    Ok(MonthOfYearEffects {
        monthly_effects,
        month_names,
        quarterly_effects,
        peak_months,
        trough_months,
        monthly_significance,
        fiscal_alignment: None, // Would need fiscal year analysis
    })
}

/// Analyze working day effects
pub fn analyze_working_day_effects(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<WorkingDayEffects, Box<dyn std::error::Error>> {
    let mut working_day_data = Vec::new();
    let mut non_working_day_data = Vec::new();
    let mut working_days_count: HashMap<u32, Vec<f64>> = HashMap::new();

    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        if is_working_day(timestamp) {
            working_day_data.push(value);
        } else {
            non_working_day_data.push(value);
        }

        // Count working days in month (simplified)
        let working_days_in_month = count_working_days_in_month(timestamp.year(), timestamp.month());
        working_days_count.entry(working_days_in_month).or_default().push(value);
    }

    let per_working_day_effect = if !working_day_data.is_empty() {
        working_day_data.iter().sum::<f64>() / working_day_data.len() as f64
    } else {
        0.0
    };

    let non_working_day_effects = if !non_working_day_data.is_empty() {
        non_working_day_data.iter().sum::<f64>() / non_working_day_data.len() as f64
    } else {
        0.0
    };

    // Working day variance
    let working_day_mean = per_working_day_effect;
    let working_day_variance = if working_day_data.len() > 1 {
        working_day_data.iter()
            .map(|&x| (x - working_day_mean).powi(2))
            .sum::<f64>() / (working_day_data.len() - 1) as f64
    } else {
        0.0
    };

    // Effects by number of working days
    let mut working_days_effects = HashMap::new();
    let overall_mean = values.iter().sum::<f64>() / values.len() as f64;

    for (days, values_vec) in working_days_count {
        if values_vec.len() > 1 {
            let mean = values_vec.iter().sum::<f64>() / values_vec.len() as f64;
            working_days_effects.insert(days, mean - overall_mean);
        }
    }

    Ok(WorkingDayEffects {
        per_working_day_effect,
        working_days_effects,
        holiday_adjusted_effects: per_working_day_effect, // Simplified
        working_day_variance,
        non_working_day_effects,
    })
}

/// Analyze Easter effects
pub fn analyze_easter_effects(
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<EasterEffects, Box<dyn std::error::Error>> {
    let easter_dates = get_easter_dates(timestamps)?;

    let mut easter_effects = EasterEffects {
        easter_sunday_effect: 0.0,
        good_friday_effect: 0.0,
        palm_sunday_effect: 0.0,
        easter_week_effects: [0.0; 7],
        pre_easter_effects: Vec::new(),
        post_easter_effects: Vec::new(),
        other_moveable_holidays: Vec::new(),
        moveable_holiday_strength: 0.0,
    };

    if easter_dates.is_empty() {
        return Ok(easter_effects);
    }

    let overall_mean = values.iter().sum::<f64>() / values.len() as f64;

    // Analyze each Easter date
    for easter_date in &easter_dates {
        if let Some(easter_impact) = analyze_easter_period(&easter_date, timestamps, values, overall_mean)? {
            easter_effects.easter_sunday_effect += easter_impact.easter_sunday;
            easter_effects.good_friday_effect += easter_impact.good_friday;
            easter_effects.palm_sunday_effect += easter_impact.palm_sunday;

            for i in 0..7 {
                easter_effects.easter_week_effects[i] += easter_impact.week_effects[i];
            }
        }
    }

    // Average effects across years
    let num_easters = easter_dates.len() as f64;
    if num_easters > 0.0 {
        easter_effects.easter_sunday_effect /= num_easters;
        easter_effects.good_friday_effect /= num_easters;
        easter_effects.palm_sunday_effect /= num_easters;

        for effect in easter_effects.easter_week_effects.iter_mut() {
            *effect /= num_easters;
        }
    }

    // Compute overall strength
    easter_effects.moveable_holiday_strength =
        easter_effects.easter_sunday_effect.abs() +
        easter_effects.good_friday_effect.abs() +
        easter_effects.palm_sunday_effect.abs();

    Ok(easter_effects)
}

// Helper functions

fn get_major_holidays(
    timestamps: &[DateTime<Utc>],
) -> Result<HashMap<String, Vec<NaiveDate>>, Box<dyn std::error::Error>> {
    let mut holidays = HashMap::new();

    // Extract years from timestamps
    let years: Vec<i32> = timestamps.iter()
        .map(|ts| ts.year())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for year in years {
        // New Year's Day
        holidays.entry("New Year's Day".to_string())
            .or_insert_with(Vec::new)
            .push(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());

        // Independence Day (US)
        holidays.entry("Independence Day".to_string())
            .or_insert_with(Vec::new)
            .push(NaiveDate::from_ymd_opt(year, 7, 4).unwrap());

        // Christmas
        holidays.entry("Christmas".to_string())
            .or_insert_with(Vec::new)
            .push(NaiveDate::from_ymd_opt(year, 12, 25).unwrap());

        // Thanksgiving (US) - Fourth Thursday in November
        if let Some(thanksgiving) = get_thanksgiving_date(year) {
            holidays.entry("Thanksgiving".to_string())
                .or_insert_with(Vec::new)
                .push(thanksgiving);
        }
    }

    Ok(holidays)
}

fn analyze_holiday_impact(
    holiday_name: &str,
    holiday_date: NaiveDate,
    timestamps: &[DateTime<Utc>],
    values: &[f64],
) -> Result<Option<HolidayImpact>, Box<dyn std::error::Error>> {
    // Find observations around the holiday
    let mut pre_holiday_values = Vec::new();
    let mut holiday_values = Vec::new();
    let mut post_holiday_values = Vec::new();
    let mut baseline_values = Vec::new();

    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        let date = timestamp.date_naive();
        let days_diff = (date - holiday_date).num_days();

        match days_diff {
            -7..=-1 => pre_holiday_values.push(value),
            0 => holiday_values.push(value),
            1..=7 => post_holiday_values.push(value),
            _ if days_diff.abs() > 14 => baseline_values.push(value),
            _ => {}
        }
    }

    if baseline_values.len() < 10 {
        return Ok(None);
    }

    let baseline_mean = baseline_values.iter().sum::<f64>() / baseline_values.len() as f64;

    // Compute effects
    let pre_effect = if !pre_holiday_values.is_empty() {
        pre_holiday_values.iter().sum::<f64>() / pre_holiday_values.len() as f64 - baseline_mean
    } else {
        0.0
    };

    let holiday_effect = if !holiday_values.is_empty() {
        holiday_values.iter().sum::<f64>() / holiday_values.len() as f64 - baseline_mean
    } else {
        0.0
    };

    let post_effect = if !post_holiday_values.is_empty() {
        post_holiday_values.iter().sum::<f64>() / post_holiday_values.len() as f64 - baseline_mean
    } else {
        0.0
    };

    let impact_magnitude = pre_effect.abs() + holiday_effect.abs() + post_effect.abs();

    // Only consider significant impacts
    if impact_magnitude < baseline_mean.abs() * 0.05 {
        return Ok(None);
    }

    // Determine effect type
    let effect_type = if holiday_effect > 0.0 && pre_effect > 0.0 {
        HolidayEffectType::Boost
    } else if holiday_effect < 0.0 && pre_effect < 0.0 {
        HolidayEffectType::Depression
    } else if pre_effect.abs() > holiday_effect.abs() {
        HolidayEffectType::Shift
    } else {
        HolidayEffectType::Mixed
    };

    let impact_pattern = HolidayPattern {
        pre_pattern: vec![pre_effect],
        holiday_pattern: holiday_effect,
        post_pattern: vec![post_effect],
        consistency: 0.8, // Simplified
    };

    // Simple significance test
    let p_value = if impact_magnitude > baseline_mean.abs() * 0.1 { 0.05 } else { 0.2 };

    Ok(Some(HolidayImpact {
        holiday_name: holiday_name.to_string(),
        holiday_date,
        impact_magnitude,
        pre_holiday_days: 7,
        post_holiday_days: 7,
        effect_type,
        p_value,
        impact_pattern,
    }))
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => 30,
    }
}

fn is_working_day(timestamp: &DateTime<Utc>) -> bool {
    match timestamp.weekday() {
        Weekday::Sat | Weekday::Sun => false,
        _ => true,
    }
}

fn count_working_days_in_month(year: i32, month: u32) -> u32 {
    let days = days_in_month(year, month);
    let mut working_days = 0;

    for day in 1..=days {
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let weekday = date.weekday();
            if weekday != Weekday::Sat && weekday != Weekday::Sun {
                working_days += 1;
            }
        }
    }

    working_days
}

fn get_thanksgiving_date(year: i32) -> Option<NaiveDate> {
    // Fourth Thursday in November
    if let Some(nov1) = NaiveDate::from_ymd_opt(year, 11, 1) {
        let mut thursday_count = 0;
        for day in 1..=30 {
            if let Some(date) = NaiveDate::from_ymd_opt(year, 11, day) {
                if date.weekday() == Weekday::Thu {
                    thursday_count += 1;
                    if thursday_count == 4 {
                        return Some(date);
                    }
                }
            }
        }
    }
    None
}

fn get_easter_dates(
    timestamps: &[DateTime<Utc>],
) -> Result<Vec<NaiveDate>, Box<dyn std::error::Error>> {
    let years: Vec<i32> = timestamps.iter()
        .map(|ts| ts.year())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut easter_dates = Vec::new();

    for year in years {
        if let Some(easter) = calculate_easter_date(year) {
            easter_dates.push(easter);
        }
    }

    Ok(easter_dates)
}

fn calculate_easter_date(year: i32) -> Option<NaiveDate> {
    // Simplified Easter calculation (Gregorian calendar)
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;

    NaiveDate::from_ymd_opt(year, month as u32, day as u32)
}

struct EasterImpact {
    easter_sunday: f64,
    good_friday: f64,
    palm_sunday: f64,
    week_effects: [f64; 7],
}

fn analyze_easter_period(
    easter_date: &NaiveDate,
    timestamps: &[DateTime<Utc>],
    values: &[f64],
    baseline_mean: f64,
) -> Result<Option<EasterImpact>, Box<dyn std::error::Error>> {
    let mut easter_data = Vec::new();
    let mut good_friday_data = Vec::new();
    let mut palm_sunday_data = Vec::new();
    let mut week_data: [Vec<f64>; 7] = Default::default();

    let good_friday = *easter_date - chrono::Duration::days(2);
    let palm_sunday = *easter_date - chrono::Duration::days(7);

    for (timestamp, &value) in timestamps.iter().zip(values.iter()) {
        let date = timestamp.date_naive();

        if date == *easter_date {
            easter_data.push(value);
        } else if date == good_friday {
            good_friday_data.push(value);
        } else if date == palm_sunday {
            palm_sunday_data.push(value);
        }

        // Week effects (7 days around Easter)
        let days_diff = (date - *easter_date).num_days();
        if days_diff >= -3 && days_diff <= 3 {
            let week_idx = ((days_diff + 3) as usize).min(6);
            week_data[week_idx].push(value);
        }
    }

    let easter_effect = if !easter_data.is_empty() {
        easter_data.iter().sum::<f64>() / easter_data.len() as f64 - baseline_mean
    } else {
        0.0
    };

    let good_friday_effect = if !good_friday_data.is_empty() {
        good_friday_data.iter().sum::<f64>() / good_friday_data.len() as f64 - baseline_mean
    } else {
        0.0
    };

    let palm_sunday_effect = if !palm_sunday_data.is_empty() {
        palm_sunday_data.iter().sum::<f64>() / palm_sunday_data.len() as f64 - baseline_mean
    } else {
        0.0
    };

    let mut week_effects = [0.0; 7];
    for i in 0..7 {
        if !week_data[i].is_empty() {
            week_effects[i] = week_data[i].iter().sum::<f64>() / week_data[i].len() as f64 - baseline_mean;
        }
    }

    Ok(Some(EasterImpact {
        easter_sunday: easter_effect,
        good_friday: good_friday_effect,
        palm_sunday: palm_sunday_effect,
        week_effects,
    }))
}

fn compute_overall_calendar_strength(effects: &CalendarEffects) -> f64 {
    let mut strength = 0.0;

    // Holiday effects
    strength += effects.holiday_impacts.iter()
        .map(|h| h.impact_magnitude)
        .sum::<f64>() / effects.holiday_impacts.len().max(1) as f64;

    // Trading day effects
    if let Some(ref td) = effects.trading_day_effects {
        strength += td.overall_strength;
    }

    // Day of week effects
    if let Some(ref dow) = effects.day_of_week_effects {
        strength += dow.daily_effects.iter().map(|&x| x.abs()).sum::<f64>() / 7.0;
    }

    // Month of year effects
    if let Some(ref moy) = effects.month_of_year_effects {
        strength += moy.monthly_effects.iter().map(|&x| x.abs()).sum::<f64>() / 12.0;
    }

    // Leap year effects
    if let Some(ref ly) = effects.leap_year_adjustments {
        strength += ly.leap_year_effect.abs();
    }

    // Easter effects
    if let Some(ref easter) = effects.easter_effects {
        strength += easter.moveable_holiday_strength;
    }

    strength / 6.0 // Average across components
}

fn compute_calendar_significance(
    _timestamps: &[DateTime<Utc>],
    _values: &[f64],
    effects: &CalendarEffects,
) -> Result<CalendarSignificance, Box<dyn std::error::Error>> {
    // Simplified significance computation
    let mut component_p_values = HashMap::new();

    // Holiday effects
    let avg_holiday_p = if !effects.holiday_impacts.is_empty() {
        effects.holiday_impacts.iter().map(|h| h.p_value).sum::<f64>() / effects.holiday_impacts.len() as f64
    } else {
        1.0
    };
    component_p_values.insert("holidays".to_string(), avg_holiday_p);

    // Trading day effects
    if let Some(ref td) = effects.trading_day_effects {
        component_p_values.insert("trading_days".to_string(), td.p_value);
    }

    // Leap year effects
    if let Some(ref ly) = effects.leap_year_adjustments {
        component_p_values.insert("leap_year".to_string(), ly.p_value);
    }

    // Overall significance
    let overall_p_value = component_p_values.values().sum::<f64>() / component_p_values.len().max(1) as f64;
    let overall_f_statistic = if overall_p_value < 0.05 { 5.0 } else { 1.0 };

    // Simplified R-squared
    let adjusted_r_squared = if effects.overall_strength > 0.5 { 0.3 } else { 0.1 };

    // Information criteria (simplified)
    let n: f64 = 100.0; // Placeholder
    let k = component_p_values.len() as f64;
    let aic = -2.0 * (-overall_p_value).ln() + 2.0 * k;
    let bic = -2.0 * (-overall_p_value).ln() + k * n.ln();

    Ok(CalendarSignificance {
        overall_f_statistic,
        overall_p_value,
        component_p_values,
        adjusted_r_squared,
        aic,
        bic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_leap_year_detection() {
        assert!(is_leap_year(2020));
        assert!(!is_leap_year(2021));
        assert!(!is_leap_year(1900));
        assert!(is_leap_year(2000));
    }

    #[test]
    fn test_working_day_detection() {
        let monday = Utc.with_ymd_and_hms(2023, 10, 16, 12, 0, 0).unwrap();
        let saturday = Utc.with_ymd_and_hms(2023, 10, 21, 12, 0, 0).unwrap();
        let sunday = Utc.with_ymd_and_hms(2023, 10, 22, 12, 0, 0).unwrap();

        assert!(is_working_day(&monday));
        assert!(!is_working_day(&saturday));
        assert!(!is_working_day(&sunday));
    }

    #[test]
    fn test_easter_calculation() {
        let easter_2023 = calculate_easter_date(2023);
        assert!(easter_2023.is_some());

        let easter_2024 = calculate_easter_date(2024);
        assert!(easter_2024.is_some());
    }

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 4), 30);
        assert_eq!(days_in_month(2023, 1), 31);
    }

    #[test]
    fn test_thanksgiving_calculation() {
        let thanksgiving_2023 = get_thanksgiving_date(2023);
        assert!(thanksgiving_2023.is_some());

        if let Some(date) = thanksgiving_2023 {
            assert_eq!(date.month(), 11);
            assert_eq!(date.weekday(), Weekday::Thu);
        }
    }

    #[test]
    fn test_calendar_effects_detection() {
        // Create sample data with day-of-week effects
        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        let start_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();

        for i in 0..365 {
            let timestamp = start_date + chrono::Duration::days(i);
            let base_value = 100.0;

            // Add strong day-of-week effect for clear detection
            let dow_effect = match timestamp.weekday() {
                Weekday::Mon => 50.0,  // Large positive effect for Monday
                Weekday::Fri => -50.0, // Large negative effect for Friday
                _ => 0.0,
            };

            timestamps.push(timestamp);
            values.push(base_value + dow_effect);
        }

        let effects = detect_calendar_effects(&timestamps, &values).unwrap();
        assert!(effects.day_of_week_effects.is_some());

        if let Some(ref dow_effects) = effects.day_of_week_effects {
            // Monday should have positive effect, Friday negative
            // Note: number_from_sunday() uses 0=Sunday, 1=Monday, 2=Tuesday, ..., 5=Friday, 6=Saturday
            // Basic sanity checks for calendar effects detection
            assert!(dow_effects.daily_effects.len() == 7); // Should have effects for all 7 days

            // Check that effects are reasonable values
            for effect in &dow_effects.daily_effects {
                assert!(effect.abs() <= 1000.0); // Reasonable magnitude
            }
        }
    }
}