//! # Seasonality Detection Module
//!
//! Provides various methods for detecting seasonal patterns in time series data
//! including Fourier analysis, periodogram analysis, and autocorrelation-based methods.

use crate::seasonality::{SeasonalPeriod, SeasonalPeriodType, SeasonalityAnalysisConfig};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Methods available for seasonality detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalityMethod {
    /// Fourier transform-based frequency analysis
    Fourier,
    /// Periodogram analysis
    Periodogram,
    /// Autocorrelation-based detection
    Autocorrelation,
    /// Combined multiple methods
    Combined,
}

/// Result of seasonality detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityDetectionResult {
    /// Detected seasonal periods
    pub seasonal_periods: Vec<SeasonalPeriod>,

    /// Overall seasonality score (0.0 to 1.0)
    pub overall_seasonality: f64,

    /// Method used for detection
    pub method: SeasonalityMethod,

    /// Fourier analysis results (if performed)
    pub fourier_analysis: Option<FourierAnalysis>,

    /// Periodogram analysis results (if performed)
    pub periodogram_analysis: Option<PeriodogramAnalysis>,

    /// Autocorrelation analysis results (if performed)
    pub autocorrelation_analysis: Option<AutocorrelationAnalysis>,

    /// Statistical significance tests
    pub significance_tests: HashMap<String, f64>,
}

impl Default for SeasonalityDetectionResult {
    fn default() -> Self {
        Self {
            seasonal_periods: Vec::new(),
            overall_seasonality: 0.0,
            method: SeasonalityMethod::Combined,
            fourier_analysis: None,
            periodogram_analysis: None,
            autocorrelation_analysis: None,
            significance_tests: HashMap::new(),
        }
    }
}

/// Fourier transform analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierAnalysis {
    /// Frequencies analyzed
    pub frequencies: Vec<f64>,

    /// Power spectrum
    pub power_spectrum: Vec<f64>,

    /// Phase spectrum
    pub phase_spectrum: Vec<f64>,

    /// Dominant frequencies and their power
    pub dominant_frequencies: Vec<(f64, f64)>,

    /// Signal-to-noise ratio
    pub snr: f64,
}

/// Periodogram analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodogramAnalysis {
    /// Periods analyzed
    pub periods: Vec<f64>,

    /// Periodogram values
    pub periodogram: Vec<f64>,

    /// Statistical significance thresholds
    pub significance_threshold: f64,

    /// Significant peaks
    pub significant_peaks: Vec<(f64, f64)>,

    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Autocorrelation-based analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationAnalysis {
    /// Lag values
    pub lags: Vec<usize>,

    /// Autocorrelation coefficients
    pub autocorr_coeffs: Vec<f64>,

    /// Confidence bands
    pub confidence_bands: Vec<(f64, f64)>,

    /// Detected seasonal lags
    pub seasonal_lags: Vec<usize>,

    /// Box-Ljung test results
    pub box_ljung_test: BoxLjungTest,
}

/// Multiple seasonality detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleSeasonalityResult {
    /// All detected seasonal periods
    pub seasonal_periods: Vec<SeasonalPeriod>,

    /// Interaction effects between periods
    pub interactions: Vec<SeasonalInteraction>,

    /// Nested seasonality hierarchy
    pub hierarchy: Vec<SeasonalHierarchy>,
}

/// Interaction between seasonal periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalInteraction {
    /// First seasonal period
    pub period1: f64,

    /// Second seasonal period
    pub period2: f64,

    /// Interaction strength
    pub interaction_strength: f64,

    /// Type of interaction
    pub interaction_type: InteractionType,
}

/// Types of seasonal interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Periods reinforce each other
    Reinforcing,
    /// Periods interfere with each other
    Interfering,
    /// Periods are independent
    Independent,
}

/// Seasonal hierarchy for nested seasonality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalHierarchy {
    /// Parent period (longer)
    pub parent_period: f64,

    /// Child periods (shorter, nested within parent)
    pub child_periods: Vec<f64>,

    /// Hierarchy strength
    pub hierarchy_strength: f64,
}

/// Box-Ljung test for autocorrelation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxLjungTest {
    /// Test statistic
    pub statistic: f64,

    /// P-value
    pub p_value: f64,

    /// Degrees of freedom
    pub degrees_of_freedom: usize,

    /// Critical value
    pub critical_value: f64,

    /// Is autocorrelation significant?
    pub is_significant: bool,
}

/// Detect seasonality using specified methods
pub fn detect_seasonality(
    data: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<SeasonalityDetectionResult, Box<dyn std::error::Error>> {
    let mut result = SeasonalityDetectionResult::default();

    // Combine results from all requested methods
    let mut all_periods = Vec::new();
    let mut overall_seasonality_scores = Vec::new();

    for method in &config.detection_methods {
        match method {
            SeasonalityMethod::Fourier => {
                let fourier_result = analyze_fourier_spectrum(data, config)?;
                let periods = fourier_to_seasonal_periods(&fourier_result, config);
                all_periods.extend(periods);
                overall_seasonality_scores.push(fourier_result.snr / (1.0 + fourier_result.snr));
                result.fourier_analysis = Some(fourier_result);
            }
            SeasonalityMethod::Periodogram => {
                let periodogram_result = compute_periodogram(data, config)?;
                let periods = periodogram_to_seasonal_periods(&periodogram_result, config);
                all_periods.extend(periods);

                let avg_significance = periodogram_result.significant_peaks.iter()
                    .map(|(_, power)| power.min(1.0))
                    .sum::<f64>() / periodogram_result.significant_peaks.len().max(1) as f64;
                overall_seasonality_scores.push(avg_significance);
                result.periodogram_analysis = Some(periodogram_result);
            }
            SeasonalityMethod::Autocorrelation => {
                let autocorr_result = analyze_autocorrelation_seasonality(data, config)?;
                let periods = autocorr_to_seasonal_periods(&autocorr_result, config);
                all_periods.extend(periods);

                let max_autocorr = autocorr_result.autocorr_coeffs.iter()
                    .skip(1)
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                overall_seasonality_scores.push(max_autocorr);
                result.autocorrelation_analysis = Some(autocorr_result);
            }
            SeasonalityMethod::Combined => {
                // Combined method uses all available methods
                result.fourier_analysis = Some(analyze_fourier_spectrum(data, config)?);
                result.periodogram_analysis = Some(compute_periodogram(data, config)?);
                result.autocorrelation_analysis = Some(analyze_autocorrelation_seasonality(data, config)?);
            }
        }
    }

    // Consolidate and rank seasonal periods
    result.seasonal_periods = consolidate_seasonal_periods(all_periods, config)?;

    // Calculate overall seasonality score
    result.overall_seasonality = if overall_seasonality_scores.is_empty() {
        0.0
    } else {
        overall_seasonality_scores.iter().sum::<f64>() / overall_seasonality_scores.len() as f64
    };

    Ok(result)
}

/// Analyze Fourier spectrum for frequency components
pub fn analyze_fourier_spectrum(
    data: &[f64],
    _config: &SeasonalityAnalysisConfig,
) -> Result<FourierAnalysis, Box<dyn std::error::Error>> {
    let n = data.len();
    if n < 4 {
        return Err("Insufficient data for Fourier analysis".into());
    }

    // Compute DFT using simple implementation (for real FFT libraries, use rustfft)
    let mut frequencies = Vec::new();
    let mut power_spectrum = Vec::new();
    let mut phase_spectrum = Vec::new();

    let nyquist_freq = 0.5;
    let freq_resolution = nyquist_freq / (n / 2) as f64;

    for k in 1..=(n / 2) {
        let freq = k as f64 * freq_resolution;
        frequencies.push(freq);

        // Compute DFT coefficient
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (i, &value) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * freq * i as f64 / n as f64;
            real_sum += value * angle.cos();
            imag_sum += value * angle.sin();
        }

        let power = (real_sum * real_sum + imag_sum * imag_sum) / n as f64;
        let phase = imag_sum.atan2(real_sum);

        power_spectrum.push(power);
        phase_spectrum.push(phase);
    }

    // Find dominant frequencies
    let mean_power = power_spectrum.iter().sum::<f64>() / power_spectrum.len() as f64;
    let power_threshold = mean_power * 2.0; // Simple threshold

    let mut dominant_frequencies = Vec::new();
    for (i, &power) in power_spectrum.iter().enumerate() {
        if power > power_threshold {
            dominant_frequencies.push((frequencies[i], power));
        }
    }

    // Sort by power (descending)
    dominant_frequencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Calculate signal-to-noise ratio
    let signal_power = dominant_frequencies.iter().map(|(_, p)| p).sum::<f64>();
    let total_power = power_spectrum.iter().sum::<f64>();
    let noise_power = total_power - signal_power;
    let snr = if noise_power > 0.0 {
        signal_power / noise_power
    } else {
        f64::INFINITY
    };

    Ok(FourierAnalysis {
        frequencies,
        power_spectrum,
        phase_spectrum,
        dominant_frequencies,
        snr,
    })
}

/// Compute periodogram for seasonal period detection
pub fn compute_periodogram(
    data: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<PeriodogramAnalysis, Box<dyn std::error::Error>> {
    let n = data.len();
    if n < config.min_period * 2 {
        return Err("Insufficient data for periodogram analysis".into());
    }

    let mut periods = Vec::new();
    let mut periodogram = Vec::new();

    // Compute mean and center the data
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered_data: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Compute periodogram values for each period
    for period in config.min_period..=config.max_period.min(n / 2) {
        let period_f = period as f64;
        periods.push(period_f);

        // Compute periodogram using sum of squares approach
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for (t, &value) in centered_data.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * t as f64 / period_f;
            cos_sum += value * angle.cos();
            sin_sum += value * angle.sin();
        }

        let periodogram_value = (cos_sum * cos_sum + sin_sum * sin_sum) / n as f64;
        periodogram.push(periodogram_value);
    }

    // Calculate significance threshold using simple approach
    let mean_periodogram = periodogram.iter().sum::<f64>() / periodogram.len() as f64;
    let variance_periodogram = periodogram.iter()
        .map(|&x| (x - mean_periodogram).powi(2))
        .sum::<f64>() / periodogram.len() as f64;
    let std_periodogram = variance_periodogram.sqrt();

    // Use 2-sigma threshold for significance
    let significance_threshold = mean_periodogram + 2.0 * std_periodogram;

    // Find significant peaks
    let mut significant_peaks = Vec::new();
    for (i, &value) in periodogram.iter().enumerate() {
        if value > significance_threshold {
            // Check if it's a local maximum
            let is_peak = (i == 0 || periodogram[i-1] < value) &&
                         (i == periodogram.len()-1 || periodogram[i+1] < value);

            if is_peak {
                significant_peaks.push((periods[i], value));
            }
        }
    }

    // Sort peaks by strength (descending)
    significant_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Compute confidence intervals (simplified)
    let confidence_intervals: Vec<(f64, f64)> = periodogram.iter()
        .map(|&x| {
            let margin = std_periodogram * 1.96; // 95% CI
            (x - margin, x + margin)
        })
        .collect();

    Ok(PeriodogramAnalysis {
        periods,
        periodogram,
        significance_threshold,
        significant_peaks,
        confidence_intervals,
    })
}

/// Analyze autocorrelation for seasonality detection
pub fn analyze_autocorrelation_seasonality(
    data: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<AutocorrelationAnalysis, Box<dyn std::error::Error>> {
    let n = data.len();
    let max_lag = config.max_period.min(n / 3);

    if n < config.min_period * 2 {
        return Err("Insufficient data for autocorrelation analysis".into());
    }

    // Compute autocorrelation coefficients
    let mut lags = Vec::new();
    let mut autocorr_coeffs = Vec::new();

    // Center the data
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered_data: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Compute variance
    let variance = centered_data.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    for lag in 0..=max_lag {
        lags.push(lag);

        if lag == 0 {
            autocorr_coeffs.push(1.0);
            continue;
        }

        let mut covariance = 0.0;
        for i in 0..(n - lag) {
            covariance += centered_data[i] * centered_data[i + lag];
        }
        covariance /= n as f64;

        let autocorr = if variance > 0.0 {
            covariance / variance
        } else {
            0.0
        };

        autocorr_coeffs.push(autocorr);
    }

    // Compute confidence bands (±1.96/sqrt(n) for 95% confidence)
    let confidence_bound = 1.96 / (n as f64).sqrt();
    let confidence_bands: Vec<(f64, f64)> = (0..=max_lag)
        .map(|_| (-confidence_bound, confidence_bound))
        .collect();

    // Detect seasonal lags (peaks above confidence bands)
    let mut seasonal_lags = Vec::new();
    for (i, &coeff) in autocorr_coeffs.iter().enumerate().skip(config.min_period) {
        if coeff.abs() > confidence_bound {
            // Check for local maxima
            let is_local_max = i > 0 && i < autocorr_coeffs.len() - 1 &&
                coeff > autocorr_coeffs[i-1] && coeff > autocorr_coeffs[i+1];

            if is_local_max {
                seasonal_lags.push(i);
            }
        }
    }

    // Compute Box-Ljung test
    let box_ljung_test = compute_box_ljung_test(&autocorr_coeffs, n, config.alpha)?;

    Ok(AutocorrelationAnalysis {
        lags,
        autocorr_coeffs,
        confidence_bands,
        seasonal_lags,
        box_ljung_test,
    })
}

/// Detect multiple seasonalities in the data
pub fn detect_multiple_seasonalities(
    data: &[f64],
    config: &SeasonalityAnalysisConfig,
) -> Result<MultipleSeasonalityResult, Box<dyn std::error::Error>> {
    // First detect all possible seasonal periods
    let detection_result = detect_seasonality(data, config)?;

    let mut seasonal_periods = detection_result.seasonal_periods;

    // Analyze interactions between periods
    let mut interactions = Vec::new();
    for i in 0..seasonal_periods.len() {
        for j in (i+1)..seasonal_periods.len() {
            let period1 = seasonal_periods[i].period;
            let period2 = seasonal_periods[j].period;

            let interaction = analyze_seasonal_interaction(data, period1, period2)?;
            interactions.push(interaction);
        }
    }

    // Build seasonal hierarchy
    let hierarchy = build_seasonal_hierarchy(&seasonal_periods)?;

    // Sort periods by strength (descending)
    seasonal_periods.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

    Ok(MultipleSeasonalityResult {
        seasonal_periods,
        interactions,
        hierarchy,
    })
}

// Helper functions

fn fourier_to_seasonal_periods(
    fourier: &FourierAnalysis,
    config: &SeasonalityAnalysisConfig,
) -> Vec<SeasonalPeriod> {
    let mut periods = Vec::new();

    for (freq, power) in &fourier.dominant_frequencies {
        if *freq > 0.0 {
            let period = 1.0 / freq;

            if period >= config.min_period as f64 && period <= config.max_period as f64 {
                let strength = (*power / fourier.power_spectrum.iter().sum::<f64>()).min(1.0);
                let confidence = if fourier.snr > 1.0 {
                    (fourier.snr / (1.0 + fourier.snr)).min(1.0)
                } else {
                    0.5
                };

                periods.push(SeasonalPeriod {
                    period,
                    strength,
                    phase: 0.0, // Would need phase calculation
                    amplitude: power.sqrt(),
                    confidence,
                    period_type: classify_period_type(period),
                });
            }
        }
    }

    periods
}

fn periodogram_to_seasonal_periods(
    periodogram: &PeriodogramAnalysis,
    _config: &SeasonalityAnalysisConfig,
) -> Vec<SeasonalPeriod> {
    let mut periods = Vec::new();

    let max_power = periodogram.periodogram.iter().fold(0.0_f64, |a, &b| a.max(b));

    for (period, power) in &periodogram.significant_peaks {
        let strength = (*power / max_power).min(1.0);
        let confidence = if *power > periodogram.significance_threshold {
            (*power / periodogram.significance_threshold - 1.0).min(1.0)
        } else {
            0.0
        };

        periods.push(SeasonalPeriod {
            period: *period,
            strength,
            phase: 0.0,
            amplitude: power.sqrt(),
            confidence,
            period_type: classify_period_type(*period),
        });
    }

    periods
}

fn autocorr_to_seasonal_periods(
    autocorr: &AutocorrelationAnalysis,
    _config: &SeasonalityAnalysisConfig,
) -> Vec<SeasonalPeriod> {
    let mut periods = Vec::new();

    for &lag in &autocorr.seasonal_lags {
        let coeff = autocorr.autocorr_coeffs[lag];
        let strength = coeff.abs();
        let confidence = if strength > 1.96 / (autocorr.lags.len() as f64).sqrt() {
            (strength * (autocorr.lags.len() as f64).sqrt() / 1.96 - 1.0).min(1.0)
        } else {
            0.0
        };

        periods.push(SeasonalPeriod {
            period: lag as f64,
            strength,
            phase: if coeff > 0.0 { 0.0 } else { std::f64::consts::PI },
            amplitude: strength,
            confidence,
            period_type: classify_period_type(lag as f64),
        });
    }

    periods
}

fn consolidate_seasonal_periods(
    periods: Vec<SeasonalPeriod>,
    config: &SeasonalityAnalysisConfig,
) -> Result<Vec<SeasonalPeriod>, Box<dyn std::error::Error>> {
    if periods.is_empty() {
        return Ok(Vec::new());
    }

    let mut consolidated = Vec::new();
    let mut used = vec![false; periods.len()];

    // Group similar periods (within 10% of each other)
    for i in 0..periods.len() {
        if used[i] {
            continue;
        }

        let mut group = vec![i];
        for j in (i+1)..periods.len() {
            if used[j] {
                continue;
            }

            let period_diff = (periods[i].period - periods[j].period).abs();
            let relative_diff = period_diff / periods[i].period.max(periods[j].period);

            if relative_diff < 0.1 { // Within 10%
                group.push(j);
                used[j] = true;
            }
        }

        used[i] = true;

        // Create consolidated period from group
        let avg_period = group.iter().map(|&idx| periods[idx].period).sum::<f64>() / group.len() as f64;
        let max_strength = group.iter().map(|&idx| periods[idx].strength).fold(0.0, f64::max);
        let avg_confidence = group.iter().map(|&idx| periods[idx].confidence).sum::<f64>() / group.len() as f64;
        let avg_amplitude = group.iter().map(|&idx| periods[idx].amplitude).sum::<f64>() / group.len() as f64;

        // Only include periods with sufficient strength and confidence
        if max_strength > 0.1 && avg_confidence > 0.1 {
            consolidated.push(SeasonalPeriod {
                period: avg_period,
                strength: max_strength,
                phase: periods[group[0]].phase, // Use first period's phase
                amplitude: avg_amplitude,
                confidence: avg_confidence,
                period_type: classify_period_type(avg_period),
            });
        }
    }

    // Sort by strength (descending)
    consolidated.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

    // Limit to top N periods
    let max_periods = config.max_period.min(10); // Reasonable limit
    consolidated.truncate(max_periods);

    Ok(consolidated)
}

fn classify_period_type(period: f64) -> SeasonalPeriodType {
    if (period - 7.0).abs() < 1.0 {
        SeasonalPeriodType::Weekly
    } else if (period - 24.0).abs() < 2.0 {
        SeasonalPeriodType::Daily
    } else if (period - 30.0).abs() < 5.0 {
        SeasonalPeriodType::Monthly
    } else if (period - 90.0).abs() < 10.0 {
        SeasonalPeriodType::Quarterly
    } else if (period - 365.0).abs() < 20.0 {
        SeasonalPeriodType::Yearly
    } else {
        SeasonalPeriodType::Custom(period)
    }
}

fn analyze_seasonal_interaction(
    data: &[f64],
    period1: f64,
    period2: f64,
) -> Result<SeasonalInteraction, Box<dyn std::error::Error>> {
    let n = data.len();

    // Simple interaction analysis using correlation
    let mut values1 = Vec::new();
    let mut values2 = Vec::new();

    for i in 0..n {
        let phase1 = 2.0 * std::f64::consts::PI * i as f64 / period1;
        let phase2 = 2.0 * std::f64::consts::PI * i as f64 / period2;
        values1.push(phase1.sin());
        values2.push(phase2.sin());
    }

    // Calculate correlation
    let mean1 = values1.iter().sum::<f64>() / n as f64;
    let mean2 = values2.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;

    for i in 0..n {
        let diff1 = values1[i] - mean1;
        let diff2 = values2[i] - mean2;

        numerator += diff1 * diff2;
        sum1_sq += diff1 * diff1;
        sum2_sq += diff2 * diff2;
    }

    let correlation = if sum1_sq > 0.0 && sum2_sq > 0.0 {
        numerator / (sum1_sq.sqrt() * sum2_sq.sqrt())
    } else {
        0.0
    };

    let interaction_type = if correlation.abs() < 0.1 {
        InteractionType::Independent
    } else if correlation > 0.0 {
        InteractionType::Reinforcing
    } else {
        InteractionType::Interfering
    };

    Ok(SeasonalInteraction {
        period1,
        period2,
        interaction_strength: correlation.abs(),
        interaction_type,
    })
}

fn build_seasonal_hierarchy(
    periods: &[SeasonalPeriod],
) -> Result<Vec<SeasonalHierarchy>, Box<dyn std::error::Error>> {
    let mut hierarchy = Vec::new();

    // Sort periods by length (descending)
    let mut sorted_periods = periods.to_vec();
    sorted_periods.sort_by(|a, b| b.period.partial_cmp(&a.period).unwrap());

    for (i, parent) in sorted_periods.iter().enumerate() {
        let mut child_periods = Vec::new();

        // Find periods that are divisors or near-divisors of the parent
        for child in sorted_periods.iter().skip(i + 1) {
            let ratio = parent.period / child.period;

            // Check if ratio is close to an integer (indicating harmonic relationship)
            if (ratio - ratio.round()).abs() < 0.1 && ratio >= 2.0 {
                child_periods.push(child.period);
            }
        }

        if !child_periods.is_empty() {
            // Calculate hierarchy strength based on how well child periods fit
            let hierarchy_strength = child_periods.len() as f64 * parent.strength / sorted_periods.len() as f64;

            hierarchy.push(SeasonalHierarchy {
                parent_period: parent.period,
                child_periods,
                hierarchy_strength,
            });
        }
    }

    Ok(hierarchy)
}

fn compute_box_ljung_test(
    autocorr_coeffs: &[f64],
    n: usize,
    alpha: f64,
) -> Result<BoxLjungTest, Box<dyn std::error::Error>> {
    let h = autocorr_coeffs.len().min(20); // Use up to 20 lags

    if h <= 1 {
        return Err("Insufficient lags for Box-Ljung test".into());
    }

    let mut statistic = 0.0;
    for k in 1..h {
        let rk = autocorr_coeffs[k];
        statistic += rk * rk / (n - k) as f64;
    }
    statistic *= n as f64 * (n + 2) as f64;

    let degrees_of_freedom = h - 1;

    // Simplified chi-square critical value (for df=10, alpha=0.05 ≈ 18.307)
    let critical_value = match degrees_of_freedom {
        1 => if alpha <= 0.05 { 3.841 } else { 2.706 },
        2 => if alpha <= 0.05 { 5.991 } else { 4.605 },
        3 => if alpha <= 0.05 { 7.815 } else { 6.251 },
        _ => if alpha <= 0.05 {
            18.307 + (degrees_of_freedom as f64 - 10.0) * 2.0
        } else {
            15.987 + (degrees_of_freedom as f64 - 10.0) * 1.5
        },
    };

    // Approximate p-value (would need proper chi-square CDF for exact value)
    let p_value = if statistic > critical_value {
        (alpha / 2.0).min(0.001)
    } else {
        (1.0 - alpha).max(0.1)
    };

    Ok(BoxLjungTest {
        statistic,
        p_value,
        degrees_of_freedom,
        critical_value,
        is_significant: statistic > critical_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodogram_analysis() {
        // Create synthetic seasonal data
        let data: Vec<f64> = (0..100).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() +
            0.5 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
        }).collect();

        let config = SeasonalityAnalysisConfig::default();
        let result = compute_periodogram(&data, &config).unwrap();

        assert!(!result.periods.is_empty());
        assert!(!result.periodogram.is_empty());
        assert!(result.significance_threshold > 0.0);
    }

    #[test]
    fn test_autocorrelation_analysis() {
        let data: Vec<f64> = (0..50).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
        }).collect();

        let config = SeasonalityAnalysisConfig::default();
        let result = analyze_autocorrelation_seasonality(&data, &config).unwrap();

        assert_eq!(result.lags.len(), result.autocorr_coeffs.len());
        assert_eq!(result.autocorr_coeffs[0], 1.0); // Lag 0 should be 1.0
    }

    #[test]
    fn test_seasonal_period_classification() {
        assert_eq!(classify_period_type(7.0), SeasonalPeriodType::Weekly);
        assert_eq!(classify_period_type(24.0), SeasonalPeriodType::Daily);
        assert_eq!(classify_period_type(30.0), SeasonalPeriodType::Monthly);
        assert_eq!(classify_period_type(365.0), SeasonalPeriodType::Yearly);

        if let SeasonalPeriodType::Custom(p) = classify_period_type(100.0) {
            assert_eq!(p, 100.0);
        } else {
            panic!("Expected Custom period type");
        }
    }

    #[test]
    fn test_seasonality_detection() {
        let data: Vec<f64> = (0..48).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() +
            0.1 * rand::random::<f64>()
        }).collect();

        let config = SeasonalityAnalysisConfig::default();
        let result = detect_seasonality(&data, &config).unwrap();

        assert!(result.overall_seasonality >= 0.0);
        assert!(result.overall_seasonality <= 1.0);
    }
}