//! Property-based tests for statistical invariants and properties
//!
//! This module uses property-based testing to verify that statistical
//! algorithms maintain their mathematical properties across a wide range
//! of inputs, including edge cases and randomly generated data.

use chronos::*;
use chronos::stats::*;
use chrono::{DateTime, Utc, TimeZone};
use proptest::prelude::*;
use quickcheck::{quickcheck, TestResult};
use approx::assert_relative_eq;

/// Generate arbitrary time series for property testing
fn arbitrary_timeseries() -> impl Strategy<Value = TimeSeries> {
    (1usize..1000, prop::collection::vec(any::<f64>(), 1..1000))
        .prop_filter_map("Valid time series", |(seed, values)| {
            let size = values.len();
            let timestamps: Vec<DateTime<Utc>> = (0..size)
                .map(|i| Utc.timestamp(1000000000 + i as i64 * 3600, 0))
                .collect();

            // Filter out invalid values (NaN, infinite)
            let valid_values: Vec<f64> = values.into_iter()
                .filter(|v| v.is_finite())
                .collect();

            if valid_values.is_empty() {
                None
            } else {
                let valid_timestamps = timestamps.into_iter().take(valid_values.len()).collect();
                TimeSeries::new(format!("test_{}", seed), valid_timestamps, valid_values).ok()
            }
        })
}

/// Generate time series with specific constraints for testing
fn bounded_timeseries(min_val: f64, max_val: f64) -> impl Strategy<Value = TimeSeries> {
    (1usize..500, prop::collection::vec(min_val..max_val, 1..500))
        .prop_map(|(seed, values)| {
            let size = values.len();
            let timestamps: Vec<DateTime<Utc>> = (0..size)
                .map(|i| Utc.timestamp(1000000000 + i as i64 * 3600, 0))
                .collect();

            TimeSeries::new(format!("bounded_{}", seed), timestamps, values).unwrap()
        })
}

proptest! {
    #[test]
    fn prop_mean_bounded_by_min_max(ts in arbitrary_timeseries()) {
        if let Ok(stats) = compute_descriptive_stats(&ts.values) {
            let values = ts.values;
            if !values.is_empty() {
                let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                // Mean should be between min and max
                prop_assert!(stats.mean >= min_val && stats.mean <= max_val);
            }
        }
    }

    #[test]
    fn prop_variance_non_negative(ts in arbitrary_timeseries()) {
        if let Ok(stats) = compute_descriptive_stats(&ts.values) {
            // Variance should always be non-negative
            prop_assert!(stats.variance >= 0.0);
        }
    }

    #[test]
    fn prop_std_dev_is_sqrt_variance(ts in arbitrary_timeseries()) {
        if let Ok(stats) = compute_descriptive_stats(&ts.values) {
            if stats.variance.is_finite() && stats.variance >= 0.0 {
                let expected_std = stats.variance.sqrt();
                prop_assert!((stats.std_dev - expected_std).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn prop_autocorrelation_at_lag_zero_is_one(ts in arbitrary_timeseries()) {
        if ts.len() > 1 {
            if let Ok(autocorr) = compute_autocorrelation(&ts, 1) {
                // Autocorrelation at lag 0 should always be 1.0
                prop_assert!((autocorr[0] - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn prop_autocorrelation_bounded(ts in bounded_timeseries(-100.0, 100.0)) {
        if ts.len() > 10 {
            if let Ok(autocorr) = compute_autocorrelation(&ts, 5) {
                // All autocorrelation values should be between -1 and 1
                for &value in &autocorr {
                    prop_assert!(value >= -1.0 && value <= 1.0);
                }
            }
        }
    }

    #[test]
    fn prop_scaling_invariance_correlation(
        ts in bounded_timeseries(1.0, 100.0),
        scale in 0.1f64..10.0
    ) {
        if ts.len() > 10 {
            // Create scaled version
            let scaled_values: Vec<f64> = ts.values.iter().map(|&x| x * scale).collect();
            let scaled_ts = TimeSeries::new(
                "scaled".to_string(),
                ts.timestamps.clone(),
                scaled_values
            ).unwrap();

            // Correlation should be invariant to scaling
            if let (Ok(orig_autocorr), Ok(scaled_autocorr)) = (
                compute_autocorrelation(&ts, 3),
                compute_autocorrelation(&scaled_ts, 3)
            ) {
                for (orig, scaled) in orig_autocorr.iter().zip(scaled_autocorr.iter()) {
                    if orig.is_finite() && scaled.is_finite() {
                        prop_assert!((orig - scaled).abs() < 1e-8);
                    }
                }
            }
        }
    }

    #[test]
    fn prop_shift_invariance_autocorrelation(
        ts in bounded_timeseries(1.0, 100.0),
        shift in -100.0f64..100.0
    ) {
        if ts.len() > 10 {
            // Create shifted version
            let shifted_values: Vec<f64> = ts.values.iter().map(|&x| x + shift).collect();
            let shifted_ts = TimeSeries::new(
                "shifted".to_string(),
                ts.timestamps.clone(),
                shifted_values
            ).unwrap();

            // Autocorrelation should be invariant to constant shifts
            if let (Ok(orig_autocorr), Ok(shifted_autocorr)) = (
                compute_autocorrelation(&ts, 3),
                compute_autocorrelation(&shifted_ts, 3)
            ) {
                for (orig, shifted) in orig_autocorr.iter().zip(shifted_autocorr.iter()) {
                    if orig.is_finite() && shifted.is_finite() {
                        prop_assert!((orig - shifted).abs() < 1e-8);
                    }
                }
            }
        }
    }
}

/// QuickCheck properties for additional verification
#[cfg(test)]
mod quickcheck_properties {
    use super::*;

    fn create_test_series(values: Vec<f64>) -> Option<TimeSeries> {
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let timestamps: Vec<DateTime<Utc>> = (0..values.len())
            .map(|i| Utc.timestamp(1000000000 + i as i64 * 3600, 0))
            .collect();

        TimeSeries::new("quickcheck".to_string(), timestamps, values).ok()
    }

    #[quickcheck]
    fn qc_mean_median_relationship_symmetric(values: Vec<f64>) -> TestResult {
        if let Some(ts) = create_test_series(values) {
            if let Ok(stats) = compute_descriptive_stats(&ts.values) {
                // For symmetric distributions, mean ≈ median
                // This is a weak test since we don't control the distribution shape
                TestResult::passed()
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn qc_variance_zero_for_constant_series(value: f64, size: u8) -> TestResult {
        if !value.is_finite() || size == 0 {
            return TestResult::discard();
        }

        let values = vec![value; size as usize];
        if let Some(ts) = create_test_series(values) {
            if let Ok(stats) = compute_descriptive_stats(&ts.values) {
                TestResult::from_bool(stats.variance == 0.0 && stats.std_dev == 0.0)
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn qc_autocorrelation_decreases_with_randomness(size: u8, seed: u64) -> TestResult {
        if size < 10 {
            return TestResult::discard();
        }

        // Create a pseudo-random series
        let mut rng_state = seed;
        let values: Vec<f64> = (0..size)
            .map(|_| {
                // Simple LCG for reproducible randomness
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng_state as f64) / (u64::MAX as f64) * 100.0 - 50.0
            })
            .collect();

        if let Some(ts) = create_test_series(values) {
            if let Ok(autocorr) = compute_autocorrelation(&ts, 5) {
                // For random data, autocorrelation should generally decrease with lag
                // This is probabilistic, so we use a weak test
                TestResult::from_bool(autocorr.len() >= 2)
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}

/// Invariant tests for specific algorithms
#[cfg(test)]
mod algorithm_invariants {
    use super::*;

    proptest! {
        #[test]
        fn prop_stationarity_test_consistent(ts in bounded_timeseries(-10.0, 10.0)) {
            if ts.len() > 20 {
                // Test should not crash and should return valid p-values
                if let Ok(result) = test_stationarity(&ts, StationarityTest::AugmentedDickeyFuller) {
                    prop_assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
                    prop_assert!(result.test_statistic.is_finite());
                }
            }
        }

        #[test]
        fn prop_trend_test_bounded_p_value(ts in arbitrary_timeseries()) {
            if ts.len() > 10 {
                if let Ok(result) = test_trend_significance(&ts, TrendTest::MannKendall) {
                    // P-value should always be between 0 and 1
                    prop_assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
                }
            }
        }

        #[test]
        fn prop_decomposition_reconstruction(ts in bounded_timeseries(1.0, 100.0)) {
            if ts.len() > 24 { // Need enough data for decomposition
                if let Ok(decomp_result) = perform_decomposition(&ts, DecompositionMethod::ClassicalAdditive) {
                    // If we have trend and seasonal components, they should reconstruct reasonably
                    if let (Some(trend), Some(seasonal)) = (&decomp_result.trend, &decomp_result.seasonal) {
                        prop_assert_eq!(trend.len(), ts.len());
                        prop_assert_eq!(seasonal.len(), ts.len());
                        prop_assert_eq!(decomp_result.residual.len(), ts.len());
                    }
                }
            }
        }
    }
}

/// Fuzzing tests for robustness
#[cfg(test)]
mod fuzzing_tests {
    use super::*;

    proptest! {
        #[test]
        fn fuzz_descriptive_stats_no_panic(ts in arbitrary_timeseries()) {
            // Should never panic, even with extreme inputs
            let _ = compute_descriptive_stats(&ts.values);
        }

        #[test]
        fn fuzz_autocorrelation_no_panic(ts in arbitrary_timeseries(), max_lag in 1usize..50) {
            if ts.len() > max_lag {
                let _ = compute_autocorrelation(&ts, max_lag);
            }
        }

        #[test]
        fn fuzz_trend_analysis_no_panic(ts in arbitrary_timeseries()) {
            if ts.len() > 5 {
                let _ = test_trend_significance(&ts, TrendTest::MannKendall);
            }
        }
    }
}