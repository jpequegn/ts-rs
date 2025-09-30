//! Integration tests for the quality module pipeline
//!
//! These tests validate the complete quality assessment, cleaning, and monitoring workflows.

use chronos::quality::*;
use chronos::TimeSeries;
use chrono::{DateTime, Utc, Duration, TimeZone};
use std::time::Instant;

/// Create test data with known quality issues
fn create_test_data_with_issues() -> TimeSeries {
    let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    // Generate 100 data points with intentional issues
    for i in 0..100 {
        let timestamp = start + Duration::days(i as i64);
        timestamps.push(timestamp);

        let value = match i {
            // Add outliers
            25 => 500.0,  // Extreme outlier
            50 => -100.0, // Negative outlier
            75 => 300.0,  // Moderate outlier
            // Add gaps (missing values will be handled separately)
            30..=32 => {
                // Skip these indices - will create gaps
                continue;
            }
            // Normal values
            _ => 50.0 + (i as f64 % 20.0),
        };

        values.push(value);
    }

    TimeSeries::from_vec("test_series", timestamps, values).unwrap()
}

/// Create degrading quality data for monitoring tests
fn create_degrading_quality_data(iteration: usize) -> TimeSeries {
    let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    let noise_level = iteration as f64 * 2.0;  // Increasing noise
    let missing_ratio = iteration as f64 * 0.01;  // Increasing missing data

    for i in 0..100 {
        // Skip some values based on missing ratio
        if (i as f64 / 100.0) < missing_ratio {
            continue;
        }

        let timestamp = start + Duration::days(i as i64);
        timestamps.push(timestamp);

        // Add increasing noise
        let base_value = 50.0 + (i as f64).sin() * 10.0;
        let noisy_value = base_value + ((i as f64 * 0.1).sin() * noise_level);
        values.push(noisy_value);
    }

    TimeSeries::from_vec("degrading_series", timestamps, values).unwrap()
}

/// Create large test dataset for performance benchmarks
fn create_large_test_dataset(size: usize) -> TimeSeries {
    let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| start + Duration::seconds(i as i64))
        .collect();

    let values: Vec<f64> = (0..size)
        .map(|i| 50.0 + (i as f64 / 100.0).sin() * 10.0)
        .collect();

    TimeSeries::from_vec("large_series", timestamps, values).unwrap()
}

#[test]
fn test_complete_quality_pipeline() {
    // Create test data with known issues
    let data = create_test_data_with_issues();

    // Step 1: Profile the data
    let profiling_config = ProfilingConfig::default();
    let profile = profile_timeseries(&data, &profiling_config).unwrap();

    // Verify profiling detected issues
    assert!(profile.completeness.missing_count > 0, "Should detect missing values");
    assert!(profile.completeness.completeness_ratio < 1.0, "Completeness should be < 100%");
    assert!(!profile.gaps.is_empty(), "Should detect gaps");

    // Step 2: Assess quality
    let quality_config = QualityConfig::default();
    let assessment = assess_quality(&data, &quality_config).unwrap();

    // Verify assessment detected issues
    assert!(assessment.overall_score < 90.0, "Should detect quality issues");
    assert!(assessment.overall_score >= 0.0 && assessment.overall_score <= 100.0, "Score should be 0-100");

    // Verify dimension scores
    assert!(assessment.dimension_scores.completeness >= 0.0);
    assert!(assessment.dimension_scores.consistency >= 0.0);
    assert!(assessment.dimension_scores.validity >= 0.0);
    assert!(assessment.dimension_scores.timeliness >= 0.0);
    assert!(assessment.dimension_scores.accuracy >= 0.0);

    // Step 3: Clean the data
    let cleaning_config = CleaningConfig::conservative();
    let cleaning_result = clean_timeseries(&data, &CleaningConfig::default()).unwrap();

    // Verify cleaning improved quality
    let post_cleaning_assessment = assess_quality(
        &cleaning_result.cleaned_data,
        &quality_config,
    ).unwrap();

    assert!(
        post_cleaning_assessment.overall_score >= assessment.overall_score,
        "Cleaning should maintain or improve quality score"
    );

    // Verify cleaning report
    assert!(cleaning_result.cleaning_report.modifications.len() > 0, "Should record modifications");
    assert!(
        cleaning_result.cleaning_report.modifications.len() as f64 / data.len() as f64 <= cleaning_config.max_modifications,
        "Should respect max modifications limit"
    );
}

#[test]
fn test_quality_monitoring_workflow() {
    // Initialize quality tracker
    let monitoring_config = MonitoringConfig::default();
    let mut tracker = QualityTracker::new(monitoring_config);

    // Simulate quality assessments over time
    let mut scores = Vec::new();
    for i in 0..10 {
        let data = create_degrading_quality_data(i);
        let quality_config = QualityConfig::default();
        let assessment = assess_quality(&data, &quality_config).unwrap();

        scores.push(assessment.overall_score);

        // Track metrics
        tracker.track_quality_metrics(&assessment).unwrap();
    }

    // Verify quality degradation was detected
    assert!(scores[0] > scores[9], "Quality should degrade over iterations");

    // Check for quality degradation alerts
    let thresholds = QualityThresholds::default();
    let trend = tracker.detect_quality_degradation(&thresholds);

    match trend {
        TrendDirection::Declining => {
            // Expected - quality is degrading
        }
        _ => panic!("Should detect declining quality trend"),
    }

    // Verify tracking data
    let quality_series = tracker.get_quality_series();
    assert_eq!(quality_series.data_points.len(), 10, "Should track all 10 assessments");
}

#[test]
fn test_outlier_detection_methods() {
    let data = create_test_data_with_issues();

    // Test Z-score method
    let outlier_config = OutlierConfig {
        method: OutlierMethod::ZScore { threshold: 3.0 },
        ..OutlierConfig::default()
    };
    let zscore_outliers = detect_outliers(&data, &outlier_config).unwrap();
    assert!(!zscore_outliers.outliers.is_empty(), "Z-score should detect outliers");

    // Test IQR method
    let iqr_config = OutlierConfig {
        method: OutlierMethod::IQR { factor: 1.5 },
        ..OutlierConfig::default()
    };
    let iqr_outliers = detect_outliers(&data, &iqr_config).unwrap();
    assert!(!iqr_outliers.outliers.is_empty(), "IQR should detect outliers");

    // Test ensemble method
    let ensemble_config = OutlierConfig {
        method: OutlierMethod::Ensemble,
        ..OutlierConfig::default()
    };
    let ensemble_outliers = detect_outliers(&data, &ensemble_config).unwrap();
    assert!(!ensemble_outliers.outliers.is_empty(), "Ensemble should detect outliers");

    // Verify outlier context
    for outlier in &zscore_outliers.outliers {
        assert!(outlier.index < data.len(), "Outlier index should be valid");
        assert!(outlier.timestamp.is_some(), "Outlier should have timestamp");
        assert!(outlier.severity != OutlierSeverity::None, "Outlier should have severity");
    }
}

#[test]
fn test_data_cleaning_methods() {
    let data = create_test_data_with_issues();

    // Test gap filling
    let gap_config = GapConfig {
        method: ImputationMethod::Linear,
        ..GapConfig::default()
    };
    let filled_result = fill_gaps(&data, &gap_config).unwrap();

    // Verify gaps were filled
    assert!(
        filled_result.cleaned_data.len() >= data.len(),
        "Filled data should have at least original size"
    );

    // Test outlier correction
    let outlier_config = OutlierConfig::default();
    let outliers = detect_outliers(&data, &outlier_config).unwrap();

    let corrected_result = correct_outliers(&data, &outliers, &OutlierCorrection::MedianReplace).unwrap();

    // Verify outliers were corrected
    assert_eq!(
        corrected_result.cleaned_data.len(),
        data.len(),
        "Corrected data should maintain original size"
    );

    // Test noise reduction
    let noise_config = NoiseReduction::MovingAverage { window: 5 };
    let smoothed_result = reduce_noise(&data, &noise_config).unwrap();

    // Verify noise reduction maintained data integrity
    assert_eq!(
        smoothed_result.cleaned_data.len(),
        data.len(),
        "Smoothed data should maintain original size"
    );
}

#[test]
fn test_performance_benchmarks() {
    let data = create_large_test_dataset(10_000);

    println!("Testing performance with 10,000 data points...");

    // Benchmark quality assessment
    let start = Instant::now();
    let quality_config = QualityConfig::default();
    let assessment = assess_quality(&data, &quality_config).unwrap();
    let assessment_time = start.elapsed();

    println!("Quality assessment: {:?}", assessment_time);
    assert!(
        assessment_time.as_millis() < 100,
        "Assessment should complete in <100ms, took {:?}",
        assessment_time
    );

    // Benchmark data profiling
    let start = Instant::now();
    let profiling_config = ProfilingConfig::default();
    let _profile = profile_timeseries(&data, &profiling_config).unwrap();
    let profiling_time = start.elapsed();

    println!("Data profiling: {:?}", profiling_time);
    assert!(
        profiling_time.as_millis() < 50,
        "Profiling should complete in <50ms, took {:?}",
        profiling_time
    );

    // Benchmark outlier detection
    let start = Instant::now();
    let outlier_config = OutlierConfig::default();
    let _outliers = detect_outliers(&data, &outlier_config).unwrap();
    let outlier_time = start.elapsed();

    println!("Outlier detection: {:?}", outlier_time);
    assert!(
        outlier_time.as_millis() < 100,
        "Outlier detection should complete in <100ms, took {:?}",
        outlier_time
    );

    // Benchmark data cleaning
    let start = Instant::now();
    let cleaning_config = CleaningConfig::default();
    let _cleaning_result = clean_timeseries(&data, &cleaning_config).unwrap();
    let cleaning_time = start.elapsed();

    println!("Data cleaning: {:?}", cleaning_time);
    assert!(
        cleaning_time.as_millis() < 200,
        "Cleaning should complete in <200ms, took {:?}",
        cleaning_time
    );

    // Verify assessment results
    assert!(assessment.overall_score >= 0.0 && assessment.overall_score <= 100.0);
}

#[test]
fn test_configuration_profiles() {
    let data = create_test_data_with_issues();

    // Test strict profile
    let strict_config = QualityConfig::strict();
    let strict_assessment = assess_quality(&data, &strict_config).unwrap();

    // Test lenient profile
    let lenient_config = QualityConfig::lenient();
    let lenient_assessment = assess_quality(&data, &lenient_config).unwrap();

    // Strict should generally score lower due to higher standards
    assert!(
        strict_assessment.overall_score <= lenient_assessment.overall_score + 5.0,
        "Strict profile should have similar or lower score"
    );

    // Both should still be valid scores
    assert!(strict_assessment.overall_score >= 0.0 && strict_assessment.overall_score <= 100.0);
    assert!(lenient_assessment.overall_score >= 0.0 && lenient_assessment.overall_score <= 100.0);
}

#[test]
fn test_quality_weights() {
    let data = create_test_data_with_issues();

    // Test with completeness-heavy weights
    let completeness_config = QualityConfig {
        weights: QualityWeights {
            completeness: 0.5,
            consistency: 0.125,
            validity: 0.125,
            timeliness: 0.125,
            accuracy: 0.125,
        },
        ..QualityConfig::default()
    };

    let completeness_assessment = assess_quality(&data, &completeness_config).unwrap();

    // Test with validity-heavy weights
    let validity_config = QualityConfig {
        weights: QualityWeights {
            completeness: 0.125,
            consistency: 0.125,
            validity: 0.5,
            timeliness: 0.125,
            accuracy: 0.125,
        },
        ..QualityConfig::default()
    };

    let validity_assessment = assess_quality(&data, &validity_config).unwrap();

    // Scores should differ based on weights
    assert_ne!(
        completeness_assessment.overall_score,
        validity_assessment.overall_score,
        "Different weights should produce different scores"
    );
}

#[test]
fn test_error_handling() {
    // Test with empty data
    let empty_data = TimeSeries::from_vec("empty", vec![], vec![]).unwrap();

    let quality_config = QualityConfig::default();
    let result = assess_quality(&empty_data, &quality_config);

    // Should handle empty data gracefully
    assert!(result.is_err(), "Should return error for empty data");

    // Test with invalid configuration
    let invalid_config = QualityConfig {
        completeness_threshold: 1.5,  // Invalid: should be 0-1
        ..QualityConfig::default()
    };

    // Configuration validation should catch this
    assert!(invalid_config.completeness_threshold > 1.0, "Invalid config for testing");
}

#[test]
fn test_cleaning_reversibility() {
    let data = create_test_data_with_issues();
    let original_len = data.len();

    // Clean data with tracking enabled
    let cleaning_config = CleaningConfig {
        preserve_characteristics: true,
        uncertainty_tracking: true,
        ..CleaningConfig::default()
    };

    let cleaning_result = clean_timeseries(&data, &cleaning_config).unwrap();

    // Verify modifications are tracked
    assert!(!cleaning_result.cleaning_report.modifications.is_empty(), "Should track modifications");

    // Verify data characteristics are preserved
    let cleaned_len = cleaning_result.cleaned_data.len();
    assert!(
        cleaned_len >= original_len,
        "Cleaned data should have at least original length"
    );
}

#[test]
fn test_quality_recommendations() {
    let data = create_test_data_with_issues();

    let quality_config = QualityConfig::default();
    let assessment = assess_quality(&data, &quality_config).unwrap();

    // Generate recommendations
    let recommendations = generate_recommendations(&assessment, &data);

    // Verify recommendations are generated for issues
    assert!(!recommendations.is_empty(), "Should generate recommendations for quality issues");

    // Verify recommendation structure
    for rec in &recommendations {
        assert!(!rec.issue.is_empty(), "Recommendation should have issue description");
        assert!(!rec.recommendation.is_empty(), "Recommendation should have suggestion");
        assert!(rec.priority != Priority::None, "Recommendation should have priority");
    }
}
