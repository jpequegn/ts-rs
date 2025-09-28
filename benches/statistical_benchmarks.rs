//! Statistical algorithm benchmarks for performance regression testing
//!
//! This benchmark suite measures the performance of core statistical
//! algorithms to detect performance regressions and guide optimization efforts.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use chronos::*;
use chronos::stats::*;
use chronos::trend::*;
use chrono::{DateTime, Utc, TimeZone};
use std::f64::consts::PI;

/// Generate test time series of various sizes for benchmarking
fn generate_test_series(name: &str, size: usize) -> TimeSeries {
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| Utc.timestamp(1000000000 + i as i64 * 3600, 0))
        .collect();
    let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

    TimeSeries::new(name.to_string(), timestamps, values).unwrap()
}

/// Generate sine wave time series for realistic data patterns
fn generate_sine_series(name: &str, size: usize, frequency: f64) -> TimeSeries {
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| Utc.timestamp(1000000000 + i as i64 * 3600, 0))
        .collect();
    let values: Vec<f64> = (0..size)
        .map(|i| (2.0 * PI * frequency * i as f64).sin() + 0.1 * (i as f64 * 0.01).sin())
        .collect();

    TimeSeries::new(name.to_string(), timestamps, values).unwrap()
}

/// Generate trending time series for trend analysis benchmarks
fn generate_trend_series(name: &str, size: usize, slope: f64) -> TimeSeries {
    let timestamps: Vec<DateTime<Utc>> = (0..size)
        .map(|i| Utc.timestamp(1000000000 + i as i64 * 3600, 0))
        .collect();
    let values: Vec<f64> = (0..size)
        .map(|i| slope * i as f64 + 0.1 * (i as f64 * 0.05).sin())
        .collect();

    TimeSeries::new(name.to_string(), timestamps, values).unwrap()
}

/// Benchmark descriptive statistics computation
fn bench_descriptive_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_stats");

    for size in [100, 1000, 10000, 100000].iter() {
        let ts = generate_test_series("bench", *size);

        group.bench_with_input(
            BenchmarkId::new("compute_descriptive_stats", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(compute_descriptive_stats(black_box(&ts)).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark autocorrelation computation
fn bench_autocorrelation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autocorrelation");

    for size in [100, 1000, 5000].iter() {
        let ts = generate_sine_series("bench", *size, 0.05);

        group.bench_with_input(
            BenchmarkId::new("autocorrelation_lag_10", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(compute_autocorrelation(black_box(&ts), 10).unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("autocorrelation_lag_50", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(compute_autocorrelation(black_box(&ts), 50).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark partial autocorrelation computation
fn bench_partial_autocorrelation(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_autocorrelation");

    for size in [100, 1000, 5000].iter() {
        let ts = generate_sine_series("bench", *size, 0.05);

        group.bench_with_input(
            BenchmarkId::new("pacf_lag_20", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(compute_partial_autocorrelation(black_box(&ts), 20).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark trend detection algorithms
fn bench_trend_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("trend_detection");

    for size in [100, 1000, 5000].iter() {
        let ts = generate_trend_series("bench", *size, 0.1);

        group.bench_with_input(
            BenchmarkId::new("mann_kendall", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(test_trend_significance(black_box(&ts), TrendTest::MannKendall).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark stationarity tests
fn bench_stationarity_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("stationarity_tests");

    for size in [100, 1000, 5000].iter() {
        let ts = generate_trend_series("bench", *size, 0.05);

        group.bench_with_input(
            BenchmarkId::new("adf_test", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(test_stationarity(black_box(&ts), StationarityTest::AugmentedDickeyFuller).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark time series decomposition
fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");

    for size in [100, 500, 1000].iter() {
        let ts = generate_sine_series("bench", *size, 0.1);

        group.bench_with_input(
            BenchmarkId::new("classical_additive", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(perform_decomposition(black_box(&ts), DecompositionMethod::ClassicalAdditive).unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("classical_multiplicative", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(perform_decomposition(black_box(&ts), DecompositionMethod::ClassicalMultiplicative).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark comprehensive analysis
fn bench_comprehensive_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_analysis");

    for size in [100, 500, 1000].iter() {
        let ts = generate_sine_series("bench", *size, 0.1);

        group.bench_with_input(
            BenchmarkId::new("full_analysis", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(analyze_comprehensive(black_box(&ts)).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark change point detection
fn bench_changepoint_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("changepoint_detection");

    for size in [100, 1000, 5000].iter() {
        let ts = generate_trend_series("bench", *size, 0.1);

        group.bench_with_input(
            BenchmarkId::new("detect_changepoints", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(detect_changepoints(black_box(&ts)).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark time series analysis with different data characteristics
fn bench_data_characteristics(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_characteristics");

    let size = 1000;

    // Benchmark with different data patterns
    let linear_ts = generate_test_series("linear", size);
    let sine_ts = generate_sine_series("sine", size, 0.1);
    let trend_ts = generate_trend_series("trend", size, 0.2);

    group.bench_function("linear_data_stats", |b| {
        b.iter(|| {
            black_box(compute_descriptive_stats(black_box(&linear_ts)).unwrap())
        })
    });

    group.bench_function("sine_data_stats", |b| {
        b.iter(|| {
            black_box(compute_descriptive_stats(black_box(&sine_ts)).unwrap())
        })
    });

    group.bench_function("trend_data_stats", |b| {
        b.iter(|| {
            black_box(compute_descriptive_stats(black_box(&trend_ts)).unwrap())
        })
    });

    group.finish();
}

/// Benchmark memory usage patterns for large datasets
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test with very large datasets to measure memory efficiency
    for size in [10000, 50000, 100000].iter() {
        let ts = generate_test_series("memory_test", *size);

        group.bench_with_input(
            BenchmarkId::new("large_dataset_stats", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(compute_descriptive_stats(black_box(&ts)).unwrap())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    statistical_benches,
    bench_descriptive_stats,
    bench_autocorrelation,
    bench_partial_autocorrelation,
    bench_trend_detection,
    bench_stationarity_tests,
    bench_decomposition,
    bench_comprehensive_analysis,
    bench_changepoint_detection,
    bench_data_characteristics,
    bench_memory_usage
);

criterion_main!(statistical_benches);