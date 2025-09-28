# Comprehensive Testing Guide

This document describes the comprehensive testing suite implemented for the Chronos time series analysis library, addressing all requirements from GitHub issue #16.

## Overview

The testing suite provides multiple layers of validation to ensure mathematical correctness, algorithmic accuracy, performance stability, and robust error handling across all components of the Chronos library.

## Testing Structure

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Verify mathematical correctness and algorithmic accuracy of individual functions.

**Coverage**:
- **Mathematical Functions** (`tests/unit/mathematical_functions.rs`)
  - Descriptive statistics (mean, variance, skewness, kurtosis)
  - Autocorrelation and partial autocorrelation
  - Trend detection algorithms (Mann-Kendall test)
  - Stationarity tests (Augmented Dickey-Fuller)
  - Numerical stability with extreme values

**Key Test Categories**:
- **Accuracy Tests**: Verify calculations match theoretical expectations
- **Edge Case Tests**: Handle single values, identical values, extreme ranges
- **Stability Tests**: Ensure numerical stability with large/small values
- **Boundary Tests**: Test algorithm boundaries and parameter limits

**Example Test**:
```rust
#[test]
fn test_mean_calculation_accuracy() {
    let ts = create_test_timeseries("test", 100);
    let stats = compute_descriptive_stats(&ts).unwrap();

    // For series 0, 1, 2, ..., 99, mean should be 49.5
    assert_relative_eq!(stats.mean, 49.5, epsilon = 1e-10);
}
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test complete workflows and end-to-end functionality.

**Coverage**:
- **CLI Workflows** (`tests/integration/cli_workflows.rs`)
  - Data import from CSV/JSON formats
  - Analysis command execution
  - Export to multiple formats
  - Error handling and recovery
  - Cross-platform path handling

**Key Test Categories**:
- **Import/Export Tests**: Full data pipeline validation
- **Analysis Workflows**: Complete analysis command execution
- **Error Handling**: Graceful failure with meaningful errors
- **Performance Tests**: Large dataset handling
- **Cross-Platform**: Path and encoding compatibility

**Example Test**:
```rust
#[test]
fn test_comprehensive_analysis_workflow() {
    let csv_content = create_test_dataset();
    let input_file = create_sample_csv(&csv_content);
    let temp_dir = TempDir::new().unwrap();

    let output = run_chronos_command(&[
        "analyze",
        "--file", input_file.path().to_str().unwrap(),
        "--comprehensive",
        "--output-dir", temp_dir.path().to_str().unwrap(),
    ]).unwrap();

    assert!(output.status.success());
}
```

### 3. Property-Based Testing (`tests/property_based/`)

**Purpose**: Verify mathematical properties and invariants hold across input ranges.

**Coverage**:
- **Statistical Properties** (`tests/property_based/statistical_properties.rs`)
  - Mean bounded by min/max values
  - Variance non-negativity
  - Autocorrelation bounded [-1, 1]
  - Scale and shift invariance
  - Statistical test p-value validity

**Key Test Categories**:
- **Invariant Tests**: Properties that must always hold
- **Scaling Tests**: Invariance to data transformations
- **Boundary Tests**: Behavior at parameter limits
- **Fuzzing Tests**: Robustness with random inputs

**Example Property Test**:
```rust
proptest! {
    #[test]
    fn prop_variance_non_negative(ts in arbitrary_timeseries()) {
        if let Ok(stats) = compute_descriptive_stats(&ts) {
            prop_assert!(stats.variance >= 0.0);
        }
    }
}
```

### 4. Benchmark Suite (`benches/`)

**Purpose**: Monitor performance and detect regressions.

**Coverage**:
- **Statistical Benchmarks** (`benches/statistical_benchmarks.rs`)
  - Descriptive statistics computation scaling
  - Autocorrelation performance across data sizes
  - Trend detection algorithm efficiency
  - Memory usage patterns

**Key Benchmark Categories**:
- **Scaling Benchmarks**: Performance vs. dataset size
- **Algorithm Benchmarks**: Relative performance of different methods
- **Memory Benchmarks**: Memory usage and efficiency
- **Real-world Benchmarks**: Performance with realistic data patterns

**Example Benchmark**:
```rust
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
}
```

### 5. Statistical Validation (`tests/statistical_validation/`)

**Purpose**: Validate against published results and reference implementations.

**Coverage**:
- **Reference Datasets** (`tests/statistical_validation/reference_datasets.rs`)
  - NIST Statistical Reference Datasets
  - Classic time series (Airline Passengers, Sunspot data)
  - Known statistical properties validation
  - Cross-package comparison benchmarks

**Key Validation Categories**:
- **NIST Datasets**: Certified statistical reference values
- **Literature Validation**: Results from academic papers
- **Cross-Package**: Comparison with R/Python implementations
- **Mathematical Validation**: Theoretical property verification

**Example Validation Test**:
```rust
#[test]
fn test_nist_norris_basic_stats() {
    let ts = create_nist_norris_dataset();
    let stats = compute_descriptive_stats(&ts).unwrap();

    // NIST certified values for the Norris dataset
    assert_relative_eq!(stats.mean, 428.75, epsilon = 1e-2);
    assert_relative_eq!(stats.std_dev, 328.2, epsilon = 1.0);
}
```

### 6. Test Data Generation (`tests/data/`)

**Purpose**: Provide comprehensive test datasets with known properties.

**Coverage**:
- **Synthetic Generators** (`tests/data/synthetic_generators.rs`)
  - White noise with configurable parameters
  - Sine waves with known frequencies
  - Linear trends with specified slopes
  - AR(1) processes with known coefficients
  - Seasonal patterns with defined periods
  - Change point series
  - Volatility clustering patterns

**Key Generator Categories**:
- **Stationary Processes**: White noise, AR processes
- **Non-stationary Processes**: Random walks, trending series
- **Seasonal Patterns**: Deterministic and stochastic seasonality
- **Complex Patterns**: Multi-component series (trend + seasonal + noise)
- **Edge Cases**: Outliers, missing values, extreme distributions

**Example Generator**:
```rust
pub fn generate_ar1_process(
    config: SyntheticConfig,
    phi: f64,
    constant: f64,
    error_variance: f64,
) -> TimeSeries {
    // Generate AR(1): x_t = c + φx_{t-1} + ε_t
    // Implementation ensures stationarity constraints
}
```

## Running Tests

### All Tests
```bash
# Run all tests (unit, integration, property-based)
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run ignored tests (comprehensive suites)
cargo test -- --ignored
```

### Specific Test Categories
```bash
# Unit tests only
cargo test --test unit

# Integration tests only
cargo test --test integration

# Property-based tests only
cargo test --test property_based

# Statistical validation tests only
cargo test --test statistical_validation
```

### Benchmarks
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench statistical_benchmarks

# Run with HTML reports
cargo bench -- --output-format html
```

### Performance Testing
```bash
# Run with release optimizations
cargo test --release

# Run with memory profiling (requires additional tools)
cargo test --release -- --test-threads=1
```

## Test Configuration

### Dependencies

The testing suite uses several specialized dependencies:

```toml
[dev-dependencies]
# Property-based testing
proptest = "1.0"
quickcheck = "1.0"

# Benchmarking
criterion = { version = "0.5", features = ["html_reports"] }

# Statistical validation
approx = "0.5"

# Test utilities
serial_test = "3.0"
test-case = "3.0"

# Mock and test doubles
mockall = "0.11"
```

### Environment Variables

```bash
# Enable additional test output
export CHRONOS_TEST_VERBOSE=1

# Set test data directory
export CHRONOS_TEST_DATA_DIR=/path/to/test/data

# Configure test timeouts
export CHRONOS_TEST_TIMEOUT=300
```

## Quality Metrics

### Test Coverage Targets
- **Unit Tests**: >95% line coverage for core algorithms
- **Integration Tests**: 100% CLI command coverage
- **Property Tests**: All critical invariants covered
- **Benchmarks**: All performance-critical functions covered

### Performance Targets
- **Descriptive Stats**: <1ms for 1K points, <100ms for 100K points
- **Autocorrelation**: <10ms for 1K points with 50 lags
- **Trend Detection**: <50ms for 5K points
- **Memory Usage**: Linear scaling with dataset size

### Validation Criteria
- **NIST Datasets**: All certified values match within tolerance
- **Literature Validation**: Results consistent with published studies
- **Cross-Package**: <1% difference from R/Python for common algorithms
- **Edge Cases**: Graceful handling without panics or invalid results

## Continuous Integration

### Test Execution Strategy
1. **Fast Tests**: Unit and property tests run on every commit
2. **Integration Tests**: Run on pull requests and releases
3. **Benchmarks**: Run nightly to detect performance regressions
4. **Validation Tests**: Run weekly against reference datasets

### Performance Regression Detection
- Benchmark results tracked over time
- Alerts on >5% performance degradation
- Memory usage monitoring and leak detection
- Scalability testing with large datasets

## Contributing to Tests

### Adding New Tests
1. **Identify Category**: Unit, integration, property-based, or validation
2. **Follow Patterns**: Use existing test structure and conventions
3. **Document Purpose**: Clear comments explaining test objectives
4. **Verify Coverage**: Ensure new functionality is tested

### Test Writing Guidelines
- Use descriptive test names that explain the scenario
- Include both positive and negative test cases
- Test edge cases and boundary conditions
- Provide clear error messages for test failures
- Use appropriate assertion types (exact vs. approximate)

### Performance Testing Guidelines
- Benchmark realistic data sizes and patterns
- Include both best-case and worst-case scenarios
- Test memory usage in addition to execution time
- Compare against previous versions to detect regressions

## Debugging Test Failures

### Common Issues
- **Floating Point Precision**: Use `approx::assert_relative_eq!` for comparisons
- **Random Data**: Use seeded generators for reproducible tests
- **Timing Issues**: Use `serial_test` for tests requiring ordered execution
- **Platform Differences**: Test on multiple platforms and handle variations

### Debugging Tools
```bash
# Run single test with detailed output
cargo test test_name -- --nocapture

# Run test under debugger
rust-gdb --args target/debug/deps/test_binary test_name

# Memory debugging with valgrind (Linux)
valgrind cargo test test_name
```

This comprehensive testing suite ensures the Chronos library maintains high quality, mathematical accuracy, and performance standards across all supported platforms and use cases.