# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Chronos** is a comprehensive CLI tool and library for time series data analysis, visualization, and statistical computation built with Rust. It provides a dual interface: both a powerful command-line tool (`chronos`) and a reusable library (`chronos` crate).

## Essential Development Commands

### Building and Running
```bash
# Build the CLI binary
cargo build --bin chronos

# Build in release mode
cargo build --release --bin chronos

# Run the CLI tool
cargo run --bin chronos -- [subcommand] [args]

# Run with help
cargo run --bin chronos -- --help
```

### Testing
```bash
# Build library (binary builds successfully, but library tests have compilation issues)
cargo build --lib

# Run benchmarks (currently working)
cargo bench

# Note: `cargo test` currently fails due to compilation errors in test modules
# Tests need fixing - see test compilation issues below
```

### Debugging and Analysis
```bash
# Run with debug output
RUST_BACKTRACE=1 cargo run --bin chronos

# Debug with lldb (for stack overflow issues)
lldb target/debug/chronos

# Check compilation warnings
cargo check
```

## Architecture Overview

### Core Data Structures

**TimeSeries (`src/timeseries.rs`)**
- Central data structure containing timestamps (`Vec<DateTime<Utc>>`) and values (`Vec<f64>`)
- Includes metadata, frequency, and missing value policies
- Validated construction with duplicate timestamp handling

**Configuration System (`src/config/`)**
- **CRITICAL**: Fixed circular dependency in `src/config/defaults.rs` (stack overflow bug)
- Supports multiple profiles (general, finance, IoT, weather) with TOML/YAML/JSON formats
- Uses figment for configuration loading with environment variable support

### Module Organization

**Analysis Modules:**
- `src/stats/` - Descriptive statistics, stationarity tests, change point detection
- `src/trend/` - Trend detection, decomposition (STL, classical), detrending methods
- `src/seasonality/` - Seasonality detection, calendar effects, adjustment methods
- `src/anomaly/` - Multiple detection methods (z-score, IQR, isolation forest)
- `src/forecasting/` - ARIMA, exponential smoothing, ensemble methods
- `src/correlation/` - Basic correlation, cross-correlation, PCA, DTW, Granger causality
- `src/ml/` - Neural network forecasting (LSTM, GRU, Transformer) with GPU acceleration

**Infrastructure Modules:**
- `src/cli/` - Comprehensive CLI with 11 subcommands (stats, trend, seasonal, etc.)
- `src/performance/` - Parallel processing, caching, memory management, progress tracking
- `src/plotting/` - Plotly-based visualization engine with export capabilities
- `src/reporting/` - Template-based reporting system (executive, technical, data quality)
- `src/plugins/` - Plugin system for extensibility (data sources, analysis, visualization)

### CLI Command Structure

The CLI provides 11 main subcommands:
- `import` - Data import with preprocessing
- `stats` - Statistical analysis
- `trend` - Trend detection and decomposition
- `seasonal` - Seasonality analysis
- `anomaly` - Anomaly detection
- `forecast` - Time series forecasting
- `correlate` - Correlation analysis
- `plot` - Visualization generation
- `report` - Comprehensive reporting
- `config` - Configuration management
- `plugin` - Plugin management

Each command supports multiple output formats (text, JSON, CSV, markdown, HTML, PDF).

## Testing Structure

### Test Organization
- `tests/unit/` - Unit tests for mathematical functions (currently 22/25 passing)
- `tests/property_based/` - Property-based tests using proptest and quickcheck
- `tests/statistical_validation/` - Tests against reference datasets (NIST, Box-Jenkins)
- `tests/integration/` - CLI workflow tests
- `benches/` - Performance benchmarks for statistical algorithms

### Known Testing Issues
**CRITICAL**: Test compilation currently fails due to:
1. Import path issues (`chronos_time_series` should be `chronos`)
2. Function signature mismatches in statistical functions
3. Missing dependencies and deprecated API usage

The comprehensive testing suite was recently implemented but needs compilation fixes.

## Key Implementation Details

### Statistical Algorithms
- Core statistics use `statrs` crate with custom implementations for time series specific functions
- Trend detection includes Mann-Kendall tests, linear regression, seasonal decomposition
- Anomaly detection supports multiple methods with ensemble scoring
- All algorithms designed for numerical stability with extreme values

### Machine Learning Module (`src/ml/`)

**Architecture**:
- `src/ml/types.rs` - Core ML types (NeuralNetwork, Layer, TrainingConfig, Device)
- `src/ml/tensor.rs` - Tensor operations with GPU acceleration via Candle
- `src/ml/activations.rs` - Activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)
- `src/ml/persistence.rs` - Model serialization and versioning
- `src/ml/recurrent.rs` - LSTM and GRU networks for sequence forecasting
- `src/ml/transformer.rs` - Transformer architecture with self-attention

**Key Features**:
- **LSTM/GRU**: Recurrent networks for sequential modeling with multi-layer stacking
- **Transformer**: Self-attention mechanism for long-range dependencies
- **GPU Support**: Automatic device selection (CUDA, Metal, CPU) with fallback
- **Normalization**: Z-score and min-max scaling with automatic denormalization
- **Training**: Validation monitoring, early stopping, training history tracking
- **Model Persistence**: Save/load models in Binary or JSON format with versioning
- **Interpretability**: Attention analysis for Transformer models

**Usage Example**:
```rust
use chronos::ml::{LSTMConfig, create_lstm_forecaster, forecast_with_lstm};

let config = LSTMConfig {
    hidden_size: 64,
    num_layers: 2,
    sequence_length: 10,
    dropout_rate: 0.2,
    ..Default::default()
};

let forecaster = create_lstm_forecaster(&config, &time_series)?;
let forecast = forecast_with_lstm(&forecaster, &input_sequence, 5)?;
```

**Dependencies**:
- `candle-core` 0.9 - Tensor operations and GPU support
- `candle-nn` 0.9 - Neural network layers and modules

**Documentation**: See `docs/ml/` for comprehensive guides:
- `docs/ml/README.md` - ML module overview and quick start
- `docs/ml/lstm_gru_guide.md` - LSTM/GRU usage guide
- `docs/ml/transformer_guide.md` - Transformer architecture guide

### Performance Features
- Parallel processing using `rayon` for large datasets
- Memory management with lazy loading and caching systems
- Progress tracking for long-running operations
- Database storage support via `rusqlite`
- GPU acceleration for ML training (5-10x speedup over CPU)

### Configuration System Bug Fix
**Recently Fixed**: Stack overflow in configuration defaults caused by circular dependency where `Config::default()` → `ProfilesConfig::default()` → `create_general_profile()` → `Config::default()`. Fixed by introducing `create_base_config()` function.

## Development Notes

### Known Compilation Warnings
The codebase currently generates ~57 warnings, mainly unused variables and dead code. These are non-blocking but should be addressed for code quality.

### Future Dependencies
Code comments indicate planned migration to Polars for high-performance data processing, currently disabled for Rust version compatibility.

### Plugin Architecture
Extensible plugin system supports three types:
- Data source plugins
- Analysis plugins
- Visualization plugins

Registry-based management with repository support for plugin distribution.

## Library Usage

When using as a library, key exports are available from the crate root:
```rust
use chronos::{TimeSeries, compute_descriptive_stats, detect_trend, analyze_comprehensive};
```

The library provides over 100 public functions across all analysis domains, making it suitable for embedding in other Rust applications.