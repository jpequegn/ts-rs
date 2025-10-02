# Chronos 📊

A powerful and efficient CLI tool for time series data analysis, visualization, and statistical computation built with Rust.

## Features

- **High-Performance Data Processing**: Built with Polars for lightning-fast data manipulation
- **Comprehensive Analysis**: Statistical analysis using robust mathematical functions
- **Beautiful Visualizations**: Generate charts and graphs with customizable styling
- **CLI-First Approach**: Intuitive command-line interface for automated workflows
- **Multiple Data Formats**: Support for CSV, JSON, and other common data formats
- **Precision Calculations**: Decimal-precise computations for financial and scientific data
- **Machine Learning Models**: Neural network forecasting with LSTM, GRU, and Transformer architectures
- **GPU Acceleration**: CUDA and Metal support for high-performance ML training

## Installation

### Prerequisites

- Rust 1.70 or higher
- Cargo (comes with Rust)

### Building from Source

```bash
git clone <repository-url>
cd chronos
cargo build --release
```

The binary will be available at `target/release/chronos`.

## Quick Start

### Analyze Time Series Data

```bash
chronos analyze --file data.csv --time-column timestamp --value-column price
```

### Generate Synthetic Data

```bash
chronos generate --points 1000 --output synthetic_data.csv
```

### Create Visualizations

```bash
chronos visualize --file data.csv --output chart.png
```

### Calculate Statistics

```bash
chronos stats --file data.csv
```

## Use Cases

### Financial Analysis
- Stock price trend analysis
- Portfolio performance tracking
- Risk assessment and volatility analysis
- Market correlation studies
- Neural network-based price forecasting

### IoT and Sensor Data
- Environmental monitoring
- Equipment performance analysis
- Anomaly detection in sensor readings
- Predictive maintenance scheduling
- ML-powered sensor value prediction

### Business Intelligence
- Sales performance tracking
- Customer behavior analysis
- Operational metrics monitoring
- Seasonal trend identification
- Deep learning demand forecasting

### Scientific Research
- Experimental data analysis
- Climate and weather pattern analysis
- Laboratory measurement processing
- Research data visualization
- Transformer-based sequence modeling

## Project Structure

```
chronos/
├── src/                    # Source code
│   └── main.rs            # Main CLI application
├── tests/                 # Unit and integration tests
├── examples/              # Usage examples
├── data/                  # Sample data files
├── Cargo.toml             # Dependencies and metadata
└── README.md              # This file
```

## Dependencies

- **clap**: Command-line argument parsing
- **polars**: High-performance data processing
- **chrono**: Date and time handling
- **statrs**: Statistical functions and distributions
- **plotters**: Chart generation and visualization
- **serde**: Serialization framework
- **csv**: CSV file parsing and writing
- **ndarray**: N-dimensional arrays for numerical computing
- **rust_decimal**: Precise decimal arithmetic
- **tabled**: Table formatting for terminal output
- **colored**: Terminal color output
- **anyhow**: Error handling
- **candle-core**: Neural network tensor operations
- **candle-nn**: Deep learning layers and modules

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source. Please see the LICENSE file for details.

## Development Status

🚧 **Early Development** - This project is in active development. Core features are being implemented.

### Roadmap

- [ ] Basic CLI structure ✅
- [ ] CSV data loading and parsing
- [ ] Statistical analysis functions
- [ ] Data visualization capabilities
- [ ] Synthetic data generation
- [ ] Advanced time series analysis
- [ ] Performance optimization
- [ ] Comprehensive testing suite
- [ ] Documentation and examples