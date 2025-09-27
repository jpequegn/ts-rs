# Troubleshooting Guide

Detailed solutions for common technical issues when using Chronos.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Import Problems](#data-import-problems)
3. [Memory and Performance Issues](#memory-and-performance-issues)
4. [Analysis Errors](#analysis-errors)
5. [Configuration Problems](#configuration-problems)
6. [Visualization Issues](#visualization-issues)
7. [Platform-Specific Issues](#platform-specific-issues)

## Installation Issues

### Issue: Rust compilation fails with OpenSSL errors

**Symptoms:**
```
error: failed to run custom build command for `openssl-sys`
could not find system library 'openssl' required by the 'openssl-sys' crate
```

**Solutions:**

**macOS:**
```bash
# Install OpenSSL via Homebrew
brew install openssl pkg-config

# Set environment variables
export OPENSSL_DIR=/opt/homebrew/opt/openssl
export PKG_CONFIG_PATH=/opt/homebrew/opt/openssl/lib/pkgconfig

# Rebuild
cargo clean && cargo build --release
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libssl-dev pkg-config build-essential
cargo clean && cargo build --release
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install openssl-devel pkg-config gcc
# or for older versions:
# sudo yum install openssl-devel pkg-config gcc
cargo clean && cargo build --release
```

### Issue: Linker errors during compilation

**Symptoms:**
```
error: linking with `cc` failed: exit code: 1
ld: library not found for -lSystem
```

**Solutions:**

**macOS with Xcode Command Line Tools:**
```bash
# Install/update Xcode command line tools
xcode-select --install

# If still failing, reinstall
sudo xcode-select --reset
xcode-select --install
```

**Alternative macOS solution:**
```bash
# Use different linker
export RUSTFLAGS="-C link-arg=-fuse-ld=lld"
cargo build --release
```

### Issue: "no default toolchain" error

**Symptoms:**
```
error: no default toolchain configured
```

**Solution:**
```bash
# Install and set default toolchain
rustup toolchain install stable
rustup default stable

# Verify
rustc --version
cargo --version
```

### Issue: Permission denied when running binary

**Symptoms:**
```bash
./target/release/chronos
-bash: ./target/release/chronos: Permission denied
```

**Solution:**
```bash
# Make binary executable
chmod +x target/release/chronos

# For system-wide installation
sudo cp target/release/chronos /usr/local/bin/
sudo chmod +x /usr/local/bin/chronos
```

## Data Import Problems

### Issue: CSV parsing fails with encoding errors

**Symptoms:**
```
Error: Invalid UTF-8 sequence
Error parsing CSV: invalid UTF-8 in field
```

**Solutions:**

1. **Check file encoding:**
```bash
# Check current encoding
file -I your_data.csv
# or
chardet your_data.csv
```

2. **Convert to UTF-8:**
```bash
# Using iconv (Linux/macOS)
iconv -f ISO-8859-1 -t UTF-8 your_data.csv > data_utf8.csv

# Using dos2unix for Windows line endings
dos2unix data_utf8.csv
```

3. **Use import options:**
```bash
chronos import --file data.csv --encoding utf-8 --skip-errors
```

### Issue: Timestamp parsing fails

**Symptoms:**
```
Error: Could not parse timestamp: '2023-01-01 10:00:00'
Error: Invalid timestamp format
```

**Diagnostic steps:**

1. **Examine your timestamps:**
```bash
# Look at first few rows
head -5 your_data.csv

# Check for consistency
cut -d',' -f1 your_data.csv | head -20
```

2. **Common timestamp format issues:**

| Issue | Example | Solution |
|-------|---------|----------|
| Mixed formats | `2023-01-01`, `01/01/2023` | Standardize format |
| Missing timezone | `2023-01-01 10:00:00` | Add timezone info |
| Excel date numbers | `44927` | Convert to proper format |
| Microseconds | `2023-01-01 10:00:00.123456` | Specify precision |

3. **Solutions:**

**Specify custom time format:**
```bash
chronos import --file data.csv \
  --time-column timestamp \
  --time-format "%d/%m/%Y %H:%M:%S" \
  --timezone "UTC"
```

**Common time format patterns:**
```bash
# US format: MM/DD/YYYY HH:MM:SS
--time-format "%m/%d/%Y %H:%M:%S"

# European format: DD.MM.YYYY HH:MM:SS
--time-format "%d.%m.%Y %H:%M:%S"

# ISO format with microseconds: YYYY-MM-DD HH:MM:SS.fff
--time-format "%Y-%m-%d %H:%M:%S.%f"

# Unix timestamp
--time-format "unix"
```

### Issue: Unexpected data types in columns

**Symptoms:**
```
Error: Could not parse value '1,234.56' as number
Error: Invalid numeric value
```

**Solutions:**

1. **Check for formatting issues:**
```bash
# Look for common issues
grep -E '[a-zA-Z]' your_data.csv | head -5  # Letters in numeric columns
grep -E ',' your_data.csv | head -5         # Thousands separators
```

2. **Clean data before import:**
```bash
# Remove thousands separators
sed 's/,//g' your_data.csv > cleaned_data.csv

# Handle European decimal separators
sed 's/,/./g' your_data.csv > cleaned_data.csv
```

3. **Use data cleaning options:**
```bash
chronos import --file data.csv \
  --clean-numeric \
  --remove-thousands-separator \
  --decimal-separator "."
```

### Issue: Memory errors with large files

**Symptoms:**
```
Error: Out of memory
thread 'main' panicked at 'memory allocation failed'
```

**Solutions:**

1. **Use streaming mode:**
```bash
chronos import --file large_data.csv \
  --streaming \
  --chunk-size 10000 \
  --output processed_data.csv
```

2. **Increase memory limits:**
```bash
chronos config set performance.memory_limit_mb 8192
chronos config set performance.use_memory_mapping true
```

3. **Split large files:**
```bash
# Split into smaller files (1M lines each)
split -l 1000000 large_data.csv chunk_

# Process each chunk
for chunk in chunk_*; do
    chronos import --file "$chunk" --output "processed_${chunk}.csv"
done

# Combine results if needed
cat processed_chunk_* > final_processed.csv
```

## Memory and Performance Issues

### Issue: Slow performance on large datasets

**Diagnostic steps:**

1. **Profile performance:**
```bash
# Enable performance monitoring
chronos config set performance.enable_profiling true

# Run with timing
time chronos stats --file large_data.csv --output stats.json
```

2. **Check system resources:**
```bash
# Monitor during execution
htop  # or top
iostat 1  # Check I/O
```

**Optimization strategies:**

1. **Enable parallel processing:**
```bash
chronos config set performance.parallel_processing true
chronos config set performance.max_threads $(nproc)  # Linux
chronos config set performance.max_threads $(sysctl -n hw.ncpu)  # macOS
```

2. **Optimize memory usage:**
```bash
chronos config set performance.memory_limit_mb 4096
chronos config set performance.enable_caching true
chronos config set performance.cache_size_mb 1024
```

3. **Use appropriate algorithms:**
```bash
# For very large datasets, use streaming algorithms
chronos stats --file large_data.csv --streaming --algorithm fast

# For forecasting, use simpler methods
chronos forecast --file large_data.csv --method linear_regression
```

### Issue: Process killed due to memory usage

**Symptoms:**
```
Killed
signal: killed
```

**Solutions:**

1. **Check system memory:**
```bash
# Linux
free -h
cat /proc/meminfo | grep MemAvailable

# macOS
vm_stat
memory_pressure
```

2. **Reduce memory footprint:**
```bash
# Use minimal analysis
chronos stats --file data.csv --minimal-output

# Process subsets
chronos stats --file data.csv --sample-rate 0.1  # 10% sample

# Use lower precision
chronos config set performance.precision_mode low
```

3. **System-level solutions:**
```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Limit other processes
# Close unnecessary applications
# Use dedicated analysis machine
```

## Analysis Errors

### Issue: Numerical instability in calculations

**Symptoms:**
```
Error: Matrix is singular
Warning: Numerical precision lost
Error: Algorithm did not converge
```

**Solutions:**

1. **Check data quality:**
```bash
chronos stats --file data.csv --missing-value-analysis --outlier-detection
```

2. **Preprocess data:**
```bash
# Remove outliers
chronos import --file data.csv --remove-outliers --outlier-method iqr

# Scale data
chronos import --file data.csv --normalize --normalization-method zscore

# Handle missing values
chronos import --file data.csv --missing-value-policy interpolate
```

3. **Use robust algorithms:**
```bash
# Use robust estimation methods
chronos stats --file data.csv --robust-estimation

# Use ensemble methods for forecasting
chronos forecast --file data.csv --method ensemble
```

### Issue: Seasonality detection fails

**Symptoms:**
```
Warning: No significant seasonality detected
Error: Seasonal decomposition failed
```

**Diagnostic steps:**

1. **Check data length:**
```bash
# Need at least 2 full seasonal cycles
wc -l data.csv
# For daily data detecting weekly seasonality: need >14 days
# For hourly data detecting daily seasonality: need >48 hours
```

2. **Verify data frequency:**
```bash
chronos import --file data.csv --frequency auto --validate-frequency
```

**Solutions:**

1. **Increase sensitivity:**
```bash
chronos seasonal --file data.csv --sensitivity 0.01 --method fourier
```

2. **Specify expected periods:**
```bash
chronos seasonal --file data.csv --periods "7,30,365" --force-periods
```

3. **Try different methods:**
```bash
# Try multiple detection methods
chronos seasonal --file data.csv --method ensemble
```

### Issue: Forecasting produces unrealistic results

**Symptoms:**
- Forecasts are negative when data should be positive
- Massive confidence intervals
- Forecasts don't follow obvious patterns

**Solutions:**

1. **Check model assumptions:**
```bash
# Test stationarity
chronos stats --file data.csv --stationarity-test

# Check residuals
chronos forecast --file data.csv --method arima --validate-residuals
```

2. **Preprocess data:**
```bash
# Transform to ensure positivity (for positive data)
chronos import --file data.csv --transform log

# Make stationary
chronos trend --file data.csv --detrend --detrending-method difference
```

3. **Use appropriate constraints:**
```bash
# Constrain forecasts to be positive
chronos forecast --file data.csv --positive-constraint

# Use bounded forecasting
chronos forecast --file data.csv --bounds "0,1000"
```

## Configuration Problems

### Issue: Configuration file not found or invalid

**Symptoms:**
```
Error: Could not load configuration file
Error: Invalid configuration format
```

**Solutions:**

1. **Initialize configuration:**
```bash
chronos config init
```

2. **Validate configuration:**
```bash
chronos config validate --verbose
```

3. **Reset to defaults:**
```bash
# Backup current config
cp ~/.config/chronos/config.toml ~/.config/chronos/config.toml.backup

# Regenerate
rm ~/.config/chronos/config.toml
chronos config init
```

4. **Check file permissions:**
```bash
ls -la ~/.config/chronos/config.toml
# Should be readable by user
chmod 644 ~/.config/chronos/config.toml
```

### Issue: Settings not taking effect

**Symptoms:**
- Configuration changes ignored
- Default values used despite custom settings

**Diagnostic steps:**

1. **Check which config file is being used:**
```bash
chronos --verbose config show
```

2. **Verify setting syntax:**
```bash
chronos config validate
```

**Solutions:**

1. **Use explicit config path:**
```bash
chronos --config /path/to/config.toml stats --file data.csv
```

2. **Check setting hierarchy:**
```
Command line options > Environment variables > Config file > Defaults
```

3. **Verify setting names:**
```bash
# List all available settings
chronos config list-settings
```

## Visualization Issues

### Issue: Plots are not generated or look wrong

**Symptoms:**
```
Error: Could not create plot
Warning: Plot export failed
Empty or corrupted image files
```

**Solutions:**

1. **Check output format support:**
```bash
# Test different formats
chronos plot --file data.csv --output test.png --format png
chronos plot --file data.csv --output test.svg --format svg
```

2. **Verify data is suitable for plotting:**
```bash
# Check data range and values
chronos stats --file data.csv --basic-stats
```

3. **Use explicit dimensions:**
```bash
chronos plot --file data.csv \
  --output plot.png \
  --width 1200 \
  --height 800 \
  --dpi 300
```

### Issue: Interactive plots don't work

**Solutions:**

1. **Use HTML format for interactivity:**
```bash
chronos plot --file data.csv --output interactive.html --format html
```

2. **Check browser compatibility:**
- Modern browsers support JavaScript
- Enable JavaScript if disabled
- Try different browsers

## Platform-Specific Issues

### macOS Issues

**Issue: "chronos" cannot be opened because the developer cannot be verified**

**Solution:**
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine target/release/chronos

# Or allow in System Preferences > Security & Privacy
```

**Issue: M1/M2 compatibility issues**

**Solution:**
```bash
# Ensure you're using the right architecture
rustup target add aarch64-apple-darwin
cargo build --release --target aarch64-apple-darwin
```

### Windows Issues

**Issue: PowerShell execution policy**

**Solution:**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue: Path separator issues**

**Solution:**
```bash
# Use forward slashes or escape backslashes
chronos import --file "C:/data/file.csv"
# or
chronos import --file "C:\\data\\file.csv"
```

### Linux Issues

**Issue: Missing GLIBC version**

**Solution:**
```bash
# Update system
sudo apt update && sudo apt upgrade

# Or build with older GLIBC target
RUSTFLAGS="-C target-feature=-crt-static" cargo build --release
```

## Getting Additional Help

### Enabling Debug Mode

For detailed troubleshooting information:

```bash
# Maximum verbosity
chronos --verbose stats --file data.csv

# Enable debug logging
RUST_LOG=debug chronos stats --file data.csv

# Enable trace logging (very detailed)
RUST_LOG=trace chronos stats --file data.csv 2> debug.log
```

### Creating Minimal Reproducible Examples

When reporting issues:

1. **Create small test dataset:**
```bash
# Generate simple test data
chronos import --generate --points 100 --output test_data.csv
```

2. **Test with minimal command:**
```bash
chronos stats --file test_data.csv --output test_stats.json
```

3. **Include system information:**
```bash
chronos --version
rustc --version
uname -a  # Linux/macOS
systeminfo  # Windows
```

### Community Resources

- **GitHub Issues**: https://github.com/jpequegn/ts-rs/issues
- **Documentation**: Complete documentation in `docs/` directory
- **Examples**: Working examples in `examples/` directory

### Professional Support

For enterprise users or complex issues:
- Priority support available
- Custom training and consulting
- On-site deployment assistance

Contact: support@chronos-analytics.com