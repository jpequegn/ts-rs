# Installation and Setup Guide

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Rust**: Version 1.70 or higher
- **Memory**: At least 4GB RAM recommended (8GB+ for large datasets)
- **Disk Space**: 500MB for installation, additional space for data files

### Required Dependencies

#### Rust Installation

If you don't have Rust installed:

```bash
# Install Rust using rustup (recommended)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### System Libraries

**On macOS:**
```bash
# Install via Homebrew
brew install pkg-config openssl
```

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential pkg-config libssl-dev
```

**On CentOS/RHEL/Fedora:**
```bash
sudo dnf install gcc pkg-config openssl-devel
# or for older versions:
# sudo yum install gcc pkg-config openssl-devel
```

## Installation Methods

### Method 1: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/jpequegn/ts-rs.git
cd ts-rs

# Build the release version
cargo build --release

# The binary will be available at target/release/chronos
./target/release/chronos --help
```

### Method 2: Install from Cargo

```bash
# Install directly from the repository
cargo install --git https://github.com/jpequegn/ts-rs chronos

# Verify installation
chronos --version
```

### Method 3: Development Build

```bash
# For development and testing
git clone https://github.com/jpequegn/ts-rs.git
cd ts-rs
cargo build

# Use the debug version
./target/debug/chronos --help
```

## Post-Installation Setup

### 1. Add to PATH (Optional)

To use `chronos` from anywhere:

```bash
# Copy binary to a directory in your PATH
sudo cp target/release/chronos /usr/local/bin/

# Or add the project's target/release directory to your PATH
echo 'export PATH="$PATH:$(pwd)/target/release"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create Configuration Directory

```bash
# Create default configuration
chronos config init

# This creates:
# ~/.config/chronos/config.toml  (Linux/macOS)
# %APPDATA%\chronos\config.toml  (Windows)
```

### 3. Verify Installation

```bash
# Test with sample data
chronos import --help
chronos stats --help

# Check version and build info
chronos --version
```

## Configuration

### Basic Configuration

Create a basic configuration file:

```bash
# Initialize with defaults
chronos config init

# View current configuration
chronos config show

# Set your preferred output directory
chronos config set output.default_directory ~/chronos-analysis

# Set default output format
chronos config set output.default_format json
```

### Advanced Configuration

Edit the configuration file directly:

```bash
# Open configuration file
$EDITOR ~/.config/chronos/config.toml
```

Example configuration:

```toml
[metadata]
active_profile = "default"
version = "0.1.0"

[analysis]
default_confidence_level = 0.95
auto_detect_frequency = true
handle_missing_values = "interpolate"

[visualization]
default_theme = "dark"
default_width = 1200
default_height = 800
output_format = "png"

[output]
default_directory = "~/chronos-analysis"
default_format = "json"
timestamp_format = "ISO8601"

[performance]
parallel_processing = true
max_threads = 0  # 0 = auto-detect
memory_limit_mb = 2048
enable_caching = true
```

## Troubleshooting Installation

### Common Issues

#### Issue: Compilation fails with OpenSSL errors
```bash
# Solution: Install OpenSSL development libraries
# macOS:
brew install openssl
export OPENSSL_DIR=/usr/local/opt/openssl

# Ubuntu/Debian:
sudo apt install libssl-dev

# CentOS/RHEL:
sudo dnf install openssl-devel
```

#### Issue: "no default toolchain" error
```bash
# Solution: Set default Rust toolchain
rustup default stable
```

#### Issue: Permission denied when running binary
```bash
# Solution: Make binary executable
chmod +x target/release/chronos
```

#### Issue: Command not found after installation
```bash
# Solution: Check PATH
which chronos
echo $PATH

# Add to PATH if needed
export PATH="$PATH:$(pwd)/target/release"
```

### Performance Optimization

#### Memory Settings
```bash
# For large datasets, increase memory limit
chronos config set performance.memory_limit_mb 4096

# Enable memory mapping for very large files
chronos config set performance.use_memory_mapping true
```

#### Parallel Processing
```bash
# Set optimal thread count (usually CPU cores)
chronos config set performance.max_threads 8

# Enable parallel processing
chronos config set performance.parallel_processing true
```

## Updating

### Update from Source
```bash
cd ts-rs
git pull origin main
cargo build --release
```

### Update via Cargo
```bash
cargo install --git https://github.com/jpequegn/ts-rs chronos --force
```

## Uninstallation

### Remove Binary
```bash
# If installed via cargo
cargo uninstall chronos

# If copied to system PATH
sudo rm /usr/local/bin/chronos
```

### Remove Configuration
```bash
# Remove configuration directory
rm -rf ~/.config/chronos/
```

## Next Steps

After successful installation:

1. Read the [Tutorial](tutorial.md) for hands-on examples
2. Check the [Command Reference](command_reference.md) for all available commands
3. Browse the [Examples Gallery](../examples/) for real-world use cases
4. Join our community for support and discussions

## Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Review [Troubleshooting Guide](troubleshooting.md)
3. Search existing [GitHub Issues](https://github.com/jpequegn/ts-rs/issues)
4. Create a new issue with:
   - Your operating system and version
   - Rust version (`rustc --version`)
   - Complete error message
   - Steps to reproduce the issue