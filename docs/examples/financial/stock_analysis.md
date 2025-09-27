# Stock Price Analysis Example

Complete workflow for analyzing stock price data using Chronos time series analysis.

## Business Context

**Objective**: Analyze Apple Inc. (AAPL) stock performance to:
- Identify price trends and patterns
- Assess volatility and risk characteristics
- Detect unusual market events
- Generate short-term price forecasts
- Understand correlation with market indices

**Business Value**:
- Investment decision support
- Risk management and portfolio optimization
- Market timing insights
- Regulatory compliance (risk reporting)

## Dataset Description

**Data Source**: Yahoo Finance (or similar financial data provider)
**Time Period**: 2020-01-01 to 2023-12-31 (4 years)
**Frequency**: Daily
**Features**:
- Date: Trading date
- Open: Opening price
- High: Highest price during the day
- Low: Lowest price during the day
- Close: Closing price (adjusted for splits/dividends)
- Volume: Number of shares traded

**Sample Data Format**:
```csv
Date,Open,High,Low,Close,Volume
2020-01-02,74.06,75.15,73.80,75.09,135480400
2020-01-03,74.29,75.14,74.13,74.36,146322800
2020-01-06,73.45,74.99,73.19,74.95,118387200
```

## Getting Sample Data

### Option 1: Use Provided Sample Data
```bash
# Download sample AAPL data
curl -o aapl_sample.csv "https://example.com/sample_data/aapl_2020_2023.csv"
```

### Option 2: Generate Synthetic Stock Data
```bash
# Generate realistic stock data for demonstration
chronos import --generate \
  --output aapl_synthetic.csv \
  --points 1460 \
  --frequency daily \
  --start-date "2020-01-01" \
  --pattern stock \
  --base-price 100 \
  --volatility 0.25 \
  --trend 0.08
```

### Option 3: Fetch Real Data (requires API key)
```bash
# Using yfinance or similar tool (not part of Chronos)
python3 -c "
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
data.to_csv('aapl_real.csv')
"
```

## Complete Analysis Workflow

### Step 1: Data Import and Validation

```bash
# Import and validate the stock data
chronos import --file aapl_sample.csv \
  --time-column Date \
  --value-column Close \
  --frequency daily \
  --validate \
  --output aapl_processed.csv

# Check data quality
chronos stats --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --missing-value-analysis \
  --outlier-detection \
  --output data_quality.json
```

**Expected Output**:
```json
{
  "total_observations": 1460,
  "missing_values": 0,
  "outliers_detected": 12,
  "data_range": {
    "start": "2020-01-02",
    "end": "2023-12-29"
  },
  "basic_stats": {
    "mean": 155.23,
    "std_dev": 45.67,
    "min": 53.15,
    "max": 199.62
  }
}
```

### Step 2: Exploratory Data Analysis

```bash
# Create basic time series plot
chronos plot --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --type line \
  --title "AAPL Stock Price (2020-2023)" \
  --output aapl_price_chart.png

# Calculate comprehensive statistics
chronos stats --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --include-distributions \
  --confidence-level 0.95 \
  --rolling-window 30 \
  --output comprehensive_stats.json
```

**Key Insights from Statistics**:
- Average closing price: $155.23
- Standard deviation: $45.67 (29% volatility)
- Distribution: Slightly right-skewed
- 30-day rolling volatility shows clustering patterns

### Step 3: Returns Analysis

```bash
# Calculate daily returns
chronos import --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --transform log-returns \
  --output aapl_returns.csv

# Analyze return characteristics
chronos stats --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --include-distributions \
  --volatility-analysis \
  --autocorrelation \
  --output returns_analysis.json
```

**Return Analysis Results**:
```json
{
  "daily_returns": {
    "mean": 0.0008,
    "std_dev": 0.0267,
    "annualized_return": 0.208,
    "annualized_volatility": 0.425,
    "sharpe_ratio": 0.49,
    "skewness": -0.15,
    "kurtosis": 5.2
  },
  "normality_test": {
    "shapiro_wilk_p": 0.001,
    "is_normal": false
  }
}
```

### Step 4: Trend Analysis

```bash
# Comprehensive trend analysis
chronos trend --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --method comprehensive \
  --decomposition stl \
  --change-point-detection \
  --trend-significance \
  --output trend_analysis.json

# Visualize trend decomposition
chronos plot --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --type decomposition \
  --output trend_decomposition.png
```

**Trend Analysis Findings**:
```json
{
  "trend_summary": {
    "direction": "upward",
    "strength": 0.73,
    "significance": "strong",
    "change_points": ["2020-03-23", "2021-01-25", "2022-01-03"]
  },
  "decomposition": {
    "trend_contribution": 0.68,
    "seasonal_contribution": 0.12,
    "residual_contribution": 0.20
  }
}
```

### Step 5: Volatility Analysis

```bash
# GARCH-style volatility clustering analysis
chronos stats --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --volatility-analysis \
  --rolling-window 21 \
  --output volatility_analysis.json

# Plot volatility over time
chronos plot --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --type volatility \
  --output volatility_chart.png
```

**Volatility Insights**:
- High volatility periods: March 2020 (COVID crash), Late 2022 (interest rate concerns)
- Average 21-day volatility: 28%
- Volatility clustering evident (high volatility followed by high volatility)

### Step 6: Seasonality Detection

```bash
# Check for seasonal patterns
chronos seasonal --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --multiple-seasonalities \
  --periods "5,21,252" \
  --strength-analysis \
  --output seasonality_analysis.json

# Analyze day-of-week effects
chronos seasonal --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --method calendar-effects \
  --output calendar_effects.json
```

**Seasonality Results**:
```json
{
  "detected_patterns": [
    {
      "period": 252,
      "description": "Annual pattern",
      "strength": 0.34,
      "significance": "moderate"
    },
    {
      "period": 21,
      "description": "Monthly pattern",
      "strength": 0.18,
      "significance": "weak"
    }
  ],
  "calendar_effects": {
    "monday_effect": -0.12,
    "friday_effect": 0.08,
    "january_effect": 0.15
  }
}
```

### Step 7: Anomaly Detection

```bash
# Detect unusual price movements
chronos anomaly --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --method ensemble \
  --sensitivity 0.05 \
  --contextual \
  --output price_anomalies.json

# Detect extreme return events
chronos anomaly --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --method statistical \
  --threshold 3.0 \
  --output return_anomalies.json
```

**Anomaly Detection Results**:
```json
{
  "total_anomalies": 28,
  "major_events": [
    {
      "date": "2020-03-16",
      "type": "extreme_drop",
      "magnitude": -12.9,
      "context": "COVID-19 market crash"
    },
    {
      "date": "2020-03-24",
      "type": "extreme_rally",
      "magnitude": 10.4,
      "context": "Federal Reserve intervention"
    },
    {
      "date": "2022-01-28",
      "type": "earnings_surprise",
      "magnitude": 7.2,
      "context": "Q4 earnings beat"
    }
  ]
}
```

### Step 8: Risk Analysis

```bash
# Calculate Value at Risk (VaR) and Expected Shortfall
chronos stats --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --risk-metrics \
  --confidence-levels "0.95,0.99" \
  --output risk_metrics.json

# Stress testing scenarios
chronos forecast --file aapl_returns.csv \
  --time-column Date \
  --value-column log_returns \
  --method monte-carlo \
  --horizon 21 \
  --scenarios 10000 \
  --output stress_test.json
```

**Risk Metrics**:
```json
{
  "value_at_risk": {
    "95%": -0.041,
    "99%": -0.063
  },
  "expected_shortfall": {
    "95%": -0.052,
    "99%": -0.078
  },
  "maximum_drawdown": -0.357,
  "average_drawdown": -0.086
}
```

### Step 9: Market Correlation Analysis

```bash
# Download SPY (S&P 500) data for comparison
chronos import --generate \
  --output spy_data.csv \
  --points 1460 \
  --frequency daily \
  --start-date "2020-01-01" \
  --pattern market-index \
  --correlation-with aapl_processed.csv 0.85

# Analyze correlation between AAPL and market
chronos correlate --files "aapl_processed.csv,spy_data.csv" \
  --time-column Date \
  --value-column Close \
  --method pearson \
  --rolling-correlation \
  --window 60 \
  --output correlation_analysis.json

# Create correlation heatmap
chronos plot --files "aapl_processed.csv,spy_data.csv" \
  --time-column Date \
  --value-column Close \
  --type correlation_heatmap \
  --output correlation_matrix.png
```

**Correlation Analysis**:
```json
{
  "static_correlation": 0.847,
  "rolling_correlation": {
    "mean": 0.821,
    "std": 0.089,
    "min": 0.612,
    "max": 0.934
  },
  "correlation_breakdown": {
    "2020": 0.785,
    "2021": 0.834,
    "2022": 0.856,
    "2023": 0.812
  }
}
```

### Step 10: Forecasting

```bash
# Generate price forecasts
chronos forecast --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --method ensemble \
  --horizon 21 \
  --confidence-intervals \
  --confidence-level 0.95 \
  --output price_forecast.json

# Visualize forecast
chronos plot --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --type forecast \
  --forecast-file price_forecast.json \
  --output forecast_visualization.png
```

**Forecast Results**:
```json
{
  "forecast_horizon": 21,
  "method": "ensemble",
  "forecasted_values": [165.23, 166.45, 167.12, ...],
  "confidence_intervals": {
    "lower_95": [158.34, 157.21, 156.87, ...],
    "upper_95": [172.12, 175.69, 177.37, ...]
  },
  "forecast_accuracy": {
    "mape": 3.2,
    "rmse": 4.67
  }
}
```

### Step 11: Generate Comprehensive Report

```bash
# Create executive summary report
chronos report --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --template financial \
  --include-forecasts \
  --include-risk-metrics \
  --output aapl_analysis_report.html

# Generate technical analysis report
chronos report --file aapl_processed.csv \
  --time-column Date \
  --value-column Close \
  --template technical \
  --include-all-analyses \
  --output aapl_technical_report.pdf
```

## Key Findings and Interpretation

### Investment Insights

**Trend Analysis**:
- Strong upward trend over 4-year period (73% trend strength)
- Major trend changes coincided with market events (COVID, interest rates)
- Overall positive momentum with periodic corrections

**Risk Assessment**:
- Annualized volatility of 42.5% indicates high-risk, high-reward investment
- 95% VaR of -4.1% suggests potential daily losses
- Maximum drawdown of 35.7% during COVID period

**Seasonality Patterns**:
- Weak but consistent January effect (+15% average)
- Monday underperformance (-12% average)
- Year-end rally patterns observed

**Market Correlation**:
- High correlation with broader market (84.7% with S&P 500)
- Correlation varies with market conditions (61%-93% range)
- Provides limited diversification benefits

### Trading and Investment Recommendations

**For Long-term Investors**:
- Strong fundamentals support continued upward trend
- Dollar-cost averaging recommended due to volatility
- Consider position sizing based on 35% maximum drawdown potential

**For Risk Management**:
- Implement stop-loss orders at 5% below recent highs
- Monitor correlation with market during portfolio construction
- Consider volatility-based position sizing

**For Short-term Traders**:
- Watch for breakouts after consolidation periods
- Friday long/Monday short bias may provide edge
- High volatility creates opportunity but requires tight risk controls

## Automation and Monitoring

### Daily Monitoring Script

```bash
#!/bin/bash
# daily_stock_monitor.sh

# Update data
chronos import --file aapl_latest.csv --append --validate

# Check for anomalies
chronos anomaly --file aapl_latest.csv --method ensemble --alert-threshold 2.0

# Update forecasts
chronos forecast --file aapl_latest.csv --horizon 5 --output daily_forecast.json

# Generate alerts if needed
if [ $? -eq 1 ]; then
    echo "Alert: Unusual market activity detected" | mail -s "AAPL Alert" trader@example.com
fi
```

### Portfolio Integration

```python
# Python integration example
import subprocess
import json

def get_stock_metrics(symbol):
    """Get latest risk metrics for a stock"""
    result = subprocess.run([
        'chronos', 'stats',
        '--file', f'{symbol}_data.csv',
        '--risk-metrics',
        '--format', 'json'
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

# Use in portfolio optimization
aapl_metrics = get_stock_metrics('AAPL')
portfolio_weight = calculate_weight(aapl_metrics['value_at_risk'])
```

## Next Steps

1. **Expand Analysis**: Include fundamental data, news sentiment, options flow
2. **Model Enhancement**: Implement GARCH models for volatility forecasting
3. **Multi-Asset Analysis**: Extend to portfolio-level analysis
4. **Real-time Integration**: Set up live data feeds and automated alerts
5. **Backtesting**: Validate trading strategies with historical simulation

## Files Generated

- `aapl_processed.csv` - Clean price data
- `aapl_returns.csv` - Log returns
- `comprehensive_stats.json` - Statistical analysis
- `trend_analysis.json` - Trend decomposition results
- `volatility_analysis.json` - Volatility metrics
- `price_anomalies.json` - Unusual price events
- `risk_metrics.json` - VaR and risk measures
- `correlation_analysis.json` - Market correlation data
- `price_forecast.json` - 21-day price forecasts
- `aapl_analysis_report.html` - Executive summary
- Various visualization files (PNG format)

This comprehensive analysis provides a solid foundation for investment decision-making and risk management in equity markets.