# Time Series Analysis Primer

A comprehensive introduction to time series analysis concepts, methods, and best practices.

## Table of Contents

1. [What is Time Series Analysis?](#what-is-time-series-analysis)
2. [Fundamental Concepts](#fundamental-concepts)
3. [Components of Time Series](#components-of-time-series)
4. [Stationarity](#stationarity)
5. [Autocorrelation and Partial Autocorrelation](#autocorrelation-and-partial-autocorrelation)
6. [Trend Analysis](#trend-analysis)
7. [Seasonality](#seasonality)
8. [Anomaly Detection](#anomaly-detection)
9. [Forecasting](#forecasting)
10. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)

## What is Time Series Analysis?

Time series analysis is a statistical technique that deals with time-ordered data. Unlike cross-sectional data where observations are independent, time series data has a temporal structure where the order of observations matters and past values can influence future ones.

### Key Characteristics

**Temporal Dependence**: Observations are not independent; they're related through time.

**Examples**:
- **Financial**: Stock prices, exchange rates, economic indicators
- **Weather**: Temperature, rainfall, atmospheric pressure
- **Business**: Sales revenue, website traffic, customer counts
- **IoT**: Sensor readings, system metrics, usage patterns
- **Healthcare**: Patient vital signs, disease outbreak patterns

### Why Time Series Analysis Matters

1. **Understanding Patterns**: Identify trends, cycles, and seasonal behaviors
2. **Forecasting**: Predict future values for planning and decision-making
3. **Anomaly Detection**: Identify unusual events or outliers
4. **Causal Analysis**: Understand relationships between different time series
5. **Process Monitoring**: Track system performance and detect issues

## Fundamental Concepts

### Time Series Components

A time series can be decomposed into several components:

```
X(t) = Trend(t) + Seasonal(t) + Cyclical(t) + Irregular(t)
```

**Visual Example**:
```
Original Series    |----|----|----|----|----|
Trend             /////////////////////// (upward slope)
Seasonal          /\/\/\/\/\/\/\/\/\/\/ (regular pattern)
Cyclical          ~~~~~~~~~~~~~~~~~~~~ (irregular waves)
Irregular         .................... (random noise)
```

### Types of Time Series

**Univariate vs. Multivariate**:
- **Univariate**: Single variable over time (e.g., daily temperature)
- **Multivariate**: Multiple variables over time (e.g., temperature, humidity, pressure)

**Discrete vs. Continuous**:
- **Discrete**: Measurements at specific time points
- **Continuous**: Continuous recording (though digitized)

**Regular vs. Irregular**:
- **Regular**: Constant time intervals (hourly, daily, monthly)
- **Irregular**: Variable time intervals (event-driven data)

### Time Series Notation

**Mathematical Notation**:
- X(t) or Xₜ: Value at time t
- t = 1, 2, ..., T: Time indices
- T: Total number of observations

**Common Operators**:
- **Lag Operator (L)**: LXₜ = Xₜ₋₁
- **Difference Operator (∇)**: ∇Xₜ = Xₜ - Xₜ₋₁
- **Backward Shift (B)**: BXₜ = Xₜ₋₁

## Components of Time Series

### Trend

The long-term direction or movement in the data.

**Types of Trends**:

**Linear Trend**: Constant rate of change
```
T(t) = α + βt
```
- **Example**: Population growth in a stable country

**Exponential Trend**: Constant percentage change
```
T(t) = α × e^(βt)
```
- **Example**: Early-stage viral spread, technology adoption

**Polynomial Trend**: Non-linear patterns
```
T(t) = α + β₁t + β₂t² + ... + βₙtⁿ
```
- **Example**: Economic cycles, product lifecycle

**No Trend**: Stationary around a constant mean
- **Example**: White noise, some economic indicators

**Identifying Trends**:
1. **Visual Inspection**: Plot the data and look for overall direction
2. **Moving Averages**: Smooth short-term fluctuations
3. **Regression Analysis**: Fit trend lines
4. **Statistical Tests**: Mann-Kendall test for monotonic trends

**Example in Chronos**:
```bash
# Detect and analyze trends
chronos trend --file data.csv --time-column date --value-column price

# Decompose to separate trend from other components
chronos trend --file data.csv --decomposition stl --output trend_analysis.json
```

### Seasonality

Regular, predictable patterns that repeat over fixed periods.

**Characteristics**:
- **Fixed Period**: Always the same length (daily, weekly, monthly, yearly)
- **Predictable**: Pattern repeats consistently
- **Calendar-Based**: Often related to calendar events

**Common Seasonal Patterns**:

**Daily Seasonality** (24-hour cycle):
- Web traffic (higher during business hours)
- Electricity demand (peaks in morning/evening)
- Temperature (daily heat cycle)

**Weekly Seasonality** (7-day cycle):
- Business sales (weekday vs. weekend patterns)
- Transportation usage (commuter patterns)
- Social media activity

**Monthly Seasonality** (30-day cycle):
- Payroll processing
- Billing cycles
- Inventory management

**Yearly Seasonality** (365-day cycle):
- Retail sales (holiday shopping)
- Tourism (seasonal destinations)
- Agricultural production
- Energy consumption (heating/cooling seasons)

**Additive vs. Multiplicative Seasonality**:

**Additive**: Seasonal fluctuations are constant over time
```
X(t) = Trend(t) + Seasonal(t) + Error(t)
```

**Multiplicative**: Seasonal fluctuations change proportionally with the level
```
X(t) = Trend(t) × Seasonal(t) × Error(t)
```

**Example in Chronos**:
```bash
# Detect seasonality patterns
chronos seasonal --file sales.csv --time-column date --value-column revenue

# Multiple seasonality detection
chronos seasonal --file hourly_data.csv --multiple-seasonalities --periods "24,168,8760"

# Seasonal adjustment
chronos seasonal --file data.csv --seasonal-adjustment --adjustment-method x13
```

### Cyclical Patterns

Longer-term fluctuations that don't have a fixed period.

**Characteristics**:
- **Variable Length**: Cycles can vary in duration
- **Economic/Business Related**: Often tied to business cycles
- **Lower Frequency**: Typically longer than seasonal patterns

**Examples**:
- Business cycles (expansion/recession)
- Real estate cycles
- Commodity price cycles
- Solar activity cycles

**Cyclical vs. Seasonal**:
| Aspect | Seasonal | Cyclical |
|--------|----------|----------|
| Duration | Fixed | Variable |
| Predictability | High | Lower |
| Frequency | Higher | Lower |
| Cause | Calendar/Weather | Economic/Social |

### Irregular Component (Noise)

Random fluctuations that can't be attributed to trend, seasonal, or cyclical patterns.

**Sources**:
- Measurement errors
- Unexpected events
- Random market movements
- Natural variability

**Types of Noise**:

**White Noise**: Independent, identically distributed random variables
```
ε(t) ~ N(0, σ²)
```

**Random Walk**: Current value depends on previous value plus noise
```
X(t) = X(t-1) + ε(t)
```

**Moving Average Noise**: Weighted average of recent noise terms
```
X(t) = ε(t) + θ₁ε(t-1) + θ₂ε(t-2) + ...
```

## Stationarity

A fundamental concept in time series analysis where statistical properties remain constant over time.

### Definition

A time series is **strictly stationary** if its statistical properties are invariant under time shifts:
```
P(X(t₁), X(t₂), ..., X(tₙ)) = P(X(t₁+τ), X(t₂+τ), ..., X(tₙ+τ))
```

A time series is **weakly stationary** (covariance stationary) if:
1. **Constant Mean**: E[X(t)] = μ for all t
2. **Constant Variance**: Var[X(t)] = σ² for all t
3. **Autocovariance depends only on lag**: Cov[X(t), X(t+k)] = γ(k)

### Why Stationarity Matters

**Statistical Inference**: Many tests and models assume stationarity
**Forecasting**: Stationary series are more predictable
**Model Fitting**: ARIMA and other models require stationarity

### Types of Non-Stationarity

**Trend Non-Stationarity**:
- Mean changes over time
- Example: Population growth, inflation

**Variance Non-Stationarity** (Heteroskedasticity):
- Variance changes over time
- Example: Financial volatility clustering

**Seasonal Non-Stationarity**:
- Seasonal patterns change over time
- Example: Changing consumer preferences

### Testing for Stationarity

**Visual Methods**:
1. **Time Plot**: Look for trends and changing variance
2. **ACF Plot**: Non-stationary series show slow decay
3. **Rolling Statistics**: Plot moving mean and variance

**Statistical Tests**:

**Augmented Dickey-Fuller (ADF) Test**:
- H₀: Series has unit root (non-stationary)
- H₁: Series is stationary
- p < 0.05 suggests stationarity

**KPSS Test**:
- H₀: Series is stationary
- H₁: Series has unit root
- p < 0.05 suggests non-stationarity

**Phillips-Perron Test**:
- Similar to ADF but robust to serial correlation
- Better for series with structural breaks

**Example in Chronos**:
```bash
# Test for stationarity
chronos stats --file data.csv --stationarity-test

# Results interpretation
{
  "adf_test": {
    "statistic": -3.45,
    "p_value": 0.01,
    "is_stationary": true
  },
  "kpss_test": {
    "statistic": 0.35,
    "p_value": 0.10,
    "is_stationary": true
  }
}
```

### Making Series Stationary

**Differencing**: Remove trends
```bash
# First difference: X(t) - X(t-1)
chronos trend --file data.csv --detrend --detrending-method difference

# Second difference: [X(t) - X(t-1)] - [X(t-1) - X(t-2)]
chronos trend --file data.csv --detrend --detrending-method difference --order 2
```

**Detrending**: Remove deterministic trends
```bash
# Linear detrending
chronos trend --file data.csv --detrend --detrending-method linear

# Polynomial detrending
chronos trend --file data.csv --detrend --detrending-method polynomial --order 2
```

**Transformation**: Stabilize variance
```bash
# Log transformation
chronos import --file data.csv --transform log

# Box-Cox transformation
chronos import --file data.csv --transform box-cox --lambda 0.5
```

**Seasonal Differencing**: Remove seasonality
```bash
# Seasonal difference with period 12 (monthly data)
chronos seasonal --file data.csv --seasonal-difference --period 12
```

## Autocorrelation and Partial Autocorrelation

### Autocorrelation Function (ACF)

Measures the linear relationship between a series and its lagged values.

**Definition**:
```
ρ(k) = Corr[X(t), X(t-k)] = γ(k) / γ(0)
```

where γ(k) is the autocovariance at lag k.

**Sample ACF**:
```
r(k) = Σ(t=k+1 to T) (X(t) - X̄)(X(t-k) - X̄) / Σ(t=1 to T) (X(t) - X̄)²
```

**Interpretation**:
- ρ(0) = 1 (perfect correlation with itself)
- -1 ≤ ρ(k) ≤ 1
- Significant correlations suggest dependence

**Patterns**:

**White Noise**: All lags ≈ 0
```
ACF: |----*----*----*----*
     0    1    2    3    4 (lag)
```

**Random Walk**: Slow, gradual decay
```
ACF: |****||||||||||||||||
     0    1    2    3    4 (lag)
```

**AR(1)**: Exponential decay
```
ACF: |****|||||||||||||||
     0    1    2    3    4 (lag)
```

**Seasonal**: Peaks at seasonal lags
```
ACF: |*---*---*---*---*
     0   12  24  36  48 (monthly data)
```

### Partial Autocorrelation Function (PACF)

Measures the correlation between X(t) and X(t-k) after removing the effects of intermediate lags.

**Definition**:
The partial autocorrelation at lag k is the correlation between X(t) and X(t-k) that is not accounted for by lags 1, 2, ..., k-1.

**Calculation**: Solve Yule-Walker equations
```
ρ(1) = φ(k,1)ρ(0) + φ(k,2)ρ(1) + ... + φ(k,k)ρ(k-1)
ρ(2) = φ(k,1)ρ(1) + φ(k,2)ρ(0) + ... + φ(k,k)ρ(k-2)
...
ρ(k) = φ(k,1)ρ(k-1) + φ(k,2)ρ(k-2) + ... + φ(k,k)ρ(0)
```

PACF(k) = φ(k,k)

**Patterns**:

**AR(p)**: Cuts off after lag p
```
PACF: |****|-------|-------
      0   1   2   3   4 (AR(1))
```

**MA(q)**: Gradually decays
```
PACF: |****||||||||||||
      0   1   2   3   4 (MA(1))
```

**Example in Chronos**:
```bash
# Compute and plot ACF/PACF
chronos stats --file data.csv --autocorrelation --partial-autocorrelation

# Visualize ACF/PACF
chronos plot --file data.csv --type acf --output acf_plot.png
chronos plot --file data.csv --type pacf --output pacf_plot.png
```

### Model Identification

Use ACF and PACF patterns to identify appropriate models:

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| AR(p) | Gradual decay | Cuts off after lag p |
| MA(q) | Cuts off after lag q | Gradual decay |
| ARMA(p,q) | Gradual decay | Gradual decay |

## Trend Analysis

### Types of Trends

**Deterministic Trend**: Predictable, mathematical function
```
X(t) = f(t) + ε(t)
```
- Can be removed by detrending
- Stationary after trend removal

**Stochastic Trend**: Random walk component
```
X(t) = X(t-1) + μ + ε(t)
```
- Requires differencing to remove
- Unit root behavior

### Trend Detection Methods

**Visual Inspection**:
```bash
# Basic time series plot
chronos plot --file data.csv --time-column date --value-column value --type line
```

**Moving Averages**:
- Smooth short-term fluctuations
- Reveal underlying trend
```bash
# Add moving average trend line
chronos stats --file data.csv --rolling-window 30 --output trend_smooth.json
```

**Regression Analysis**:
- Fit polynomial trends
- Quantify trend strength
```bash
chronos trend --file data.csv --method regression --polynomial-order 2
```

**Mann-Kendall Test**:
- Non-parametric trend test
- Robust to outliers and non-normal distributions
```bash
chronos trend --file data.csv --test-significance --method mann-kendall
```

### Trend Removal (Detrending)

**Linear Detrending**:
```
Detrended(t) = X(t) - (α + βt)
```

**Polynomial Detrending**:
```
Detrended(t) = X(t) - (α + β₁t + β₂t² + ... + βₙtⁿ)
```

**Differencing**:
```
Differenced(t) = X(t) - X(t-1)
```

**HP Filter** (Hodrick-Prescott):
- Separates trend and cyclical components
- Minimizes: Σ(X(t) - T(t))² + λΣ((T(t+1) - T(t)) - (T(t) - T(t-1)))²

**Example in Chronos**:
```bash
# Different detrending methods
chronos trend --file data.csv --detrend --detrending-method linear
chronos trend --file data.csv --detrend --detrending-method polynomial --order 3
chronos trend --file data.csv --detrend --detrending-method hp-filter --lambda 1600
chronos trend --file data.csv --detrend --detrending-method difference
```

## Seasonality

### Detecting Seasonality

**Visual Methods**:
1. **Seasonal Plots**: Overlay data by season
2. **Subseries Plots**: Separate plot for each season
3. **Autocorrelation**: Look for peaks at seasonal lags

**Statistical Methods**:

**Spectral Analysis**: Find dominant frequencies
```bash
chronos seasonal --file data.csv --method fourier --output frequency_analysis.json
```

**X-13ARIMA-SEATS**: Robust seasonal decomposition
```bash
chronos seasonal --file data.csv --method x13 --output seasonal_decomp.json
```

**STL Decomposition**: Flexible, robust method
```bash
chronos seasonal --file data.csv --method stl --output stl_decomp.json
```

### Seasonal Patterns

**Fixed Seasonality**: Constant seasonal effects
- Retail sales patterns
- Temperature cycles

**Evolving Seasonality**: Changing seasonal patterns
- Consumer preferences
- Technology adoption

**Multiple Seasonality**: Multiple seasonal cycles
- Hourly, daily, weekly, yearly patterns in electricity demand

### Seasonal Adjustment

**Purpose**: Remove seasonal effects to reveal underlying trends

**Methods**:

**X-13ARIMA-SEATS**:
- Official method for economic statistics
- Handles trading day effects and holidays
```bash
chronos seasonal --file data.csv --seasonal-adjustment --method x13
```

**STL Decomposition**:
- Robust to outliers
- Flexible seasonal pattern
```bash
chronos seasonal --file data.csv --seasonal-adjustment --method stl
```

**Moving Average**:
- Simple seasonal adjustment
- Good for stable seasonal patterns
```bash
chronos seasonal --file data.csv --seasonal-adjustment --method moving-average
```

## Anomaly Detection

### Types of Anomalies

**Point Anomalies**: Individual observations that are unusual
- Single spike in website traffic
- One-time equipment failure

**Contextual Anomalies**: Observations that are normal in one context but unusual in another
- Air conditioning usage in winter
- High sales on a typical Tuesday

**Collective Anomalies**: Collection of observations that together form an anomaly
- Gradual system degradation
- Coordinated cyber attack

### Detection Methods

**Statistical Methods**:

**Z-Score**: Based on standard deviations from mean
```
Z = |X - μ| / σ
Anomaly if Z > threshold (typically 2.5 or 3.0)
```

**Modified Z-Score**: Robust version using median
```
M = 0.6745 × |X - median| / MAD
Anomaly if M > 3.5
```

**IQR Method**: Based on quartiles
```
Anomaly if X < Q1 - 1.5×IQR or X > Q3 + 1.5×IQR
```

**Machine Learning Methods**:

**Isolation Forest**: Isolates anomalies in feature space
- Good for multivariate data
- No assumptions about data distribution

**One-Class SVM**: Learns normal behavior boundary
- Effective for high-dimensional data
- Can handle non-linear patterns

**Local Outlier Factor (LOF)**: Density-based method
- Identifies local anomalies
- Good for varying density data

**Time Series Specific**:

**Seasonal Hybrid ESD**: Accounts for seasonality
- Handles seasonal time series
- Robust to multiple anomalies

**Prophet**: Facebook's forecasting method
- Built-in anomaly detection
- Handles trends and seasonality

**Example in Chronos**:
```bash
# Statistical anomaly detection
chronos anomaly --file data.csv --method statistical --threshold 3.0

# Machine learning approach
chronos anomaly --file data.csv --method isolation-forest --sensitivity 0.05

# Ensemble method (recommended)
chronos anomaly --file data.csv --method ensemble --contextual --seasonal-adjustment
```

### Handling Anomalies

**Investigation**: Determine if anomalies are:
- Data errors (correct or remove)
- Genuine events (keep and document)
- System issues (investigate root cause)

**Treatment Options**:
1. **Remove**: Delete anomalous observations
2. **Replace**: Substitute with interpolated values
3. **Transform**: Apply robust transformations
4. **Model**: Include anomaly indicators in models

## Forecasting

### Forecasting Process

1. **Problem Definition**: What, why, when to forecast
2. **Data Collection**: Gather relevant historical data
3. **Exploratory Analysis**: Understand patterns and relationships
4. **Model Selection**: Choose appropriate forecasting method
5. **Model Fitting**: Estimate model parameters
6. **Evaluation**: Assess forecast accuracy
7. **Deployment**: Generate and monitor forecasts

### Forecasting Methods

**Naive Methods**:

**Naive Forecast**: Tomorrow = Today
```
ŷ(t+1) = y(t)
```

**Seasonal Naive**: Next period = Same period last cycle
```
ŷ(t+1) = y(t+1-m)  where m is seasonal period
```

**Drift Method**: Linear trend from first to last observation
```
ŷ(t+h) = y(t) + h × (y(t) - y(1))/(t-1)
```

**Exponential Smoothing**:

**Simple Exponential Smoothing**: Weighted average
```
ŷ(t+1) = αy(t) + (1-α)ŷ(t)
```

**Holt's Linear Trend**: Adds trend component
```
Level:    ℓ(t) = αy(t) + (1-α)(ℓ(t-1) + b(t-1))
Trend:    b(t) = β(ℓ(t) - ℓ(t-1)) + (1-β)b(t-1)
Forecast: ŷ(t+h) = ℓ(t) + hb(t)
```

**Holt-Winters**: Adds seasonal component
- Additive: ŷ(t+h) = ℓ(t) + hb(t) + s(t+h-m)
- Multiplicative: ŷ(t+h) = (ℓ(t) + hb(t))s(t+h-m)

**ARIMA Models**: Autoregressive Integrated Moving Average
- AR(p): Current value depends on p previous values
- I(d): d differences to make series stationary
- MA(q): Current value depends on q previous errors

**Example in Chronos**:
```bash
# Simple forecasting
chronos forecast --file data.csv --method naive --horizon 30

# Exponential smoothing
chronos forecast --file data.csv --method ets --horizon 30 --seasonal

# ARIMA model
chronos forecast --file data.csv --method arima --horizon 30 --auto-select

# Ensemble forecasting (recommended)
chronos forecast --file data.csv --method ensemble --horizon 30 --confidence-intervals
```

### Forecast Evaluation

**Point Forecast Accuracy**:

**Mean Absolute Error (MAE)**:
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

**Root Mean Square Error (RMSE)**:
```
RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]
```

**Mean Absolute Percentage Error (MAPE)**:
```
MAPE = (100/n) Σ|yᵢ - ŷᵢ|/|yᵢ|
```

**Symmetric MAPE (sMAPE)**:
```
sMAPE = (100/n) Σ2|yᵢ - ŷᵢ|/(|yᵢ| + |ŷᵢ|)
```

**Interval Forecast Accuracy**:
- **Coverage**: Percentage of observations within prediction intervals
- **Width**: Average width of prediction intervals
- **Skill**: Improvement over benchmark method

### Best Practices

**Cross-Validation**: Use time series cross-validation
```
Training: [1, 2, 3, 4, 5, ...]
Test:     [           6]

Training: [1, 2, 3, 4, 5, 6, ...]
Test:     [                 7]
```

**Ensemble Methods**: Combine multiple forecasts
- Often more accurate than individual methods
- Robust to model misspecification

**Forecast Uncertainty**: Always include prediction intervals
- Quantify forecast uncertainty
- Enable risk-based decision making

## Common Pitfalls and Best Practices

### Data Quality Issues

**Missing Values**:
- Check for patterns in missingness
- Choose appropriate imputation method
- Consider impact on analysis

**Outliers**:
- Investigate cause of outliers
- Use robust methods when appropriate
- Document treatment decisions

**Measurement Errors**:
- Validate data sources
- Cross-check with alternative sources
- Implement data quality checks

### Model Selection

**Overfitting**:
- Use out-of-sample validation
- Prefer simpler models
- Apply regularization techniques

**Underfitting**:
- Include relevant variables
- Consider non-linear relationships
- Test for omitted variables

**Assumptions**:
- Check model assumptions
- Use diagnostic plots
- Apply robust methods when assumptions violated

### Interpretation

**Correlation vs. Causation**:
- Correlation doesn't imply causation
- Consider confounding variables
- Use causal inference methods when needed

**Statistical vs. Practical Significance**:
- Large samples can produce significant but tiny effects
- Consider practical importance
- Report effect sizes

**Multiple Testing**:
- Adjust for multiple comparisons
- Use false discovery rate control
- Focus on pre-specified hypotheses

### Best Practices

**Start Simple**:
1. Plot the data
2. Check basic statistics
3. Test simple models first
4. Add complexity gradually

**Document Everything**:
- Record data sources and transformations
- Document modeling decisions
- Keep track of model performance

**Validate Thoroughly**:
- Use appropriate validation methods
- Test on multiple datasets
- Monitor performance over time

**Consider Context**:
- Domain knowledge is crucial
- Involve subject matter experts
- Understand business constraints

**Stay Updated**:
- Time series patterns can change
- Update models regularly
- Monitor for concept drift

### Example Workflow in Chronos

```bash
# 1. Explore the data
chronos import --file data.csv --time-column date --value-column value --validate
chronos plot --file data.csv --type line --output exploration.png

# 2. Basic analysis
chronos stats --file data.csv --include-distributions --stationarity-test

# 3. Check for patterns
chronos trend --file data.csv --test-significance
chronos seasonal --file data.csv --multiple-seasonalities

# 4. Detect anomalies
chronos anomaly --file data.csv --method ensemble --output anomalies.json

# 5. Prepare for modeling
chronos trend --file data.csv --detrend --detrending-method difference --output stationary.csv

# 6. Forecast
chronos forecast --file stationary.csv --method ensemble --horizon 30 --confidence-intervals

# 7. Generate comprehensive report
chronos report --file data.csv --template comprehensive --output final_report.html
```

This primer provides a solid foundation for understanding time series analysis. For hands-on practice, work through the [Tutorial](../user/tutorial.md) and explore the [Examples Gallery](../examples/) for domain-specific applications.