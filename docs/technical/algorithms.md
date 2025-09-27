# Algorithm Documentation and References

Comprehensive documentation of the statistical and time series analysis algorithms implemented in Chronos.

## Table of Contents

1. [Statistical Analysis](#statistical-analysis)
2. [Time Series Decomposition](#time-series-decomposition)
3. [Trend Detection and Analysis](#trend-detection-and-analysis)
4. [Seasonality Detection](#seasonality-detection)
5. [Anomaly Detection](#anomaly-detection)
6. [Forecasting Methods](#forecasting-methods)
7. [Stationarity Testing](#stationarity-testing)
8. [Correlation Analysis](#correlation-analysis)
9. [Performance Optimization](#performance-optimization)

## Statistical Analysis

### Descriptive Statistics

**Implementation**: Standard univariate statistical measures

**Algorithms**:
- **Mean**: Arithmetic mean μ = (1/n) Σ xᵢ
- **Median**: Middle value when sorted
- **Standard Deviation**: σ = √[(1/n) Σ(xᵢ - μ)²]
- **Skewness**: γ₁ = E[(X-μ)³]/σ³
- **Kurtosis**: γ₂ = E[(X-μ)⁴]/σ⁴ - 3

**Robust Estimators**:
- **Median Absolute Deviation (MAD)**: MAD = median(|xᵢ - median(x)|)
- **Interquartile Range (IQR)**: IQR = Q₃ - Q₁

**References**:
- Hogg, R. V., McKean, J., & Craig, A. T. (2018). *Introduction to Mathematical Statistics*. 8th Edition. Pearson.
- Huber, P. J. (2004). *Robust Statistics*. John Wiley & Sons.

### Distribution Analysis

**Normality Testing**:

**Shapiro-Wilk Test**:
- **Null Hypothesis**: Data follows normal distribution
- **Test Statistic**: W = (Σ aᵢx₍ᵢ₎)² / Σ(xᵢ - x̄)²
- **Range**: Effective for n ≤ 5000

**Anderson-Darling Test**:
- **Test Statistic**: A² = -n - (1/n)Σ(2i-1)[ln F(x₍ᵢ₎) + ln(1-F(x₍ₙ₊₁₋ᵢ₎))]
- **Advantage**: More sensitive to tail behavior

**Kolmogorov-Smirnov Test**:
- **Test Statistic**: D = max|Fₙ(x) - F₀(x)|
- **Use**: General goodness-of-fit test

**References**:
- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3-4), 591-611.
- Anderson, T. W., & Darling, D. A. (1952). Asymptotic theory of certain "goodness of fit" criteria. *Annals of Mathematical Statistics*, 23(2), 193-212.

### Autocorrelation Analysis

**Sample Autocorrelation Function (ACF)**:
```
ρ̂(k) = γ̂(k) / γ̂(0)
γ̂(k) = (1/n) Σ(xₜ - μ̂)(xₜ₊ₖ - μ̂)
```

**Partial Autocorrelation Function (PACF)**:
- Derived from Yule-Walker equations
- Measures direct correlation at lag k, removing intermediate effects

**Ljung-Box Test** (Modified Box-Pierce):
```
Q = n(n+2) Σ(k=1 to h) [ρ̂²(k)/(n-k)]
```
- Tests for serial correlation in residuals
- Asymptotically χ² distributed

**References**:
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. 5th Edition. Wiley.
- Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.

## Time Series Decomposition

### STL Decomposition (Seasonal and Trend decomposition using Loess)

**Algorithm**:
1. **Detrending**: Apply loess smoother to remove trend
2. **Seasonal Extraction**: Extract seasonal component using seasonal loess
3. **Trend Estimation**: Apply loess to seasonally adjusted series
4. **Iteration**: Repeat until convergence

**Parameters**:
- `n_p`: Number of observations in each seasonal subseries
- `n_s`: Seasonal smoothing parameter
- `n_t`: Trend smoothing parameter
- `n_l`: Low-pass filter parameter

**Advantages**:
- Robust to outliers
- Handles changing seasonal patterns
- Non-parametric approach

**Implementation Details**:
```rust
// Pseudocode for STL
fn stl_decomposition(series: &[f64], period: usize) -> STLResult {
    let mut trend = vec![0.0; series.len()];
    let mut seasonal = vec![0.0; series.len()];

    for iteration in 0..max_iterations {
        // Step 1: Detrending
        let detrended = subtract_arrays(series, &trend);

        // Step 2: Seasonal smoothing
        seasonal = seasonal_loess(&detrended, period);

        // Step 3: Trend estimation
        let deseasoned = subtract_arrays(series, &seasonal);
        trend = trend_loess(&deseasoned);

        // Check convergence
        if is_converged(&trend, &seasonal) { break; }
    }

    let residual = subtract_all(series, &trend, &seasonal);
    STLResult { trend, seasonal, residual }
}
```

**References**:
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.

### X-13ARIMA-SEATS

**Components**:
1. **RegARIMA**: Regression with ARIMA errors for outlier detection
2. **X-11**: Seasonal adjustment using moving averages
3. **SEATS**: Signal Extraction in ARIMA Time Series

**X-11 Algorithm**:
- Applies symmetric and asymmetric moving averages
- Iterative process to estimate trend and seasonal components
- Handles trading day effects and holidays

**SEATS Algorithm**:
- Model-based approach using ARIMA decomposition
- Optimal signal extraction using Wiener-Kolmogorov filters
- Provides model diagnostics and forecasting capability

**References**:
- U.S. Census Bureau (2017). *X-13ARIMA-SEATS Reference Manual*. Version 1.1.
- Gómez, V., & Maravall, A. (1996). Programs TRAMO and SEATS. *Bank of Spain Working Paper 9628*.

### Classical Decomposition

**Additive Model**: X(t) = T(t) + S(t) + R(t)
**Multiplicative Model**: X(t) = T(t) × S(t) × R(t)

**Algorithm**:
1. **Trend Estimation**: Centered moving average
2. **Detrending**: X(t) - T(t) (additive) or X(t)/T(t) (multiplicative)
3. **Seasonal Estimation**: Average detrended values by season
4. **Residual**: Remainder after removing trend and seasonal

**Limitations**:
- Assumes constant seasonal pattern
- Not robust to outliers
- Edge effects at series boundaries

## Trend Detection and Analysis

### Mann-Kendall Trend Test

**Non-parametric test for monotonic trend**:

**Test Statistic**:
```
S = Σ(i=1 to n-1) Σ(j=i+1 to n) sgn(xⱼ - xᵢ)

where sgn(θ) = {
    1 if θ > 0
    0 if θ = 0
   -1 if θ < 0
}
```

**Variance** (with ties correction):
```
Var(S) = [n(n-1)(2n+5) - Σtᵢ(tᵢ-1)(2tᵢ+5)] / 18
```

**Standardized Test Statistic**:
```
Z = {
    (S-1)/√Var(S) if S > 0
    0             if S = 0
    (S+1)/√Var(S) if S < 0
}
```

**Sen's Slope Estimator**:
```
β = median{(xⱼ-xᵢ)/(j-i) : i < j}
```

**Advantages**:
- Non-parametric (no distribution assumptions)
- Robust to outliers
- Handles missing data

**References**:
- Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13(3), 245-259.
- Kendall, M. G. (1975). *Rank Correlation Methods*. 4th Edition. Charles Griffin.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63(324), 1379-1389.

### Change Point Detection

**PELT (Pruned Exact Linear Time)**:

**Cost Function**: C(y₁:τ) + βf(τ) where:
- C(y₁:τ): Cost of segment from 1 to τ
- βf(τ): Penalty for segment
- β: Penalty parameter

**Algorithm**:
1. Dynamic programming approach
2. Pruning step reduces computational complexity
3. Exact solution in O(n) expected time

**CUSUM-based Detection**:
```
Sₜ = max(0, Sₜ₋₁ + (xₜ - μ₀ - k))
```
- Detects shifts in mean level
- Control limit h triggers change point detection

**References**:
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

## Seasonality Detection

### Fourier Analysis

**Discrete Fourier Transform (DFT)**:
```
X(k) = Σ(n=0 to N-1) x(n) e^(-i2πkn/N)
```

**Power Spectral Density**:
```
PSD(k) = |X(k)|² / N
```

**Dominant Frequencies**:
- Identify peaks in power spectrum
- Convert to seasonal periods: P = N/k

**Implementation**:
```rust
fn detect_seasonality_fft(data: &[f64]) -> Vec<f64> {
    let fft_result = fft(data);
    let power_spectrum = fft_result.iter()
        .map(|c| c.norm_sqr())
        .collect::<Vec<_>>();

    let peaks = find_peaks(&power_spectrum, min_prominence);
    peaks.iter()
        .map(|&peak_idx| data.len() as f64 / peak_idx as f64)
        .collect()
}
```

**References**:
- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297-301.

### Periodogram Analysis

**Lomb-Scargle Periodogram** (for irregularly spaced data):
```
P(ω) = (1/2σ²)[{Σ(xᵢ-x̄)cos(ω(tᵢ-τ))}² / Σcos²(ω(tᵢ-τ)) + {Σ(xᵢ-x̄)sin(ω(tᵢ-τ))}² / Σsin²(ω(tᵢ-τ))]
```

where τ is defined by:
```
tan(2ωτ) = Σsin(2ωtᵢ) / Σcos(2ωtᵢ)
```

**Welch's Method** (for regularly spaced data):
- Segments data into overlapping windows
- Applies window function (Hanning, Hamming)
- Averages periodograms to reduce variance

**References**:
- Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophysics and Space Science*, 39(2), 447-462.
- Scargle, J. D. (1982). Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data. *The Astrophysical Journal*, 263, 835-853.

### Seasonal Strength Measurement

**STL-based Seasonal Strength**:
```
Fₛ = max(0, 1 - Var(R) / Var(S + R))
```
where:
- R: Residual component
- S: Seasonal component

**Interpretation**:
- Fₛ = 0: No seasonality
- Fₛ = 1: Perfect seasonality
- Fₛ > 0.64: Strong seasonality

**References**:
- Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering for time series data. *Data Mining and Knowledge Discovery*, 13(3), 335-364.

## Anomaly Detection

### Statistical Methods

**Z-Score Method**:
```
Z = |x - μ| / σ
```
- Threshold typically 2.5 or 3.0
- Assumes normal distribution

**Modified Z-Score** (robust):
```
Mᵢ = 0.6745(xᵢ - median) / MAD
```
- Uses median and MAD instead of mean and std
- More robust to outliers

**Interquartile Range (IQR) Method**:
```
Outlier if: x < Q₁ - 1.5×IQR or x > Q₃ + 1.5×IQR
```

### Isolation Forest

**Algorithm**:
1. **Tree Construction**: Randomly select feature and split value
2. **Path Length**: Count steps to isolate data point
3. **Anomaly Score**: s(x,n) = 2^(-E(h(x))/c(n))

where:
- E(h(x)): Average path length over all trees
- c(n): Average path length of unsuccessful search in BST

**Anomaly Score Interpretation**:
- Close to 1: Anomaly
- Much smaller than 0.5: Normal
- Around 0.5: Entire sample has no distinct anomalies

**Implementation**:
```rust
struct IsolationTree {
    split_feature: usize,
    split_value: f64,
    left: Option<Box<IsolationTree>>,
    right: Option<Box<IsolationTree>>,
    size: usize,
}

fn isolation_score(x: &[f64], trees: &[IsolationTree]) -> f64 {
    let avg_path_length: f64 = trees.iter()
        .map(|tree| path_length(x, tree, 0))
        .sum::<f64>() / trees.len() as f64;

    2f64.powf(-avg_path_length / c_factor(trees[0].size))
}
```

**References**:
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *IEEE 8th International Conference on Data Mining*, 413-422.

### Local Outlier Factor (LOF)

**Algorithm**:
1. **k-distance**: Distance to k-th nearest neighbor
2. **Reachability Distance**: reachₖ(A,B) = max(k-distance(B), d(A,B))
3. **Local Reachability Density**: lrdₖ(A) = 1 / (Σ reachₖ(A,B) / |Nₖ(A)|)
4. **Local Outlier Factor**: LOFₖ(A) = (Σ lrdₖ(B) / lrdₖ(A)) / |Nₖ(A)|

**Interpretation**:
- LOF ≈ 1: Normal point
- LOF >> 1: Outlier

**References**:
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. *ACM SIGMOD Record*, 29(2), 93-104.

### One-Class SVM

**Objective Function**:
```
min (1/2)||w||² + (1/νn)Σξᵢ - ρ
```

subject to:
```
⟨w,φ(xᵢ)⟩ ≥ ρ - ξᵢ, ξᵢ ≥ 0
```

**Decision Function**:
```
f(x) = sgn(⟨w,φ(x)⟩ - ρ)
```

**Kernel Functions**:
- **RBF**: K(x,y) = exp(-γ||x-y||²)
- **Polynomial**: K(x,y) = (⟨x,y⟩ + r)^d

**References**:
- Schölkopf, B., Williamson, R. C., Smola, A. J., Shawe-Taylor, J., & Platt, J. C. (2000). Support vector method for novelty detection. *Advances in Neural Information Processing Systems*, 12, 582-588.

## Forecasting Methods

### ARIMA Models

**AutoRegressive Integrated Moving Average (p,d,q)**:

**General Form**:
```
φ(B)(1-B)ᵈXₜ = θ(B)εₜ
```

where:
- φ(B): AR polynomial
- θ(B): MA polynomial
- B: Backshift operator
- d: Degree of differencing

**AR(p) Component**:
```
Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ
```

**MA(q) Component**:
```
Xₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θₑεₜ₋ₑ
```

**Parameter Estimation**:
- **Maximum Likelihood Estimation (MLE)**
- **Conditional Sum of Squares**
- **Yule-Walker Equations** (for AR models)

**Model Selection**:
- **AIC**: -2log(L) + 2k
- **BIC**: -2log(L) + k×log(n)
- **AICc**: AIC + 2k(k+1)/(n-k-1)

**Seasonal ARIMA (p,d,q)(P,D,Q)ₛ**:
```
φ(B)Φ(Bˢ)(1-B)ᵈ(1-Bˢ)ᴰXₜ = θ(B)Θ(Bˢ)εₜ
```

**References**:
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. 5th Edition. Wiley.

### Exponential Smoothing (ETS)

**Simple Exponential Smoothing**:
```
ŷₜ₊₁ = αxₜ + (1-α)ŷₜ
```

**Holt's Linear Trend**:
```
ℓₜ = αxₜ + (1-α)(ℓₜ₋₁ + bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)bₜ₋₁
ŷₜ₊ₕ = ℓₜ + hbₜ
```

**Holt-Winters Seasonal**:

**Additive**:
```
ℓₜ = α(xₜ - sₜ₋ₘ) + (1-α)(ℓₜ₋₁ + bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)bₜ₋₁
sₜ = γ(xₜ - ℓₜ) + (1-γ)sₜ₋ₘ
ŷₜ₊ₕ = ℓₜ + hbₜ + sₜ₊ₕ₋ₘ
```

**Multiplicative**:
```
ℓₜ = α(xₜ/sₜ₋ₘ) + (1-α)(ℓₜ₋₁ + bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)bₜ₋₁
sₜ = γ(xₜ/ℓₜ) + (1-γ)sₜ₋ₘ
ŷₜ₊ₕ = (ℓₜ + hbₜ)sₜ₊ₕ₋ₘ
```

**State Space Form**:
- Error: Additive (A) or Multiplicative (M)
- Trend: None (N), Additive (A), or Multiplicative (M)
- Seasonal: None (N), Additive (A), or Multiplicative (M)

**References**:
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.

### Prophet

**Decomposable Model**:
```
y(t) = g(t) + s(t) + h(t) + εₜ
```

where:
- g(t): Trend function
- s(t): Seasonal component
- h(t): Holiday effects
- εₜ: Error term

**Trend Models**:

**Linear Growth**:
```
g(t) = (k + a(t)ᵀδ)t + (m + a(t)ᵀγ)
```

**Logistic Growth**:
```
g(t) = C(t) / (1 + exp(-(k + a(t)ᵀδ)(t - (m + a(t)ᵀγ))))
```

**Seasonal Component**:
```
s(t) = Σ(n=1 to N) (aₙcos(2πnt/P) + bₙsin(2πnt/P))
```

**Holiday Effects**:
```
h(t) = Σ(i∈D(t)) κᵢ(1 + βᵢZ(t))
```

**References**:
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.

## Stationarity Testing

### Augmented Dickey-Fuller (ADF) Test

**Test Regression**:
```
Δyₜ = α + βt + γyₜ₋₁ + Σ(i=1 to p)δᵢΔyₜ₋ᵢ + εₜ
```

**Null Hypothesis**: γ = 0 (unit root exists, non-stationary)
**Alternative**: γ < 0 (stationary)

**Test Statistic**:
```
τ = γ̂ / SE(γ̂)
```

**Critical Values**: Depend on model specification (constant, trend)

**Model Selection**:
- **No constant, no trend**: Pure random walk
- **Constant only**: Random walk with drift
- **Constant and trend**: Random walk with drift and trend

**References**:
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366a), 427-431.

### Phillips-Perron (PP) Test

**Test Statistic**:
```
Z(τ) = τ√(γ₀/f₀) - (T(f₀-γ₀)(SE(β̂)))/2f₀^(1/2)
```

where:
- γ₀: Variance of error term
- f₀: Residual spectrum at frequency zero

**Advantages over ADF**:
- Robust to heteroskedasticity
- Robust to serial correlation
- Non-parametric correction

**References**:
- Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time series regression. *Biometrika*, 75(2), 335-346.

### KPSS Test

**Null Hypothesis**: Series is stationary
**Alternative**: Unit root exists

**Test Statistic**:
```
KPSS = (1/T²) Σ(t=1 to T) S²ₜ / s²(l)
```

where:
- Sₜ = Σ(i=1 to t) êᵢ (partial sum of residuals)
- s²(l): Consistent estimator of long-run variance

**Complementary to ADF/PP**:
- KPSS: H₀ = stationary
- ADF/PP: H₀ = non-stationary

**References**:
- Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1-3), 159-178.

## Correlation Analysis

### Granger Causality

**Definition**: X Granger-causes Y if past values of X help predict Y beyond what past values of Y alone can predict.

**VAR Model**:
```
Yₜ = α₀ + Σ(i=1 to p) α₁ᵢYₜ₋ᵢ + Σ(i=1 to p) α₂ᵢXₜ₋ᵢ + ε₁ₜ
Xₜ = β₀ + Σ(i=1 to p) β₁ᵢYₜ₋ᵢ + Σ(i=1 to p) β₂ᵢXₜ₋ᵢ + ε₂ₜ
```

**Test Procedure**:
1. Estimate unrestricted VAR
2. Estimate restricted VAR (H₀: α₂ᵢ = 0 ∀i)
3. F-test: F = [(RSS_r - RSS_u)/p] / [RSS_u/(T-2p-1)]

**References**:
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.

### Cointegration Testing

**Engle-Granger Approach**:
1. Test each series for unit root
2. Estimate cointegrating regression: yₜ = α + βxₜ + uₜ
3. Test residuals for stationarity

**Johansen Test**:
```
Δyₜ = Πyₜ₋₁ + Σ(i=1 to k-1) ΓᵢΔyₜ₋ᵢ + εₜ
```

where Π = αβ' (α: adjustment, β: cointegrating vectors)

**Test Statistics**:
- **Trace**: λ_trace(r) = -T Σ(i=r+1 to n) ln(1-λ̂ᵢ)
- **Maximum Eigenvalue**: λ_max(r,r+1) = -T ln(1-λ̂ᵣ₊₁)

**References**:
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
- Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.

### Dynamic Time Warping (DTW)

**Objective**: Find optimal alignment between two time series

**Distance Matrix**:
```
D(i,j) = d(xᵢ,yⱼ) + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}
```

**Constraints**:
- **Boundary**: Path must start at (1,1) and end at (m,n)
- **Continuity**: Path can only move to adjacent cells
- **Monotonicity**: Path must be non-decreasing

**Sakoe-Chiba Band**: |i-j| ≤ r (limits warping)

**References**:
- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.

## Performance Optimization

### Parallel Processing Algorithms

**Data Parallelism**:
- Split time series into chunks
- Process chunks in parallel
- Aggregate results

**Task Parallelism**:
- Execute different analysis methods simultaneously
- Use thread pools for I/O operations

**Implementation**:
```rust
use rayon::prelude::*;

fn parallel_statistical_analysis(series: &[TimeSeries]) -> Vec<StatResult> {
    series.par_iter()
        .map(|ts| compute_statistics(ts))
        .collect()
}

fn parallel_window_analysis(data: &[f64], window_size: usize) -> Vec<f64> {
    data.par_windows(window_size)
        .map(|window| window.iter().sum::<f64>() / window.len() as f64)
        .collect()
}
```

### Memory Optimization

**Streaming Algorithms**:
- **Welford's Algorithm** for online variance:
```
δ = x - μₙ₋₁
μₙ = μₙ₋₁ + δ/n
δ₂ = x - μₙ
M₂,ₙ = M₂,ₙ₋₁ + δ × δ₂
σ²ₙ = M₂,ₙ/(n-1)
```

**Reservoir Sampling** for large datasets:
```rust
fn reservoir_sample<T>(stream: impl Iterator<Item = T>, k: usize) -> Vec<T> {
    let mut reservoir = Vec::with_capacity(k);
    let mut rng = thread_rng();

    for (i, item) in stream.enumerate() {
        if i < k {
            reservoir.push(item);
        } else {
            let j = rng.gen_range(0..=i);
            if j < k {
                reservoir[j] = item;
            }
        }
    }
    reservoir
}
```

**References**:
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics*, 4(3), 419-420.
- Vitter, J. S. (1985). Random sampling with a reservoir. *ACM Transactions on Mathematical Software*, 11(1), 37-57.

### Numerical Stability

**Kahan Summation** for accurate floating-point arithmetic:
```rust
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for &value in values {
        let y = value - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}
```

**Cholesky Decomposition** for positive definite matrices:
- More stable than LU decomposition
- Used in covariance matrix computations

**QR Decomposition** for least squares:
- Numerically stable for overdetermined systems
- Used in ARIMA parameter estimation

**References**:
- Kahan, W. (1965). Pracniques: further remarks on reducing truncation errors. *Communications of the ACM*, 8(1), 40.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*. 4th Edition. Johns Hopkins University Press.

## Implementation Notes

### Numerical Considerations

**Floating Point Precision**:
- Use `f64` for statistical calculations
- Check for overflow/underflow in iterative algorithms
- Implement robust numerical methods

**Convergence Criteria**:
- Relative tolerance: |xₙ₊₁ - xₙ|/|xₙ| < ε
- Absolute tolerance: |xₙ₊₁ - xₙ| < ε
- Maximum iterations to prevent infinite loops

**Edge Cases**:
- Empty datasets
- Single-point series
- All identical values
- Missing data patterns

### Testing and Validation

**Unit Tests**:
- Known analytical solutions
- Synthetic data with known properties
- Edge cases and boundary conditions

**Property-Based Testing**:
- Statistical properties (mean, variance)
- Invariant relationships
- Consistency across implementations

**Benchmarking**:
- Compare with established libraries (R, Python)
- Performance testing with large datasets
- Memory usage profiling

This comprehensive algorithm documentation provides the theoretical foundation for all analysis methods implemented in Chronos, enabling users to understand the mathematical basis of their analyses and make informed decisions about method selection.