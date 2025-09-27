# Scientific References and Bibliography

Comprehensive bibliography of scientific papers, books, and resources that form the theoretical foundation of Chronos time series analysis methods.

## Table of Contents

1. [Foundational Time Series Analysis](#foundational-time-series-analysis)
2. [Statistical Methods](#statistical-methods)
3. [Trend Analysis and Decomposition](#trend-analysis-and-decomposition)
4. [Seasonality and Periodicity](#seasonality-and-periodicity)
5. [Stationarity and Unit Root Testing](#stationarity-and-unit-root-testing)
6. [Forecasting Methods](#forecasting-methods)
7. [Anomaly Detection](#anomaly-detection)
8. [Correlation and Causality](#correlation-and-causality)
9. [Machine Learning Applications](#machine-learning-applications)
10. [Financial Time Series](#financial-time-series)
11. [Software and Computational Methods](#software-and-computational-methods)

## Foundational Time Series Analysis

### Classic Textbooks

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015)**
*Time Series Analysis: Forecasting and Control* (5th Edition)
John Wiley & Sons
**DOI**: 10.1002/9781118619193
**Note**: The definitive reference for ARIMA modeling and time series methodology.

**Hamilton, J. D. (1994)**
*Time Series Analysis*
Princeton University Press
**ISBN**: 978-0691042893
**Note**: Comprehensive mathematical treatment of time series econometrics.

**Brockwell, P. J., & Davis, R. A. (2016)**
*Introduction to Time Series and Forecasting* (3rd Edition)
Springer
**DOI**: 10.1007/978-3-319-29854-2
**Note**: Excellent balance of theory and practical applications.

**Shumway, R. H., & Stoffer, D. S. (2017)**
*Time Series Analysis and Its Applications: With R Examples* (4th Edition)
Springer
**DOI**: 10.1007/978-3-319-52452-8
**Note**: Strong focus on spectral analysis and state space models.

**Chatfield, C. (2003)**
*The Analysis of Time Series: An Introduction* (6th Edition)
Chapman and Hall/CRC
**ISBN**: 978-1584883173
**Note**: Accessible introduction with practical focus.

### Survey Papers

**De Gooijer, J. G., & Hyndman, R. J. (2006)**
"25 years of time series forecasting"
*International Journal of Forecasting*, 22(3), 443-473
**DOI**: 10.1016/j.ijforecast.2006.01.001
**Note**: Comprehensive survey of forecasting methods and their evolution.

**Hyndman, R. J., & Athanasopoulos, G. (2021)**
*Forecasting: Principles and Practice* (3rd Edition)
OTexts
**URL**: https://otexts.com/fpp3/
**Note**: Modern, practical approach to forecasting with R examples.

## Statistical Methods

### Descriptive Statistics

**Hogg, R. V., McKean, J., & Craig, A. T. (2018)**
*Introduction to Mathematical Statistics* (8th Edition)
Pearson
**ISBN**: 978-0134686998
**Note**: Foundation for statistical inference and hypothesis testing.

**Huber, P. J. (2004)**
*Robust Statistics*
John Wiley & Sons
**DOI**: 10.1002/9780470434697
**Note**: Theory and methods for robust statistical estimation.

### Distribution Analysis and Testing

**Shapiro, S. S., & Wilk, M. B. (1965)**
"An analysis of variance test for normality (complete samples)"
*Biometrika*, 52(3-4), 591-611
**DOI**: 10.1093/biomet/52.3-4.591
**Note**: Original paper introducing the Shapiro-Wilk normality test.

**Anderson, T. W., & Darling, D. A. (1952)**
"Asymptotic theory of certain 'goodness of fit' criteria based on stochastic processes"
*Annals of Mathematical Statistics*, 23(2), 193-212
**DOI**: 10.1214/aoms/1177729437
**Note**: Foundation for the Anderson-Darling goodness-of-fit test.

**Kolmogorov, A. (1933)**
"Sulla determinazione empirica di una legge di distribuzione"
*Giornale dell'Istituto Italiano degli Attuari*, 4, 83-91
**Note**: Original Kolmogorov-Smirnov test development.

### Autocorrelation Analysis

**Ljung, G. M., & Box, G. E. P. (1978)**
"On a measure of lack of fit in time series models"
*Biometrika*, 65(2), 297-303
**DOI**: 10.1093/biomet/65.2.297
**Note**: Modified Box-Pierce test for serial correlation.

**Bartlett, M. S. (1946)**
"On the theoretical specification and sampling properties of autocorrelated time-series"
*Supplement to the Journal of the Royal Statistical Society*, 8(1), 27-41
**DOI**: 10.2307/2983611
**Note**: Theoretical foundation for autocorrelation function estimation.

## Trend Analysis and Decomposition

### STL Decomposition

**Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990)**
"STL: A seasonal-trend decomposition procedure based on loess"
*Journal of Official Statistics*, 6(1), 3-73
**Note**: Seminal paper introducing STL decomposition methodology.

**Cleveland, W. S. (1979)**
"Robust locally weighted regression and smoothing scatterplots"
*Journal of the American Statistical Association*, 74(368), 829-836
**DOI**: 10.1080/01621459.1979.10481038
**Note**: Foundation of LOESS smoothing used in STL.

### X-13ARIMA-SEATS

**U.S. Census Bureau (2017)**
*X-13ARIMA-SEATS Reference Manual* (Version 1.1)
U.S. Census Bureau
**URL**: https://www.census.gov/ts/x13as/docX13AS.pdf
**Note**: Official documentation for X-13ARIMA-SEATS methodology.

**Gómez, V., & Maravall, A. (1996)**
"Programs TRAMO and SEATS"
*Bank of Spain Working Paper 9628*
**Note**: Original SEATS methodology for model-based seasonal adjustment.

**Maravall, A. (1995)**
"Unobserved components in economic time series"
In *Handbook of Applied Econometrics*, H. Pesaran & M. Wickens (Eds.)
Blackwell Publishers
**Note**: Theoretical foundation of unobserved components models.

### Change Point Detection

**Killick, R., Fearnhead, P., & Eckley, I. A. (2012)**
"Optimal detection of changepoints with a linear computational cost"
*Journal of the American Statistical Association*, 107(500), 1590-1598
**DOI**: 10.1080/01621459.2012.737745
**Note**: PELT algorithm for efficient change point detection.

**Page, E. S. (1954)**
"Continuous inspection schemes"
*Biometrika*, 41(1/2), 100-115
**DOI**: 10.2307/2333009
**Note**: Original CUSUM methodology for change detection.

**Lavielle, M. (2005)**
"Using penalized contrasts for the change-point problem"
*Signal Processing*, 85(8), 1501-1510
**DOI**: 10.1016/j.sigpro.2005.01.012
**Note**: Penalized likelihood approaches to change point detection.

### Trend Testing

**Mann, H. B. (1945)**
"Nonparametric tests against trend"
*Econometrica*, 13(3), 245-259
**DOI**: 10.2307/1907187
**Note**: Original Mann-Kendall trend test.

**Kendall, M. G. (1975)**
*Rank Correlation Methods* (4th Edition)
Charles Griffin
**Note**: Comprehensive treatment of rank-based correlation methods.

**Sen, P. K. (1968)**
"Estimates of the regression coefficient based on Kendall's tau"
*Journal of the American Statistical Association*, 63(324), 1379-1389
**DOI**: 10.1080/01621459.1968.10480934
**Note**: Sen's slope estimator for trend magnitude.

## Seasonality and Periodicity

### Fourier Analysis

**Cooley, J. W., & Tukey, J. W. (1965)**
"An algorithm for the machine calculation of complex Fourier series"
*Mathematics of Computation*, 19(90), 297-301
**DOI**: 10.2307/2003354
**Note**: Fast Fourier Transform algorithm revolutionizing spectral analysis.

**Welch, P. (1967)**
"The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms"
*IEEE Transactions on Audio and Electroacoustics*, 15(2), 70-73
**DOI**: 10.1109/TAU.1967.1161901
**Note**: Welch's method for spectral density estimation.

### Periodogram Analysis

**Lomb, N. R. (1976)**
"Least-squares frequency analysis of unequally spaced data"
*Astrophysics and Space Science*, 39(2), 447-462
**DOI**: 10.1007/BF00648343
**Note**: Lomb-Scargle periodogram for irregularly sampled data.

**Scargle, J. D. (1982)**
"Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data"
*The Astrophysical Journal*, 263, 835-853
**DOI**: 10.1086/160554
**Note**: Statistical properties of the Lomb-Scargle periodogram.

### Seasonal Strength Measurement

**Wang, X., Smith, K. A., & Hyndman, R. J. (2006)**
"Characteristic-based clustering for time series data"
*Data Mining and Knowledge Discovery*, 13(3), 335-364
**DOI**: 10.1007/s10618-005-0039-x
**Note**: Defines seasonal strength measures used in time series analysis.

## Stationarity and Unit Root Testing

### Augmented Dickey-Fuller Test

**Dickey, D. A., & Fuller, W. A. (1979)**
"Distribution of the estimators for autoregressive time series with a unit root"
*Journal of the American Statistical Association*, 74(366a), 427-431
**DOI**: 10.1080/01621459.1979.10482531
**Note**: Foundational paper for unit root testing.

**Dickey, D. A., & Fuller, W. A. (1981)**
"Likelihood ratio statistics for autoregressive time series with a unit root"
*Econometrica*, 49(4), 1057-1072
**DOI**: 10.2307/1912517
**Note**: Extended ADF test with trend and constant terms.

**Said, S. E., & Dickey, D. A. (1984)**
"Testing for unit roots in autoregressive-moving average models of unknown order"
*Biometrika*, 71(3), 599-607
**DOI**: 10.1093/biomet/71.3.599
**Note**: Augmented Dickey-Fuller test for ARMA processes.

### Phillips-Perron Test

**Phillips, P. C., & Perron, P. (1988)**
"Testing for a unit root in time series regression"
*Biometrika*, 75(2), 335-346
**DOI**: 10.1093/biomet/75.2.335
**Note**: Non-parametric unit root test robust to serial correlation.

### KPSS Test

**Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992)**
"Testing the null hypothesis of stationarity against the alternative of a unit root"
*Journal of Econometrics*, 54(1-3), 159-178
**DOI**: 10.1016/0304-4076(92)90104-Y
**Note**: Stationarity test complementary to unit root tests.

## Forecasting Methods

### ARIMA Models

**Akaike, H. (1974)**
"A new look at the statistical model identification"
*IEEE Transactions on Automatic Control*, 19(6), 716-723
**DOI**: 10.1109/TAC.1974.1100705
**Note**: Akaike Information Criterion for model selection.

**Schwarz, G. (1978)**
"Estimating the dimension of a model"
*Annals of Statistics*, 6(2), 461-464
**DOI**: 10.1214/aos/1176344136
**Note**: Bayesian Information Criterion for model selection.

### Exponential Smoothing

**Holt, C. C. (2004)**
"Forecasting seasonals and trends by exponentially weighted moving averages"
*International Journal of Forecasting*, 20(1), 5-10
**DOI**: 10.1016/j.ijforecast.2003.09.015
**Note**: Reprint of original 1957 paper on Holt's linear trend method.

**Winters, P. R. (1960)**
"Forecasting sales by exponentially weighted moving averages"
*Management Science*, 6(3), 324-342
**DOI**: 10.1287/mnsc.6.3.324
**Note**: Holt-Winters method for seasonal time series.

**Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008)**
*Forecasting with Exponential Smoothing: The State Space Approach*
Springer
**DOI**: 10.1007/978-3-540-71918-2
**Note**: Comprehensive treatment of exponential smoothing methods.

### State Space Models

**Durbin, J., & Koopman, S. J. (2012)**
*Time Series Analysis by State Space Methods* (2nd Edition)
Oxford University Press
**ISBN**: 978-0199641178
**Note**: Definitive reference for state space methodology.

**Harvey, A. C. (1989)**
*Forecasting, Structural Time Series Models and the Kalman Filter*
Cambridge University Press
**ISBN**: 978-0521405737
**Note**: Classic text on structural time series models.

### Prophet

**Taylor, S. J., & Letham, B. (2018)**
"Forecasting at scale"
*The American Statistician*, 72(1), 37-45
**DOI**: 10.1080/00031305.2017.1380080
**Note**: Facebook's Prophet forecasting methodology.

### Vector Autoregression

**Sims, C. A. (1980)**
"Macroeconomics and reality"
*Econometrica*, 48(1), 1-48
**DOI**: 10.2307/1912017
**Note**: Influential paper introducing VAR methodology to economics.

**Lütkepohl, H. (2005)**
*New Introduction to Multiple Time Series Analysis*
Springer
**DOI**: 10.1007/978-3-540-27752-1
**Note**: Comprehensive treatment of multivariate time series methods.

## Anomaly Detection

### Statistical Methods

**Tukey, J. W. (1977)**
*Exploratory Data Analysis*
Addison-Wesley
**ISBN**: 978-0201076165
**Note**: Introduction of box plots and outlier detection methods.

**Rousseeuw, P. J., & Leroy, A. M. (1987)**
*Robust Regression and Outlier Detection*
John Wiley & Sons
**DOI**: 10.1002/0471725382
**Note**: Comprehensive treatment of robust statistical methods.

**Iglewicz, B., & Hoaglin, D. C. (1993)**
"How to detect and handle outliers"
*ASQC Quality Press*
**ISBN**: 978-0873892476
**Note**: Practical guide to outlier detection methods.

### Machine Learning Approaches

**Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)**
"Isolation forest"
*IEEE 8th International Conference on Data Mining*, 413-422
**DOI**: 10.1109/ICDM.2008.17
**Note**: Isolation Forest algorithm for anomaly detection.

**Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000)**
"LOF: identifying density-based local outliers"
*ACM SIGMOD Record*, 29(2), 93-104
**DOI**: 10.1145/335191.335388
**Note**: Local Outlier Factor algorithm.

**Schölkopf, B., Williamson, R. C., Smola, A. J., Shawe-Taylor, J., & Platt, J. C. (2000)**
"Support vector method for novelty detection"
*Advances in Neural Information Processing Systems*, 12, 582-588
**Note**: One-class SVM for novelty detection.

### Time Series Specific

**Rosner, B. (1983)**
"Percentage points for a generalized ESD many-outlier procedure"
*Technometrics*, 25(2), 165-172
**DOI**: 10.1080/00401706.1983.10487848
**Note**: Extreme Studentized Deviate test for multiple outliers.

**Vallis, O., Hochenbaum, J., & Kejariwal, A. (2014)**
"A novel technique for long-term anomaly detection in the cloud"
*6th USENIX Workshop on Hot Topics in Cloud Computing*
**Note**: Seasonal Hybrid ESD algorithm for time series anomaly detection.

## Correlation and Causality

### Granger Causality

**Granger, C. W. J. (1969)**
"Investigating causal relations by econometric models and cross-spectral methods"
*Econometrica*, 37(3), 424-438
**DOI**: 10.2307/1912791
**Note**: Original paper introducing Granger causality concept.

**Granger, C. W. J. (1988)**
"Some recent development in a concept of causality"
*Journal of Econometrics*, 39(1-2), 199-211
**DOI**: 10.1016/0304-4076(88)90045-0
**Note**: Developments and refinements of causality testing.

### Cointegration

**Engle, R. F., & Granger, C. W. J. (1987)**
"Co-integration and error correction: representation, estimation, and testing"
*Econometrica*, 55(2), 251-276
**DOI**: 10.2307/1913236
**Note**: Foundational paper on cointegration analysis.

**Johansen, S. (1988)**
"Statistical analysis of cointegration vectors"
*Journal of Economic Dynamics and Control*, 12(2-3), 231-254
**DOI**: 10.1016/0165-1889(88)90041-3
**Note**: Johansen test for cointegration.

**Johansen, S. (1991)**
"Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models"
*Econometrica*, 59(6), 1551-1580
**DOI**: 10.2307/2938278
**Note**: Maximum likelihood estimation of cointegrated VAR models.

### Dynamic Time Warping

**Sakoe, H., & Chiba, S. (1978)**
"Dynamic programming algorithm optimization for spoken word recognition"
*IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49
**DOI**: 10.1109/TASSP.1978.1163055
**Note**: Original dynamic time warping algorithm.

**Müller, M. (2007)**
*Information Retrieval for Music and Motion*
Springer
**DOI**: 10.1007/978-3-540-74048-3
**Note**: Comprehensive treatment of DTW and related methods.

## Machine Learning Applications

### Deep Learning for Time Series

**Hochreiter, S., & Schmidhuber, J. (1997)**
"Long short-term memory"
*Neural Computation*, 9(8), 1735-1780
**DOI**: 10.1162/neco.1997.9.8.1735
**Note**: LSTM networks for sequence modeling.

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017)**
"Attention is all you need"
*Advances in Neural Information Processing Systems*, 30
**Note**: Transformer architecture with applications to time series.

**Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021)**
"Temporal fusion transformers for interpretable multi-horizon time series forecasting"
*International Journal of Forecasting*, 37(4), 1748-1764
**DOI**: 10.1016/j.ijforecast.2021.03.012
**Note**: Attention-based models for time series forecasting.

### Online Learning

**Cesa-Bianchi, N., & Lugosi, G. (2006)**
*Prediction, Learning, and Games*
Cambridge University Press
**DOI**: 10.1017/CBO9780511546921
**Note**: Theory of online learning and prediction.

**Shalev-Shwartz, S. (2011)**
"Online learning and online convex optimization"
*Foundations and Trends in Machine Learning*, 4(2), 107-194
**DOI**: 10.1561/2200000018
**Note**: Comprehensive survey of online learning methods.

## Financial Time Series

### Volatility Modeling

**Engle, R. F. (1982)**
"Autoregressive conditional heteroskedasticity with estimates of the variance of United Kingdom inflation"
*Econometrica*, 50(4), 987-1007
**DOI**: 10.2307/1912773
**Note**: Original ARCH model for volatility clustering.

**Bollerslev, T. (1986)**
"Generalized autoregressive conditional heteroskedasticity"
*Journal of Econometrics*, 31(3), 307-327
**DOI**: 10.1016/0304-4076(86)90063-1
**Note**: GARCH model extending ARCH methodology.

**Nelson, D. B. (1991)**
"Conditional heteroskedasticity in asset returns: A new approach"
*Econometrica*, 59(2), 347-370
**DOI**: 10.2307/2938260
**Note**: EGARCH model for asymmetric volatility.

### Risk Management

**Jorion, P. (2006)**
*Value at Risk: The New Benchmark for Managing Financial Risk* (3rd Edition)
McGraw-Hill
**ISBN**: 978-0071464956
**Note**: Comprehensive treatment of VaR methodology.

**McNeil, A. J., Frey, R., & Embrechts, P. (2015)**
*Quantitative Risk Management: Concepts, Techniques and Tools* (Revised Edition)
Princeton University Press
**ISBN**: 978-0691166278
**Note**: Modern approach to financial risk management.

### Market Microstructure

**Hasbrouck, J. (2007)**
*Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading*
Oxford University Press
**ISBN**: 978-0195301649
**Note**: Econometric analysis of high-frequency financial data.

**Aït-Sahalia, Y., & Jacod, J. (2014)**
*High-Frequency Financial Econometrics*
Princeton University Press
**ISBN**: 978-0691161433
**Note**: Statistical methods for high-frequency financial data.

## Software and Computational Methods

### Numerical Methods

**Golub, G. H., & Van Loan, C. F. (2013)**
*Matrix Computations* (4th Edition)
Johns Hopkins University Press
**ISBN**: 978-1421407944
**Note**: Standard reference for numerical linear algebra.

**Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007)**
*Numerical Recipes: The Art of Scientific Computing* (3rd Edition)
Cambridge University Press
**ISBN**: 978-0521880688
**Note**: Practical algorithms for scientific computing.

### Floating Point Arithmetic

**Kahan, W. (1965)**
"Pracniques: further remarks on reducing truncation errors"
*Communications of the ACM*, 8(1), 40
**DOI**: 10.1145/363707.363723
**Note**: Kahan summation algorithm for numerical stability.

**Higham, N. J. (2002)**
*Accuracy and Stability of Numerical Algorithms* (2nd Edition)
SIAM
**DOI**: 10.1137/1.9780898718027
**Note**: Comprehensive treatment of numerical stability issues.

### Random Sampling

**Vitter, J. S. (1985)**
"Random sampling with a reservoir"
*ACM Transactions on Mathematical Software*, 11(1), 37-57
**DOI**: 10.1145/3147.3165
**Note**: Reservoir sampling algorithm for large datasets.

**Welford, B. P. (1962)**
"Note on a method for calculating corrected sums of squares and products"
*Technometrics*, 4(3), 419-420
**DOI**: 10.1080/00401706.1962.10490022
**Note**: Online algorithm for computing variance.

### Programming and Software Engineering

**Bentley, J. (2000)**
*Programming Pearls* (2nd Edition)
Addison-Wesley
**ISBN**: 978-0201657883
**Note**: Classic algorithms and programming techniques.

**Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009)**
*Introduction to Algorithms* (3rd Edition)
MIT Press
**ISBN**: 978-0262033848
**Note**: Comprehensive algorithms textbook.

## Research Papers by Topic

### Changepoint Detection

**Truong, C., Oudre, L., & Vayatis, N. (2020)**
"Selective review of offline change point detection methods"
*Signal Processing*, 167, 107299
**DOI**: 10.1016/j.sigpro.2019.107299
**Note**: Comprehensive survey of change point detection methods.

**Aminikhanghahi, S., & Cook, D. J. (2017)**
"A survey of methods for time series change point detection"
*Knowledge and Information Systems*, 51(2), 339-367
**DOI**: 10.1007/s10115-016-0987-z
**Note**: Survey of time series change point detection methods.

### Forecasting Competitions and Evaluations

**Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018)**
"The M4 Competition: Results, findings, conclusion and way forward"
*International Journal of Forecasting*, 34(4), 802-808
**DOI**: 10.1016/j.ijforecast.2018.06.001
**Note**: Results from the M4 forecasting competition.

**Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020)**
"The M4 Competition: 100,000 time series and 61 forecasting methods"
*International Journal of Forecasting*, 36(1), 54-74
**DOI**: 10.1016/j.ijforecast.2019.04.014
**Note**: Comprehensive analysis of forecasting method performance.

### Seasonal Adjustment

**Hood, C. C. (2005)**
"An empirical comparison of methods for benchmarking seasonally adjusted series to annual totals"
*Survey Methodology*, 31(2), 225-235
**Note**: Comparison of seasonal adjustment benchmarking methods.

**Ladiray, D., & Quenneville, B. (2001)**
*Seasonal Adjustment with the X-11 Method*
Springer
**DOI**: 10.1007/978-1-4613-0175-2
**Note**: Comprehensive treatment of X-11 seasonal adjustment.

## Standards and Guidelines

### Statistical Standards

**ISO 3534-1:2006**
*Statistics -- Vocabulary and symbols -- Part 1: General statistical terms and terms used in probability*
International Organization for Standardization
**Note**: Standard definitions for statistical terms.

**NIST/SEMATECH (2012)**
*e-Handbook of Statistical Methods*
NIST/SEMATECH
**URL**: https://www.itl.nist.gov/div898/handbook/
**Note**: Comprehensive handbook of statistical methods and best practices.

### Time Series Analysis Guidelines

**Eurostat (2015)**
*ESS Guidelines on Seasonal Adjustment*
Publications Office of the European Union
**DOI**: 10.2785/317290
**Note**: Official guidelines for seasonal adjustment in official statistics.

**IMF (2017)**
*Quarterly National Accounts Manual: Concepts, Data Sources, and Compilation*
International Monetary Fund
**ISBN**: 978-1-48439-956-9
**Note**: Guidelines for time series analysis in national accounts.

## Online Resources and Software Documentation

### R Packages Documentation

**Hyndman, R. J., & Khandakar, Y. (2008)**
"Automatic time series forecasting: the forecast package for R"
*Journal of Statistical Software*, 27(3), 1-22
**DOI**: 10.18637/jss.v027.i03
**Note**: Documentation for the forecast package in R.

**Hyndman, R. J., Athanasopoulos, G., Bergmeir, C., Caceres, G., Chhay, L., O'Hara-Wild, M., ... & Zhou, Z. (2020)**
"forecast: Forecasting functions for time series and linear models"
*R package version 8.13*
**URL**: https://pkg.robjhyndman.com/forecast/
**Note**: Comprehensive R package for time series forecasting.

### Python Libraries

**Seabold, S., & Perktold, J. (2010)**
"statsmodels: Econometric and statistical modeling with python"
*9th Python in Science Conference*
**DOI**: 10.25080/Majora-92bf1922-011
**Note**: Statistical modeling library for Python.

**McKinney, W. (2010)**
"Data structures for statistical computing in python"
*Proceedings of the 9th Python in Science Conference*, 445, 51-56
**DOI**: 10.25080/Majora-92bf1922-00a
**Note**: Pandas library for data manipulation in Python.

## Citation Guidelines

When using Chronos in research or commercial applications, please cite relevant papers based on the methods used:

**For STL Decomposition**: Cleveland et al. (1990)
**For ARIMA Models**: Box et al. (2015)
**For Anomaly Detection**: Liu et al. (2008) for Isolation Forest
**For Trend Testing**: Mann (1945) and Kendall (1975)
**For Stationarity Testing**: Dickey & Fuller (1979) for ADF test
**For Granger Causality**: Granger (1969)
**For Seasonal Adjustment**: U.S. Census Bureau (2017) for X-13ARIMA-SEATS

## Keeping References Current

This bibliography is maintained to reflect current best practices and recent developments in time series analysis. For the most recent publications and emerging methods, consult:

- *Journal of Forecasting*
- *International Journal of Forecasting*
- *Journal of Time Series Analysis*
- *Journal of Business & Economic Statistics*
- *Computational Statistics & Data Analysis*
- Conference proceedings: NIPS, ICML, KDD, ICDM

## Contributing to the Bibliography

To suggest additions or corrections to this bibliography:

1. Ensure the reference is peer-reviewed and relevant to time series analysis
2. Follow the citation format used in this document
3. Include DOI or URL when available
4. Add a brief note explaining the relevance to Chronos methodology
5. Submit via pull request with clear justification for inclusion

This comprehensive bibliography provides the scientific foundation for understanding and validating the methods implemented in Chronos, ensuring users can trace theoretical foundations and explore advanced applications of time series analysis techniques.