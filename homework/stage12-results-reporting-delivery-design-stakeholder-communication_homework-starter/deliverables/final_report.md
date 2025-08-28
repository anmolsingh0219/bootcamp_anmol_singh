# Options Pricing Model: Risk Assessment Report

**Analysis Period:** Options Dataset (496 contracts)  
**Date:** January 2025

---

## Executive Summary

**Key Findings:**
- Model explains only 8.3% of price variation (R² = 0.083)
- Missing data treatment causes 19.3% performance difference
- Significant prediction bias across strike price segments

**Recommendation:**
Model requires substantial improvement before operational use.

---

## Performance Results

### Scenario Comparison
- Mean Imputation: MAE $72.8, R² 8.3%
- Median Imputation: MAE $72.9, R² 8.3%
- Drop Missing: MAE $58.8, R² 15.6%

**Key Finding:** Dropping missing observations improves performance by 19.3%.

### Segment Analysis
- Low Strike (64 options): $107 under-prediction bias, $200 MAE
- Mid Strike (142 options): $-10 bias, $94 MAE
- High Strike (290 options): $42 over-prediction bias, $126 MAE

**Key Finding:** Model performance varies significantly by strike price range.

### Uncertainty Analysis
Bootstrap 95% Confidence Intervals:
- MAE: $66.6 - $79.2
- RMSE: $92.5 - $114.7
- R²: 0.000 - 0.162

**Key Finding:** Wide confidence intervals indicate high uncertainty.

---

## Sensitivity Analysis

| Assumption | Baseline | Alternative | Impact |
|---|---|---|---|
| Missing Data | Mean Imputation ($72.8 MAE) | Drop Missing ($58.8 MAE) | 19.3% improvement |
| Sample Size | 496 observations | 421 complete cases | 15% reduction |
| Segment Focus | All segments | Mid-strike only | Reduced bias |

---

## Risk Assessment

**High Risk Areas:**
- Low explanatory power (R² = 8.3%)
- Missing data creates systematic bias
- Large prediction errors for low-strike options

**Model Limitations:**
- Cannot reliably predict individual option prices
- Performance degrades with missing volatility data
- Significant bias across different strike ranges

---

## Recommendations

**Immediate:**
- Do not use for production pricing
- Investigate missing data patterns
- Develop segment-specific models

**Technical:**
- Expand feature set beyond implied volatility
- Implement data quality monitoring
- Consider alternative modeling approaches

---

## Technical Details

**Data:**
- 496 options contracts
- 15% missing volatility data
- Strike segments: 64 low, 142 mid, 290 high

**Model:**
- Linear regression with implied volatility
- Bootstrap validation (600 iterations)
- 95% confidence intervals calculated

**Files:**
- model_performance_comparison.png
- subgroup_risk_assessment.png
- bootstrap_confidence_intervals.png
- sensitivity_analysis.csv