import pandas as pd
import numpy as np
from typing import Tuple, Optional


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR-based outlier detection. Works well for skewed distributions."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score outlier detection. Assumes normal distribution."""
    mu = series.mean()
    sigma = series.std(ddof=0)  # Population standard deviation
    z = (series - mu) / (sigma if sigma != 0 else 1.0)
    return z.abs() > threshold


def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """Cap extreme values at percentiles, preserving sample size."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def analyze_outlier_impact(series: pd.Series, outlier_mask: pd.Series) -> pd.DataFrame:
    """Compare statistics with and without outliers."""
    # Original statistics
    original_stats = series.describe()[['mean', '50%', 'std']].rename({'50%': 'median'})
    
    # Filtered statistics (without outliers)
    filtered_series = series[~outlier_mask]
    filtered_stats = filtered_series.describe()[['mean', '50%', 'std']].rename({'50%': 'median'})
    
    # Winsorized statistics
    winsorized_series = winsorize_series(series)
    winsorized_stats = winsorized_series.describe()[['mean', '50%', 'std']].rename({'50%': 'median'})
    
    # Combine results
    comparison = pd.concat({
        'original': original_stats,
        'filtered': filtered_stats,
        'winsorized': winsorized_stats
    }, axis=1)
    
    return comparison


def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Modified Z-Score using median absolute deviation. More robust than standard Z-score."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else pd.Series([0] * len(series), index=series.index)
    return np.abs(modified_z_scores) > threshold
