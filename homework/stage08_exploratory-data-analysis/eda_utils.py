"""
EDA Utility Functions for Stage 08 Homework
Provides reusable functions for exploratory data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def setup_plot_style():
    """Configure seaborn style for consistent plotting"""
    sns.set(context='talk', style='whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)

def generate_numeric_profile(df, numeric_cols):
    """
    Generate comprehensive numeric profile with descriptive stats, skew, and kurtosis
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric column names
    
    Returns:
        pandas DataFrame with extended descriptive statistics
    """
    desc = df[numeric_cols].describe().T
    desc['skew'] = [skew(df[c].dropna()) for c in desc.index]
    desc['kurtosis'] = [kurtosis(df[c].dropna()) for c in desc.index]
    desc['missing_count'] = [df[c].isna().sum() for c in desc.index]
    desc['missing_pct'] = [df[c].isna().sum() / len(df) * 100 for c in desc.index]
    return desc

def plot_distributions(df, cols, figsize=(15, 10)):
    """
    Create histogram + KDE plots for multiple variables
    
    Args:
        df: pandas DataFrame
        cols: list of column names to plot
        figsize: tuple for figure size
    """
    n_cols = len(cols)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    for i, col in enumerate(cols):
        if i < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, numeric_cols, categorical_col=None, figsize=(15, 10)):
    """
    Create boxplots for outlier detection
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric columns
        categorical_col: optional categorical column for grouping
        figsize: tuple for figure size
    """
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            if categorical_col:
                sns.boxplot(data=df, x=categorical_col, y=col, ax=axes[i])
                axes[i].set_title(f'{col} by {categorical_col}')
            else:
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(f'{col} Outliers')
    
    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlations(df, numeric_cols, figsize=(10, 8)):
    """
    Create correlation heatmap
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric columns
        figsize: tuple for figure size
    
    Returns:
        correlation matrix
    """
    corr = df[numeric_cols].corr(numeric_only=True)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', 
                vmin=-1, vmax=1, center=0, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr

def plot_bivariate_relationships(df, x_cols, y_col, hue_col=None, figsize=(15, 10)):
    """
    Create scatter plots for bivariate relationships
    
    Args:
        df: pandas DataFrame
        x_cols: list of x-axis columns
        y_col: y-axis column name
        hue_col: optional column for color coding
        figsize: tuple for figure size
    """
    n_plots = len(x_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.ravel()
    
    for i, x_col in enumerate(x_cols):
        if i < len(axes):
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                          alpha=0.7, ax=axes[i])
            axes[i].set_title(f'{x_col} vs {y_col}')
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)
    
    # Hide unused subplots
    for i in range(len(x_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def detect_outliers(df, col, method='iqr', threshold=1.5):
    """
    Detect outliers using IQR or Z-score method
    
    Args:
        df: pandas DataFrame
        col: column name
        method: 'iqr' or 'zscore'
        threshold: threshold value (1.5 for IQR, 3 for Z-score)
    
    Returns:
        boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = z_scores > threshold
    
    return outliers

def summarize_missing_data(df):
    """
    Summarize missing data patterns
    
    Args:
        df: pandas DataFrame
    
    Returns:
        pandas DataFrame with missing data summary
    """
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isna().sum(),
        'Missing_Percentage': (df.isna().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    missing_summary = missing_summary.sort_values('Missing_Count', ascending=False)
    
    return missing_summary

def generate_eda_report(df, target_col=None):
    """
    Generate comprehensive EDA report
    
    Args:
        df: pandas DataFrame
        target_col: optional target variable for focused analysis
    
    Returns:
        dict with key findings
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    report = {
        'shape': df.shape,
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(categorical_cols),
        'missing_data': summarize_missing_data(df),
        'numeric_profile': generate_numeric_profile(df, numeric_cols)
    }
    
    if target_col and target_col in numeric_cols:
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        report['target_correlations'] = correlations
    
    return report
