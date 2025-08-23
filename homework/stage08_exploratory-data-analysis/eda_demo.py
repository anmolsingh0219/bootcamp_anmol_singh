"""
Demo script showing how to use EDA utility functions
Run this script to see the EDA functions in action
"""

import numpy as np
import pandas as pd
from eda_utils import *

def main():
    # Setup plotting style
    setup_plot_style()
    
    # Generate sample data (same as notebook)
    np.random.seed(8)
    n = 160
    df = pd.DataFrame({
        'date': pd.date_range('2021-02-01', periods=n, freq='D'),
        'region': np.random.choice(['North','South','East','West'], size=n),
        'age': np.random.normal(40, 8, size=n).clip(22, 70).round(1),
        'income': np.random.lognormal(mean=10.6, sigma=0.3, size=n).round(2),
        'transactions': np.random.poisson(lam=3, size=n),
    })
    
    # Create spend variable
    base = df['income'] * 0.0015 + df['transactions']*18 + np.random.normal(0, 40, size=n)
    df['spend'] = np.maximum(0, base).round(2)
    
    # Inject missingness and outliers
    df.loc[np.random.choice(df.index, 5, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 3, replace=False), 'spend'] = np.nan
    df.loc[np.random.choice(df.index, 2, replace=False), 'transactions'] = df['transactions'].max()+12
    
    print("=== EDA Report ===")
    
    # Generate comprehensive report
    numeric_cols = ['age', 'income', 'transactions', 'spend']
    report = generate_eda_report(df, target_col='spend')
    
    print(f"Dataset shape: {report['shape']}")
    print(f"Numeric columns: {report['numeric_cols']}")
    print(f"Categorical columns: {report['categorical_cols']}")
    
    print("\n=== Missing Data ===")
    print(report['missing_data'])
    
    print("\n=== Numeric Profile ===")
    print(report['numeric_profile'])
    
    print("\n=== Target Correlations ===")
    print(report['target_correlations'])
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    
    # Distribution plots
    plot_distributions(df, numeric_cols)
    
    # Boxplots for outlier detection
    plot_boxplots(df, numeric_cols)
    
    # Boxplots by region
    plot_boxplots(df, ['age', 'income'], categorical_col='region')
    
    # Correlation matrix
    corr_matrix = plot_correlations(df, numeric_cols)
    
    # Bivariate relationships
    plot_bivariate_relationships(df, ['income', 'age', 'transactions'], 'spend', hue_col='region')
    
    # Outlier detection
    print("\n=== Outlier Detection ===")
    transaction_outliers = detect_outliers(df, 'transactions', method='iqr')
    print(f"Transaction outliers (IQR method): {transaction_outliers.sum()}")
    
    income_outliers = detect_outliers(df, 'income', method='zscore', threshold=3)
    print(f"Income outliers (Z-score method): {income_outliers.sum()}")

if __name__ == "__main__":
    main()
