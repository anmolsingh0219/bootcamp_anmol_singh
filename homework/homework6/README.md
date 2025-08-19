# Homework 6: Data Preprocessing

## Overview
This homework demonstrates systematic data cleaning and preprocessing using reusable functions. We apply techniques learned in the lecture to handle missing data, normalize features, and prepare data for analysis.

## Data Cleaning Strategy

### 1. Missing Data Handling
Our approach follows the lecture methodology for different types of missingness:

- **MCAR (Missing Completely At Random)**: Safe to fill with median or drop
- **MAR (Missing At Random)**: Can be imputed based on other features  
- **MNAR (Missing Not At Random)**: Requires domain knowledge

**Implementation:**
- `fill_missing_median()`: Fills missing numerical values with column medians
- Median chosen over mean for robustness to outliers
- Applied to columns with simulated MCAR/MAR patterns

### 2. Row Filtering
- `drop_missing()`: Removes rows with excessive missing data
- Threshold-based approach: keep rows with ≥50% non-null values
- Balances data retention with quality requirements

### 3. Data Normalization
- `normalize_data()`: MinMax scaling to [0,1] range
- Makes features comparable across different scales
- Essential for algorithms sensitive to feature magnitude

## File Structure
```
homework6/
├── src/
│   ├── __init__.py
│   └── cleaning.py          # Reusable cleaning functions
├── data/
│   ├── raw/
│   │   └── sample_data.csv  # Original Alpha Vantage stock data
│   └── processed/
│       └── sample_data_cleaned.csv  # Cleaned dataset
├── notebooks/
│   └── stage06_data-preprocessing_homework-starter.ipynb
└── README.md
```

## Functions Documentation

### `fill_missing_median(df, columns)`
- Fills missing values with median for specified columns
- Returns new DataFrame with missing values filled
- Prints summary of filled values

### `drop_missing(df, threshold=0.5)`
- Drops rows with missing values below threshold
- Threshold = fraction of non-null values required
- Returns DataFrame with filtered rows

### `normalize_data(df, columns)`
- Applies MinMax scaling to specified columns
- Scales values to [0, 1] range
- Handles constant-value columns gracefully

## Data Processing Steps

1. **Load Raw Data**: Alpha Vantage stock price data (AAPL)
2. **Simulate Missingness**: Add MCAR, MAR, and MNAR patterns for demonstration
3. **Fill Missing Values**: Use median imputation for numerical columns
4. **Filter Rows**: Remove rows with <50% non-null values
5. **Normalize Features**: Scale price, volume, and market cap to [0,1] range
6. **Save Results**: Export cleaned data to processed directory

## Key Assumptions

- **Missing data is MCAR/MAR**: Safe to impute with statistical measures
- **Median is robust**: Better than mean for skewed financial data
- **Normalization preserves relationships**: Relative patterns maintained after scaling
- **Threshold filtering is appropriate**: 50% cutoff balances quality vs. quantity

## Usage
```python
from src import cleaning

# Fill missing values
df_filled = cleaning.fill_missing_median(df, ['price', 'volume'])

# Drop sparse rows  
df_filtered = cleaning.drop_missing(df_filled, threshold=0.5)

# Normalize features
df_normalized = cleaning.normalize_data(df_filtered, ['price', 'volume'])
```

## Dependencies
- pandas
- numpy

## Results
- Original dataset: 100 rows × 8 columns
- Cleaned dataset: Maintains data integrity while handling missing values
- All preprocessing steps documented with clear assumptions
