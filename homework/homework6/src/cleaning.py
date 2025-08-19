"""
Data Cleaning Functions for Homework 6: Data Preprocessing

Simple, reusable functions for common data cleaning operations.
"""

import pandas as pd
import numpy as np
from typing import List


def fill_missing_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Fill missing values in specified columns with their median values.
    
    Args:
        df: Input DataFrame
        columns: List of column names to fill
        
    Returns:
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            median_val = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median_val)
            print(f"Filled {df[col].isna().sum()} missing values in '{col}' with median: {median_val:.2f}")
    
    return df_copy


def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop rows with missing values based on threshold.
    
    Args:
        df: Input DataFrame
        threshold: Fraction of non-null values required to keep row
        
    Returns:
        DataFrame with rows dropped
    """
    df_copy = df.copy()
    original_rows = len(df_copy)
    
    # Drop rows that don't meet threshold
    df_copy = df_copy.dropna(thresh=int(threshold * df_copy.shape[1]))
    
    dropped_rows = original_rows - len(df_copy)
    print(f"Dropped {dropped_rows} rows with <{threshold:.1%} non-null values")
    
    return df_copy


def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize numerical columns using MinMax scaling (0-1 range).
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
        
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            
            if max_val != min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
                print(f"Normalized '{col}' to range [0, 1]")
            else:
                print(f"Warning: '{col}' has constant values, skipping normalization")
    
    return df_copy
