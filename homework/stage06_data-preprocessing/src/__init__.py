"""
Data Preprocessing Package for Homework 6

This package provides utilities for data cleaning and preprocessing.
"""

from .cleaning import (
    fill_missing_median,
    drop_missing,
    normalize_data
)

__all__ = [
    'fill_missing_median',
    'drop_missing', 
    'normalize_data'
]