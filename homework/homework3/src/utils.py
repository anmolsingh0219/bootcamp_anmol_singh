def get_summary_stats(df, group_col, value_col):
    """Calculate summary statistics grouped by a column.
    
    Args:
        df: pandas DataFrame
        group_col: column name to group by
        value_col: column name to calculate statistics for
        
    Returns:
        pandas DataFrame with summary statistics
    """
    return df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
