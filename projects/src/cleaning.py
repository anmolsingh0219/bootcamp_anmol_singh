import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def clean_options_data(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning for options data.
    
    Args:
        df (pd.DataFrame): Raw options data
        config (Dict[str, Any]): Configuration parameters for filtering
        
    Returns:
        pd.DataFrame: Cleaned options data
        
    Assumptions:
        - Options with zero or negative prices are invalid
        - Very high implied volatility (>500%) indicates data errors
        - Options with negative time to expiry are expired/invalid
        - Extreme moneyness values (S/K) indicate potential errors
    """
    if config is None:
        config = {
            'MIN_VOLUME': 0,
            'MIN_OPEN_INTEREST': 1, 
            'MIN_TIME_TO_EXPIRY': 0.002,  # ~8 hours
            'MAX_IMPLIED_VOL': 5.0,       # 500%
            'MIN_MONEYNESS': 0.1,         # 10% of strike
            'MAX_MONEYNESS': 3.0          # 300% of strike
        }
    
    logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
    cleaned_df = df.copy()
    
    # Calculate derived fields needed for cleaning
    cleaned_df['time_to_expiry'] = (cleaned_df['expiration_date'] - pd.Timestamp.now()).dt.days / 365.0
    cleaned_df['market_price'] = (cleaned_df['bid'] + cleaned_df['ask']) / 2.0
    
    # Handle cases where bid/ask are NaN - use last_price if available
    mask_no_bid_ask = cleaned_df['market_price'].isna()
    cleaned_df.loc[mask_no_bid_ask, 'market_price'] = cleaned_df.loc[mask_no_bid_ask, 'last_price']
    
    # Calculate moneyness (S/K ratio)
    cleaned_df['moneyness'] = cleaned_df['underlying_price'] / cleaned_df['strike']
    
    # Apply cleaning filters with detailed logging
    initial_count = len(cleaned_df)
    
    # Filter 1: Volume requirements
    volume_filter = cleaned_df['volume'].fillna(0) >= config['MIN_VOLUME']
    cleaned_df = cleaned_df[volume_filter]
    logger.info(f"Volume filter: Removed {initial_count - len(cleaned_df)} rows")
    
    # Filter 2: Open interest requirements
    oi_filter = cleaned_df['open_interest'].fillna(0) >= config['MIN_OPEN_INTEREST']
    cleaned_df = cleaned_df[oi_filter]
    logger.info(f"Open interest filter: Removed {len(df) - len(cleaned_df)} total rows so far")
    
    # Filter 3: Time to expiry
    time_filter = (cleaned_df['time_to_expiry'] > config['MIN_TIME_TO_EXPIRY']) & (cleaned_df['time_to_expiry'] <= 2.0)
    cleaned_df = cleaned_df[time_filter]
    logger.info(f"Time to expiry filter: Removed {len(df) - len(cleaned_df)} total rows so far")
    
    # Filter 4: Implied volatility
    iv_filter = (cleaned_df['implied_volatility'].notna() & 
                 (cleaned_df['implied_volatility'] > 0) & 
                 (cleaned_df['implied_volatility'] < config['MAX_IMPLIED_VOL']))
    cleaned_df = cleaned_df[iv_filter]
    logger.info(f"Implied volatility filter: Removed {len(df) - len(cleaned_df)} total rows so far")
    
    # Filter 5: Market price validity
    price_filter = (cleaned_df['market_price'].notna() & (cleaned_df['market_price'] > 0))
    cleaned_df = cleaned_df[price_filter]
    logger.info(f"Price validity filter: Removed {len(df) - len(cleaned_df)} total rows so far")
    
    # Filter 6: Strike and underlying price validity
    strike_filter = (cleaned_df['strike'] > 0) & (cleaned_df['underlying_price'] > 0)
    cleaned_df = cleaned_df[strike_filter]
    logger.info(f"Strike/underlying price filter: Removed {len(df) - len(cleaned_df)} total rows so far")
    
    # Filter 7: Moneyness bounds
    moneyness_filter = ((cleaned_df['moneyness'] > config['MIN_MONEYNESS']) & 
                       (cleaned_df['moneyness'] < config['MAX_MONEYNESS']))
    cleaned_df = cleaned_df[moneyness_filter]
    
    logger.info(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    logger.info(f"Total removed: {len(df) - len(cleaned_df)} rows ({(len(df) - len(cleaned_df))/len(df)*100:.2f}%)")
    
    return cleaned_df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input data
        strategy (str): Strategy for handling missing values ('drop', 'fill', 'interpolate')
        
    Returns:
        pd.DataFrame: Data with missing values handled
        
    Assumptions:
        - Critical fields (price, strike, expiry) should not have missing values
        - Volume/OI can be filled with 0 (indicating no trading)
        - Implied volatility missing values indicate invalid options
    """
    logger.info(f"Handling missing values using strategy: {strategy}")
    
    result_df = df.copy()
    
    # Critical fields that shouldn't be missing
    critical_fields = ['market_price', 'strike', 'underlying_price', 'time_to_expiry', 'implied_volatility']
    
    if strategy == 'drop':
        # Drop rows with missing critical values
        for field in critical_fields:
            if field in result_df.columns:
                before_count = len(result_df)
                result_df = result_df.dropna(subset=[field])
                logger.info(f"Dropped {before_count - len(result_df)} rows due to missing {field}")
    
    elif strategy == 'fill':
        # Fill volume and open_interest with 0
        if 'volume' in result_df.columns:
            result_df['volume'] = result_df['volume'].fillna(0)
        if 'open_interest' in result_df.columns:
            result_df['open_interest'] = result_df['open_interest'].fillna(0)
        
        # Fill bid/ask with last_price if available
        if 'bid' in result_df.columns and 'last_price' in result_df.columns:
            result_df['bid'] = result_df['bid'].fillna(result_df['last_price'])
        if 'ask' in result_df.columns and 'last_price' in result_df.columns:
            result_df['ask'] = result_df['ask'].fillna(result_df['last_price'])
    
    logger.info(f"Missing value handling complete. Shape: {result_df.shape}")
    return result_df

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        Dict[str, Any]: Quality metrics and validation results
    """
    logger.info("Validating data quality...")
    
    quality_metrics = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Price validation
    if 'market_price' in df.columns:
        quality_metrics['negative_prices'] = (df['market_price'] <= 0).sum()
        quality_metrics['extreme_prices'] = (df['market_price'] > df['underlying_price']).sum()
    
    # Volatility validation
    if 'implied_volatility' in df.columns:
        quality_metrics['extreme_volatility'] = (df['implied_volatility'] > 5.0).sum()
        quality_metrics['zero_volatility'] = (df['implied_volatility'] <= 0).sum()
    
    # Time validation
    if 'time_to_expiry' in df.columns:
        quality_metrics['expired_options'] = (df['time_to_expiry'] <= 0).sum()
        quality_metrics['long_term_options'] = (df['time_to_expiry'] > 2.0).sum()
    
    logger.info(f"Data quality validation complete. Found {quality_metrics['duplicate_rows']} duplicates")
    
    return quality_metrics

def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input data
        subset (list): Columns to consider for duplicate detection
        
    Returns:
        pd.DataFrame: Data with duplicates removed
        
    Assumptions:
        - Options with same symbol, strike, expiry are true duplicates
        - Keep the most recent observation based on fetch_timestamp
    """
    if subset is None:
        subset = ['symbol', 'strike', 'expiration_date', 'contract_type']
    
    logger.info(f"Removing duplicates based on: {subset}")
    
    initial_count = len(df)
    
    # Sort by fetch_timestamp to keep most recent
    if 'fetch_timestamp' in df.columns:
        df_sorted = df.sort_values('fetch_timestamp', ascending=False)
    else:
        df_sorted = df.copy()
    
    # Remove duplicates, keeping first (most recent)
    result_df = df_sorted.drop_duplicates(subset=subset, keep='first')
    
    logger.info(f"Removed {initial_count - len(result_df)} duplicate rows")
    
    return result_df
