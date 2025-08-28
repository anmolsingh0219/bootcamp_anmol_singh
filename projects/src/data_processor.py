import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OptionsDataProcessor:
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
    
    def clean_raw_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Clean raw data using the dedicated cleaning module.
        
        Args:
            df (pd.DataFrame): Raw options data
            config (Dict[str, Any]): Configuration parameters
            
        Returns:
            pd.DataFrame: Cleaned options data
        """
        from cleaning import clean_options_data
        return clean_options_data(df, config)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.copy()
        
        # Basic features
        feature_df['log_moneyness'] = np.log(feature_df['moneyness'])
        feature_df['sqrt_time_to_expiry'] = np.sqrt(feature_df['time_to_expiry'])
        feature_df['vol_time_product'] = feature_df['implied_volatility'] * feature_df['time_to_expiry']
        
        # Interaction features
        feature_df['vol_moneyness'] = feature_df['implied_volatility'] * feature_df['moneyness']
        feature_df['rate_time'] = feature_df['risk_free_rate'] * feature_df['time_to_expiry']
        
        # Market microstructure features
        feature_df['bid_ask_spread'] = feature_df['ask'] - feature_df['bid']
        feature_df['relative_spread'] = feature_df['bid_ask_spread'] / feature_df['market_price']
        feature_df['log_volume'] = np.log1p(feature_df['volume'])
        feature_df['log_open_interest'] = np.log1p(feature_df['open_interest'])
        
        # Option type indicators (ITM/OTM/ATM)
        feature_df['is_itm'] = (feature_df['moneyness'] > 1.0).astype(int)
        feature_df['is_atm'] = ((feature_df['moneyness'] >= 0.95) & 
                               (feature_df['moneyness'] <= 1.05)).astype(int)
        feature_df['is_otm'] = (feature_df['moneyness'] < 1.0).astype(int)
        
        # Time-based features
        feature_df['days_to_expiry'] = feature_df['time_to_expiry'] * 365
        feature_df['is_short_term'] = (feature_df['days_to_expiry'] <= 30).astype(int)
        feature_df['is_long_term'] = (feature_df['days_to_expiry'] >= 90).astype(int)
        
        logger.info(f"Feature engineering complete. Added {len(feature_df.columns) - len(df.columns)} new features")
        
        return feature_df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'isolation_forest', 
                       contamination: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers using the dedicated outliers module.
        
        Args:
            df (pd.DataFrame): Input data
            method (str): Detection method
            contamination (float): Expected proportion of outliers
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (clean_data, outliers)
        """
        from outliers import OutlierDetector
        detector = OutlierDetector()
        return detector.detect_outliers(df, method=method, contamination=contamination)
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard', 
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize/standardize numerical features.
        
        Args:
            df (pd.DataFrame): Input data
            method (str): Normalization method ('standard', 'minmax')
            fit (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Normalized data
        """
        logger.info(f"Starting feature normalization using {method}")
        
        normalized_df = df.copy()
        
        features_to_normalize = [
            'underlying_price', 'strike', 'implied_volatility', 'time_to_expiry',
            'volume', 'open_interest', 'market_price', 'log_moneyness',
            'sqrt_time_to_expiry', 'vol_time_product', 'vol_moneyness',
            'rate_time', 'bid_ask_spread', 'relative_spread',
            'log_volume', 'log_open_interest', 'days_to_expiry'
        ]
        
        features_to_normalize = [col for col in features_to_normalize if col in normalized_df.columns]
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Fit and transform
            normalized_df[features_to_normalize] = self.scaler.fit_transform(
                normalized_df[features_to_normalize].fillna(0)
            )
            self.feature_columns = features_to_normalize
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            
            # Transform only
            normalized_df[self.feature_columns] = self.scaler.transform(
                normalized_df[self.feature_columns].fillna(0)
            )
        
        logger.info(f"Feature normalization complete for {len(features_to_normalize)} features")
        
        return normalized_df
    
    def calculate_target_variable(self, df: pd.DataFrame, bs_prices: pd.Series) -> pd.DataFrame:
        """
        Calculate the target variable (pricing error) for modeling.
        
        Args:
            df (pd.DataFrame): Options data with market prices
            bs_prices (pd.Series): Black-Scholes theoretical prices
            
        Returns:
            pd.DataFrame: Data with target variable added
        """
        logger.info("Calculating target variable (pricing error)")
        
        result_df = df.copy()
        result_df['bs_price'] = bs_prices
        result_df['pricing_error'] = result_df['market_price'] - result_df['bs_price']
        result_df['relative_error'] = result_df['pricing_error'] / result_df['bs_price']
        
        # Log statistics
        logger.info(f"Pricing error statistics:")
        logger.info(f"  Mean: ${result_df['pricing_error'].mean():.4f}")
        logger.info(f"  Std:  ${result_df['pricing_error'].std():.4f}")
        logger.info(f"  Min:  ${result_df['pricing_error'].min():.4f}")
        logger.info(f"  Max:  ${result_df['pricing_error'].max():.4f}")
        
        return result_df
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str) -> None:
        df.to_csv(filepath, index=False)
