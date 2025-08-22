import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class OptionsDataProcessor:
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
    
    def clean_raw_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        cleaned_df = df.copy()
        
        cleaned_df['time_to_expiry'] = (cleaned_df['expiration_date'] - pd.Timestamp.now()).dt.days / 365.0
        cleaned_df['market_price'] = (cleaned_df['bid'] + cleaned_df['ask']) / 2.0
        
        # Handle cases where bid/ask are NaN - use last_price if available
        mask_no_bid_ask = cleaned_df['market_price'].isna()
        cleaned_df.loc[mask_no_bid_ask, 'market_price'] = cleaned_df.loc[mask_no_bid_ask, 'last_price']
        
        # Calculate moneyness (S/K ratio)
        cleaned_df['moneyness'] = cleaned_df['underlying_price'] / cleaned_df['strike']

        filters = [
            (cleaned_df['volume'].fillna(0) >= config.get('MIN_VOLUME', 1)),
            (cleaned_df['open_interest'].fillna(0) >= config.get('MIN_OPEN_INTEREST', 1)),
            cleaned_df['time_to_expiry'] > config.get('MIN_TIME_TO_EXPIRY', 0.01),
            cleaned_df['time_to_expiry'] <= 2.0,
            cleaned_df['implied_volatility'].notna(),
            cleaned_df['implied_volatility'] > 0,
            cleaned_df['implied_volatility'] < 5.0,
            cleaned_df['market_price'].notna(),
            cleaned_df['market_price'] > 0,
            cleaned_df['strike'] > 0,
            cleaned_df['underlying_price'] > 0,
            cleaned_df['moneyness'] > 0.1,
            cleaned_df['moneyness'] < 3.0
        ]
        
        combined_filter = pd.Series(True, index=cleaned_df.index)
        for filter_condition in filters:
            combined_filter &= filter_condition
        
        cleaned_df = cleaned_df[combined_filter].copy()
        
        logger.info(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
        logger.info(f"Removed {len(df) - len(cleaned_df)} rows during cleaning")
        
        return cleaned_df
    
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
        
        outlier_features = ['market_price', 'implied_volatility', 'moneyness', 
                           'time_to_expiry', 'volume', 'open_interest']
        outlier_features = [col for col in outlier_features if col in df.columns]
        
        if method == 'isolation_forest':
            # Use Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[outlier_features].fillna(0))
            is_outlier = outlier_labels == -1
            
        elif method == 'zscore':
            # Use Z-score method (|z| > 3)
            z_scores = np.abs(stats.zscore(df[outlier_features].fillna(0), axis=0))
            is_outlier = (z_scores > 3).any(axis=1)
            
        elif method == 'iqr':
            # Use IQR method
            Q1 = df[outlier_features].quantile(0.25)
            Q3 = df[outlier_features].quantile(0.75)
            IQR = Q3 - Q1
            is_outlier = ((df[outlier_features] < (Q1 - 1.5 * IQR)) | 
                         (df[outlier_features] > (Q3 + 1.5 * IQR))).any(axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        clean_data = df[~is_outlier].copy()
        outliers = df[is_outlier].copy()
        
        logger.info(f"Outlier detection complete. Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
        
        return clean_data, outliers
    
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
