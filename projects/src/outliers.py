import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy import stats
from typing import Tuple, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class OutlierDetector:
    """
    Comprehensive outlier detection for options data.
    
    Supports multiple detection methods:
    - Isolation Forest (anomaly detection)
    - Z-score (statistical method)
    - IQR (interquartile range)
    - Domain-specific rules for options
    """
    
    def __init__(self):
        self.outlier_features = [
            'market_price', 'implied_volatility', 'moneyness', 
            'time_to_expiry', 'volume', 'open_interest'
        ]
        
    def detect_outliers(self, df: pd.DataFrame, method: str = 'isolation_forest', 
                       contamination: float = 0.1, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers using specified method.
        
        Args:
            df (pd.DataFrame): Input data
            method (str): Detection method ('isolation_forest', 'zscore', 'iqr', 'domain_specific')
            contamination (float): Expected proportion of outliers (for isolation forest)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (clean_data, outliers)
            
        Assumptions:
            - Outliers represent data quality issues or extreme market conditions
            - 10% contamination rate is reasonable for financial data
            - Domain knowledge should guide outlier definitions
        """
        logger.info(f"Detecting outliers using method: {method}")
        
        # Select features available in the dataset
        available_features = [col for col in self.outlier_features if col in df.columns]
        logger.info(f"Using features for outlier detection: {available_features}")
        
        if method == 'isolation_forest':
            is_outlier = self._isolation_forest_detection(df, available_features, contamination)
            
        elif method == 'zscore':
            threshold = kwargs.get('threshold', 3.0)
            is_outlier = self._zscore_detection(df, available_features, threshold)
            
        elif method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            is_outlier = self._iqr_detection(df, available_features, multiplier)
            
        elif method == 'domain_specific':
            is_outlier = self._domain_specific_detection(df)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        clean_data = df[~is_outlier].copy()
        outliers = df[is_outlier].copy()
        
        logger.info(f"Outlier detection complete. Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
        
        return clean_data, outliers
    
    def _isolation_forest_detection(self, df: pd.DataFrame, features: List[str], 
                                   contamination: float) -> pd.Series:
        """Isolation Forest outlier detection."""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(df[features].fillna(0))
        return pd.Series(outlier_labels == -1, index=df.index)
    
    def _zscore_detection(self, df: pd.DataFrame, features: List[str], 
                         threshold: float) -> pd.Series:
        """Z-score based outlier detection."""
        z_scores = np.abs(stats.zscore(df[features].fillna(0), axis=0))
        return pd.Series((z_scores > threshold).any(axis=1), index=df.index)
    
    def _iqr_detection(self, df: pd.DataFrame, features: List[str], 
                      multiplier: float) -> pd.Series:
        """IQR based outlier detection."""
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        is_outlier = ((df[features] < lower_bound) | (df[features] > upper_bound)).any(axis=1)
        return pd.Series(is_outlier, index=df.index)
    
    def _domain_specific_detection(self, df: pd.DataFrame) -> pd.Series:
        """
        Domain-specific outlier detection for options data.
        
        Based on financial domain knowledge:
        - Options priced above underlying (for calls) are suspicious
        - Extremely high implied volatility (>1000%) indicates errors
        - Zero volume with high open interest is unusual
        - Bid > Ask spreads indicate data errors
        """
        outlier_conditions = []
        
        # Condition 1: Calls priced above underlying (impossible without dividends)
        if 'contract_type' in df.columns and 'market_price' in df.columns:
            call_mask = df['contract_type'] == 'call'
            impossible_calls = call_mask & (df['market_price'] > df['underlying_price'])
            outlier_conditions.append(impossible_calls)
        
        # Condition 2: Extreme implied volatility
        if 'implied_volatility' in df.columns:
            extreme_iv = df['implied_volatility'] > 10.0  # 1000%
            outlier_conditions.append(extreme_iv)
        
        # Condition 3: Bid > Ask (impossible spread)
        if 'bid' in df.columns and 'ask' in df.columns:
            invalid_spread = df['bid'] > df['ask']
            outlier_conditions.append(invalid_spread)
        
        # Condition 4: Zero volume with very high open interest (stale data)
        if 'volume' in df.columns and 'open_interest' in df.columns:
            stale_data = (df['volume'] == 0) & (df['open_interest'] > 1000)
            outlier_conditions.append(stale_data)
        
        # Combine all conditions
        if outlier_conditions:
            combined_outliers = pd.concat(outlier_conditions, axis=1).any(axis=1)
        else:
            combined_outliers = pd.Series(False, index=df.index)
        
        return combined_outliers
    
    def sensitivity_analysis(self, df: pd.DataFrame, target_column: str = 'pricing_error') -> Dict[str, Any]:
        """
        Perform sensitivity analysis comparing results with and without outliers.
        
        Args:
            df (pd.DataFrame): Input data with target column
            target_column (str): Column to analyze sensitivity for
            
        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        logger.info(f"Performing sensitivity analysis on {target_column}")
        
        results = {}
        methods = ['isolation_forest', 'zscore', 'iqr', 'domain_specific']
        
        # Original statistics
        if target_column in df.columns:
            results['original'] = {
                'count': len(df),
                'mean': df[target_column].mean(),
                'std': df[target_column].std(),
                'min': df[target_column].min(),
                'max': df[target_column].max(),
                'median': df[target_column].median()
            }
        
        # Test each method
        for method in methods:
            try:
                clean_data, outliers = self.detect_outliers(df, method=method)
                
                if target_column in clean_data.columns:
                    results[method] = {
                        'count': len(clean_data),
                        'outliers_removed': len(outliers),
                        'outlier_percentage': len(outliers) / len(df) * 100,
                        'mean': clean_data[target_column].mean(),
                        'std': clean_data[target_column].std(),
                        'min': clean_data[target_column].min(),
                        'max': clean_data[target_column].max(),
                        'median': clean_data[target_column].median(),
                        'mean_change': clean_data[target_column].mean() - results['original']['mean'],
                        'std_change': clean_data[target_column].std() - results['original']['std']
                    }
            except Exception as e:
                logger.warning(f"Error in sensitivity analysis for {method}: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def visualize_outliers(self, df: pd.DataFrame, outliers: pd.DataFrame, 
                          feature: str = 'market_price') -> plt.Figure:
        """
        Create visualization comparing data with and without outliers.
        
        Args:
            df (pd.DataFrame): Original data
            outliers (pd.DataFrame): Detected outliers
            feature (str): Feature to visualize
            
        Returns:
            plt.Figure: Matplotlib figure with comparison plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Outlier Analysis: {feature}', fontsize=16)
        
        clean_data = df[~df.index.isin(outliers.index)]
        
        # Box plot comparison
        axes[0, 0].boxplot([clean_data[feature].dropna(), df[feature].dropna()], 
                          labels=['Without Outliers', 'With Outliers'])
        axes[0, 0].set_title('Box Plot Comparison')
        axes[0, 0].set_ylabel(feature)
        
        # Histogram comparison
        axes[0, 1].hist(clean_data[feature].dropna(), alpha=0.7, label='Without Outliers', bins=30)
        axes[0, 1].hist(df[feature].dropna(), alpha=0.7, label='With Outliers', bins=30)
        axes[0, 1].set_title('Distribution Comparison')
        axes[0, 1].set_xlabel(feature)
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Scatter plot (if pricing_error exists)
        if 'pricing_error' in df.columns and feature != 'pricing_error':
            axes[1, 0].scatter(clean_data[feature], clean_data['pricing_error'], 
                              alpha=0.6, label='Without Outliers')
            axes[1, 0].scatter(outliers[feature], outliers['pricing_error'], 
                              alpha=0.8, color='red', label='Outliers')
            axes[1, 0].set_xlabel(feature)
            axes[1, 0].set_ylabel('Pricing Error')
            axes[1, 0].set_title('Scatter Plot with Outliers Highlighted')
            axes[1, 0].legend()
        
        # Summary statistics table
        stats_data = {
            'Metric': ['Count', 'Mean', 'Std', 'Min', 'Max'],
            'With Outliers': [
                len(df[feature].dropna()),
                f"{df[feature].mean():.4f}",
                f"{df[feature].std():.4f}",
                f"{df[feature].min():.4f}",
                f"{df[feature].max():.4f}"
            ],
            'Without Outliers': [
                len(clean_data[feature].dropna()),
                f"{clean_data[feature].mean():.4f}",
                f"{clean_data[feature].std():.4f}",
                f"{clean_data[feature].min():.4f}",
                f"{clean_data[feature].max():.4f}"
            ]
        }
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=list(zip(*stats_data.values()))[1:],
                                colLabels=list(stats_data.keys()),
                                rowLabels=stats_data['Metric'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        return fig

def flag_outliers(df: pd.DataFrame, method: str = 'isolation_forest', 
                 contamination: float = 0.1) -> pd.DataFrame:
    """
    Convenience function to flag outliers without removing them.
    
    Args:
        df (pd.DataFrame): Input data
        method (str): Detection method
        contamination (float): Expected proportion of outliers
        
    Returns:
        pd.DataFrame: Data with outlier flags added
    """
    detector = OutlierDetector()
    clean_data, outliers = detector.detect_outliers(df, method=method, contamination=contamination)
    
    result_df = df.copy()
    result_df['is_outlier'] = result_df.index.isin(outliers.index)
    result_df['outlier_method'] = method
    
    return result_df

def remove_outliers(df: pd.DataFrame, method: str = 'isolation_forest', 
                   contamination: float = 0.1) -> pd.DataFrame:
    """
    Convenience function to remove outliers from dataset.
    
    Args:
        df (pd.DataFrame): Input data
        method (str): Detection method
        contamination (float): Expected proportion of outliers
        
    Returns:
        pd.DataFrame: Data with outliers removed
    """
    detector = OutlierDetector()
    clean_data, _ = detector.detect_outliers(df, method=method, contamination=contamination)
    
    return clean_data
