import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class OptionsModelTrainer:
    """
    Comprehensive model trainer for options pricing error prediction.
    
    Supports multiple model types with automated hyperparameter tuning
    and proper train/test splitting for time series data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'pricing_error') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for modeling, excluding non-predictive columns.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Target variable column name
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, List[str]]: Features, target, feature names
        """
        # Exclude non-predictive columns
        exclude_cols = [
            'symbol', 'expiration_date', 'fetch_timestamp', 'simulated_trade_date',
            'market_price', 'last_price', 'bid', 'ask',  # These contain target info
            target_col, 'bs_price', 'relative_error', 'abs_pricing_error',  # Target-related
            'underlying_symbol', 'data_source'  # Additional non-numeric columns
        ]
        
        # Get all columns first
        all_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select only numeric columns for modeling
        X_temp = df[all_cols].copy()
        numeric_cols = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle categorical columns that should be numeric
        categorical_to_numeric = {
            'contract_type': {'call': 1, 'put': 0}
        }
        
        for col, mapping in categorical_to_numeric.items():
            if col in X_temp.columns:
                X_temp[col] = X_temp[col].map(mapping)
                if col not in numeric_cols:
                    numeric_cols.append(col)
        
        # Use only numeric columns
        feature_cols = numeric_cols
        X = X_temp[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Fill remaining missing values in numeric columns
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                # Skip non-numeric columns that somehow got through
                continue
            X[col] = X[col].fillna(X[col].median())
        
        logger.info(f"Prepared features: {len(feature_cols)} features, {len(X)} samples")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def time_aware_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create time-aware train/test split for options data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion for test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Sort by time to expiry for time-aware split
        if 'time_to_expiry' in X.columns:
            sort_idx = X['time_to_expiry'].argsort()
            X_sorted = X.iloc[sort_idx]
            y_sorted = y.iloc[sort_idx]
        else:
            X_sorted, y_sorted = X, y
        
        # Use the last portion as test set (most recent in terms of time to expiry)
        split_idx = int(len(X_sorted) * (1 - test_size))
        
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]
        
        logger.info(f"Time-aware split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create model configurations for training.
        
        Returns:
            Dict[str, Any]: Model configurations
        """
        models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'pipeline': False  # RF handles features well without scaling
            },
            'linear_regression': {
                'model': LinearRegression(),
                'params': {},  # No hyperparameters to tune
                'pipeline': True  # Needs scaling
            },
            'ridge_regression': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'pipeline': True  # Needs scaling
            }
        }
        
        return models
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_name: str, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Model name
            cv_folds (int): Cross-validation folds
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Training {model_name}...")
        
        models_config = self.create_models()
        config = models_config[model_name]
        
        # Create pipeline if needed
        if config['pipeline']:
            pipeline_steps = [
                ('scaler', StandardScaler()),
                ('model', config['model'])
            ]
            model_pipeline = Pipeline(pipeline_steps)
            
            # Adjust parameter names for pipeline
            params = {}
            for key, value in config['params'].items():
                params[f'model__{key}'] = value
        else:
            model_pipeline = config['model']
            params = config['params']
        
        # Hyperparameter tuning
        if params:
            grid_search = GridSearchCV(
                model_pipeline, 
                params, 
                cv=cv_folds, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = -grid_search.best_score_
        else:
            # No hyperparameters to tune
            model_pipeline.fit(X_train, y_train)
            best_model = model_pipeline
            best_params = {}
            cv_scores = cross_val_score(model_pipeline, X_train, y_train, 
                                      cv=cv_folds, scoring='neg_mean_absolute_error')
            cv_score = -cv_scores.mean()
        
        # Store model and scaler
        self.models[model_name] = best_model
        if config['pipeline']:
            self.scalers[model_name] = best_model.named_steps['scaler']
        
        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance[model_name] = best_model.feature_importances_
        elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['model'], 'feature_importances_'):
            self.feature_importance[model_name] = best_model.named_steps['model'].feature_importances_
        
        logger.info(f"{model_name} training complete. CV MAE: {cv_score:.4f}")
        
        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'model_name': model_name
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Dict[str, Any]]: All training results
        """
        results = {}
        models_config = self.create_models()
        
        for model_name in models_config.keys():
            try:
                results[model_name] = self.train_model(X_train, y_train, model_name)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X (pd.DataFrame): Features
            model_name (str): Model name
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        return model.predict(X)
    
    def save_models(self, save_dir: str = '../models/') -> Dict[str, str]:
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
            
        Returns:
            Dict[str, str]: Saved model paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            filename = f'{model_name}_model_{timestamp}.pkl'
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            saved_paths[model_name] = filepath
            logger.info(f"Saved {model_name} to {filepath}")
        
        return saved_paths
    
    def load_model(self, filepath: str, model_name: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to saved model
            model_name (str): Name to assign to loaded model
        """
        model = joblib.load(filepath)
        self.models[model_name] = model
        logger.info(f"Loaded model {model_name} from {filepath}")
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for a trained model.
        
        Args:
            model_name (str): Model name
            feature_names (List[str]): Feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in self.feature_importance:
            return pd.DataFrame()
        
        importance = self.feature_importance[model_name]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true (pd.Series): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Metrics dictionary
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true)
    }
