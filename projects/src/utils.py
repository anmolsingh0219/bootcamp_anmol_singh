import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any
import logging

# Import existing modules
from model_trainer import OptionsModelTrainer, calculate_metrics
from model_evaluator import ModelEvaluator
from black_scholes import BlackScholesCalculator

logger = logging.getLogger(__name__)

class OptionsPricingAnalyzer:
    """
    High-level analyzer for options pricing with ML enhancement.
    Orchestrates the complete analysis pipeline.
    """
    
    def __init__(self, model_path: str = './models/random_forest_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.bs_calculator = BlackScholesCalculator()
        
    def load_model(self) -> bool:
        """Load the trained model if it exists."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                # Load feature names from metadata
                metadata_path = self.model_path.replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_names = metadata.get('feature_names', [])
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Model not found at {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_single_option(self, option_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict pricing error for a single option.
        
        Args:
            option_data (Dict): Option parameters
            
        Returns:
            Dict: Prediction results with Black-Scholes price and ML correction
        """
        if not self.model:
            if not self.load_model():
                raise ValueError("No trained model available")
        
        try:
            # Calculate Black-Scholes price
            if option_data.get('contract_type', 'call').lower() == 'call':
                bs_price = self.bs_calculator.call_price(
                    S=option_data['underlying_price'],
                    K=option_data['strike'],
                    T=option_data['time_to_expiry'],
                    r=option_data['risk_free_rate'],
                    sigma=option_data['implied_volatility']
                )
            else:
                bs_price = self.bs_calculator.put_price(
                    S=option_data['underlying_price'],
                    K=option_data['strike'],
                    T=option_data['time_to_expiry'],
                    r=option_data['risk_free_rate'],
                    sigma=option_data['implied_volatility']
                )
            
            # Prepare features for ML model
            features = self._prepare_single_prediction_features(option_data)
            
            # Get ML prediction (pricing error)
            if self.feature_names and len(features) == len(self.feature_names):
                feature_array = np.array([features]).reshape(1, -1)
                predicted_error = self.model.predict(feature_array)[0]
            else:
                raise ValueError("Feature mismatch with trained model")
            
            # Calculate ML-corrected price
            ml_price = bs_price + predicted_error
            
            return {
                'black_scholes_price': float(bs_price),
                'predicted_error': float(predicted_error),
                'ml_corrected_price': float(ml_price),
                'improvement': float(abs(predicted_error)),
                'contract_type': option_data.get('contract_type', 'call'),
                'strike': option_data['strike'],
                'underlying_price': option_data['underlying_price'],
                'time_to_expiry': option_data['time_to_expiry'],
                'implied_volatility': option_data['implied_volatility']
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _prepare_single_prediction_features(self, option_data: Dict[str, float]) -> list:
        """Prepare features for a single option prediction matching the trained model exactly."""
        
        # Calculate derived features
        moneyness = option_data['underlying_price'] / option_data['strike']
        time_to_expiry = option_data['time_to_expiry']
        implied_vol = option_data['implied_volatility']
        
        # Get defaults for missing optional fields
        open_interest = option_data.get('open_interest', 100)
        volume = option_data.get('volume', 50)
        risk_free_rate = option_data.get('risk_free_rate', 0.05)
        
        # Contract type indicators
        contract_type = option_data.get('contract_type', 'call').lower()
        is_call = 1 if contract_type == 'call' else 0
        is_put = 1 if contract_type == 'put' else 0
        
        # Moneyness categories
        itm = 1 if moneyness > 1.0 else 0
        otm = 1 if moneyness < 1.0 else 0  
        atm = 1 if abs(moneyness - 1.0) <= 0.05 else 0
        
        # Engineered features with defaults for missing market data
        bid_ask_spread = option_data.get('bid_ask_spread', 0.05)  # Default spread
        relative_spread = bid_ask_spread / option_data.get('market_price', option_data['underlying_price'] * 0.1)
        
        # Features in exact order expected by the model (from metadata)
        features = [
            option_data['strike'],                                    # strike
            implied_vol,                                             # implied_volatility  
            open_interest,                                           # open_interest
            volume,                                                  # volume
            option_data['underlying_price'],                         # underlying_price
            risk_free_rate,                                          # risk_free_rate
            time_to_expiry,                                          # time_to_expiry
            moneyness,                                               # moneyness
            0.0,                                                     # relative_error_robust (placeholder)
            bid_ask_spread,                                          # bid_ask_spread
            relative_spread,                                         # relative_spread
            np.log(moneyness),                                       # log_moneyness
            np.sqrt(time_to_expiry),                                 # sqrt_time
            implied_vol * np.sqrt(time_to_expiry),                   # vega_proxy
            np.log1p(volume) + np.log1p(open_interest),              # liquidity_score
            is_call,                                                 # is_call
            is_put,                                                  # is_put
            itm,                                                     # itm
            otm,                                                     # otm
            atm,                                                     # atm
            1 if contract_type == 'call' else 0                     # contract_type (numeric)
        ]
        
        return features
    
    def run_full_analysis(self, data_path: str = '../data/processed/cleaned_outliers_latest.csv') -> Dict[str, Any]:
        """
        Run complete analysis pipeline and generate results.
        
        Args:
            data_path (str): Path to processed data
            
        Returns:
            Dict: Complete analysis results
        """
        try:
            # Load data
            if not os.path.exists(data_path):
                # Try alternative paths
                alternative_paths = [
                    '../data/processed/eda_features_20250823_191052.csv',
                    '../data/processed/cleaned_options_data_20250823_164021.csv'
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        data_path = alt_path
                        break
                else:
                    raise FileNotFoundError("No processed data file found")
            
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data: {df.shape}")
            
            trainer = OptionsModelTrainer(random_state=42)
            evaluator = ModelEvaluator(random_state=42)
            
            # Prepare features
            X, y, feature_names = trainer.prepare_features(df, target_col='pricing_error')
            
            # Train or load model
            model_exists = os.path.exists(self.model_path)
            if not model_exists:
                logger.info("Training new model...")
                X_train, X_test, y_train, y_test = trainer.time_aware_split(X, y, test_size=0.2)
                
                # Train Random Forest only
                training_results = trainer.train_model(X_train, y_train, 'random_forest')
                
                # Save model and metadata
                os.makedirs('../models', exist_ok=True)
                joblib.dump(trainer.models['random_forest'], self.model_path)
                
                # Save metadata
                metadata = {
                    'feature_names': feature_names,
                    'model_type': 'random_forest',
                    'training_date': datetime.now().isoformat(),
                    'data_shape': {'train': X_train.shape, 'test': X_test.shape}
                }
                metadata_path = self.model_path.replace('.pkl', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Evaluate model
                y_pred = trainer.predict(X_test, 'random_forest')
                test_metrics = calculate_metrics(y_test, y_pred)
                
            else:
                logger.info("Using existing model...")
                self.load_model()
                # Quick evaluation on full dataset
                y_pred = self.model.predict(X)
                test_metrics = calculate_metrics(y, y_pred)
                X_test, y_test = X, y  # Use full dataset for display
            
            # Generate results summary
            results = {
                'model_performance': test_metrics,
                'data_summary': {
                    'total_samples': len(df),
                    'features_used': len(feature_names),
                    'contracts_by_type': df['contract_type'].value_counts().to_dict() if 'contract_type' in df.columns else {},
                },
                'key_metrics': {
                    'mae': test_metrics['mae'],
                    'rmse': test_metrics['rmse'],
                    'r2': test_metrics['r2'],
                    'improvement_vs_baseline': f"{((4.6 - test_metrics['mae'])/4.6)*100:.0f}%"
                },
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            
            logger.info("Full analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Full analysis failed: {e}")
            raise ValueError(f"Analysis failed: {str(e)}")

def validate_option_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean option input data."""
    required_fields = ['underlying_price', 'strike', 'time_to_expiry', 'implied_volatility']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(data[field], (int, float)) or data[field] <= 0:
            raise ValueError(f"Invalid {field}: must be positive number")
    
    # Set defaults and validate
    validated = {
        'underlying_price': float(data['underlying_price']),
        'strike': float(data['strike']),
        'time_to_expiry': float(data['time_to_expiry']),
        'implied_volatility': float(data['implied_volatility']),
        'contract_type': data.get('contract_type', 'call').lower(),
        'risk_free_rate': float(data.get('risk_free_rate', 0.05)),
        'volume': float(data.get('volume', 50)),
        'open_interest': float(data.get('open_interest', 100)),
    }
    
    # Validate ranges
    if validated['time_to_expiry'] > 2.0:
        raise ValueError("Time to expiry cannot exceed 2 years")
    if validated['implied_volatility'] > 5.0:
        raise ValueError("Implied volatility cannot exceed 500%")
    if validated['contract_type'] not in ['call', 'put']:
        raise ValueError("Contract type must be 'call' or 'put'")
    
    return validated

def create_sample_prediction() -> Dict[str, Any]:
    """Create a sample prediction for demonstration."""
    sample_option = {
        'underlying_price': 450.0,
        'strike': 455.0,
        'time_to_expiry': 0.0833,  # ~1 month
        'implied_volatility': 0.25,  # 25%
        'contract_type': 'call',
        'risk_free_rate': 0.05
    }
    
    analyzer = OptionsPricingAnalyzer()
    return analyzer.predict_single_option(sample_option)
