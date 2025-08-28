import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

def load_model(model_path):
    """Load pickled model and metadata"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def preprocess_features(implied_vol, underlying_price, strike, time_to_expiry):
    """Preprocess input features for prediction"""
    moneyness = underlying_price / strike
    vol_time = implied_vol * time_to_expiry
    return [implied_vol, moneyness, vol_time]

def validate_inputs(implied_vol, moneyness, vol_time):
    """Validate input ranges based on training data"""
    errors = []
    
    if not (0.1 <= implied_vol <= 2.0):
        errors.append("Implied volatility must be between 0.1 and 2.0")
    
    if not (0.5 <= moneyness <= 2.0):
        errors.append("Moneyness must be between 0.5 and 2.0")
    
    if not (0.0 <= vol_time <= 0.5):
        errors.append("Vol-time interaction must be between 0.0 and 0.5")
    
    return errors

def calculate_confidence_interval(prediction, mae, confidence=0.95):
    """Calculate prediction confidence interval"""
    z_score = 1.96 if confidence == 0.95 else 1.645
    margin = z_score * mae
    return {
        'prediction': float(prediction),
        'lower_bound': float(prediction - margin),
        'upper_bound': float(prediction + margin),
        'confidence': confidence
    }

def load_and_predict(model_path, implied_vol, moneyness, vol_time):
    """Load model and make prediction with validation"""
    # Validate inputs
    errors = validate_inputs(implied_vol, moneyness, vol_time)
    if errors:
        return {'error': 'Validation failed', 'details': errors}
    
    try:
        # Load model
        model_data = load_model(model_path)
        model = model_data['model']
        mae = model_data['mae']
        features = model_data.get('features', ['implied_volatility', 'moneyness', 'vol_time'])
        
        # Create DataFrame with feature names to avoid warnings
        features_df = pd.DataFrame({
            features[0]: [implied_vol],
            features[1]: [moneyness], 
            features[2]: [vol_time]
        })
        
        prediction = model.predict(features_df)[0]
        
        # Calculate confidence interval
        result = calculate_confidence_interval(prediction, mae)
        result['inputs'] = {
            'implied_volatility': implied_vol,
            'moneyness': moneyness,
            'vol_time': vol_time
        }
        return result
        
    except Exception as e:
        return {'error': 'Prediction failed', 'details': str(e)}
