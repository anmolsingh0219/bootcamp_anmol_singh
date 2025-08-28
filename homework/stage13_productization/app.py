from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import logging
from src.utils import validate_inputs, calculate_confidence_interval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model on startup
try:
    with open('model/options_pricing_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
    mae = model_data['mae']
    logger.info(f"Model loaded successfully. MAE: {mae:.2f}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route('/')
def home():
    """API documentation endpoint"""
    docs = """
    <h1>Options Pricing Model API</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li><b>GET /predict/&lt;vol&gt;/&lt;moneyness&gt;/&lt;vol_time&gt;</b> - Single prediction</li>
        <li><b>POST /predict</b> - Prediction with JSON body</li>
        <li><b>GET /plot</b> - Sample visualization</li>
        <li><b>GET /health</b> - Health check</li>
    </ul>
    <h3>Example:</h3>
    <p>GET /predict/0.25/1.05/0.05 → Predict price for 25% vol, 5% moneyness, 5% vol-time</p>
    <p>POST /predict with {"implied_volatility": 0.25, "moneyness": 1.05, "vol_time": 0.05}</p>
    """
    return docs

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'features': features if model else None
    })

@app.route('/predict/<float:implied_vol>/<float:moneyness>/<float:vol_time>')
def predict_get(implied_vol, moneyness, vol_time):
    """GET prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Validate inputs
        errors = validate_inputs(implied_vol, moneyness, vol_time)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        features_df = pd.DataFrame({
            features[0]: [implied_vol],
            features[1]: [moneyness], 
            features[2]: [vol_time]
        })
        prediction = model.predict(features_df)[0]
        
        # Add confidence interval
        result = calculate_confidence_interval(prediction, mae)
        result['inputs'] = {
            'implied_volatility': implied_vol,
            'moneyness': moneyness,
            'vol_time': vol_time
        }
        
        logger.info(f"Prediction made: {prediction:.2f}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_post():
    """POST prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features
        implied_vol = data.get('implied_volatility')
        moneyness = data.get('moneyness')
        vol_time = data.get('vol_time')
        
        if any(x is None for x in [implied_vol, moneyness, vol_time]):
            return jsonify({
                'error': 'Missing required fields',
                'required': ['implied_volatility', 'moneyness', 'vol_time']
            }), 400
        
        # Validate inputs
        errors = validate_inputs(implied_vol, moneyness, vol_time)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        features_df = pd.DataFrame({
            features[0]: [implied_vol],
            features[1]: [moneyness], 
            features[2]: [vol_time]
        })
        prediction = model.predict(features_df)[0]
        
        # Add confidence interval
        result = calculate_confidence_interval(prediction, mae)
        result['inputs'] = data
        
        logger.info(f"POST prediction made: {prediction:.2f}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"POST prediction error: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/plot')
def plot():
    """Generate sample plot"""
    try:
        vol_range = np.linspace(0.1, 0.5, 50)
        moneyness = 1.0
        vol_time = 0.05
        
        predictions = []
        for vol in vol_range:
            features_array = np.array([[vol, moneyness, vol_time]])
            pred = model.predict(features_array)[0]
            predictions.append(pred)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(vol_range, predictions, 'b-', linewidth=2)
        plt.xlabel('Implied Volatility')
        plt.ylabel('Predicted Price ($)')
        plt.title('Options Price vs Implied Volatility\n(Moneyness=1.0, Vol-Time=0.05)')
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_bytes = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'<img src="data:image/png;base64,{img_bytes}" style="max-width:100%"/>'
        
    except Exception as e:
        logger.error(f"Plot error: {e}")
        return jsonify({'error': 'Plot generation failed', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
