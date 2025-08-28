from flask import Flask, request, jsonify
import sys
import os
sys.path.append('src')

from src.utils import OptionsPricingAnalyzer, validate_option_input, create_sample_prediction
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize analyzer
analyzer = OptionsPricingAnalyzer('./models/random_forest_model.pkl')

@app.route('/', methods=['GET'])
def home():
    """API documentation and health check."""
    return jsonify({
        'service': 'ML-Enhanced Options Pricing API',
        'status': 'active',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict option price with ML correction',
            '/predict/<underlying>/<strike>/<time>/<volatility>': 'GET - Quick prediction with path parameters',
            '/run_full_analysis': 'POST - Run complete analysis pipeline',
            '/sample': 'GET - Get sample prediction for testing',
            '/health': 'GET - Health check'
        },
        'example_request': {
            'underlying_price': 450.0,
            'strike': 455.0,
            'time_to_expiry': 0.0833,
            'implied_volatility': 0.25,
            'contract_type': 'call'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_loaded = analyzer.load_model()
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'timestamp': analyzer.bs_calculator.__class__.__name__ + ' available'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict option price with ML correction."""
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate input
        validated_data = validate_option_input(data)
        
        # Make prediction
        result = analyzer.predict_single_option(validated_data)
        
        return jsonify({
            'success': True,
            'prediction': result['predicted_error'],
            'black_scholes_price': result['black_scholes_price'], 
            'ml_corrected_price': result['ml_corrected_price'],
            'input': validated_data
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/<float:underlying>/<float:strike>/<float:time>/<float:volatility>', methods=['GET'])
def predict_with_params(underlying, strike, time, volatility):
    """Quick prediction with path parameters (call option default)."""
    try:
        option_data = {
            'underlying_price': underlying,
            'strike': strike,
            'time_to_expiry': time,
            'implied_volatility': volatility,
            'contract_type': 'call'
        }
        
        validated_data = validate_option_input(option_data)
        result = analyzer.predict_single_option(validated_data)
        
        return jsonify({
            'success': True,
            'prediction': result['predicted_error'],
            'black_scholes_price': result['black_scholes_price'],
            'ml_corrected_price': result['ml_corrected_price'],
            'note': 'Used default call option type and risk-free rate'
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/<float:underlying>/<float:strike>/<float:time>/<float:volatility>/<contract_type>', methods=['GET'])
def predict_with_type(underlying, strike, time, volatility, contract_type):
    """Prediction with contract type specified."""
    try:
        option_data = {
            'underlying_price': underlying,
            'strike': strike,
            'time_to_expiry': time,
            'implied_volatility': volatility,
            'contract_type': contract_type.lower()
        }
        
        validated_data = validate_option_input(option_data)
        result = analyzer.predict_single_option(validated_data)
        
        return jsonify({
            'success': True,
            'prediction': result['predicted_error'],
            'black_scholes_price': result['black_scholes_price'],
            'ml_corrected_price': result['ml_corrected_price']
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/run_full_analysis', methods=['POST'])
def run_full_analysis():
    """Run complete analysis pipeline."""
    try:
        # Get optional parameters
        data = request.get_json() or {}
        data_path = data.get('data_path', './data/processed/cleaned_outliers_latest.csv')
        
        # Run analysis
        results = analyzer.run_full_analysis(data_path)
        
        return jsonify({
            'success': True,
            'analysis_results': results,
            'message': 'Full analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/sample', methods=['GET'])
def sample_prediction():
    """Get sample prediction for testing."""
    try:
        result = create_sample_prediction()
        return jsonify({
            'success': True,
            'prediction': result['predicted_error'],
            'black_scholes_price': result['black_scholes_price'],
            'ml_corrected_price': result['ml_corrected_price'],
            'input_data': {
                'underlying_price': 450.0,
                'strike': 455.0,
                'time_to_expiry': 0.0833,
                'implied_volatility': 0.25,
                'contract_type': 'call'
            },
            'note': 'This is a sample prediction for API testing'
        })
    except Exception as e:
        logger.error(f"Sample prediction error: {e}")
        return jsonify({'error': 'Could not generate sample prediction'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/',
            '/predict',
            '/predict/<underlying>/<strike>/<time>/<volatility>',
            '/run_full_analysis',
            '/sample',
            '/health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    model_loaded = analyzer.load_model()
    if model_loaded:
        logger.info("Model loaded successfully")
    else:
        logger.warning("No model found - will train on first analysis request")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
