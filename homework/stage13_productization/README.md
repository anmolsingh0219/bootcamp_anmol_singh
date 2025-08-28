# Options Pricing Model - Production Deployment

## Project Overview

This project productizes an options pricing model. The model predicts options market prices based on implied volatility, moneyness, and volatility-time interactions.

### Key Findings
- **Best Model Performance**: MAE $58.8 (Drop Missing scenario)
- **R-Squared**: 15.6% (limited explanatory power)
- **Missing Data Impact**: 19.3% performance improvement when dropping missing values
- **Risk Assessment**: Model requires enhancement before production trading use

## Project Structure

```
stage13_productization/
├── app.py                     # Flask API
├── app_streamlit.py          # Streamlit dashboard
├── requirements.txt          # Dependencies
├── README.md                # This file
├── data/                    # Training data (linked from Stage 11)
├── model/                   # Pickled models
│   └── options_pricing_model.pkl
├── src/                     # Utility functions
│   └── utils.py
├── notebooks/               # Development notebooks
│   └── stage13_productization_homework-starter.ipynb
└── reports/                 # Output reports
```

## Quick Start

### 1. Environment Setup
```bash
# Create dedicated environment for Stage 13
python -m venv stage13_env

# Activate environment (Windows)
stage13_env\Scripts\activate

# Or for PowerShell
stage13_env\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import flask, streamlit; print('Dependencies installed successfully')"
```

### 2. Train and Save Model
Run the notebook to train and pickle the model:
```bash
jupyter notebook notebooks/stage13_productization_homework-starter.ipynb
```

### 3. Run Flask API
```bash
python app.py
```
API will be available at http://127.0.0.1:5000

### 4. Run Streamlit Dashboard
```bash
streamlit run app_streamlit.py
```
Dashboard will open in your browser.

## API Usage

### Endpoints

**GET /predict/{vol}/{moneyness}/{vol_time}**
- Example: `/predict/0.25/1.05/0.05`
- Returns: Price prediction with confidence interval

**POST /predict**
- Body: `{"implied_volatility": 0.25, "moneyness": 1.05, "vol_time": 0.05}`
- Returns: Price prediction with confidence interval

**GET /plot**
- Returns: Volatility sensitivity chart

**GET /health**
- Returns: API health status

### Example Usage

```python
import requests

# Single prediction
response = requests.get('http://127.0.0.1:5000/predict/0.25/1.05/0.05')
print(response.json())

# POST prediction
data = {'implied_volatility': 0.25, 'moneyness': 1.05, 'vol_time': 0.05}
response = requests.post('http://127.0.0.1:5000/predict', json=data)
print(response.json())
```

## Model Assumptions & Limitations

### Assumptions
1. **Linear Relationship**: Implied volatility linearly predicts market price
2. **Missing at Random**: Volatility missingness unrelated to price patterns
3. **Temporal Stability**: Relationships remain stable across market conditions

### Limitations
- **Low R² (15.6%)**: Limited explanatory power
- **Missing Data Sensitivity**: 19.3% performance degradation with imputation
- **Segment Bias**: Significant prediction errors for low-strike options
- **Wide Confidence Intervals**: High prediction uncertainty

### Input Validation Ranges
- **Implied Volatility**: 0.1 - 2.0
- **Moneyness**: 0.5 - 2.0
- **Vol-Time Interaction**: 0.0 - 0.5

## Risk Assessment

**Risk Level: MEDIUM-HIGH**

### High Risk Areas
- Model explains only 15.6% of price variation
- Missing data creates systematic bias
- Large prediction errors for extreme moneyness

### Mitigation Recommendations
1. Use only for directional guidance, not precise pricing
2. Implement real-time data quality monitoring
3. Develop segment-specific models for different strike ranges
4. Regular model retraining and validation

## Development Lifecycle

### Data Pipeline
1. **Stage 5-6**: Data storage and preprocessing
2. **Stage 7-8**: Outlier detection and EDA
3. **Stage 9**: Feature engineering (moneyness, vol-time interaction)
4. **Stage 10**: Model training (linear regression)
5. **Stage 11**: Risk evaluation and bootstrap validation
6. **Stage 12**: Stakeholder reporting
7. **Stage 13**: Production deployment (this stage)

### Model Versions
- **v1.0**: Baseline linear regression with implied volatility
- **v1.1**: Added moneyness and vol-time features
- **v1.2**: Optimized with drop-missing strategy (current)

## Testing Evidence

### Local Testing
1. Start Flask API: `python app.py`
2. Test endpoints using notebook or curl
3. Verify error handling with invalid inputs
4. Check confidence interval calculations

### Dashboard Testing
1. Start Streamlit: `streamlit run app_streamlit.py`
2. Test input sliders and validation
3. Verify sensitivity analysis charts
4. Check performance metrics display

## Reproducibility Checklist

- [ ] Model pickle file exists in `/model/`
- [ ] All dependencies in `requirements.txt`
- [ ] API endpoints return valid JSON
- [ ] Dashboard loads without errors
- [ ] Input validation works correctly
- [ ] Confidence intervals calculated properly

## Next Steps

1. **Enhanced Features**: Add Greeks (delta, gamma, theta, vega)
2. **Alternative Models**: Implement Black-Scholes benchmark
3. **Real-time Data**: Connect to live options data feeds
4. **Authentication**: Add API key-based security
5. **Monitoring**: Implement prediction accuracy tracking
6. **Scaling**: Containerize for cloud deployment

For technical questions, refer to the development notebooks in previous stages or check the API `/health` endpoint for current model status.
