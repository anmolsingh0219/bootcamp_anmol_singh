# Project: ML-Enhanced European Option Pricing

This project aims to develop a machine learning model that corrects for the known pricing errors of the Black-Scholes model, providing a more accurate and reliable pricing tool for traders in real-time.

## Project Overview

The Black-Scholes model is the industry standard for pricing European options, but its assumptions are unrealistic, leading to persistent pricing errors. This project develops a machine learning model to automatically predict and correct these errors, providing traders with a faster, more accurate pricing tool.

## Project Structure

```
projects/
├── README.md                           # This file - project overview and setup
├── .gitignore                         # Git ignore patterns
├── .gitkeep                          # Keep empty directories in git
├── data/                             # Data storage
│   ├── raw/                          # Raw market data from sources
│   │   └── options_market_data.csv   # Historical options market data
│   └── processed/                    # Cleaned and processed datasets
│       └── training_data.csv         # Final training dataset with BS errors
├── src/                              # Source code
│   ├── __init__.py                   # Package initialization
│   ├── data_fetcher.py               # Script to download and process market data
│   ├── black_scholes.py              # Black-Scholes pricing implementation
│   ├── data_processor.py             # Data cleaning and feature engineering
│   ├── cleaning.py                   # Dedicated data cleaning functions
│   ├── outliers.py                   # Outlier detection and analysis
│   └── pricer_tool.py               # Final ML-enhanced pricing tool (coming next)
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_fetching_and_processing.ipynb  # Data pipeline and processing
│   ├── 02_exploratory_data_analysis.ipynb     # EDA and feature engineering
│   ├── 03_sensitivity_outliers.ipynb          # Outlier analysis and sensitivity testing
│   ├── 04_model_training.ipynb                # Model development and training (coming next)
│   └── 05_model_evaluation.ipynb              # Performance evaluation and validation (coming next)
└── docs/                             # Documentation
    ├── stakeholder_memo.md           # Project requirements and scope
    ├── model_validation_report.md    # Final model performance report
    └── user_guide.md                 # How to use the pricing tool
```

## Goals → Lifecycle → Deliverables

| **Project Goal** | **Lifecycle Stage** | **Key Deliverable(s)** |
| -------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| 1. Generate a robust training dataset of Black-Scholes pricing errors. | **Data Sourcing & Engineering** | A script (`src/generate_data.py`) to download market data and calculate the `Actual Price - Black-Scholes Price` error for each option. Saved dataset in `data/processed/`. |
| 2. Develop a highly accurate error prediction model. | **Model Development & Training** | A Jupyter Notebook (`notebooks/02_model_training.ipynb`) detailing the training of a Random Forest model to predict the pricing error. The trained model saved as a `.pkl` file. |
| 3. Validate the model's performance and accuracy gains. | **Model Evaluation** | A summary report (`docs/model_validation_report.md`) showing performance metrics (e.g., Mean Absolute Error) and charts comparing the corrected price vs. the actual market price. |
| 4. Provide the end-user with a fast, enhanced pricing tool. | **Deployment & Artifact Creation** | A final script (`src/pricer_tool.py`) that loads the trained model and allows a user to input option parameters to get an instant, ML-corrected price. |

## Getting Started

### Prerequisites
- Python 3.8+
- Alpha Vantage API key (free tier: 500 calls/day)
- Required packages (install via `pip install -r requirements.txt`)

### Environment Setup
1. **Clone/Download** this project
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Environment Variables**: Create a `.env` file in the project root with:
   ```
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
   TARGET_SYMBOL=SPY
   RISK_FREE_RATE_TICKER=^IRX
   ```

### Quick Start Workflow
1. **Data Pipeline**: Run `notebooks/01_data_fetching_and_processing.ipynb` to fetch and process options data
2. **Exploratory Analysis**: Use `notebooks/02_exploratory_data_analysis.ipynb` for comprehensive EDA
3. **Outlier Analysis**: Run `notebooks/03_sensitivity_outliers.ipynb` for outlier detection and sensitivity analysis
4. **Model Training**: Run `notebooks/04_model_training_evaluation.ipynb` to train and evaluate models
5. **API Usage**: Use `python app.py` for REST API or `streamlit run app_streamlit.py` for interactive dashboard

### Production Usage
- **Flask API**: `python app.py` (runs on http://localhost:5000)
- **Streamlit Dashboard**: `streamlit run app_streamlit.py` (runs on http://localhost:8501)
- **Batch Processing**: Use the API `/run_full_analysis` endpoint or dashboard batch analysis

### Project Workflow
```
Data Fetching → Data Cleaning → Feature Engineering → EDA → Outlier Detection → Model Training → Evaluation → Deployment
```

## Key Features

- **Real-time Pricing**: Get instant, ML-corrected European option prices
- **Error Correction**: Automatically adjusts for known Black-Scholes model limitations
- **Market Data Integration**: Uses Yahoo Finance for historical options data
- **Performance Metrics**: Comprehensive model validation and accuracy reporting
- **User-Friendly Interface**: Simple Python script for traders and analysts

## Model Approach

The project uses a **Random Forest** model to predict the pricing error between actual market prices and Black-Scholes theoretical prices. The model takes standard option parameters as input:

- Current stock price
- Strike price  
- Time to expiration
- Risk-free rate
- Implied volatility

The final price is calculated as: `ML-Corrected Price = Black-Scholes Price + Predicted Error`

## Feature Engineering

The project implements comprehensive feature engineering to enhance model performance:

### Core Features
- **Moneyness**: S/K ratio indicating in/out-of-the-money status
- **Time Features**: Time to expiration, days to expiry, short/long-term flags
- **Volatility Features**: Implied volatility and volatility-time interactions

### Engineered Features
- **log_moneyness**: Log-transformed moneyness for better distribution
- **sqrt_time_to_expiry**: Square root of time for volatility scaling
- **vol_time_product**: Volatility × Time interaction
- **vol_moneyness**: Volatility × Moneyness interaction
- **rate_time**: Risk-free rate × Time interaction

### Market Microstructure
- **bid_ask_spread**: Ask - Bid price difference
- **relative_spread**: Spread relative to market price
- **log_volume**: Log-transformed trading volume
- **log_open_interest**: Log-transformed open interest
- **liquidity_score**: Combined volume and open interest metric

### Option Classification
- **is_itm/otm/atm**: Binary flags for moneyness categories
- **is_call/put**: Option type indicators
- **is_short_term/long_term**: Time-based classification

All features are designed to capture the key factors that cause Black-Scholes pricing errors in real markets.

## API Endpoints

### Flask REST API (`python app.py`)

**Base URL**: `http://localhost:5000`

#### Available Endpoints:
- `GET /` - API documentation and health check
- `POST /predict` - Predict option price with JSON payload
- `GET /predict/<underlying>/<strike>/<time>/<volatility>` - Quick prediction with URL parameters  
- `POST /run_full_analysis` - Run complete analysis pipeline
- `GET /sample` - Get sample prediction for testing
- `GET /health` - Health check

#### Example API Request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "underlying_price": 450.0,
    "strike": 455.0,
    "time_to_expiry": 0.0833,
    "implied_volatility": 0.25,
    "contract_type": "call"
  }'
```

#### Example Response:
```json
{
  "success": true,
  "prediction": {
    "black_scholes_price": 12.3456,
    "predicted_error": -0.2345,
    "ml_corrected_price": 12.1111,
    "improvement": 0.2345,
    "contract_type": "call"
  }
}
```

### Streamlit Dashboard (`streamlit run app_streamlit.py`)

Interactive web interface with:
- Single option pricing calculator
- Batch analysis from CSV upload
- Model performance monitoring  
- Real-time visualization and results

## Setup Instructions

### From Fresh Git Clone:

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd projects
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Analysis Pipeline**
   ```bash
   # Option 1: Run notebooks in order
   jupyter notebook notebooks/04_model_training_evaluation.ipynb
   
   # Option 2: Use API for full analysis
   python app.py
   # Then POST to http://localhost:5000/run_full_analysis
   ```

4. **Start Applications**
   ```bash
   # Flask API
   python app.py
   
   # Streamlit Dashboard (separate terminal)
   streamlit run app_streamlit.py
   ```

## Success Metrics

- **Mean Absolute Error (MAE)**: Reduction compared to baseline Black-Scholes model
- **Root Mean Square Error (RMSE)**: Overall prediction accuracy
- **R-squared**: Model explanatory power
- **Latency**: Sub-second pricing response time

