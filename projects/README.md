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
│   ├── generate_data.py              # Script to download and process market data
│   ├── black_scholes.py              # Black-Scholes pricing implementation
│   ├── data_preprocessing.py         # Data cleaning and feature engineering
│   └── pricer_tool.py               # Final ML-enhanced pricing tool
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb     # Initial data analysis and visualization
│   ├── 02_model_training.ipynb       # Model development and training
│   └── 03_model_evaluation.ipynb     # Performance evaluation and validation
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
3. **Model Training**: Open `notebooks/03_model_training.ipynb` to train the Random Forest model (coming next)
4. **Pricing Tool**: Use the final pricing tool for ML-corrected option prices (coming next)

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

## Success Metrics

- **Mean Absolute Error (MAE)**: Reduction compared to baseline Black-Scholes model
- **Root Mean Square Error (RMSE)**: Overall prediction accuracy
- **R-squared**: Model explanatory power
- **Latency**: Sub-second pricing response time

## Contributors

This project is part of the NYU FRE Bootcamp coursework focusing on practical machine learning applications in quantitative finance.
