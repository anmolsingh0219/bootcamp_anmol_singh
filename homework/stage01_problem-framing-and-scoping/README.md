# Project: ML-Enhanced European Option Pricing

This project aims to develop a machine learning model that corrects for the known pricing errors of the Black-Scholes model, providing a more accurate and reliable pricing tool for traders in real-time.

## Project Scope

The pricing of even simple European options is notoriously imperfect. The foundational Black-Scholes model, while fast, relies on unrealistic assumptions (e.g., constant volatility, frictionless markets) that cause its theoretical prices to deviate from actual market prices. This discrepancy, or "error," represents both risk and missed opportunity. This project aims to solve this accuracy problem by developing a machine learning model to predict and correct for the Black-Scholes pricing error.

The primary stakeholder is an options trader or a quantitative analyst (quant). This user needs highly accurate pricing data to execute trades, manage risk (the "Greeks"), and identify mispricings. Relying on a flawed model leads to suboptimal decisions. The useful answer for this stakeholder is predictive; we are forecasting the pricing error of a traditional model. The final output will be a functional artifact: a Python tool that takes market inputs, calculates the Black-Scholes price, and then applies an ML-driven correction to return a more accurate final price instantly.

## Goals → Lifecycle → Deliverables

| **Project Goal** | **Lifecycle Stage** | **Key Deliverable(s)** |
| -------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| 1. Generate a robust training dataset of Black-Scholes pricing errors. | **Data Sourcing & Engineering** | A script (`/src/generate_data.py`) to download market data and calculate the `Actual Price - Black-Scholes Price` error for each option. Saved dataset in `/data/`. |
| 2. Develop a highly accurate error prediction model. | **Model Development & Training** | A Jupyter Notebook (`/notebooks/model_training.ipynb`) detailing the training of a Random Forest model to predict the pricing error. The trained model saved as a `.pkl` file. |
| 3. Validate the model's performance and accuracy gains. | **Model Evaluation** | A summary report in the notebook showing performance metrics (e.g., Mean Absolute Error) and charts comparing the corrected price vs. the actual market price. |
| 4. Provide the end-user with a fast, enhanced pricing tool. | **Deployment & Artifact Creation** | A final script (`/src/pricer_tool.py`) that loads the trained model and allows a user to input option parameters to get an instant, ML-corrected price. |