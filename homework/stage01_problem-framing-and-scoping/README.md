# Project: ML-Powered Bermudan Option Pricing

This project aims to develop a machine learning model to rapidly and accurately predict the price of Bermudan put options, addressing the computational bottlenecks of traditional pricing methods.

## Project Scope

The pricing of exotic options, such as **Bermudan options**, presents a significant computational challenge in finance. Unlike simpler European options, Bermudan options can be exercised on several specific dates before expiration. Traditional pricing methods, like Monte Carlo simulations or binomial lattices, are often computationally intensive and slow, making it difficult for traders to assess risks and opportunities in real-time. This project aims to solve this efficiency problem by developing a machine learning model to rapidly and accurately predict the price of Bermudan put options on a single underlying asset.

The primary stakeholder is an **exotic options trader** or a **quantitative analyst (quant)** on a derivatives desk. This user needs immediate pricing data to make informed trading decisions, manage their portfolio's risk exposure (the "Greeks"), and identify potential arbitrage opportunities. A slow pricing model directly translates to lost opportunities and increased risk. The useful answer for this stakeholder is **predictive**; we are forecasting the option's price based on market parameters. The final output will be a functional **artifact**: a Python script or notebook that takes market inputs (e.g., stock price, strike price, volatility, interest rate, time to maturity, and exercise dates) and returns a predicted price and its associated risk sensitivities almost instantly.

## Goals → Lifecycle → Deliverables

| **Project Goal** | **Lifecycle Stage** | **Key Deliverable(s)** |
| -------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| 1. Generate a robust training dataset of option prices.  | **Data Generation & Engineering** | A script (`/src/generate_data.py`) using a traditional (slow) pricer to create a large dataset of input parameters and their corresponding "true" prices. Saved dataset in `/data/`. |
| 2. Develop a highly accurate pricing model.              | **Model Development & Training** | A Jupyter Notebook (`/notebooks/model_training.ipynb`) detailing the experimentation and training of a neural network or gradient boosting model. The trained model saved as a file (e.g., `.h5` or `.pkl`). |
| 3. Validate the model's performance against industry standards. | **Model Evaluation** | A summary report or section in the notebook with performance metrics (e.g., Mean Squared Error, R-squared) and visualizations comparing ML predictions to true prices. |
| 4. Provide the end-user with a fast, usable pricing tool. | **Deployment & Artifact Creation** | A final, clean script or notebook (`/src/pricer_tool.py`) that loads the trained model and allows a user to input market parameters to get an instant price prediction. |