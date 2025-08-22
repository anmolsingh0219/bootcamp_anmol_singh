# Project: ML-Enhanced European Option Pricing

**Stage:** Problem Framing & Scoping (Stage 01)

---

## Problem Statement

The Black-Scholes model is the industry standard for pricing European options, but its assumptions are unrealistic, leading to persistent pricing errors. Our traders currently have to manually adjust for these known flaws, which is inefficient and inconsistent. This project will develop a machine learning model to automatically predict and correct these errors, providing a faster, more accurate pricing tool to give our desk a competitive edge.

---

## Stakeholder & User

* **Decision Maker:** The **Head of Options Trading** is the primary sponsor, seeking to improve the desk's profitability and risk management.
* **Primary User:** An **Options Trader** needs this tool for immediate, reliable pricing to make rapid trading and hedging decisions.

---

## Useful Answer & Decision

* **Type:** The model is **predictive**, forecasting the error (Market Price - Black-Scholes Price).
* **Metrics:** Success is defined by a statistically significant reduction in pricing error (Mean Absolute Error) compared to the baseline Black-Scholes model, with near-zero latency.
* **Artifact:** The deliverable is a **Python tool** that can be used as a standalone script or integrated into the trader's existing workflow for on-demand pricing.

---

## Assumptions & Constraints

* **Assumption:** We can source sufficient historical market data from public sources (Yahoo Finance) to train a robust model.
* **Assumption:** The complex patterns of the Black-Scholes error are learnable and can be generalized by a Random Forest model.
* **Constraint:** The model's inputs must be readily available market parameters to ensure real-time usability.

---

## Known Unknowns / Risks

* **Model Performance:** The model's accuracy may degrade in extreme market conditions (a "black swan" event) not present in the training data. This will be mitigated by testing against historical volatility spikes.
* **User Adoption:** Traders might be skeptical of an ML-based correction. We will build trust by running the model in parallel with current methods and providing clear reports on its historical accuracy.

---

## Lifecycle Mapping

| Goal                      | Stage                 | Deliverable                                        |
| :------------------------ | :-------------------- | :------------------------------------------------- |
| 1. Create a training dataset. | **Data Sourcing** | A large `.csv` file of options data with calculated pricing errors. |
| 2. Build the error model. | **Model Development** | A trained and validated Random Forest model file (`.pkl`). |
| 3. Deliver a usable tool.   | **Deployment** | A documented and functional Python pricing script (`pricer_tool.py`). |

---

## Repo Plan

* **/data/:** Stores the raw and processed training/testing datasets.
* **/src/:** Contains all production code, including data generation and the final pricing tool.
* **/notebooks/:** For research, model training, and performance evaluation.
* **/docs/:** Contains this memo and the final model validation report.