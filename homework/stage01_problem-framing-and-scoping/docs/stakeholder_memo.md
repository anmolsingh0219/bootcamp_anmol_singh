# Project: ML-Powered Bermudan Option Pricing

**Stage:** Problem Framing & Scoping (Stage 01)
**Date:** August 14, 2025

---

## Problem Statement

Pricing exotic instruments like **Bermudan options** is a major computational bottleneck. Our current methods are too slow for the fast-paced nature of modern markets, causing traders to miss opportunities and mismanage risk. This project will develop a machine learning model to provide instant, accurate pricing, giving our trading desk a significant competitive advantage.

---

## Stakeholder & User

* **Decision Maker:** The **Head of Derivatives Trading** is the primary sponsor, seeking to improve the desk's P&L and efficiency.
* **Primary User:** An **Exotic Options Trader** needs this tool to make immediate, data-driven trading and hedging decisions throughout the day.

---

## Useful Answer & Decision

* **Type:** The model is **predictive**, forecasting an option's theoretical price.
* **Metrics:** Success is defined by **high accuracy** (matching traditional models) and **low latency** (millisecond-level predictions).
* **Artifact:** The deliverable is a **Python API** that integrates directly into the trader's existing software for on-demand pricing.

---

## Assumptions & Constraints

* **Assumption:** We can generate a sufficiently large and accurate dataset of option prices to train a robust model.
* **Assumption:** The complex patterns of option pricing can be effectively learned and generalized by a neural network.
* **Constraint:** The model must be highly explainable to pass our internal model risk review and gain trader trust.

---

## Known Unknowns / Risks

* **Model Performance:** The model might be inaccurate during extreme market events not seen in its training data. We'll mitigate this by stress-testing the model against historical crisis scenarios.
* **User Adoption:** Traders may be hesitant to trust a new "black box" system. To build confidence, we will run the model in parallel with existing methods and provide clear explainability reports.

---

## Lifecycle Mapping

| Goal                      | Stage                 | Deliverable                                        |
| :------------------------ | :-------------------- | :------------------------------------------------- |
| 1. Create a training dataset. | **Data Generation** | A large `.parquet` file of labeled option prices.  |
| 2. Build the pricing model. | **Model Development** | A trained and validated model file (`.h5`).        |
| 3. Deliver a usable tool.   | **Deployment** | A documented and functional Python API.            |

---

## Repo Plan

* **/data/:** Stores the training and testing datasets.
* **/src/:** Contains all production code, including the final model and API.
* **/notebooks/:** For research, experimentation, and exploratory data analysis.
* **/docs/:** Contains this memo and the final model validation reports.