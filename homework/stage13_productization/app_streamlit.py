import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import validate_inputs, calculate_confidence_interval

st.set_page_config(
    page_title="Options Pricing Model",
    page_icon="💹",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        with open('model/options_pricing_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def main():
    st.title("Options Pricing Model Dashboard")
    st.markdown("Interactive tool for options price prediction based on volatility and moneyness")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.error("Model could not be loaded. Please check model file.")
        return
    
    model = model_data['model']
    mae = model_data['mae']
    r2 = model_data['r2']
    
    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.metric("Mean Absolute Error", f"${mae:.2f}")
        st.metric("R-Squared", f"{r2:.3f}")
        st.metric("Training Samples", model_data['n_samples'])
        
        st.header("Feature Ranges")
        st.info("**Implied Volatility:** 0.1 - 2.0")
        st.info("**Moneyness:** 0.5 - 2.0") 
        st.info("**Vol-Time:** 0.0 - 0.5")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Price Prediction")
        
        # Input widgets
        implied_vol = st.slider(
            "Implied Volatility", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.25, 
            step=0.01,
            help="Expected volatility of the underlying asset"
        )
        
        moneyness = st.slider(
            "Moneyness (S/K)", 
            min_value=0.5, 
            max_value=2.0, 
            value=1.0, 
            step=0.01,
            help="Ratio of stock price to strike price"
        )
        
        vol_time = st.slider(
            "Volatility-Time Interaction", 
            min_value=0.0, 
            max_value=0.5, 
            value=0.05, 
            step=0.01,
            help="Implied volatility × time to expiration"
        )
        
        # Validate and predict
        errors = validate_inputs(implied_vol, moneyness, vol_time)
        
        if errors:
            st.error("Input validation failed:")
            for error in errors:
                st.write(f"• {error}")
        else:
            features_df = pd.DataFrame({
                'implied_volatility': [implied_vol],
                'moneyness': [moneyness], 
                'vol_time': [vol_time]
            })
            prediction = model.predict(features_df)[0]
            
            # Calculate confidence interval
            result = calculate_confidence_interval(prediction, mae)
            
            st.success("Prediction Results")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Predicted Price", f"${result['prediction']:.2f}")
            with col_b:
                st.metric("Lower Bound", f"${result['lower_bound']:.2f}")
            with col_c:
                st.metric("Upper Bound", f"${result['upper_bound']:.2f}")
            
            st.info(f"95% Confidence Interval: ${result['lower_bound']:.2f} - ${result['upper_bound']:.2f}")
    
    with col2:
        st.header("Sensitivity Analysis")
        
        # sensitivity plot
        vol_range = np.linspace(0.1, 0.5, 20)
        predictions = []
        
        for vol in vol_range:
            features_df = pd.DataFrame({
                'implied_volatility': [vol],
                'moneyness': [moneyness], 
                'vol_time': [vol_time]
            })
            pred = model.predict(features_df)[0]
            predictions.append(pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(vol_range, predictions, 'b-', linewidth=2, marker='o', markersize=4)
        ax.axvline(implied_vol, color='red', linestyle='--', alpha=0.7, label='Current Input')
        ax.set_xlabel('Implied Volatility')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('Price Sensitivity to Volatility')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # Show impact
        base_idx = np.argmin(np.abs(vol_range - implied_vol))
        if base_idx < len(predictions) - 1:
            vol_impact = predictions[base_idx + 1] - predictions[base_idx]
            st.metric("Volatility Impact", f"${vol_impact:.2f}", 
                     help="Price change for +0.02 volatility increase")
    
    # Historical comparison
    st.header("Model Performance Context")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Performance metrics comparison
        metrics_df = pd.DataFrame({
            'Scenario': ['Mean Imputation', 'Drop Missing', 'Current Model'],
            'MAE': [72.80, 58.83, mae],
            'R²': [0.083, 0.156, r2]
        })
        
        st.subheader("Performance Comparison")
        st.dataframe(metrics_df, use_container_width=True)
    
    with col_b:
        # Risk factors
        st.subheader("Risk Factors")
        
        risk_level = "🟡 MEDIUM"
        if r2 < 0.1:
            risk_level = "🔴 HIGH"
        elif r2 > 0.2:
            risk_level = "🟢 LOW"
        
        st.markdown(f"**Model Risk Level:** {risk_level}")
        
        risk_factors = [
            f"Low R² ({r2:.1%}) indicates limited explanatory power",
            "Model sensitive to missing data patterns", 
            "Performance varies by strike price segment",
            "Confidence intervals are wide (±$12-13)"
        ]
        
        for factor in risk_factors:
            st.write(f"• {factor}")

if __name__ == "__main__":
    main()
