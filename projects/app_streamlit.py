import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import json
sys.path.append('src')

from src.utils import OptionsPricingAnalyzer, validate_option_input

# Page configuration
st.set_page_config(
    page_title="ML-Enhanced Options Pricing",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = OptionsPricingAnalyzer('./models/random_forest_model.pkl')
    st.session_state.analyzer.load_model()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def main():
    """Main dashboard application."""
    
    st.title("ML-Enhanced Options Pricing Dashboard")
    st.markdown("**Predict option prices with machine learning correction of Black-Scholes model**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "Option Pricing", 
        "Batch Analysis", 
        "Model Performance",
        "About"
    ])
    
    if page == "Option Pricing":
        option_pricing_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Model Performance":
        model_performance_page()
    else:
        about_page()

def option_pricing_page():
    """Single option pricing page."""
    
    st.header("Single Option Pricing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Option Parameters")
        
        # Input parameters
        underlying_price = st.number_input(
            "Underlying Price ($)", 
            min_value=1.0, 
            max_value=1000.0, 
            value=450.0, 
            step=1.0,
            help="Current price of the underlying asset"
        )
        
        strike = st.number_input(
            "Strike Price ($)", 
            min_value=1.0, 
            max_value=1000.0, 
            value=455.0, 
            step=1.0,
            help="Exercise price of the option"
        )
        
        time_to_expiry = st.number_input(
            "Time to Expiry (Years)", 
            min_value=0.001, 
            max_value=2.0, 
            value=0.0833, 
            step=0.001,
            format="%.4f",
            help="Time until option expiration (0.0833 ≈ 1 month)"
        )
        
        implied_volatility = st.number_input(
            "Implied Volatility (%)", 
            min_value=1.0, 
            max_value=500.0, 
            value=25.0, 
            step=1.0,
            help="Market implied volatility percentage"
        ) / 100  # Convert to decimal
        
        contract_type = st.selectbox(
            "Contract Type", 
            ["call", "put"],
            help="Type of option contract"
        )
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=5.0, 
            step=0.1,
            help="Risk-free interest rate percentage"
        ) / 100  # Convert to decimal
        
        # Advanced parameters (collapsible)
        with st.expander("Advanced Parameters"):
            volume = st.number_input("Volume", min_value=0, value=50, help="Trading volume")
            open_interest = st.number_input("Open Interest", min_value=0, value=100, help="Open interest")
        
        # Predict button
        if st.button("Calculate Option Price", type="primary"):
            try:
                option_data = {
                    'underlying_price': underlying_price,
                    'strike': strike,
                    'time_to_expiry': time_to_expiry,
                    'implied_volatility': implied_volatility,
                    'contract_type': contract_type,
                    'risk_free_rate': risk_free_rate,
                    'volume': volume,
                    'open_interest': open_interest
                }
                
                # Validate and predict
                validated_data = validate_option_input(option_data)
                result = st.session_state.analyzer.predict_single_option(validated_data)
                
                # Store in history
                result['timestamp'] = pd.Timestamp.now()
                st.session_state.prediction_history.append(result)
                
                # Display results in second column
                with col2:
                    display_prediction_results(result)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    with col2:
        if not st.session_state.prediction_history:
            st.info("Enter option parameters and click 'Calculate Option Price' to see results.")
        else:
            # Show latest prediction
            display_prediction_results(st.session_state.prediction_history[-1])
    
    # Prediction history
    if st.session_state.prediction_history:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df[['contract_type', 'strike', 'black_scholes_price', 'ml_corrected_price', 'improvement']], use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

def display_prediction_results(result):
    """Display prediction results."""
    st.subheader("Pricing Results")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Black-Scholes Price", 
            f"${result['black_scholes_price']:.4f}",
            help="Traditional Black-Scholes theoretical price"
        )
    
    with col2:
        st.metric(
            "ML-Corrected Price", 
            f"${result['ml_corrected_price']:.4f}",
            delta=f"${result['predicted_error']:.4f}",
            help="Machine learning enhanced price"
        )
    
    with col3:
        st.metric(
            "Pricing Error", 
            f"${abs(result['predicted_error']):.4f}",
            help="Predicted Black-Scholes pricing error"
        )
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Black-Scholes', 'ML-Corrected'],
        y=[result['black_scholes_price'], result['ml_corrected_price']],
        marker_color=['lightblue', 'darkblue'],
        text=[f"${result['black_scholes_price']:.4f}", f"${result['ml_corrected_price']:.4f}"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Price Comparison",
        yaxis_title="Option Price ($)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    with st.expander("Detailed Breakdown"):
        st.json({
            'Contract Details': {
                'Type': result['contract_type'].title(),
                'Strike': f"${result['strike']:.2f}",
                'Underlying': f"${result['underlying_price']:.2f}",
                'Time to Expiry': f"{result['time_to_expiry']:.4f} years",
                'Implied Volatility': f"{result['implied_volatility']*100:.2f}%"
            },
            'Pricing Results': {
                'Black-Scholes Price': f"${result['black_scholes_price']:.4f}",
                'Predicted Error': f"${result['predicted_error']:.4f}",
                'ML-Corrected Price': f"${result['ml_corrected_price']:.4f}",
                'Absolute Improvement': f"${result['improvement']:.4f}"
            }
        })

def batch_analysis_page():
    """Batch analysis page."""
    st.header("Batch Analysis")
    
    st.markdown("Upload a CSV file with option parameters for batch pricing analysis.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="CSV should contain columns: underlying_price, strike, time_to_expiry, implied_volatility, contract_type"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Run Batch Analysis"):
                with st.spinner("Processing batch predictions..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        try:
                            option_data = {
                                'underlying_price': float(row['underlying_price']),
                                'strike': float(row['strike']),
                                'time_to_expiry': float(row['time_to_expiry']),
                                'implied_volatility': float(row['implied_volatility']),
                                'contract_type': str(row.get('contract_type', 'call')).lower(),
                                'risk_free_rate': float(row.get('risk_free_rate', 0.05))
                            }
                            
                            validated_data = validate_option_input(option_data)
                            result = st.session_state.analyzer.predict_single_option(validated_data)
                            results.append(result)
                            
                        except Exception as e:
                            st.warning(f"Row {i+1} failed: {str(e)}")
                            results.append(None)
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Display results
                    valid_results = [r for r in results if r is not None]
                    if valid_results:
                        results_df = pd.DataFrame(valid_results)
                        
                        st.subheader("Batch Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", len(valid_results))
                        with col2:
                            avg_error = np.mean([abs(r['predicted_error']) for r in valid_results])
                            st.metric("Average Error", f"${avg_error:.4f}")
                        with col3:
                            total_improvement = sum([r['improvement'] for r in valid_results])
                            st.metric("Total Improvement", f"${total_improvement:.2f}")
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "batch_pricing_results.csv",
                            "text/csv"
                        )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Upload a CSV file to begin batch analysis.")

def model_performance_page():
    """Model performance and analysis page."""
    st.header("Model Performance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Run Full Analysis")
        
        if st.button("Execute Complete Analysis", type="primary"):
            with st.spinner("Running full analysis pipeline..."):
                try:
                    results = st.session_state.analyzer.run_full_analysis()
                    
                    st.success("Analysis completed successfully!")
                    
                    # Display key metrics
                    metrics = results['key_metrics']
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("MAE", f"${metrics['mae']:.4f}")
                    with col_b:
                        st.metric("RMSE", f"${metrics['rmse']:.4f}")
                    with col_c:
                        st.metric("R²", f"{metrics['r2']:.3f}")
                    
                    st.metric("Improvement vs Baseline", metrics['improvement_vs_baseline'])
                    
                    # Detailed results
                    with st.expander("Detailed Analysis Results"):
                        st.json(results)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    with col2:
        st.subheader("Model Information")
        
        model_loaded = st.session_state.analyzer.model is not None
        st.write(f"**Model Status**: {'Loaded' if model_loaded else 'Not Loaded'}")
        
        if model_loaded:
            st.write(f"**Model Path**: {st.session_state.analyzer.model_path}")
            
            # Try to load metadata
            metadata_path = st.session_state.analyzer.model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                st.write(f"**Model Type**: {metadata.get('model_type', 'Unknown')}")
                st.write(f"**Features**: {len(metadata.get('feature_names', []))}")
                st.write(f"**Training Date**: {metadata.get('training_date', 'Unknown')}")
        
        # Model health check
        if st.button("Health Check"):
            try:
                sample_data = {
                    'underlying_price': 450.0,
                    'strike': 455.0,
                    'time_to_expiry': 0.0833,
                    'implied_volatility': 0.25,
                    'contract_type': 'call'
                }
                
                result = st.session_state.analyzer.predict_single_option(sample_data)
                st.success("Model is healthy and responding correctly")
                st.write(f"Sample prediction: ${result['ml_corrected_price']:.4f}")
                
            except Exception as e:
                st.error(f"Model health check failed: {str(e)}")

def about_page():
    """About page with project information."""
    st.header("About ML-Enhanced Options Pricing")
    
    st.markdown("""
    ## Project Overview
    
    This dashboard provides machine learning enhanced option pricing that corrects systematic errors 
    in the Black-Scholes model. The ML model predicts pricing errors and provides more accurate 
    option valuations for trading and risk management.
    
    ## Key Features
    
    - **Single Option Pricing**: Get ML-corrected prices for individual options
    - **Batch Analysis**: Process multiple options from CSV files
    - **Model Performance**: Run complete analysis pipeline and view model metrics
    - **Interactive Dashboard**: User-friendly interface for all pricing needs
    
    ## Model Performance
    
    - **Accuracy**: 78% improvement over baseline linear regression
    - **Average Error**: $1.03 (vs $4.60 baseline)
    - **Model Type**: Random Forest Regressor
    - **Features**: 15+ engineered features from market data
    
    ## How It Works
    
    1. **Input**: Option parameters (strike, underlying price, time, volatility)
    2. **Black-Scholes**: Calculate traditional theoretical price
    3. **ML Prediction**: Predict pricing error using trained Random Forest model
    4. **Correction**: Add predicted error to Black-Scholes price for final result
    
    ## Business Value
    
    - **Improved Accuracy**: More precise option valuations
    - **Risk Reduction**: Better understanding of true option values
    - **Competitive Advantage**: Superior pricing vs traditional methods
    - **Automated**: Fast, consistent pricing without manual intervention
    
    ## Technical Details
    
    - **Framework**: Python, Streamlit, scikit-learn
    - **Model**: Random Forest with time-aware cross-validation
    - **Features**: Implied volatility, time decay, moneyness, market microstructure
    - **Validation**: 5-fold cross-validation with comprehensive testing
    """)
if __name__ == "__main__":
    main()
