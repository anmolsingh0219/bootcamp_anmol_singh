import numpy as np
from scipy.stats import norm
import pandas as pd
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

class BlackScholesCalculator:
    """Black-Scholes option pricing calculator."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter for Black-Scholes formula.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            sigma (float): Volatility
            
        Returns:
            float: d1 value
        """
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d2 parameter for Black-Scholes formula.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            sigma (float): Volatility
            
        Returns:
            float: d2 value
        """
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            sigma (float): Volatility
            
        Returns:
            float: Call option price
        """
        # Handle edge cases
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return max(S - K, 0) if T <= 0 else 0
        
        try:
            d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
            d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma)
            
            call_price = (S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val))
            return max(call_price, 0)  # Ensure non-negative price
        except Exception as e:
            logger.warning(f"Error calculating call price: {e}")
            return 0
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            sigma (float): Volatility
            
        Returns:
            float: Put option price
        """
        # Handle edge cases
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return max(K - S, 0) if T <= 0 else 0
        
        try:
            d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
            d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma)
            
            put_price = (K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val))
            return max(put_price, 0)  # Ensure non-negative price
        except Exception as e:
            logger.warning(f"Error calculating put price: {e}")
            return 0
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho).
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (years)
            r (float): Risk-free rate
            sigma (float): Volatility
            
        Returns:
            dict: Dictionary containing Greek values
        """
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma)
            d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma)
            
            # Delta (price sensitivity to underlying)
            delta = norm.cdf(d1_val)
            
            # Gamma (delta sensitivity to underlying)
            gamma = norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
            
            # Theta (price sensitivity to time decay)
            theta = ((-S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2_val)) / 365
            
            # Vega (price sensitivity to volatility)
            vega = S * norm.pdf(d1_val) * np.sqrt(T) / 100
            
            # Rho (price sensitivity to interest rate)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

def vectorized_black_scholes(df: pd.DataFrame, option_type: str = 'call') -> pd.Series:
    """
    Vectorized Black-Scholes calculation for DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with columns S, K, T, r, sigma
        option_type (str): 'call' or 'put'
        
    Returns:
        pd.Series: Black-Scholes prices
    """
    if option_type.lower() == 'call':
        return df.apply(
            lambda row: BlackScholesCalculator.call_price(
                S=row['S'], K=row['K'], T=row['T'], 
                r=row['r'], sigma=row['sigma']
            ), axis=1
        )
    else:
        return df.apply(
            lambda row: BlackScholesCalculator.put_price(
                S=row['S'], K=row['K'], T=row['T'], 
                r=row['r'], sigma=row['sigma']
            ), axis=1
        )
