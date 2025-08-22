import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Market Data
    TARGET_SYMBOL = os.getenv('TARGET_SYMBOL', 'SPY')
    RISK_FREE_RATE_TICKER = os.getenv('RISK_FREE_RATE_TICKER', '^IRX')
    
    # Data Processing Settings  
    MIN_VOLUME = 0  # Allow any volume (including NaN/0)
    MIN_OPEN_INTEREST = 0  # Allow any open interest
    MIN_TIME_TO_EXPIRY = 0.001  # Allow options with at least ~8 hours to expiry
    
    DATA_RAW_PATH = 'data/raw/'
    DATA_PROCESSED_PATH = 'data/processed/'
    
    @classmethod
    def validate_config(cls):
        if not cls.ALPHA_VANTAGE_API_KEY:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY not found in environment variables"
            )
        return True
