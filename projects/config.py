import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Market Data
    TARGET_SYMBOL = os.getenv('TARGET_SYMBOL', 'SPY')
    RISK_FREE_RATE_TICKER = os.getenv('RISK_FREE_RATE_TICKER', '^IRX')
    
    # Data Processing Settings  
    MIN_VOLUME = 0  # Allow any volume (including NaN/0)
    MIN_OPEN_INTEREST = 1  # Allow any open interest
    MIN_TIME_TO_EXPIRY = 0.002  # Allow options with at least ~8 hours to expiry
    
    DATA_RAW_PATH = 'data/raw/'
    DATA_PROCESSED_PATH = 'data/processed/'
    
    @classmethod
    def validate_config(cls):
        # Configuration is valid as long as we have the target symbol
        return True
