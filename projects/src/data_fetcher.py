import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import logging
import requests
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsDataFetcher:
    
    def __init__(self, alpha_vantage_api_key: str = None):
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.alpha_vantage_ts = TimeSeries(key=alpha_vantage_api_key) if alpha_vantage_api_key else None
        logger.info("OptionsDataFetcher initialized successfully")
    
    def get_risk_free_rate(self, ticker: str = '^IRX') -> float:
        """
        Risk-free rate from Yahoo Finance.
        
        Args:
            ticker (str): Yahoo Finance ticker for risk-free rate (default: ^IRX)
            
        Returns:
            float: Risk-free rate as decimal
        """
        try:
            rf_ticker = yf.Ticker(ticker)
            rf_data = rf_ticker.history(period='1d')
            risk_free_rate = rf_data['Close'].iloc[-1] / 100
            logger.info(f"Risk-free rate fetched: {risk_free_rate:.4f}")
            return risk_free_rate
        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {e}")
            return 0.05
    
    def get_current_stock_price(self, symbol: str) -> float:
        """
        Current stock price from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Current stock price
        """
        try:
            stock = yf.Ticker(symbol)
            price = stock.history(period='1d')['Close'].iloc[-1]
            logger.info(f"Current {symbol} price: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"Error fetching stock price for {symbol}: {e}")
            raise
    
    def fetch_options_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch options data using Yahoo Finance only.
        
        Args:
            symbol (str): Underlying stock symbol
            
        Returns:
            List[Dict]: List of options data from Yahoo Finance
        """
        logger.info(f"Fetching options data from Yahoo Finance for {symbol}...")
        return self._fetch_from_yfinance(symbol)
    
    def _fetch_from_alpha_vantage(self, symbol: str, date: str = None) -> List[Dict[str, Any]]:
        """
        Fetch real options data from Alpha Vantage HISTORICAL_OPTIONS API.
        
        Args:
            symbol (str): Stock symbol
            date (str): Date in YYYY-MM-DD format (optional, defaults to previous trading day)
            
        Returns:
            List[Dict]: Real options data from Alpha Vantage
        """
        if not self.alpha_vantage_api_key:
            logger.error("Alpha Vantage API key not provided")
            return []
            
        try:
            # Build API URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                'function': 'HISTORICAL_OPTIONS',
                'symbol': symbol,
                'apikey': self.alpha_vantage_api_key
            }
            
            # Add date parameter if provided
            if date:
                params['date'] = date
            else:
                # Use previous trading day by default
                today = datetime.now()
                # Go back a few days to ensure we get a trading day
                prev_date = today - timedelta(days=3)
                params['date'] = prev_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching options data from Alpha Vantage for {symbol} on {params.get('date', 'latest')}")
            
            # Make API request
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return []
                
            if 'Information' in data:
                logger.warning(f"Alpha Vantage API info: {data['Information']}")
                return []
            
            # Parse the options data
            options_data = []
            
            # The API returns data with the date as the key
            for date_key, date_data in data.items():
                if date_key.startswith('20'):  # Date keys start with year
                    contracts = date_data.get('contracts', [])
                    
                    for contract in contracts:
                        try:
                            # Extract contract details
                            option_data = {
                                'symbol': contract.get('contractID', ''),
                                'underlying_symbol': symbol,
                                'strike': float(contract.get('strike', 0)),
                                'expiration_date': pd.to_datetime(contract.get('expiration', '')),
                                'contract_type': contract.get('type', '').lower(),  # 'call' or 'put'
                                'last_price': float(contract.get('last', 0)),
                                'bid': float(contract.get('bid', 0)),
                                'ask': float(contract.get('ask', 0)),
                                'volume': int(contract.get('volume', 0)),
                                'open_interest': int(contract.get('openInterest', 0)),
                                'implied_volatility': float(contract.get('impliedVolatility', 0)),
                                'delta': float(contract.get('delta', 0)),
                                'gamma': float(contract.get('gamma', 0)),
                                'theta': float(contract.get('theta', 0)),
                                'vega': float(contract.get('vega', 0)),
                                'rho': float(contract.get('rho', 0)),
                                'fetch_timestamp': datetime.now(),
                                'data_date': pd.to_datetime(date_key)
                            }
                            
                            # Only include call options with reasonable data
                            if (option_data['contract_type'] == 'call' and 
                                option_data['strike'] > 0 and 
                                option_data['last_price'] > 0):
                                options_data.append(option_data)
                                
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Skipping contract due to data error: {e}")
                            continue
            
            # Limit to around 50 contracts, prioritize by volume and open interest
            if len(options_data) > 50:
                # Sort by volume * open_interest (liquidity proxy)
                options_data.sort(key=lambda x: x['volume'] * x['open_interest'], reverse=True)
                options_data = options_data[:50]
            
            logger.info(f"Successfully fetched {len(options_data)} real options contracts from Alpha Vantage")
            return options_data
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching from Alpha Vantage: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage options data: {e}")
            return []
    
    def _fetch_from_yfinance(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch options data using Yahoo Finance (no rate limits).
        
        Args:
            symbol (str): Underlying stock symbol
            
        Returns:
            List[Dict]: List of options data from Yahoo Finance
        """
        all_options_data = []
        
        try:
            logger.info(f"Fetching options data for {symbol} from Yahoo Finance...")
            
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return []
            
            # Filter out same-day expirations and use dates with longer time to expiry
            today = datetime.now().date()
            filtered_expirations = [exp for exp in expirations 
                                  if pd.to_datetime(exp).date() > today]
            
            if not filtered_expirations:
                logger.warning(f"No future expiration dates found for {symbol}")
                return []
            
            # Use the first few expiration dates to get some data, prioritize closer dates
            for exp_date in filtered_expirations[:4]:  # Limit to first 4 expiration dates
                try:
                    logger.info(f"Fetching options chain for expiration: {exp_date}")
                    
                    # Get options chain for this expiration
                    opt_chain = ticker.option_chain(exp_date)
                    calls = opt_chain.calls
                    
                    # Process call options only, limit to first 15 per expiration for ~50 total
                    for _, row in calls.head(15).iterrows():
                        option_data = {
                            'symbol': f"{symbol}{exp_date.replace('-', '')}C{int(row['strike']*1000):08d}",
                            'underlying_symbol': symbol,
                            'strike': row['strike'],
                            'expiration_date': pd.to_datetime(exp_date),
                            'contract_type': 'call',
                            'last_price': row['lastPrice'],
                            'bid': row['bid'],
                            'ask': row['ask'],
                            'volume': row['volume'],
                            'open_interest': row['openInterest'],
                            'implied_volatility': row['impliedVolatility'],
                            'fetch_timestamp': datetime.now()
                        }
                        all_options_data.append(option_data)
                        
                except Exception as e:
                    logger.warning(f"Error fetching options for expiration {exp_date}: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(all_options_data)} options from Yahoo Finance")
            return all_options_data
            
        except Exception as e:
            logger.error(f"Error fetching options data from Yahoo Finance: {e}")
            return []
    
    def process_options_data(self, options_list: List, stock_price: float, 
                           risk_free_rate: float) -> pd.DataFrame:
        """
        Process options data into a structured DataFrame.
        Works with both Alpha Vantage and Yahoo Finance data structures.
        
        Args:
            options_list (List): Options data from Alpha Vantage or Yahoo Finance
            stock_price (float): Current underlying stock price
            risk_free_rate (float): Risk-free rate
            
        Returns:
            pd.DataFrame: Processed options data
        """
        processed_data = []
        
        for option_data in options_list:
            # Only process call options
            if option_data.get('contract_type') == 'call':
                try:
                    # Create base data structure
                    processed_option = {
                        'symbol': option_data['symbol'],
                        'underlying_symbol': option_data['underlying_symbol'],
                        'strike': option_data['strike'],
                        'expiration_date': option_data['expiration_date'],
                        'contract_type': option_data['contract_type'],
                        'implied_volatility': option_data['implied_volatility'],
                        'open_interest': option_data['open_interest'],
                        'volume': option_data['volume'],
                        'bid': option_data['bid'],
                        'ask': option_data['ask'],
                        'last_price': option_data['last_price'],
                        'market_price': (option_data['bid'] + option_data['ask']) / 2.0 if option_data['bid'] > 0 and option_data['ask'] > 0 else option_data['last_price'],
                        'underlying_price': stock_price,
                        'risk_free_rate': risk_free_rate,
                        'fetch_timestamp': option_data['fetch_timestamp']
                    }
                    
                    # Add Greeks if available (from Alpha Vantage)
                    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
                    for greek in greeks:
                        if greek in option_data:
                            processed_option[greek] = option_data[greek]
                    
                    # Add data date if available (from Alpha Vantage)
                    if 'data_date' in option_data:
                        processed_option['data_date'] = option_data['data_date']
                    
                    processed_data.append(processed_option)
                    
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping option due to missing data: {e}")
                    continue
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(df)} option contracts into DataFrame")
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Raw options data to CSV file.
        
        Args:
            df (pd.DataFrame): Options data DataFrame
            filepath (str): Path to save the data
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Raw data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
            raise
