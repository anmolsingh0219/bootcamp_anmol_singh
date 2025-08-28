import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsDataFetcher:
    
    def __init__(self):
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
    
    def get_historical_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol
            period (str): Period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period=period)
            logger.info(f"Fetched {len(hist_data)} days of historical data for {symbol}")
            return hist_data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    def fetch_options_data(self, symbol: str, historical_focus: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch options data using Yahoo Finance with focus on historical-style contracts.
        
        Args:
            symbol (str): Underlying stock symbol
            historical_focus (bool): If True, focus on short-term contracts to simulate historical data
            
        Returns:
            List[Dict]: List of options data from Yahoo Finance
        """
        logger.info(f"Fetching {'historical-style' if historical_focus else 'standard'} options data from Yahoo Finance for {symbol}...")
        return self._fetch_from_yfinance(symbol, focus_short_term=historical_focus)
    
    def fetch_historical_style_options(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch options data that simulates historical options contracts.
        This focuses on short-term options (0-30 days to expiration) to create
        a dataset that represents historical options trading patterns.
        
        Args:
            symbol (str): Underlying stock symbol
            
        Returns:
            List[Dict]: List of historical-style options data
        """
        logger.info(f"Fetching historical-style options data for {symbol}...")
        return self._fetch_from_yfinance(symbol, focus_short_term=True)
    
    def fetch_enhanced_historical_dataset(self, symbol: str, sample_size: int = 500) -> List[Dict[str, Any]]:
        """
        Fetch an enhanced historical-style dataset by combining short-term options
        with historical stock price context to create a richer dataset.
        
        Args:
            symbol (str): Underlying stock symbol
            sample_size (int): Target number of option contracts to fetch
            
        Returns:
            List[Dict]: Enhanced historical-style options dataset
        """
        logger.info(f"Creating enhanced historical dataset for {symbol} with target size {sample_size}...")
        
        # Get current options data
        options_data = self._fetch_from_yfinance(symbol, focus_short_term=True)
        
        # Get historical stock prices for context
        try:
            hist_stock_data = self.get_historical_stock_data(symbol, period='3mo')
            historical_prices = hist_stock_data['Close'].tolist()
            historical_dates = hist_stock_data.index.tolist()
            
            # Enhance the dataset with historical price context
            for i, option in enumerate(options_data):
                if i < len(historical_prices):
                    # Add historical price context
                    option['historical_underlying_price'] = historical_prices[i]
                    option['historical_date'] = historical_dates[i]
                    option['price_change_context'] = (option['historical_underlying_price'] / historical_prices[0] - 1) * 100
                
            logger.info(f"Enhanced dataset with {len(historical_prices)} historical price points")
            
        except Exception as e:
            logger.warning(f"Could not enhance with historical prices: {e}")
        
        return options_data[:sample_size]
    

    
    def _fetch_from_yfinance(self, symbol: str, focus_short_term: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch options data using Yahoo Finance, focusing on short-term contracts to simulate historical data.
        
        Args:
            symbol (str): Underlying stock symbol
            focus_short_term (bool): If True, prioritize short-term options that represent "historical" trading
            
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
            
            today = datetime.now().date()
            
            if focus_short_term:
                # Focus on short-term options (0-30 days) to simulate historical contracts
                filtered_expirations = []
                for exp in expirations:
                    exp_date = pd.to_datetime(exp).date()
                    days_to_exp = (exp_date - today).days
                    # Include contracts expiring within 30 days (including past dates if available)
                    if -5 <= days_to_exp <= 30:  # Allow slightly past dates and near-term
                        filtered_expirations.append(exp)
                
                logger.info(f"Found {len(filtered_expirations)} short-term expiration dates")
            else:
                # Use all available future expirations
                filtered_expirations = [exp for exp in expirations 
                                      if pd.to_datetime(exp).date() >= today]
            
            if not filtered_expirations:
                logger.warning(f"No suitable expiration dates found for {symbol}")
                # Fall back to any available expirations
                filtered_expirations = expirations[:10]
            
            # Target around 500 rows
            target_rows = 500
            rows_per_expiration = max(10, target_rows // min(len(filtered_expirations), 20))
            
            logger.info(f"Using {len(filtered_expirations)} expiration dates, targeting {rows_per_expiration} contracts per date")
            
            for exp_date in filtered_expirations[:20]:  # Use up to 20 expiration dates
                if len(all_options_data) >= target_rows:
                    break
                    
                try:
                    logger.info(f"Fetching options chain for expiration: {exp_date}")
                    
                    # Get options chain for this expiration
                    opt_chain = ticker.option_chain(exp_date)
                    calls = opt_chain.calls
                    puts = opt_chain.puts
                    
                    # Calculate time to expiration for data quality
                    exp_datetime = pd.to_datetime(exp_date)
                    time_to_exp = (exp_datetime - pd.Timestamp.now()).days / 365.0
                    
                    # Process calls
                    calls_to_process = min(len(calls), rows_per_expiration // 2)
                    for _, row in calls.head(calls_to_process).iterrows():
                        if len(all_options_data) >= target_rows:
                            break
                        
                        # Skip options with invalid data
                        if pd.isna(row['lastPrice']) or row['lastPrice'] <= 0:
                            continue
                            
                        option_data = {
                            'symbol': f"{symbol}{exp_date.replace('-', '')}C{int(row['strike']*1000):08d}",
                            'underlying_symbol': symbol,
                            'strike': row['strike'],
                            'expiration_date': exp_datetime,
                            'contract_type': 'call',
                            'last_price': row['lastPrice'],
                            'bid': row['bid'] if not pd.isna(row['bid']) else 0,
                            'ask': row['ask'] if not pd.isna(row['ask']) else 0,
                            'volume': row['volume'] if not pd.isna(row['volume']) else 0,
                            'open_interest': row['openInterest'] if not pd.isna(row['openInterest']) else 0,
                            'implied_volatility': row['impliedVolatility'] if not pd.isna(row['impliedVolatility']) else 0,
                            'time_to_expiry': time_to_exp,
                            'fetch_timestamp': datetime.now(),
                            'data_source': 'yahoo_finance'
                        }
                        all_options_data.append(option_data)
                    
                    # Process puts
                    puts_to_process = min(len(puts), rows_per_expiration // 2)
                    for _, row in puts.head(puts_to_process).iterrows():
                        if len(all_options_data) >= target_rows:
                            break
                            
                        # Skip options with invalid data
                        if pd.isna(row['lastPrice']) or row['lastPrice'] <= 0:
                            continue
                            
                        option_data = {
                            'symbol': f"{symbol}{exp_date.replace('-', '')}P{int(row['strike']*1000):08d}",
                            'underlying_symbol': symbol,
                            'strike': row['strike'],
                            'expiration_date': exp_datetime,
                            'contract_type': 'put',
                            'last_price': row['lastPrice'],
                            'bid': row['bid'] if not pd.isna(row['bid']) else 0,
                            'ask': row['ask'] if not pd.isna(row['ask']) else 0,
                            'volume': row['volume'] if not pd.isna(row['volume']) else 0,
                            'open_interest': row['openInterest'] if not pd.isna(row['openInterest']) else 0,
                            'implied_volatility': row['impliedVolatility'] if not pd.isna(row['impliedVolatility']) else 0,
                            'time_to_expiry': time_to_exp,
                            'fetch_timestamp': datetime.now(),
                            'data_source': 'yahoo_finance'
                        }
                        all_options_data.append(option_data)
                        
                except Exception as e:
                    logger.warning(f"Error fetching options for expiration {exp_date}: {e}")
                    continue
            
            # Add some synthetic "historical" characteristics to the data
            for option in all_options_data:
                # Simulate historical trading by slightly adjusting timestamps
                # This represents the concept that we're looking at options as if from a historical perspective
                days_back = min(30, max(1, int(option['time_to_expiry'] * 365)))
                option['simulated_trade_date'] = datetime.now() - timedelta(days=days_back)
            
            logger.info(f"Successfully fetched {len(all_options_data)} options contracts representing historical-style data")
            return all_options_data
            
        except Exception as e:
            logger.error(f"Error fetching options data from Yahoo Finance: {e}")
            return []
    
    def process_options_data(self, options_list: List, stock_price: float, 
                           risk_free_rate: float) -> pd.DataFrame:
        """
        Process options data into a structured DataFrame.
        
        Args:
            options_list (List): Options data from Yahoo Finance
            stock_price (float): Current underlying stock price
            risk_free_rate (float): Risk-free rate
            
        Returns:
            pd.DataFrame: Processed options data
        """
        processed_data = []
        
        for option_data in options_list:
            # Process both call and put options
            if option_data.get('contract_type') in ['call', 'put']:
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
                    
                    # Add new historical-focused fields
                    if 'time_to_expiry' in option_data:
                        processed_option['time_to_expiry'] = option_data['time_to_expiry']
                    if 'data_source' in option_data:
                        processed_option['data_source'] = option_data['data_source']
                    if 'simulated_trade_date' in option_data:
                        processed_option['simulated_trade_date'] = option_data['simulated_trade_date']
                    
                    # Add Greeks if available
                    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
                    for greek in greeks:
                        if greek in option_data:
                            processed_option[greek] = option_data[greek]
                    
                    processed_data.append(processed_option)
                    
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping option due to missing data: {e}")
                    continue
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(df)} option contracts into DataFrame")
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save raw options data to CSV file.
        
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
