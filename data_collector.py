"""
Data Collection Script for Copper and Aluminum Historical Prices
Uses Yahoo Finance to fetch historical commodity data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


class CommodityDataCollector:
    """Collect historical price data for commodities"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Ticker symbols for commodities
        self.tickers = {
            'copper': 'HG=F',  # Copper Futures
            'aluminum': 'ALI=F'  # Aluminum Futures
        }
    
    def fetch_data(self, commodity, start_date=None, end_date=None):
        """
        Fetch historical data for a specific commodity
        
        Parameters:
        -----------
        commodity : str
            Name of the commodity ('copper' or 'aluminum')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        
        Returns:
        --------
        pd.DataFrame
            Historical price data
        """
        if commodity not in self.tickers:
            raise ValueError(f"Commodity '{commodity}' not supported. Choose from {list(self.tickers.keys())}")
        
        ticker = self.tickers[commodity]
        
        # Default to 5 years of data if not specified
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching {commodity} data from {start_date} to {end_date}...")
        
        try:
            # Download data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                print(f"Warning: No data found for {commodity}. Trying alternative source...")
                # Try alternative ticker
                if commodity == 'aluminum':
                    df = yf.download('JJU', start=start_date, end=end_date, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
            
            # Ensure we have the required columns
            if not df.empty:
                df['Commodity'] = commodity
                # Remove any NaN rows
                df = df.dropna()
            
            print(f"✓ Fetched {len(df)} records for {commodity}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {commodity}: {e}")
            return pd.DataFrame()
    
    def save_data(self, commodity, data):
        """Save data to CSV file"""
        filename = os.path.join(self.data_dir, f"{commodity}_historical.csv")
        data.to_csv(filename)
        print(f"✓ Saved {commodity} data to {filename}")
    
    def load_data(self, commodity):
        """Load data from CSV file"""
        filename = os.path.join(self.data_dir, f"{commodity}_historical.csv")
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"✓ Loaded {len(data)} records for {commodity}")
            return data
        else:
            print(f"File {filename} not found")
            return None
    
    def collect_all_data(self, start_date=None, end_date=None):
        """Collect and save data for all commodities"""
        all_data = {}
        
        for commodity in self.tickers.keys():
            data = self.fetch_data(commodity, start_date, end_date)
            if not data.empty:
                self.save_data(commodity, data)
                all_data[commodity] = data
        
        return all_data


def main():
    """Main execution function"""
    # Initialize collector
    collector = CommodityDataCollector()
    
    # Collect data for the past 5 years
    print("="*60)
    print("COMMODITY DATA COLLECTION")
    print("="*60)
    
    all_data = collector.collect_all_data()
    
    # Display summary
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    
    for commodity, data in all_data.items():
        if not data.empty:
            print(f"\n{commodity.upper()}:")
            print(f"  Records: {len(data)}")
            print(f"  Date Range: {data.index.min()} to {data.index.max()}")
            print(f"  Columns: {', '.join(data.columns)}")


if __name__ == "__main__":
    main()
