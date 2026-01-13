"""
Data Collection Script for Aluminum Production Data
Reads historical production data from Excel file
"""

import pandas as pd
from datetime import datetime
import os
import numpy as np


class CommodityDataCollector:
    """Collect historical production data for commodities from Excel file"""
    
    def __init__(self, data_dir='data', excel_file='Inventory level & Production.xlsx'):
        self.data_dir = data_dir
        self.excel_file = excel_file
        os.makedirs(data_dir, exist_ok=True)
        
        # Commodity mapping
        self.commodities = {
            'aluminum': 'Production',  # Sheet name in Excel
            'copper': 'Production'  # Using same sheet for now
        }
    
    
    def fetch_data(self, commodity, start_date=None, end_date=None):
        """
        Fetch historical data for a specific commodity from Excel file
        
        Parameters:
        -----------
        commodity : str
            Name of the commodity ('copper' or 'aluminum')
        start_date : str
            Start date in 'YYYY-MM-DD' format (optional, filters data)
        end_date : str
            End date in 'YYYY-MM-DD' format (optional, filters data)
        
        Returns:
        --------
        pd.DataFrame
            Historical production data with OHLCV format
        """
        if commodity not in self.commodities:
            raise ValueError(f"Commodity '{commodity}' not supported. Choose from {list(self.commodities.keys())}")
        
        sheet_name = self.commodities[commodity]
        
        print(f"Reading {commodity} data from Excel file...")
        
        try:
            # Read the Excel file from the Production sheet
            df_raw = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            # The structure has dates starting from row 3 (index 3) across columns
            # First, let's extract the date row and production data
            
            # Find the row with dates (row index 3 based on earlier inspection)
            date_row_idx = None
            for idx in range(len(df_raw)):
                # Check if the row contains datetime objects
                row_values = df_raw.iloc[idx].tolist()
                if any(isinstance(val, (pd.Timestamp, datetime)) for val in row_values):
                    date_row_idx = idx
                    break
            
            if date_row_idx is None:
                raise ValueError("Could not find date row in Excel file")
            
            # Extract dates from the date row
            date_row = df_raw.iloc[date_row_idx]
            dates = [val for val in date_row if isinstance(val, (pd.Timestamp, datetime))]
            
            # Get production lines data (rows after the date row)
            production_data = []
            for idx in range(date_row_idx + 1, len(df_raw)):
                row = df_raw.iloc[idx]
                # Extract numeric production values corresponding to dates
                values = []
                for col_idx, val in enumerate(row):
                    # Check if this column corresponds to a date column
                    if col_idx < len(date_row) and isinstance(date_row.iloc[col_idx], (pd.Timestamp, datetime)):
                        if pd.notna(val) and isinstance(val, (int, float)):
                            values.append(val)
                        else:
                            values.append(np.nan)
                
                if len(values) > 0 and not all(pd.isna(values)):
                    production_data.append(values[:len(dates)])
            
            # Create a time series dataframe
            if not production_data or len(dates) == 0:
                print(f"Warning: No valid production data found for {commodity}")
                return pd.DataFrame()
            
            # Sum production across all lines for each date
            production_totals = []
            for date_idx in range(len(dates)):
                total = sum(row[date_idx] if date_idx < len(row) and pd.notna(row[date_idx]) else 0 
                           for row in production_data)
                production_totals.append(total)
            
            # Create dataframe with date index
            df = pd.DataFrame({
                'Date': dates,
                'Production': production_totals
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index()
            
            # Convert production data to OHLCV format (required by the predictor)
            # Using production as the "Close" price and creating synthetic OHLC
            df['Close'] = df['Production']
            df['Open'] = df['Close'] * (1 + np.random.uniform(-0.02, 0.02, len(df)))
            df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.03, len(df)))
            df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.03, len(df)))
            df['Volume'] = df['Production'] * np.random.uniform(0.8, 1.2, len(df))
            df['Adj Close'] = df['Close']
            df['Commodity'] = commodity
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            # Remove NaN rows
            df = df.dropna()
            
            print(f"✓ Loaded {len(df)} records for {commodity}")
            print(f"  Date Range: {df.index.min()} to {df.index.max()}")
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Commodity']]
            
        except Exception as e:
            print(f"Error reading data for {commodity}: {e}")
            import traceback
            traceback.print_exc()
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
        """Collect and save data for all commodities from Excel file"""
        all_data = {}
        
        for commodity in self.commodities.keys():
            data = self.fetch_data(commodity, start_date, end_date)
            if not data.empty:
                self.save_data(commodity, data)
                all_data[commodity] = data
        
        return all_data


def main():
    """Main execution function"""
    # Initialize collector
    collector = CommodityDataCollector()
    
    # Collect data from Excel file
    print("="*60)
    print("COMMODITY DATA COLLECTION FROM EXCEL")
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
            print(f"  Sample data:")
            print(data.head(3))


if __name__ == "__main__":
    main()
