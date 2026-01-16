"""
Inventory and Consumption Data Collection Script
Reads consumption data, on-hand inventory, and min/max levels from Excel
"""

import pandas as pd
from datetime import datetime
import os
import numpy as np


class CommodityDataCollector:
    """Collect inventory and consumption data from Excel file"""
    
    def __init__(self, data_dir='data', excel_file='Inventory level & Production.xlsx'):
        self.data_dir = data_dir
        self.excel_file = excel_file
        os.makedirs(data_dir, exist_ok=True)
        self.inventory_data = {}
    
    def fetch_inventory_data(self):
        """Fetch inventory and consumption data from Excel file"""
        print(f"Reading inventory data from Excel...")
        
        try:
            df_raw = pd.read_excel(self.excel_file, sheet_name='Material')
            
            # Find header row
            header_row_idx = None
            for idx in range(len(df_raw)):
                row_str = ' '.join([str(v) for v in df_raw.iloc[idx].tolist() if pd.notna(v)]).lower()
                if 'item description' in row_str and 'on hand' in row_str:
                    header_row_idx = idx
                    break
            
            if header_row_idx is None:
                print("Error: Could not find header row")
                return None
            
            # Get dates from row after header
            date_row = df_raw.iloc[header_row_idx + 1]
            dates = [val for val in date_row.tolist()[4:16] if isinstance(val, (pd.Timestamp, datetime))]
            
            # Process materials
            materials_data = {}
            for row_idx in range(header_row_idx + 2, min(header_row_idx + 7, len(df_raw))):
                row = df_raw.iloc[row_idx]
                item_name = row.iloc[1]
                
                if pd.isna(item_name):
                    continue
                
                item_name = str(item_name).strip()
                on_hand = float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0
                
                # Get consumption values (columns 4-15 for Jan-Dec)
                consumption = []
                for col_idx in range(4, 16):  # 12 months
                    val = row.iloc[col_idx]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        consumption.append(float(val))
                    else:
                        consumption.append(np.nan)
                
                # Get min/max inventory levels (columns 16 and 17)
                min_inv = float(row.iloc[16]) if pd.notna(row.iloc[16]) else 100
                max_inv = float(row.iloc[17]) if pd.notna(row.iloc[17]) else 200
                
                materials_data[item_name] = {
                    'on_hand': on_hand,
                    'min': min_inv,
                    'max': max_inv,
                    'dates': dates,
                    'consumption': consumption
                }
            
            print(f"✓ Read {len(materials_data)} materials")
            for mat in materials_data.keys():
                print(f"  - {mat}")
            
            self.inventory_data = materials_data
            return materials_data
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def convert_to_time_series(self, material_name):
        """Convert to time series format"""
        if material_name not in self.inventory_data:
            return pd.DataFrame()
        
        info = self.inventory_data[material_name]
        
        df = pd.DataFrame({
            'Date': info['dates'],
            'Consumption': info['consumption']
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index().dropna()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df['On_Hand'] = info['on_hand']
        df['Min_Inventory'] = info['min']
        df['Max_Inventory'] = info['max']
        
        # OHLCV format for compatibility
        df['Close'] = df['Consumption']
        df['Open'] = df['Close'] * (1 + np.random.randn(len(df)) * 0.02)
        df['High'] = df[['Open', 'Close']].max(axis=1) * 1.02
        df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.98
        df['Volume'] = df['Consumption'] * 100
        df['Adj Close'] = df['Close']
        df['Material'] = material_name
        
        print(f"✓ {material_name}: {len(df)} records, Avg: {df['Consumption'].mean():.2f} MT")
        return df
    
    def save_data(self, material_name, data):
        """Save to CSV"""
        fname = material_name.replace(' ', '_').replace('/', '_')
        filepath = os.path.join(self.data_dir, f"{fname}_consumption.csv")
        data.to_csv(filepath)
        print(f"✓ Saved to {filepath}")
    
    def load_data(self, material_name):
        """Load from CSV"""
        fname = material_name.replace(' ', '_').replace('/', '_')
        filepath = os.path.join(self.data_dir, f"{fname}_consumption.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        return None
    
    def collect_all_materials(self):
        """Collect all materials"""
        materials_data = self.fetch_inventory_data()
        if not materials_data:
            return {}
        
        all_data = {}
        for name in materials_data.keys():
            df = self.convert_to_time_series(name)
            if not df.empty:
                self.save_data(name, df)
                all_data[name] = df
        return all_data


if __name__ == "__main__":
    collector = CommodityDataCollector()
    all_data = collector.collect_all_materials()
    print(f"\n✓ Collected {len(all_data)} materials")
