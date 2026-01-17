#!/usr/bin/env python3
"""
Data loader for dashboard - converts CSV data to JSON
"""

import pandas as pd
import json
import os

def load_all_scrap_data():
    """Load all historical scrap data"""
    csv_path = '../data/casting_scrap_historical.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Convert to format for JavaScript
    data = {
        'months': df['Month'].tolist(),
        'years': df['Year'].tolist(),
        'labels': [f"{row['Year']}-{str(row['Month']).zfill(2)}" for _, row in df.iterrows()],
        'total': df['Total'].tolist(),
        'defect': df['Defect'].tolist(),
        'process': df['Process'].tolist()
    }
    
    return data

def load_predictions():
    """Load prediction data"""
    predictions = {}
    
    for scrap_type in ['Total', 'Defect', 'Process']:
        csv_path = f'../results/optimized_scrap_{scrap_type}_predictions.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            predictions[scrap_type.lower()] = df.to_dict('records')
    
    return predictions

def main():
    # Load all data
    scrap_data = load_all_scrap_data()
    predictions = load_predictions()
    
    if scrap_data:
        # Save to JSON file for dashboard
        output = {
            'historical': scrap_data,
            'predictions': predictions,
            'metadata': {
                'total_months': len(scrap_data['months']),
                'last_updated': pd.Timestamp.now().isoformat()
            }
        }
        
        with open('dashboard/data.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("✅ Data exported successfully!")
        print(f"   Total months: {len(scrap_data['months'])}")
        print(f"   Total scrap range: {min(scrap_data['total']):.1f} - {max(scrap_data['total']):.1f} kg")
        print(f"   Output: dashboard/data.json")

if __name__ == '__main__':
    main()
