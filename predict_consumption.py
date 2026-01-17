#!/usr/bin/env python3
"""
Material Consumption Predictor
Predicts material consumption and generates accuracy metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

def create_features(df, target='Consumption'):
    """Create time series features"""
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Lag features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df[target].shift(lag).fillna(method='bfill')
    
    # Rolling statistics
    for window in [2, 3]:
        df[f'ma_{window}'] = df[target].rolling(window=window, min_periods=1).mean()
        df[f'std_{window}'] = df[target].rolling(window=window, min_periods=1).std().fillna(0)
    
    # Trend
    df['trend'] = df[target].diff().fillna(0)
    
    # Fill any remaining NaN
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def predict_consumption(material_file, material_name):
    """Predict consumption for a material"""
    df = pd.read_csv(f'data/{material_file}')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create features
    df_features = create_features(df)
    
    # Use first 60% for training, rest for testing
    train_size = int(len(df_features) * 0.6)
    train_size = max(3, train_size)
    
    predictions = []
    actuals = []
    
    for i in range(train_size, len(df_features)):
        train_data = df_features.iloc[:i]
        test_data = df_features.iloc[i:i+1]
        
        feature_cols = [c for c in train_data.columns 
                       if c not in ['Date', 'Consumption', 'Material'] 
                       and train_data[c].dtype in ['float64', 'int64']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Consumption']
        X_test = test_data[feature_cols]
        
        # Simple model
        model = xgb.XGBRegressor(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            random_state=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_test)[0]
        
        predictions.append(pred)
        actuals.append(test_data['Consumption'].values[0])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Get dates for predictions
    pred_dates = df.iloc[train_size:train_size+len(predictions)]['Date'].dt.strftime('%Y-%m').tolist()
    
    return {
        'material': material_name,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist(),
        'dates': pred_dates,
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }

def main():
    """Generate consumption predictions for all materials"""
    materials = [
        ('Boron_4_%_consumption.csv', 'Boron 4%'),
        ('Iron_Metal_(80%)_consumption.csv', 'Iron Metal (80%)'),
        ('Magnesium(99.90%)_consumption.csv', 'Magnesium (99.90%)'),
        ('Si_Metal_98.5%_consumption.csv', 'Si Metal 98.5%'),
        ('Tibor_Rod_5_1_consumption.csv', 'Tibor Rod 5:1')
    ]
    
    results = {}
    
    print("\n" + "="*70)
    print("MATERIAL CONSUMPTION PREDICTION")
    print("="*70 + "\n")
    
    for file, name in materials:
        try:
            print(f"Processing: {name}...")
            result = predict_consumption(file, name)
            results[name] = result
            print(f"  MAE: {result['mae']:.2f} | MAPE: {result['mape']:.2f}%")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save to JSON for dashboard
    with open('results/consumption_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Results saved to: results/consumption_predictions.json")
    print(f"{'='*70}\n")
    
    return results

if __name__ == '__main__':
    main()
