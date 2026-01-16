"""
Individual Material Optimizers
Each material gets a specialized model tuned to its unique pattern
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class BoronOptimizer:
    """Specialized optimizer for Boron - very stable pattern"""
    
    def __init__(self):
        self.name = "Boron 4 %"
        self.scaler = RobustScaler()
        self.model = None
    
    def create_features(self, df):
        """Minimal features for stable patterns"""
        df = df.copy()
        
        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Basic lags
        for lag in [1, 2]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        # Simple moving averages
        df['sma_3'] = df['Consumption'].rolling(window=3, min_periods=1).mean()
        df['std_3'] = df['Consumption'].rolling(window=3, min_periods=1).std()
        
        # Month encoding
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            # Generate month from index position
            df['month'] = (np.arange(len(df)) % 12) + 1
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        """Train with very conservative parameters"""
        data_with_features = self.create_features(data_df)
        
        predictions = []
        actuals = []
        
        min_train = 6
        
        for i in range(min_train, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            feature_cols = [c for c in train_data.columns if c not in ['Date', 'Consumption', 'Material']]
            feature_cols = [c for c in feature_cols if train_data[c].dtype in ['float64', 'int64']]
            
            X_train = train_data[feature_cols]
            y_train = train_data['Consumption']
            X_test = test_data[feature_cols]
            
            # Very shallow model for stable pattern
            model = xgb.XGBRegressor(
                max_depth=1,
                learning_rate=0.05,
                n_estimators=30,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=5,
                objective='reg:squarederror',
                random_state=42
            )
            
            model.fit(X_train, y_train, verbose=False)
            pred = model.predict(X_test)[0]
            
            predictions.append(pred)
            actuals.append(test_data['Consumption'].values[0])
        
        return np.array(predictions), np.array(actuals)


class TiborOptimizer:
    """Specialized optimizer for Tibor - oscillating pattern"""
    
    def __init__(self):
        self.name = "Tibor Rod 5/1"
        self.scaler = RobustScaler()
    
    def create_features(self, df):
        """Features for oscillating patterns"""
        df = df.copy()
        
        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Extended lags for oscillation
        for lag in [1, 2, 3, 4]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        # Oscillation detection
        df['diff_1'] = df['Consumption'].diff(1)
        df['diff_2'] = df['Consumption'].diff(2)
        
        # Peak/trough detection
        df['is_peak'] = ((df['Consumption'].shift(1) < df['Consumption']) & 
                         (df['Consumption'] > df['Consumption'].shift(-1))).astype(int)
        df['is_trough'] = ((df['Consumption'].shift(1) > df['Consumption']) & 
                           (df['Consumption'] < df['Consumption'].shift(-1))).astype(int)
        
        # Alternating pattern
        df['alternating'] = (df['diff_1'] * df['diff_1'].shift(1) < 0).astype(int)
        
        # Rolling stats
        for window in [3, 4]:
            df[f'sma_{window}'] = df['Consumption'].rolling(window=window, min_periods=1).mean()
            df[f'std_{window}'] = df['Consumption'].rolling(window=window, min_periods=1).std()
        
        # Month
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        """Ensemble specifically for oscillations"""
        data_with_features = self.create_features(data_df)
        
        predictions = []
        actuals = []
        
        min_train = 6
        
        for i in range(min_train, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            feature_cols = [c for c in train_data.columns if c not in ['Date', 'Consumption', 'Material']]
            feature_cols = [c for c in feature_cols if train_data[c].dtype in ['float64', 'int64']]
            
            X_train = train_data[feature_cols]
            y_train = train_data['Consumption']
            X_test = test_data[feature_cols]
            
            # Shallow tree for oscillation
            xgb_model = xgb.XGBRegressor(
                max_depth=1,
                learning_rate=0.04,
                n_estimators=50,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=4,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Random forest to capture alternation
            rf_model = RandomForestRegressor(
                n_estimators=30,
                max_depth=2,
                min_samples_split=3,
                random_state=42
            )
            
            xgb_model.fit(X_train, y_train, verbose=False)
            rf_model.fit(X_train, y_train)
            
            pred_xgb = xgb_model.predict(X_test)[0]
            pred_rf = rf_model.predict(X_test)[0]
            
            # Weighted ensemble
            pred = 0.6 * pred_xgb + 0.4 * pred_rf
            
            predictions.append(pred)
            actuals.append(test_data['Consumption'].values[0])
        
        return np.array(predictions), np.array(actuals)


class SiMetalOptimizer:
    """Specialized optimizer for Si Metal - use simple weighted average"""
    
    def __init__(self):
        self.name = "Si Metal 98.5%"
        self.scaler = RobustScaler()
    
    def create_features(self, df):
        """Minimal features"""
        df = df.copy()
        
        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Only basic lags
        for lag in [1, 2]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        # Simple SMA
        df['sma_3'] = df['Consumption'].rolling(window=3, min_periods=1).mean()
        
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        """Simple weighted moving average approach"""
        data_with_features = self.create_features(data_df)
        
        predictions = []
        actuals = []
        
        min_train = 4  # Need less training data for simple model
        
        for i in range(min_train, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            # Simple weighted average of last 3 values
            recent_values = train_data['Consumption'].tail(3).values
            if len(recent_values) >= 3:
                # More weight to recent values
                pred = 0.5 * recent_values[-1] + 0.3 * recent_values[-2] + 0.2 * recent_values[-3]
            elif len(recent_values) == 2:
                pred = 0.6 * recent_values[-1] + 0.4 * recent_values[-2]
            else:
                pred = recent_values[-1]
            
            predictions.append(pred)
            actuals.append(test_data['Consumption'].values[0])
        
        return np.array(predictions), np.array(actuals)


class IronOptimizer:
    """Specialized optimizer for Iron - simple exponential smoothing"""
    
    def __init__(self):
        self.name = "Iron Metal (80%)"
        self.scaler = RobustScaler()
    
    def create_features(self, df):
        """Minimal features"""
        df = df.copy()
        
        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Only lags
        for lag in [1, 2]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        df['ema_3'] = df['Consumption'].ewm(span=3, min_periods=1).mean()
        
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        """Exponential smoothing with naive forecast blend"""
        data_with_features = self.create_features(data_df)
        
        predictions = []
        actuals = []
        
        min_train = 4
        alpha = 0.4  # Smoothing parameter
        
        for i in range(min_train, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            # Exponential moving average
            recent_ema = train_data['Consumption'].ewm(span=3, adjust=False).mean().iloc[-1]
            
            # Last value (naive forecast)
            last_value = train_data['Consumption'].iloc[-1]
            
            # Blend EMA and last value
            pred = alpha * recent_ema + (1 - alpha) * last_value
            
            predictions.append(pred)
            actuals.append(test_data['Consumption'].values[0])
        
        return np.array(predictions), np.array(actuals)


class MagnesiumOptimizer:
    """Specialized optimizer for Magnesium - simple moving average"""
    
    def __init__(self):
        self.name = "Magnesium(99.90%)"
        self.scaler = RobustScaler()
    
    def create_features(self, df):
        """Minimal features"""
        df = df.copy()
        
        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        for lag in [1, 2]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        df['sma_3'] = df['Consumption'].rolling(window=3, min_periods=1).mean()
        
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        """Simple average of recent values"""
        data_with_features = self.create_features(data_df)
        
        predictions = []
        actuals = []
        
        min_train = 4
        
        for i in range(min_train, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            # Simple moving average of last 3 values
            recent_values = train_data['Consumption'].tail(3).values
            pred = np.mean(recent_values)
            
            predictions.append(pred)
            actuals.append(test_data['Consumption'].values[0])
        
        return np.array(predictions), np.array(actuals)


def calculate_metrics(predictions, actuals):
    """Calculate performance metrics"""
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mae, rmse, mape
