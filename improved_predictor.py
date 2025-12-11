"""
Improved Price Direction Predictor using Ensemble Approach
Combines regression and classification for better directional accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import joblib
import os


class ImprovedCommodityPredictor:
    """Enhanced XGBoost predictor with better directional accuracy"""
    
    def __init__(self, commodity_name, model_dir='models'):
        self.commodity_name = commodity_name
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.price_model = None  # For price prediction
        self.direction_model = None  # For direction classification
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def create_advanced_features(self, df):
        """Create enhanced technical indicators"""
        data = df.copy()
        
        # Basic price features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price range features
        if 'High' in data.columns and 'Low' in data.columns:
            data['Price_Range'] = data['High'] - data['Low']
            data['Price_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
            data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
            data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        
        # Multiple timeframe moving averages
        for window in [3, 5, 7, 10, 15, 20, 30, 50, 100, 200]:
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
            
            # Price position relative to MA
            data[f'Price_to_SMA_{window}'] = (data['Close'] - data[f'SMA_{window}']) / data[f'SMA_{window}']
            data[f'Price_to_EMA_{window}'] = (data['Close'] - data[f'EMA_{window}']) / data[f'EMA_{window}']
            
            # MA slope
            data[f'SMA_Slope_{window}'] = data[f'SMA_{window}'].diff() / data[f'SMA_{window}']
            data[f'EMA_Slope_{window}'] = data[f'EMA_{window}'].diff() / data[f'EMA_{window}']
        
        # MA crossovers
        data['SMA_5_20_Cross'] = (data['SMA_5'] > data['SMA_20']).astype(int)
        data['SMA_10_50_Cross'] = (data['SMA_10'] > data['SMA_50']).astype(int)
        data['EMA_5_20_Cross'] = (data['EMA_5'] > data['EMA_20']).astype(int)
        
        # Volatility measures
        for window in [5, 10, 20, 30]:
            data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
            data[f'Volatility_Ratio_{window}'] = data[f'Volatility_{window}'] / data[f'Volatility_{window}'].rolling(window=window).mean()
            
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
        data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
        data['RSI_Momentum'] = data['RSI'].diff()
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']
        data['MACD_Cross'] = (data['MACD'] > data['MACD_Signal']).astype(int)
        
        # Bollinger Bands
        for window in [20]:
            sma = data['Close'].rolling(window=window).mean()
            std = data['Close'].rolling(window=window).std()
            data[f'BB_High_{window}'] = sma + (2 * std)
            data[f'BB_Low_{window}'] = sma - (2 * std)
            data[f'BB_Mid_{window}'] = sma
            data[f'BB_Width_{window}'] = (data[f'BB_High_{window}'] - data[f'BB_Low_{window}']) / data[f'BB_Mid_{window}']
            data[f'BB_Position_{window}'] = (data['Close'] - data[f'BB_Low_{window}']) / (data[f'BB_High_{window}'] - data[f'BB_Low_{window}'] + 1e-10)
            data[f'BB_Squeeze_{window}'] = data[f'BB_Width_{window}'] / data[f'BB_Width_{window}'].rolling(window=50).mean()
        
        # Rate of Change (ROC)
        for period in [1, 3, 5, 7, 10, 14, 21, 30]:
            data[f'ROC_{period}'] = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100
            data[f'Returns_Lag_{period}'] = data['Returns'].shift(period)
        
        # Momentum
        for window in [5, 10, 20]:
            data[f'Momentum_{window}'] = data['Close'] - data['Close'].shift(window)
            data[f'Momentum_Pct_{window}'] = ((data['Close'] - data['Close'].shift(window)) / data['Close'].shift(window)) * 100
        
        # Trend strength
        data['Trend_Strength'] = abs(data['Close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        # Volume features
        if 'Volume' in data.columns:
            data['Volume_Change'] = data['Volume'].pct_change()
            data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
            data['Price_Volume'] = data['Close'] * data['Volume']
            data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        
        # Time features
        data['DayOfWeek'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
        data['DayOfMonth'] = data.index.day
        data['WeekOfYear'] = data.index.isocalendar().week
        
        # Cyclical time encoding
        data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        data['Day_Sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
        data['Day_Cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
        
        # Target variables
        data['Target_Price'] = data['Close'].shift(-1)
        data['Target_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        return data
    
    def prepare_data(self, data, test_size=0.2):
        """Prepare data for training both models"""
        exclude_cols = ['Target_Price', 'Target_Direction', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols]
        y_price = data['Target_Price']
        y_direction = data['Target_Direction']
        
        self.feature_columns = feature_cols
        
        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_price_train = y_price.iloc[:split_idx]
        y_price_test = y_price.iloc[split_idx:]
        y_dir_train = y_direction.iloc[:split_idx]
        y_dir_test = y_direction.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_price_train, y_price_test, y_dir_train, y_dir_test, X_train.index, X_test.index
    
    def train(self, X_train, y_price_train, y_dir_train, X_val=None, y_price_val=None, y_dir_val=None):
        """Train both price and direction models"""
        print(f"\\nTraining models for {self.commodity_name}...")
        
        # Price prediction model
        price_params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.03,
            'n_estimators': 700,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.1,
            'reg_alpha': 0.5,
            'reg_lambda': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Direction classification model
        direction_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'gamma': 0.05,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.price_model = xgb.XGBRegressor(**price_params)
        self.direction_model = xgb.XGBClassifier(**direction_params)
        
        # Train price model
        self.price_model.fit(X_train, y_price_train, verbose=False)
        
        # Train direction model
        self.direction_model.fit(X_train, y_dir_train, verbose=False)
        
        print(f"\u2713 Models training completed")
    
    def predict(self, X, use_direction=True):
        """Make predictions using both models"""
        price_pred = self.price_model.predict(X)
        
        if use_direction:
            direction_pred = self.direction_model.predict(X)
            # Adjust price prediction based on direction
            current_prices = X[:, 0] if len(X.shape) > 1 else X[0]  # Assuming first feature is price-related
            return price_pred, direction_pred
        
        return price_pred
    
    def evaluate(self, X_test, y_price_test, y_dir_test):
        """Evaluate both models"""
        price_predictions = self.price_model.predict(X_test)
        dir_predictions = self.direction_model.predict(X_test)
        
        # Price metrics
        mae = mean_absolute_error(y_price_test, price_predictions)
        mse = mean_squared_error(y_price_test, price_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_price_test, price_predictions)
        mape = np.mean(np.abs((y_price_test - price_predictions) / y_price_test)) * 100
        
        # Direction metrics
        dir_accuracy = accuracy_score(y_dir_test, dir_predictions) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': dir_accuracy
        }
        
        return metrics, price_predictions, dir_predictions
    
    def save_model(self):
        """Save both models"""
        joblib.dump(self.price_model, os.path.join(self.model_dir, f"{self.commodity_name}_price_model.pkl"))
        joblib.dump(self.direction_model, os.path.join(self.model_dir, f"{self.commodity_name}_direction_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, f"{self.commodity_name}_scaler.pkl"))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, f"{self.commodity_name}_features.pkl"))
        print(f"\u2713 Models saved")
    
    def load_model(self):
        """Load both models"""
        self.price_model = joblib.load(os.path.join(self.model_dir, f"{self.commodity_name}_price_model.pkl"))
        self.direction_model = joblib.load(os.path.join(self.model_dir, f"{self.commodity_name}_direction_model.pkl"))
        self.scaler = joblib.load(os.path.join(self.model_dir, f"{self.commodity_name}_scaler.pkl"))
        self.feature_columns = joblib.load(os.path.join(self.model_dir, f"{self.commodity_name}_features.pkl"))
