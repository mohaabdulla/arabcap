"""
XGBoost Production Volume Forecasting Model
Predicts monthly aluminum production volumes based on historical data
Includes feature engineering, training, and prediction capabilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class CommodityPricePredictor:
    """XGBoost-based production volume forecasting model"""
    
    def __init__(self, commodity_name, model_dir='models'):
        self.commodity_name = commodity_name
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = RobustScaler()  # Better for handling outliers
        self.feature_columns = None
        self.prediction_history = []  # For smoothing predictions
        
    def create_features(self, df):
        """
        Create production-specific features and technical indicators
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with production volume data (OHLCV format)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added features
        """
        data = df.copy()
        
        # Ensure we have the required columns
        if 'Close' not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column (production volume)")
        
        # Production-based features (Close = production volume)
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Production_Change'] = data['Close'].diff()
        
        # Production range analysis
        if 'High' in data.columns and 'Low' in data.columns:
            data['Volume_Range'] = data['High'] - data['Low']
            data['Volume_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        # Capacity utilization features (if available)
        if 'Capacity_Utilization' in data.columns:
            data['Util_Change'] = data['Capacity_Utilization'].diff()
            data['Util_MA_3'] = data['Capacity_Utilization'].rolling(3).mean()
        
        if 'Production_vs_Avg' in data.columns:
            data['Prod_vs_Avg_Change'] = data['Production_vs_Avg'].diff()
        
        # Determine max window size based on data length
        # Use 1/3 of available data as max window to preserve enough samples
        max_window = max(3, len(data) // 3)
        
        # Moving averages - adjusted for limited data
        windows = [w for w in [3, 5, 7, 10, 15] if w < max_window]
        if not windows:
            windows = [2, 3]  # Minimum windows
            
        for window in windows:
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
            data[f'Price_to_SMA_{window}'] = data['Close'] / data[f'SMA_{window}']
            data[f'SMA_Slope_{window}'] = data[f'SMA_{window}'].diff()
        
        # Volatility - shorter windows
        vol_windows = [w for w in [3, 5, 7] if w < max_window]
        if not vol_windows:
            vol_windows = [2, 3]
            
        for window in vol_windows:
            data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
            data[f'Volatility_High_Low_{window}'] = (data['High'] - data['Low']).rolling(window=window).std() if 'High' in data.columns else 0
        
        # RSI - manual calculation with adjusted window
        rsi_window = min(14, max_window - 1)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
        data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
        
        # MACD - manual calculation with adjusted windows
        macd_fast = min(12, max(5, max_window // 2))
        macd_slow = min(26, max(10, max_window - 3))
        macd_signal = min(9, max(3, max_window // 3))
        
        exp1 = data['Close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=macd_slow, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']
        data['MACD_Cross'] = ((data['MACD'] > data['MACD_Signal']).astype(int).diff()).fillna(0)
        
        # Bollinger Bands - manual calculation with adjusted window
        bb_window = min(10, max(5, max_window - 2))
        sma = data['Close'].rolling(window=bb_window).mean()
        std = data['Close'].rolling(window=bb_window).std()
        data[f'BB_High_{bb_window}'] = sma + (2 * std)
        data[f'BB_Low_{bb_window}'] = sma - (2 * std)
        data[f'BB_Mid_{bb_window}'] = sma
        data[f'BB_Width_{bb_window}'] = (data[f'BB_High_{bb_window}'] - data[f'BB_Low_{bb_window}']) / data[f'BB_Mid_{bb_window}']
        data[f'BB_Position_{bb_window}'] = (data['Close'] - data[f'BB_Low_{bb_window}']) / (data[f'BB_High_{bb_window}'] - data[f'BB_Low_{bb_window}'])
        
        # Price momentum and rate of change - shorter lags
        lags = [lag for lag in [1, 2, 3, 5, 7] if lag < max_window]
        if not lags:
            lags = [1, 2]
            
        for lag in lags:
            data[f'Price_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
            data[f'ROC_{lag}'] = ((data['Close'] - data['Close'].shift(lag)) / data['Close'].shift(lag)) * 100
        
        # Momentum oscillator - adjusted
        mom_windows = [w for w in [5, 10] if w < max_window]
        if not mom_windows:
            mom_windows = [2, 3]
            
        for window in mom_windows[:2]:  # Use max 2 momentum windows
            data[f'Momentum_{window}'] = data['Close'] - data['Close'].shift(window)
        
        # Price acceleration
        data['Price_Acceleration'] = data['Returns'].diff()
        
        # Volume features (if available) - adjusted
        if 'Volume' in data.columns:
            data['Volume_Change'] = data['Volume'].pct_change()
            vol_ma_windows = [w for w in [3, 5] if w < max_window]
            if not vol_ma_windows:
                vol_ma_windows = [2]
            for window in vol_ma_windows:
                data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
        
        # Time-based features
        data['DayOfWeek'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
        data['DayOfMonth'] = data.index.day
        
        # Advanced trend features
        data['Price_Distance_from_Mean'] = (data['Close'] - data['Close'].rolling(min(7, max_window-1)).mean()) / data['Close'].rolling(min(7, max_window-1)).std()
        data['Trend_Strength'] = data['Close'].rolling(min(5, max_window-2)).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
        
        # Price percentile position (relative to recent history)
        for window in [w for w in [5, 10] if w < max_window]:
            data[f'Price_Percentile_{window}'] = data['Close'].rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
        
        # Mean reversion indicators
        for window in [w for w in [3, 5, 7] if w < max_window]:
            rolling_mean = data['Close'].rolling(window).mean()
            rolling_std = data['Close'].rolling(window).std()
            data[f'Z_Score_{window}'] = (data['Close'] - rolling_mean) / (rolling_std + 1e-10)
        
        # Target variable - next day's closing price
        data['Target'] = data['Close'].shift(-1)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Replace inf values with NaN and drop them
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        return data
    
    def prepare_data(self, data, test_size=0.2):
        """
        Prepare data for training
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with features
        test_size : float
            Proportion of data to use for testing
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Remove target and non-feature columns
        exclude_cols = ['Target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols]
        y = data['Target']
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        # Time series split - respect temporal order
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.index, X_test.index
    
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        """
        Train XGBoost model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation target
        sample_weights : array-like, optional
            Sample weights (more weight to recent samples)
        """
        print(f"\nTraining XGBoost model for {self.commodity_name}...")
        
        # Create exponential weights favoring recent data
        if sample_weights is None:
            sample_weights = np.exp(np.linspace(0, 1, len(y_train)))
            sample_weights = sample_weights / sample_weights.sum() * len(y_train)
        
        # Optimized XGBoost parameters for production forecasting
        # Adjusted for limited data (24 samples) to prevent overfitting
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,  # Shallower trees to prevent overfitting with limited data
            'learning_rate': 0.05,  # Lower learning rate for better generalization
            'n_estimators': 150,  # Fewer estimators for small dataset
            'min_child_weight': 2,  # Higher to prevent overfitting
            'subsample': 0.85,  # Use most of limited data but not all
            'colsample_bytree': 0.85,  # Use most features but allow some dropout
            'colsample_bylevel': 0.85,
            'gamma': 0.1,  # Stronger regularization
            'reg_alpha': 0.5,  # Increased L1 regularization
            'reg_lambda': 1.5,  # Increased L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'eval_metric': 'rmse'
        }
        
        # Create and train model
        self.model = xgb.XGBRegressor(**params)
        
        # Prepare evaluation set if validation data provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                verbose=False
            )
        
        print(f"✓ Model training completed")
    
    def predict(self, X, smooth=False, smooth_window=3):
        """
        Make predictions with optional smoothing
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        smooth : bool
            Whether to apply smoothing to predictions
        smooth_window : int
            Window size for smoothing (default 3)
        
        Returns:
        --------
        array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        
        if smooth and len(self.prediction_history) >= smooth_window - 1:
            # Apply exponential moving average smoothing
            alpha = 2 / (smooth_window + 1)
            for i in range(len(predictions)):
                if len(self.prediction_history) > 0:
                    predictions[i] = alpha * predictions[i] + (1 - alpha) * self.prediction_history[-1]
                self.prediction_history.append(predictions[i])
                # Keep history limited
                if len(self.prediction_history) > smooth_window * 2:
                    self.prediction_history.pop(0)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, predictions
    
    def save_model(self):
        """Save model and scaler"""
        model_path = os.path.join(self.model_dir, f"{self.commodity_name}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{self.commodity_name}_scaler.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, f"{self.commodity_name}_features.pkl"))
        
        print(f"✓ Model saved to {model_path}")
    
    def load_model(self):
        """Load model and scaler"""
        model_path = os.path.join(self.model_dir, f"{self.commodity_name}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{self.commodity_name}_scaler.pkl")
        features_path = os.path.join(self.model_dir, f"{self.commodity_name}_features.pkl")
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(features_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"Model file not found at {model_path}")
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance
