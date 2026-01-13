"""
Advanced Ensemble Predictor with Feature Selection and Optimization
Uses XGBoost + LightGBM + CatBoost for higher accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import os
from scipy import stats


class AdvancedEnsemblePredictor:
    """Advanced ensemble predictor with multiple models and feature selection"""
    
    def __init__(self, commodity_name, model_dir='models'):
        self.commodity_name = commodity_name
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Ensemble of models
        self.xgb_dir_model = None
        self.lgb_dir_model = None
        self.catb_dir_model = None
        self.xgb_price_model = None
        self.lgb_price_model = None
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.selected_features = None
        
    def create_ultra_features(self, df):
        """Create comprehensive feature set with market regime detection"""
        data = df.copy()
        
        # Basic price features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price statistics
        if 'High' in data.columns and 'Low' in data.columns:
            data['Price_Range'] = data['High'] - data['Low']
            data['Price_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
            data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
            data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
            data['Body_Size'] = abs(data['Close'] - data['Open'])
            data['Body_Pct'] = data['Body_Size'] / data['Price_Range'].replace(0, np.nan)
        
        # Determine max window size based on data length
        max_window = max(3, len(data) // 3)
        
        # Multiple timeframe moving averages - adjusted for limited data
        windows = [w for w in [3, 5, 7, 10, 15] if w < max_window]
        if not windows:
            windows = [2, 3]
        
        for window in windows:
            sma = data['Close'].rolling(window=window).mean()
            ema = data['Close'].ewm(span=window, adjust=False).mean()
            
            data[f'SMA_{window}'] = sma
            data[f'EMA_{window}'] = ema
            data[f'Price_to_SMA_{window}'] = (data['Close'] - sma) / sma
            data[f'Price_to_EMA_{window}'] = (data['Close'] - ema) / ema
            data[f'SMA_Slope_{window}'] = sma.pct_change()
            data[f'EMA_Slope_{window}'] = ema.pct_change()
            
            # Distance from MA in std units
            std = data['Close'].rolling(window=window).std()
            data[f'Price_ZScore_{window}'] = (data['Close'] - sma) / (std + 1e-10)
        
        # MA crossovers and relative positions - adjusted for available windows
        if len(windows) >= 2:
            data[f'SMA_{windows[0]}_{windows[1]}_Ratio'] = data[f'SMA_{windows[0]}'] / (data[f'SMA_{windows[1]}'] + 1e-10)
            data[f'EMA_{windows[0]}_{windows[1]}_Ratio'] = data[f'EMA_{windows[0]}'] / (data[f'EMA_{windows[1]}'] + 1e-10)
        if len(windows) >= 3:
            data[f'SMA_{windows[0]}_{windows[2]}_Ratio'] = data[f'SMA_{windows[0]}'] / (data[f'SMA_{windows[2]}'] + 1e-10)
        
        # Golden/Death cross signals - skip if not enough data
        # data['Golden_Cross'] = ((data['SMA_50'] > data['SMA_200']) & 
        #                         (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))).astype(int)
        data['Death_Cross'] = ((data['SMA_50'] < data['SMA_200']) & 
                               (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))).astype(int)
        
        # Volatility measures
        for window in [5, 10, 20, 30, 60]:
            data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
            data[f'Volatility_Ratio_{window}'] = (data[f'Volatility_{window}'] / 
                                                   data[f'Volatility_{window}'].rolling(window=20).mean())
            # Parkinson volatility (uses High-Low)
            if 'High' in data.columns and 'Low' in data.columns:
                data[f'Parkinson_Vol_{window}'] = np.sqrt(
                    (np.log(data['High'] / data['Low']) ** 2).rolling(window=window).mean() / (4 * np.log(2))
                )
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            data[f'RSI_{period}_Oversold'] = (data[f'RSI_{period}'] < 30).astype(int)
            data[f'RSI_{period}_Overbought'] = (data[f'RSI_{period}'] > 70).astype(int)
            data[f'RSI_{period}_Momentum'] = data[f'RSI_{period}'].diff()
        
        # Stochastic Oscillator
        for window in [14]:
            low_min = data['Low'].rolling(window=window).min() if 'Low' in data.columns else data['Close'].rolling(window=window).min()
            high_max = data['High'].rolling(window=window).max() if 'High' in data.columns else data['Close'].rolling(window=window).max()
            data[f'Stoch_K_{window}'] = 100 * (data['Close'] - low_min) / (high_max - low_min + 1e-10)
            data[f'Stoch_D_{window}'] = data[f'Stoch_K_{window}'].rolling(window=3).mean()
        
        # MACD with multiple settings
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            data[f'MACD_{fast}_{slow}'] = macd
            data[f'MACD_Signal_{fast}_{slow}'] = macd_signal
            data[f'MACD_Diff_{fast}_{slow}'] = macd - macd_signal
            data[f'MACD_Cross_{fast}_{slow}'] = (macd > macd_signal).astype(int)
        
        # Bollinger Bands with multiple windows
        for window in [20, 50]:
            sma = data['Close'].rolling(window=window).mean()
            std = data['Close'].rolling(window=window).std()
            data[f'BB_High_{window}'] = sma + (2 * std)
            data[f'BB_Low_{window}'] = sma - (2 * std)
            data[f'BB_Mid_{window}'] = sma
            data[f'BB_Width_{window}'] = (data[f'BB_High_{window}'] - data[f'BB_Low_{window}']) / sma
            data[f'BB_Position_{window}'] = (data['Close'] - data[f'BB_Low_{window}']) / (data[f'BB_High_{window}'] - data[f'BB_Low_{window}'] + 1e-10)
            data[f'BB_Squeeze_{window}'] = data[f'BB_Width_{window}'] / data[f'BB_Width_{window}'].rolling(window=50).mean()
            # Price distance from bands
            data[f'BB_Upper_Dist_{window}'] = (data[f'BB_High_{window}'] - data['Close']) / data['Close']
            data[f'BB_Lower_Dist_{window}'] = (data['Close'] - data[f'BB_Low_{window}']) / data['Close']
        
        # Rate of Change
        for period in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
            data[f'ROC_{period}'] = ((data['Close'] - data['Close'].shift(period)) / 
                                     (data['Close'].shift(period) + 1e-10)) * 100
            data[f'Returns_Lag_{period}'] = data['Returns'].shift(period)
        
        # Momentum indicators
        for window in [5, 10, 20, 30]:
            data[f'Momentum_{window}'] = data['Close'] - data['Close'].shift(window)
            data[f'Momentum_Pct_{window}'] = ((data['Close'] - data['Close'].shift(window)) / 
                                              (data['Close'].shift(window) + 1e-10)) * 100
        
        # Trend strength and direction
        for window in [10, 20, 50]:
            data[f'Trend_Strength_{window}'] = abs(
                data['Close'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            )
            data[f'Trend_Direction_{window}'] = np.sign(
                data['Close'].rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            )
        
        # Autocorrelation features
        for lag in [1, 5, 10]:
            data[f'Autocorr_{lag}'] = data['Returns'].rolling(window=30).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        # Market regime indicators
        data['High_Volatility_Regime'] = (data['Volatility_20'] > data['Volatility_20'].rolling(100).quantile(0.75)).astype(int)
        data['Trending_Regime'] = (abs(data['Trend_Strength_20']) > data['Trend_Strength_20'].rolling(100).quantile(0.75)).astype(int)
        
        # Price patterns
        data['Higher_High'] = ((data['Close'] > data['Close'].shift(1)) & 
                               (data['Close'].shift(1) > data['Close'].shift(2))).astype(int)
        data['Lower_Low'] = ((data['Close'] < data['Close'].shift(1)) & 
                             (data['Close'].shift(1) < data['Close'].shift(2))).astype(int)
        
        # Volume features
        if 'Volume' in data.columns:
            for window in [5, 10, 20]:
                data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
                data[f'Volume_Ratio_{window}'] = data['Volume'] / (data[f'Volume_MA_{window}'] + 1e-10)
                data[f'Volume_Std_{window}'] = data['Volume'].rolling(window=window).std()
            
            data['Volume_Price_Trend'] = (np.sign(data['Close'].diff()) * data['Volume']).rolling(20).sum()
            data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
            data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
        
        # Time features with cyclical encoding
        data['DayOfWeek'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
        data['DayOfMonth'] = data.index.day
        data['WeekOfYear'] = data.index.isocalendar().week
        
        # Cyclical encoding
        data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        data['Day_Sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
        data['Day_Cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
        data['Quarter_Sin'] = np.sin(2 * np.pi * data['Quarter'] / 4)
        data['Quarter_Cos'] = np.cos(2 * np.pi * data['Quarter'] / 4)
        
        # Target variables
        data['Target_Price'] = data['Close'].shift(-1)
        data['Target_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        return data
    
    def select_important_features(self, X, y, top_n=100):
        """Select most important features using XGBoost"""
        print(f"  Selecting top {top_n} features from {X.shape[1]}...")
        
        temp_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        temp_model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': range(len(temp_model.feature_importances_)),
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_indices = importance.head(top_n)['feature'].values
        print(f"  ✓ Selected {len(selected_indices)} most important features")
        
        return selected_indices
    
    def prepare_data(self, data, test_size=0.2, select_features=True):
        """Prepare data with feature selection"""
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
        
        # Feature selection
        if select_features and self.selected_features is None:
            self.selected_features = self.select_important_features(
                X_train_scaled, y_dir_train, top_n=min(100, X_train_scaled.shape[1])
            )
        
        if self.selected_features is not None:
            X_train_scaled = X_train_scaled[:, self.selected_features]
            X_test_scaled = X_test_scaled[:, self.selected_features]
        
        return X_train_scaled, X_test_scaled, y_price_train, y_price_test, y_dir_train, y_dir_test, X_train.index, X_test.index
    
    def train(self, X_train, y_price_train, y_dir_train):
        """Train ensemble of models"""
        print(f"\nTraining ensemble models for {self.commodity_name}...")
        
        # Direction models (ensemble)
        print("  Training XGBoost direction model...")
        self.xgb_dir_model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=7,
            learning_rate=0.03,
            n_estimators=800,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=1,
            gamma=0.05,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_dir_model.fit(X_train, y_dir_train, verbose=False)
        
        print("  Training LightGBM direction model...")
        self.lgb_dir_model = lgb.LGBMClassifier(
            objective='binary',
            max_depth=7,
            learning_rate=0.03,
            n_estimators=800,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_dir_model.fit(X_train, y_dir_train)
        
        print("  Training CatBoost direction model...")
        self.catb_dir_model = cb.CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        self.catb_dir_model.fit(X_train, y_dir_train)
        
        # Price models
        print("  Training XGBoost price model...")
        self.xgb_price_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=8,
            learning_rate=0.02,
            n_estimators=900,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_price_model.fit(X_train, y_price_train, verbose=False)
        
        print("  Training LightGBM price model...")
        self.lgb_price_model = lgb.LGBMRegressor(
            objective='regression',
            max_depth=8,
            learning_rate=0.02,
            n_estimators=900,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_price_model.fit(X_train, y_price_train)
        
        print(f"  ✓ All ensemble models trained")
    
    def predict_direction(self, X):
        """Ensemble prediction for direction"""
        # Get predictions from each model
        xgb_pred = self.xgb_dir_model.predict_proba(X)[:, 1]
        lgb_pred = self.lgb_dir_model.predict_proba(X)[:, 1]
        catb_pred = self.catb_dir_model.predict_proba(X)[:, 1]
        
        # Weighted average (can tune weights)
        ensemble_proba = (0.4 * xgb_pred + 0.35 * lgb_pred + 0.25 * catb_pred)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def predict_price(self, X):
        """Ensemble prediction for price"""
        xgb_pred = self.xgb_price_model.predict(X)
        lgb_pred = self.lgb_price_model.predict(X)
        
        # Average predictions
        ensemble_pred = (0.55 * xgb_pred + 0.45 * lgb_pred)
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_price_test, y_dir_test):
        """Evaluate ensemble models"""
        price_predictions = self.predict_price(X_test)
        dir_predictions, _ = self.predict_direction(X_test)
        
        # Price metrics
        mae = mean_absolute_error(y_price_test, price_predictions)
        rmse = np.sqrt(np.mean((y_price_test - price_predictions) ** 2))
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
        """Save all models"""
        joblib.dump(self.xgb_dir_model, os.path.join(self.model_dir, f"{self.commodity_name}_xgb_dir.pkl"))
        joblib.dump(self.lgb_dir_model, os.path.join(self.model_dir, f"{self.commodity_name}_lgb_dir.pkl"))
        joblib.dump(self.catb_dir_model, os.path.join(self.model_dir, f"{self.commodity_name}_catb_dir.pkl"))
        joblib.dump(self.xgb_price_model, os.path.join(self.model_dir, f"{self.commodity_name}_xgb_price.pkl"))
        joblib.dump(self.lgb_price_model, os.path.join(self.model_dir, f"{self.commodity_name}_lgb_price.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, f"{self.commodity_name}_adv_scaler.pkl"))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, f"{self.commodity_name}_adv_features.pkl"))
        joblib.dump(self.selected_features, os.path.join(self.model_dir, f"{self.commodity_name}_selected_features.pkl"))
        print(f"  ✓ Ensemble models saved")
