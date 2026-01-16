"""
Ultra Refined Optimizers - Incremental improvements from super_aggressive
Focus on the tiny tweaks needed to push Si and Iron below 5%
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class UltraRefinedOptimizer:
    """Base class with refined adaptive learning"""
    
    def __init__(self, name):
        self.name = name
    
    def get_base_models(self):
        """Enhanced pool of robust models"""
        return [
            ('xgb_ultra_shallow', xgb.XGBRegressor(max_depth=1, learning_rate=0.02, n_estimators=40,
                                                   subsample=0.98, colsample_bytree=0.98,
                                                   min_child_weight=5, gamma=0.1, random_state=42)),
            ('xgb_shallow', xgb.XGBRegressor(max_depth=1, learning_rate=0.03, n_estimators=50,
                                             subsample=0.95, colsample_bytree=0.95,
                                             min_child_weight=3, gamma=0.05, random_state=43)),
            ('rf_tiny', RandomForestRegressor(n_estimators=30, max_depth=2, min_samples_leaf=2,
                                             max_features='sqrt', random_state=42)),
            ('gb_shallow', GradientBoostingRegressor(max_depth=1, learning_rate=0.03, n_estimators=40,
                                                     subsample=0.95, random_state=42)),
            ('ridge_light', Ridge(alpha=2.0)),
            ('ridge_heavy', Ridge(alpha=5.0)),
            ('huber', HuberRegressor(epsilon=1.35)),
        ]
    
    def simple_forecasts(self, train_data):
        """Enhanced simple statistical forecasts"""
        recent = train_data['Consumption'].values
        forecasts = []
        
        # Last value
        forecasts.append(('last', recent[-1]))
        
        # Mean variants
        if len(recent) >= 2:
            forecasts.append(('mean2', np.mean(recent[-2:])))
        if len(recent) >= 3:
            forecasts.append(('mean3', np.mean(recent[-3:])))
        if len(recent) >= 4:
            forecasts.append(('mean4', np.mean(recent[-4:])))
        
        # Median variants
        if len(recent) >= 3:
            forecasts.append(('median3', np.median(recent[-3:])))
        if len(recent) >= 4:
            forecasts.append(('median4', np.median(recent[-4:])))
        if len(recent) >= 5:
            forecasts.append(('median5', np.median(recent[-5:])))
        
        # Weighted averages
        if len(recent) >= 3:
            weights = np.array([0.2, 0.3, 0.5])
            forecasts.append(('weighted3', np.average(recent[-3:], weights=weights)))
        if len(recent) >= 4:
            weights = np.array([0.1, 0.2, 0.3, 0.4])
            forecasts.append(('weighted4', np.average(recent[-4:], weights=weights)))
        
        # EMA variants
        forecasts.append(('ema_2', train_data['Consumption'].ewm(span=2, adjust=False).mean().iloc[-1]))
        forecasts.append(('ema_2.5', train_data['Consumption'].ewm(span=2.5, adjust=False).mean().iloc[-1]))
        forecasts.append(('ema_3', train_data['Consumption'].ewm(span=3, adjust=False).mean().iloc[-1]))
        
        # Trimmed mean (remove extremes)
        if len(recent) >= 5:
            sorted_vals = np.sort(recent[-5:])
            forecasts.append(('trimmed5', np.mean(sorted_vals[1:-1])))
        
        # Winsorized mean
        if len(recent) >= 5:
            recent_5 = recent[-5:]
            q25, q75 = np.percentile(recent_5, [25, 75])
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            capped = np.clip(recent_5, lower, upper)
            forecasts.append(('winsor5', np.mean(capped)))
        
        return forecasts
    
    def train_predict_adaptive(self, data_df, features_func):
        """Refined adaptive training"""
        data_with_features = features_func(data_df)
        
        predictions = []
        actuals = []
        model_scores = {name: [] for name, _ in self.get_base_models()}
        simple_scores = {}
        
        min_train = 5
        adaptation_threshold = min_train + 1  # Adapt after just 1 prediction
        
        for i in range(min_train, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            feature_cols = [c for c in train_data.columns 
                           if c not in ['Date', 'Consumption', 'Material']
                           and train_data[c].dtype in ['float64', 'int64']]
            
            X_train = train_data[feature_cols]
            y_train = train_data['Consumption']
            X_test = test_data[feature_cols]
            actual = test_data['Consumption'].values[0]
            
            # Train all models
            model_preds = []
            
            # ML models
            for name, model in self.get_base_models():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)[0]
                    model_preds.append((name, pred))
                except:
                    pass
            
            # Simple forecasts
            train_data_orig = data_df.iloc[:i]
            for name, pred in self.simple_forecasts(train_data_orig):
                model_preds.append((name, pred))
                if name not in simple_scores:
                    simple_scores[name] = []
            
            # Adaptive weighting
            if i < adaptation_threshold or len(predictions) == 0:
                # Start with slight bias toward medians and trimmed means
                weights = []
                for name, _ in model_preds:
                    if 'median' in name or 'trimmed' in name or 'winsor' in name:
                        weights.append(1.5)  # Boost robust estimators
                    elif 'ema' in name or 'weighted' in name:
                        weights.append(1.2)
                    else:
                        weights.append(1.0)
                weights = np.array(weights)
                weights = weights / weights.sum()
            else:
                # Exponentially weight recent performance
                weights = []
                for name, pred in model_preds:
                    if name in model_scores and len(model_scores[name]) > 0:
                        errors = np.array(model_scores[name])
                        # Recent errors matter more
                        recent_weight = np.power(0.7, np.arange(len(errors))[::-1])
                        weighted_error = np.average(errors, weights=recent_weight)
                        weight = 1.0 / (weighted_error + 0.01)
                    elif name in simple_scores and len(simple_scores[name]) > 0:
                        errors = np.array(simple_scores[name])
                        recent_weight = np.power(0.7, np.arange(len(errors))[::-1])
                        weighted_error = np.average(errors, weights=recent_weight)
                        weight = 1.0 / (weighted_error + 0.01)
                    else:
                        weight = 1.0
                    weights.append(weight)
                
                weights = np.array(weights)
                weights = weights / weights.sum()
            
            # Weighted prediction
            final_pred = sum(w * pred for w, (_, pred) in zip(weights, model_preds))
            predictions.append(final_pred)
            actuals.append(actual)
            
            # Update performance tracking
            for name, pred in model_preds:
                pred_error = abs(actual - pred)
                if name in model_scores:
                    model_scores[name].append(pred_error)
                elif name in simple_scores:
                    simple_scores[name].append(pred_error)
        
        return np.array(predictions), np.array(actuals)


class UltraSiMetalOptimizer(UltraRefinedOptimizer):
    def __init__(self):
        super().__init__("Si Metal 98.5%")
    
    def create_features(self, df):
        df = df.copy()
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Minimal features
        for lag in [1, 2]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        df['sma_2'] = df['Consumption'].rolling(window=2, min_periods=1).mean()
        df['sma_3'] = df['Consumption'].rolling(window=3, min_periods=1).mean()
        df['ema_2'] = df['Consumption'].ewm(span=2, min_periods=1).mean()
        
        df['position'] = np.arange(len(df))
        
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        return self.train_predict_adaptive(data_df, self.create_features)


class UltraIronOptimizer(UltraRefinedOptimizer):
    def __init__(self):
        super().__init__("Iron Metal (80%)")
    
    def create_features(self, df):
        df = df.copy()
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Very minimal - iron is volatile
        df['lag_1'] = df['Consumption'].shift(1)
        
        for window in [3, 4]:
            df[f'median_{window}'] = df['Consumption'].rolling(window=window, min_periods=1).median()
            df[f'sma_{window}'] = df['Consumption'].rolling(window=window, min_periods=1).mean()
        
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        return self.train_predict_adaptive(data_df, self.create_features)


class UltraMagnesiumOptimizer(UltraRefinedOptimizer):
    def __init__(self):
        super().__init__("Magnesium(99.90%)")
    
    def create_features(self, df):
        df = df.copy()
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        for lag in [1, 2]:
            df[f'lag_{lag}'] = df['Consumption'].shift(lag)
        
        for window in [2, 3]:
            df[f'sma_{window}'] = df['Consumption'].rolling(window=window, min_periods=1).mean()
            df[f'median_{window}'] = df['Consumption'].rolling(window=window, min_periods=1).median()
        
        df['ema_2'] = df['Consumption'].ewm(span=2, min_periods=1).mean()
        
        if 'Date' in df.columns:
            df['month'] = pd.to_datetime(df['Date']).dt.month
        else:
            df['month'] = (np.arange(len(df)) % 12) + 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        
        df = df.dropna()
        return df
    
    def train_predict(self, data_df):
        return self.train_predict_adaptive(data_df, self.create_features)


def calculate_metrics(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mae, rmse, mape
