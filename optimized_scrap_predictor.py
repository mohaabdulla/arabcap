"""
Optimized Casting Scrap Predictor
Specialized model with advanced time series features for scrap prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class OptimizedScrapPredictor:
    """Advanced predictor for casting scrap with specialized features"""
    
    def __init__(self, target_column):
        self.target = target_column
        self.model = None
    
    def create_features(self, df):
        """Create advanced time series features"""
        df = df.copy()
        
        # Extended lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df[self.target].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df[f'sma_{window}'] = df[self.target].rolling(window=window, min_periods=1).mean()
            df[f'std_{window}'] = df[self.target].rolling(window=window, min_periods=1).std()
            df[f'min_{window}'] = df[self.target].rolling(window=window, min_periods=1).min()
            df[f'max_{window}'] = df[self.target].rolling(window=window, min_periods=1).max()
            df[f'range_{window}'] = df[f'max_{window}'] - df[f'min_{window}']
        
        # Exponential moving averages
        for span in [3, 6, 12]:
            df[f'ema_{span}'] = df[self.target].ewm(span=span, min_periods=1).mean()
        
        # Momentum indicators
        df['momentum_1'] = df[self.target].diff(1)
        df['momentum_3'] = df[self.target].diff(3)
        df['roc_1'] = df[self.target].pct_change(1)
        df['roc_3'] = df[self.target].pct_change(3)
        
        # Acceleration
        df['acceleration'] = df['momentum_1'].diff(1)
        
        # Volatility
        df['volatility_3'] = df[self.target].rolling(window=3, min_periods=1).std()
        df['volatility_6'] = df[self.target].rolling(window=6, min_periods=1).std()
        
        # Month features (cyclical encoding)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['quarter'] = ((df['Month'] - 1) // 3) + 1
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Year trend
        df['time_index'] = np.arange(len(df))
        
        # Seasonal decomposition indicators
        df['detrended'] = df[self.target] - df['sma_12']
        
        # Distance from mean/median
        df['dist_from_mean'] = df[self.target] - df[self.target].expanding().mean()
        df['dist_from_median'] = df[self.target] - df[self.target].expanding().median()
        
        df = df.dropna()
        return df
    
    def walk_forward_validation(self, data_df, min_train_size=15):
        """Walk-forward validation with ensemble"""
        data_with_features = self.create_features(data_df)
        
        # Adjust min_train_size if data is too short
        min_train_size = min(min_train_size, len(data_with_features) - 2)
        
        predictions = []
        actuals = []
        
        print(f"  Training from month {min_train_size} onwards ({len(data_with_features) - min_train_size} test points)...")
        
        for i in range(min_train_size, len(data_with_features)):
            train_data = data_with_features.iloc[:i]
            test_data = data_with_features.iloc[i:i+1]
            
            feature_cols = [c for c in train_data.columns 
                           if c not in ['Year', 'Month', self.target, 'Date'] 
                           and train_data[c].dtype in ['float64', 'int64']]
            
            X_train = train_data[feature_cols]
            y_train = train_data[self.target]
            X_test = test_data[feature_cols]
            
            # XGBoost model
            xgb_model = xgb.XGBRegressor(
                max_depth=3,
                learning_rate=0.05,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Random Forest for ensemble
            rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=4,
                min_samples_split=3,
                random_state=42
            )
            
            xgb_model.fit(X_train, y_train, verbose=False)
            rf_model.fit(X_train, y_train)
            
            pred_xgb = xgb_model.predict(X_test)[0]
            pred_rf = rf_model.predict(X_test)[0]
            
            # Weighted ensemble (70% XGBoost, 30% Random Forest)
            pred = 0.7 * pred_xgb + 0.3 * pred_rf
            
            predictions.append(pred)
            actuals.append(test_data[self.target].values[0])
        
        return np.array(predictions), np.array(actuals)
    
    def calculate_metrics(self, predictions, actuals):
        """Calculate performance metrics"""
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        return mae, rmse, mape
    
    def visualize_results(self, predictions, actuals, data_df, min_train_size=15):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Optimized Casting Scrap Prediction: {self.target}', fontsize=16, fontweight='bold')
        
        # Adjust min_train_size to match actual data length
        data_with_features = self.create_features(data_df)
        actual_min_train = min(min_train_size, len(data_with_features) - 2)
        
        # Get dates for x-axis
        dates = data_df.iloc[actual_min_train:actual_min_train+len(actuals)]['Month'].values
        
        # 1. Predictions vs Actuals
        axes[0, 0].plot(dates, actuals, 'o-', label='Actual', linewidth=2, markersize=8)
        axes[0, 0].plot(dates, predictions, 's-', label='Predicted', linewidth=2, markersize=8)
        axes[0, 0].set_title('Predictions vs Actuals', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel(self.target)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        errors = actuals - predictions
        axes[0, 1].hist(errors, bins=10, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Error Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot
        axes[1, 0].scatter(actuals, predictions, alpha=0.6, s=100)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_title('Actual vs Predicted', fontweight='bold')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentage errors
        pct_errors = np.abs((actuals - predictions) / actuals) * 100
        axes[1, 1].bar(range(len(pct_errors)), pct_errors, alpha=0.7)
        axes[1, 1].axhline(y=5, color='red', linestyle='--', linewidth=2, label='5% threshold')
        axes[1, 1].set_title('Percentage Errors', fontweight='bold')
        axes[1, 1].set_xlabel('Test Sample')
        axes[1, 1].set_ylabel('Absolute % Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Run optimized scrap predictions"""
    print("\n" + "="*70)
    print("OPTIMIZED CASTING SCRAP PREDICTION SYSTEM")
    print("="*70 + "\n")
    
    # Load data
    data_df = pd.read_csv('data/casting_scrap_historical.csv')
    print(f"Loaded data: {len(data_df)} months (2024-2025)")
    print(f"Columns: {list(data_df.columns)}\n")
    
    results = []
    
    for target in ['Process', 'Defect', 'Total']:
        print(f"\n{'='*70}")
        print(f"PREDICTING: {target}")
        print(f"{'='*70}\n")
        
        predictor = OptimizedScrapPredictor(target)
        
        # Create features
        data_with_features = predictor.create_features(data_df)
        print(f"Created {len([c for c in data_with_features.columns if c not in ['Year', 'Month', target]])} features")
        
        # Run walk-forward validation
        print("Running optimized walk-forward validation...")
        predictions, actuals = predictor.walk_forward_validation(data_df)
        
        # Calculate metrics
        mae, rmse, mape = predictor.calculate_metrics(predictions, actuals)
        
        print(f"\nResults for {target}:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Visualize
        fig = predictor.visualize_results(predictions, actuals, data_df)
        output_path = f'results/optimized_scrap_{target}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")
        
        # Save predictions
        data_with_features = predictor.create_features(data_df)
        actual_min_train = min(15, len(data_with_features) - 2)
        
        results_df = pd.DataFrame({
            'Month': data_df.iloc[actual_min_train:actual_min_train+len(actuals)]['Month'].values,
            'Year': data_df.iloc[actual_min_train:actual_min_train+len(actuals)]['Year'].values,
            'Actual': actuals,
            'Predicted': predictions,
            'Error': actuals - predictions,
            'Abs_Pct_Error': np.abs((actuals - predictions) / actuals) * 100
        })
        csv_path = f'results/optimized_scrap_{target}_predictions.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
        
        results.append({
            'Target': target,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL PREDICTIONS")
    print(f"{'='*70}\n")
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("OPTIMIZED CASTING SCRAP PREDICTION COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
