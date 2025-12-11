"""
Backtesting Module for Commodity Price Prediction
Implements walk-forward validation and comprehensive performance metrics
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class BacktestEngine:
    """Backtesting engine for time series predictions"""
    
    def __init__(self, predictor, data):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        predictor : CommodityPricePredictor
            Trained prediction model
        data : pd.DataFrame
            Data with features for backtesting
        """
        self.predictor = predictor
        self.data = data
        self.results = None
    
    def walk_forward_validation(self, initial_train_size=0.7, step_size=1, retrain_frequency=30):
        """
        Perform walk-forward validation
        
        Parameters:
        -----------
        initial_train_size : float
            Initial proportion of data for training
        step_size : int
            Number of days to step forward in each iteration
        retrain_frequency : int
            Retrain model every N days
        
        Returns:
        --------
        pd.DataFrame
            Predictions and actual values for each time step
        """
        print(f"\nStarting walk-forward validation...")
        print(f"Initial training size: {initial_train_size*100:.0f}%")
        print(f"Retrain frequency: every {retrain_frequency} days")
        
        results = []
        
        # Prepare data
        data_with_features = self.predictor.create_features(self.data)
        
        # Exclude non-feature columns
        exclude_cols = ['Target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']
        feature_cols = [col for col in data_with_features.columns if col not in exclude_cols]
        
        # Initial split
        initial_train_idx = int(len(data_with_features) * initial_train_size)
        
        days_since_retrain = 0
        
        # Walk forward through the data
        for i in range(initial_train_idx, len(data_with_features) - 1, step_size):
            # Training data up to current point
            train_data = data_with_features.iloc[:i]
            
            # Test data point
            test_data = data_with_features.iloc[i:i+1]
            
            X_train = train_data[feature_cols]
            y_train = train_data['Target']
            X_test = test_data[feature_cols]
            y_test = test_data['Target'].values[0]
            
            # Retrain model if needed
            if days_since_retrain == 0 or days_since_retrain >= retrain_frequency:
                X_train_scaled = self.predictor.scaler.fit_transform(X_train)
                self.predictor.train(X_train_scaled, y_train)
                days_since_retrain = 0
            
            # Make prediction
            X_test_scaled = self.predictor.scaler.transform(X_test)
            prediction = self.predictor.predict(X_test_scaled)[0]
            
            # Store results
            results.append({
                'Date': test_data.index[0],
                'Actual': y_test,
                'Predicted': prediction,
                'Error': prediction - y_test,
                'Absolute_Error': abs(prediction - y_test),
                'Percentage_Error': abs((prediction - y_test) / y_test) * 100
            })
            
            days_since_retrain += step_size
            
            # Progress indicator
            if len(results) % 50 == 0:
                print(f"  Processed {len(results)} predictions...")
        
        self.results = pd.DataFrame(results)
        print(f"✓ Walk-forward validation completed: {len(self.results)} predictions")
        
        return self.results
    
    def calculate_metrics(self):
        """Calculate comprehensive backtest metrics"""
        if self.results is None:
            raise ValueError("No backtest results available. Run walk_forward_validation first.")
        
        actual = self.results['Actual'].values
        predicted = self.results['Predicted'].values
        
        # Basic metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy (did we predict the right direction?)
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # Trading metrics (assuming simple strategy)
        # Buy when predicted > actual (price expected to rise)
        trade_signals = predicted > actual
        returns = []
        
        for i in range(len(actual) - 1):
            if trade_signals[i]:
                # Calculate return if we bought
                returns.append((actual[i+1] - actual[i]) / actual[i])
        
        if len(returns) > 0:
            total_return = (np.prod([1 + r for r in returns]) - 1) * 100
            avg_return = np.mean(returns) * 100
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            total_return = 0
            avg_return = 0
            sharpe_ratio = 0
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2 Score': r2,
            'Directional Accuracy (%)': directional_accuracy,
            'Total Return (%)': total_return,
            'Average Return per Trade (%)': avg_return,
            'Sharpe Ratio': sharpe_ratio,
            'Number of Predictions': len(self.results),
            'Number of Trades': len(returns)
        }
        
        return metrics
    
    def get_error_statistics(self):
        """Get detailed error statistics"""
        if self.results is None:
            raise ValueError("No backtest results available")
        
        errors = self.results['Error']
        
        stats = {
            'Mean Error': errors.mean(),
            'Median Error': errors.median(),
            'Std Error': errors.std(),
            'Min Error': errors.min(),
            'Max Error': errors.max(),
            '25th Percentile': errors.quantile(0.25),
            '75th Percentile': errors.quantile(0.75),
            'Mean Absolute Error': self.results['Absolute_Error'].mean(),
            'Mean Percentage Error': self.results['Percentage_Error'].mean()
        }
        
        return stats
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.results is None:
            raise ValueError("No backtest results available")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted Prices
        axes[0].plot(self.results['Date'], self.results['Actual'], 
                    label='Actual Price', linewidth=2, alpha=0.7)
        axes[0].plot(self.results['Date'], self.results['Predicted'], 
                    label='Predicted Price', linewidth=2, alpha=0.7)
        axes[0].set_title(f'{self.predictor.commodity_name.upper()} - Actual vs Predicted Prices', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction Errors
        axes[1].plot(self.results['Date'], self.results['Error'], 
                    color='red', alpha=0.6)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].fill_between(self.results['Date'], self.results['Error'], 0, 
                            alpha=0.3, color='red')
        axes[1].set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Error (Predicted - Actual)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        axes[2].hist(self.results['Percentage_Error'], bins=50, 
                    color='skyblue', edgecolor='black', alpha=0.7)
        axes[2].axvline(x=self.results['Percentage_Error'].mean(), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {self.results["Percentage_Error"].mean():.2f}%')
        axes[2].set_title('Distribution of Percentage Errors', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Absolute Percentage Error (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        return fig
    
    def export_results(self, filepath):
        """Export backtest results to CSV"""
        if self.results is None:
            raise ValueError("No backtest results available")
        
        self.results.to_csv(filepath, index=False)
        print(f"✓ Results exported to {filepath}")


def print_metrics_report(metrics):
    """Pretty print metrics report"""
    print("\n" + "="*70)
    print("BACKTESTING PERFORMANCE METRICS")
    print("="*70)
    
    print("\nPrediction Accuracy Metrics:")
    print(f"  MAE (Mean Absolute Error):        ${metrics['MAE']:.4f}")
    print(f"  RMSE (Root Mean Squared Error):   ${metrics['RMSE']:.4f}")
    print(f"  MAPE (Mean Abs Percentage Error): {metrics['MAPE']:.2f}%")
    print(f"  R² Score:                         {metrics['R2 Score']:.4f}")
    print(f"  Directional Accuracy:             {metrics['Directional Accuracy (%)']:.2f}%")
    
    print("\nTrading Performance Metrics:")
    print(f"  Total Return:                     {metrics['Total Return (%)']:.2f}%")
    print(f"  Average Return per Trade:         {metrics['Average Return per Trade (%)']:.4f}%")
    print(f"  Sharpe Ratio:                     {metrics['Sharpe Ratio']:.4f}")
    
    print("\nBacktest Statistics:")
    print(f"  Total Predictions:                {metrics['Number of Predictions']}")
    print(f"  Total Trades:                     {metrics['Number of Trades']}")
    
    print("="*70)
