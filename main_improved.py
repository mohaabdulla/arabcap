"""
Improved Main Script with Better Directional Accuracy
Uses ensemble approach with classification + regression
"""

import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import CommodityDataCollector
from improved_predictor import ImprovedCommodityPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_style('whitegrid')


def backtest_improved_model(predictor, data, initial_train_pct=0.7, retrain_freq=30):
    """Walk-forward backtesting with direction focus"""
    print("\\nRunning improved backtesting...")
    
    # Create features
    data_with_features = predictor.create_advanced_features(data)
    
    exclude_cols = ['Target_Price', 'Target_Direction', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']
    feature_cols = [col for col in data_with_features.columns if col not in exclude_cols]
    
    initial_train_idx = int(len(data_with_features) * initial_train_pct)
    
    results = []
    days_since_retrain = 0
    
    for i in range(initial_train_idx, len(data_with_features) - 1):
        train_data = data_with_features.iloc[:i]
        test_data = data_with_features.iloc[i:i+1]
        
        X_train = train_data[feature_cols]
        y_price_train = train_data['Target_Price']
        y_dir_train = train_data['Target_Direction']
        
        X_test = test_data[feature_cols]
        y_price_test = test_data['Target_Price'].values[0]
        y_dir_test = test_data['Target_Direction'].values[0]
        
        # Retrain if needed
        if days_since_retrain == 0 or days_since_retrain >= retrain_freq:
            X_train_scaled = predictor.scaler.fit_transform(X_train)
            predictor.train(X_train_scaled, y_price_train, y_dir_train)
            days_since_retrain = 0
        
        # Predict
        X_test_scaled = predictor.scaler.transform(X_test)
        price_pred = predictor.price_model.predict(X_test_scaled)[0]
        dir_pred = predictor.direction_model.predict(X_test_scaled)[0]
        
        results.append({
            'Date': test_data.index[0],
            'Actual_Price': y_price_test,
            'Predicted_Price': price_pred,
            'Actual_Direction': y_dir_test,
            'Predicted_Direction': dir_pred,
            'Direction_Correct': (dir_pred == y_dir_test)
        })
        
        days_since_retrain += 1
        
        if len(results) % 50 == 0:
            print(f"  Processed {len(results)} predictions...")
    
    results_df = pd.DataFrame(results)
    return results_df


def calculate_improved_metrics(results_df):
    """Calculate comprehensive metrics"""
    # Direction accuracy
    dir_accuracy = (results_df['Direction_Correct'].sum() / len(results_df)) * 100
    
    # Price metrics
    mae = np.mean(np.abs(results_df['Actual_Price'] - results_df['Predicted_Price']))
    rmse = np.sqrt(np.mean((results_df['Actual_Price'] - results_df['Predicted_Price']) ** 2))
    mape = np.mean(np.abs((results_df['Actual_Price'] - results_df['Predicted_Price']) / results_df['Actual_Price'])) * 100
    
    # R2
    ss_res = np.sum((results_df['Actual_Price'] - results_df['Predicted_Price']) ** 2)
    ss_tot = np.sum((results_df['Actual_Price'] - np.mean(results_df['Actual_Price'])) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Trading simulation
    results_df['Actual_Return'] = results_df['Actual_Price'].pct_change()
    results_df['Strategy_Return'] = np.where(
        results_df['Predicted_Direction'] == 1,
        results_df['Actual_Return'],
        -results_df['Actual_Return']
    )
    
    total_return = (1 + results_df['Strategy_Return'].fillna(0)).prod() - 1
    win_rate = (results_df['Strategy_Return'] > 0).sum() / len(results_df) * 100
    
    metrics = {
        'Directional_Accuracy': dir_accuracy,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Total_Strategy_Return': total_return * 100,
        'Win_Rate': win_rate,
        'Predictions': len(results_df)
    }
    
    return metrics


def train_and_test_commodity(commodity_name, data):
    """Complete training and testing pipeline"""
    print("\\n" + "="*70)
    print(f"PROCESSING {commodity_name.upper()} WITH IMPROVED MODEL")
    print("="*70)
    
    predictor = ImprovedCommodityPredictor(commodity_name)
    
    # Feature engineering
    print("\\n1. Advanced Feature Engineering...")
    data_with_features = predictor.create_advanced_features(data)
    num_features = len([c for c in data_with_features.columns if c not in ['Target_Price', 'Target_Direction', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']])
    print(f"   \u2713 Created {num_features} advanced features")
    
    # Prepare data
    print("\\n2. Preparing Data...")
    X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test, train_idx, test_idx = predictor.prepare_data(data_with_features, test_size=0.2)
    print(f"   \u2713 Training samples: {len(X_train)}")
    print(f"   \u2713 Testing samples: {len(X_test)}")
    
    # Train models
    print("\\n3. Training Dual Models (Price + Direction)...")
    predictor.train(X_train, y_price_train, y_dir_train)
    
    # Evaluate
    print("\\n4. Evaluating on Test Set...")
    metrics, price_preds, dir_preds = predictor.evaluate(X_test, y_price_test, y_dir_test)
    
    print("\\n   Test Set Metrics:")
    print(f"   MAE:                  ${metrics['MAE']:.4f}")
    print(f"   RMSE:                 ${metrics['RMSE']:.4f}")
    print(f"   MAPE:                 {metrics['MAPE']:.2f}%")
    print(f"   R²:                   {metrics['R2']:.4f}")
    print(f"   Direction Accuracy:   {metrics['Direction_Accuracy']:.2f}%")
    
    # Save models
    predictor.save_model()
    
    # Backtesting
    print("\\n5. Walk-Forward Backtesting...")
    backtest_results = backtest_improved_model(predictor, data, retrain_freq=30)
    
    print("\\n6. Calculating Backtest Metrics...")
    backtest_metrics = calculate_improved_metrics(backtest_results)
    
    print("\\n" + "="*70)
    print("BACKTESTING RESULTS")
    print("="*70)
    print(f"Directional Accuracy:     {backtest_metrics['Directional_Accuracy']:.2f}%")
    print(f"MAE:                      ${backtest_metrics['MAE']:.4f}")
    print(f"RMSE:                     ${backtest_metrics['RMSE']:.4f}")
    print(f"MAPE:                     {backtest_metrics['MAPE']:.2f}%")
    print(f"R² Score:                 {backtest_metrics['R2']:.4f}")
    print(f"Total Strategy Return:    {backtest_metrics['Total_Strategy_Return']:.2f}%")
    print(f"Win Rate:                 {backtest_metrics['Win_Rate']:.2f}%")
    print(f"Total Predictions:        {backtest_metrics['Predictions']}")
    print("="*70)
    
    # Visualization
    print("\\n7. Creating Visualizations...")
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price predictions
    axes[0, 0].plot(backtest_results['Date'], backtest_results['Actual_Price'], label='Actual', alpha=0.7)
    axes[0, 0].plot(backtest_results['Date'], backtest_results['Predicted_Price'], label='Predicted', alpha=0.7)
    axes[0, 0].set_title(f'{commodity_name.upper()} - Price Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Direction accuracy over time
    backtest_results['Direction_Acc_Rolling'] = backtest_results['Direction_Correct'].rolling(30).mean() * 100
    axes[0, 1].plot(backtest_results['Date'], backtest_results['Direction_Acc_Rolling'])
    axes[0, 1].axhline(y=50, color='r', linestyle='--', label='50% Baseline')
    axes[0, 1].set_title(f'Rolling 30-Day Direction Accuracy')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(backtest_results['Actual_Direction'], backtest_results['Predicted_Direction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Direction Prediction Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Cumulative returns
    backtest_results['Cumulative_Return'] = (1 + backtest_results['Strategy_Return'].fillna(0)).cumprod()
    axes[1, 1].plot(backtest_results['Date'], backtest_results['Cumulative_Return'])
    axes[1, 1].set_title('Cumulative Strategy Returns')
    axes[1, 1].set_ylabel('Cumulative Return')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{commodity_name}_improved_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    backtest_results.to_csv(f'results/{commodity_name}_improved_predictions.csv', index=False)
    pd.DataFrame([backtest_metrics]).to_csv(f'results/{commodity_name}_improved_metrics.csv', index=False)
    
    print(f"   \u2713 Results saved to results/ directory")
    
    return {
        'predictor': predictor,
        'metrics': backtest_metrics,
        'results': backtest_results
    }


def main():
    """Main execution"""
    print("="*70)
    print("IMPROVED COMMODITY PRICE PREDICTION")
    print("Enhanced Directional Accuracy with Dual-Model Approach")
    print("="*70)
    
    # Collect data
    print("\\n[STEP 1] DATA COLLECTION")
    print("-" * 70)
    
    collector = CommodityDataCollector()
    copper_data = collector.load_data('copper')
    aluminum_data = collector.load_data('aluminum')
    
    if copper_data is None or aluminum_data is None:
        print("Downloading data...")
        all_data = collector.collect_all_data()
        copper_data = all_data.get('copper')
        aluminum_data = all_data.get('aluminum')
    
    # Train and test
    print("\\n[STEP 2] TRAINING AND BACKTESTING")
    print("-" * 70)
    
    results = {}
    
    if copper_data is not None and not copper_data.empty:
        results['copper'] = train_and_test_commodity('copper', copper_data)
    
    if aluminum_data is not None and not aluminum_data.empty:
        results['aluminum'] = train_and_test_commodity('aluminum', aluminum_data)
    
    # Summary
    print("\\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for commodity, result in results.items():
        metrics = result['metrics']
        print(f"\\n{commodity.upper()}:")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        print(f"  MAPE:                 {metrics['MAPE']:.2f}%")
        print(f"  Total Return:         {metrics['Total_Strategy_Return']:.2f}%")
    
    print("\\n" + "="*70)
    print("\u2713 ALL PROCESSING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
