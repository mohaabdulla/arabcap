"""
Advanced Main Script with Ensemble Models
Targets 55%+ directional accuracy using multiple algorithms
"""

import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import CommodityDataCollector
from advanced_predictor import AdvancedEnsemblePredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sns.set_style('whitegrid')


def backtest_advanced_model(predictor, data, initial_train_pct=0.7, retrain_freq=30):
    """Walk-forward backtesting with ensemble models"""
    print("\nRunning advanced ensemble backtesting...")
    
    data_with_features = predictor.create_ultra_features(data)
    
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
            
            # Feature selection on first iteration
            if predictor.selected_features is None:
                predictor.selected_features = predictor.select_important_features(
                    X_train_scaled, y_dir_train, top_n=min(100, X_train_scaled.shape[1])
                )
            
            # Apply feature selection
            X_train_scaled = X_train_scaled[:, predictor.selected_features]
            predictor.train(X_train_scaled, y_price_train, y_dir_train)
            days_since_retrain = 0
        
        # Predict
        X_test_scaled = predictor.scaler.transform(X_test)
        X_test_scaled = X_test_scaled[:, predictor.selected_features]
        
        price_pred = predictor.predict_price(X_test_scaled)[0]
        dir_pred, dir_proba = predictor.predict_direction(X_test_scaled)
        dir_pred = dir_pred[0]
        dir_proba = dir_proba[0]
        
        results.append({
            'Date': test_data.index[0],
            'Actual_Price': y_price_test,
            'Predicted_Price': price_pred,
            'Actual_Direction': y_dir_test,
            'Predicted_Direction': dir_pred,
            'Direction_Probability': dir_proba,
            'Direction_Correct': (dir_pred == y_dir_test)
        })
        
        days_since_retrain += 1
        
        if len(results) % 50 == 0:
            print(f"  Processed {len(results)} predictions...")
    
    results_df = pd.DataFrame(results)
    return results_df


def calculate_advanced_metrics(results_df):
    """Calculate comprehensive metrics with confidence intervals"""
    # Direction accuracy
    dir_accuracy = (results_df['Direction_Correct'].sum() / len(results_df)) * 100
    
    # Confidence-based metrics (only trade when probability > 0.55)
    high_conf = results_df[results_df['Direction_Probability'].apply(lambda x: abs(x - 0.5) > 0.05)]
    if len(high_conf) > 0:
        high_conf_accuracy = (high_conf['Direction_Correct'].sum() / len(high_conf)) * 100
    else:
        high_conf_accuracy = 0
    
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
    
    # High confidence trading
    results_df['High_Conf_Return'] = np.where(
        results_df['Direction_Probability'].apply(lambda x: abs(x - 0.5) > 0.05),
        results_df['Strategy_Return'],
        0
    )
    
    total_return = (1 + results_df['Strategy_Return'].fillna(0)).prod() - 1
    high_conf_return = (1 + results_df['High_Conf_Return'].fillna(0)).prod() - 1
    win_rate = (results_df['Strategy_Return'] > 0).sum() / len(results_df) * 100
    
    # Sharpe ratio
    if results_df['Strategy_Return'].std() > 0:
        sharpe = results_df['Strategy_Return'].mean() / results_df['Strategy_Return'].std() * np.sqrt(252)
    else:
        sharpe = 0
    
    metrics = {
        'Directional_Accuracy': dir_accuracy,
        'High_Confidence_Accuracy': high_conf_accuracy,
        'High_Confidence_Trades': len(high_conf),
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Total_Strategy_Return': total_return * 100,
        'High_Conf_Return': high_conf_return * 100,
        'Win_Rate': win_rate,
        'Sharpe_Ratio': sharpe,
        'Predictions': len(results_df)
    }
    
    return metrics


def train_and_test_advanced(commodity_name, data):
    """Complete training and testing with advanced ensemble"""
    print("\n" + "="*70)
    print(f"PROCESSING {commodity_name.upper()} WITH ADVANCED ENSEMBLE")
    print("="*70)
    
    predictor = AdvancedEnsemblePredictor(commodity_name)
    
    # Feature engineering
    print("\n1. Ultra Feature Engineering...")
    data_with_features = predictor.create_ultra_features(data)
    num_features = len([c for c in data_with_features.columns if c not in ['Target_Price', 'Target_Direction', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']])
    print(f"   ✓ Created {num_features} ultra features")
    
    # Prepare data
    print("\n2. Preparing Data with Feature Selection...")
    X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test, train_idx, test_idx = predictor.prepare_data(
        data_with_features, test_size=0.2, select_features=True
    )
    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Testing samples: {len(X_test)}")
    print(f"   ✓ Using {X_train.shape[1]} selected features")
    
    # Train ensemble
    print("\n3. Training Advanced Ensemble (XGBoost + LightGBM + CatBoost)...")
    predictor.train(X_train, y_price_train, y_dir_train)
    
    # Evaluate
    print("\n4. Evaluating Ensemble on Test Set...")
    metrics, price_preds, dir_preds = predictor.evaluate(X_test, y_price_test, y_dir_test)
    
    print("\n   Test Set Metrics:")
    print(f"   MAE:                  ${metrics['MAE']:.4f}")
    print(f"   RMSE:                 ${metrics['RMSE']:.4f}")
    print(f"   MAPE:                 {metrics['MAPE']:.2f}%")
    print(f"   R²:                   {metrics['R2']:.4f}")
    print(f"   Direction Accuracy:   {metrics['Direction_Accuracy']:.2f}%")
    
    # Save models
    predictor.save_model()
    
    # Backtesting
    print("\n5. Walk-Forward Backtesting with Ensemble...")
    backtest_results = backtest_advanced_model(predictor, data, retrain_freq=30)
    
    print("\n6. Calculating Advanced Metrics...")
    backtest_metrics = calculate_advanced_metrics(backtest_results)
    
    print("\n" + "="*70)
    print("ADVANCED ENSEMBLE BACKTESTING RESULTS")
    print("="*70)
    print(f"Overall Directional Accuracy:      {backtest_metrics['Directional_Accuracy']:.2f}%")
    print(f"High Confidence Accuracy:          {backtest_metrics['High_Confidence_Accuracy']:.2f}%")
    print(f"High Confidence Trades:            {backtest_metrics['High_Confidence_Trades']}")
    print(f"MAE:                               ${backtest_metrics['MAE']:.4f}")
    print(f"RMSE:                              ${backtest_metrics['RMSE']:.4f}")
    print(f"MAPE:                              {backtest_metrics['MAPE']:.2f}%")
    print(f"R² Score:                          {backtest_metrics['R2']:.4f}")
    print(f"Total Strategy Return:             {backtest_metrics['Total_Strategy_Return']:.2f}%")
    print(f"High Confidence Return:            {backtest_metrics['High_Conf_Return']:.2f}%")
    print(f"Win Rate:                          {backtest_metrics['Win_Rate']:.2f}%")
    print(f"Sharpe Ratio:                      {backtest_metrics['Sharpe_Ratio']:.4f}")
    print(f"Total Predictions:                 {backtest_metrics['Predictions']}")
    print("="*70)
    
    # Visualization
    print("\n7. Creating Advanced Visualizations...")
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Price predictions
    axes[0, 0].plot(backtest_results['Date'], backtest_results['Actual_Price'], label='Actual', alpha=0.7, linewidth=2)
    axes[0, 0].plot(backtest_results['Date'], backtest_results['Predicted_Price'], label='Predicted', alpha=0.7, linewidth=2)
    axes[0, 0].set_title(f'{commodity_name.upper()} - Ensemble Price Predictions', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Direction accuracy over time
    backtest_results['Direction_Acc_Rolling'] = backtest_results['Direction_Correct'].rolling(30).mean() * 100
    axes[0, 1].plot(backtest_results['Date'], backtest_results['Direction_Acc_Rolling'], linewidth=2)
    axes[0, 1].axhline(y=50, color='r', linestyle='--', label='50% Baseline', linewidth=2)
    axes[0, 1].axhline(y=backtest_metrics['Directional_Accuracy'], color='g', linestyle=':', 
                       label=f"Overall: {backtest_metrics['Directional_Accuracy']:.1f}%", linewidth=2)
    axes[0, 1].set_title(f'Rolling 30-Day Direction Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([35, 70])
    
    # Confusion matrix
    cm = confusion_matrix(backtest_results['Actual_Direction'], backtest_results['Predicted_Direction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[1, 0], 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    axes[1, 0].set_title('Direction Prediction Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Direction')
    axes[1, 0].set_ylabel('Actual Direction')
    
    # Cumulative returns
    backtest_results['Cumulative_Return'] = (1 + backtest_results['Strategy_Return'].fillna(0)).cumprod()
    backtest_results['Cumulative_HC_Return'] = (1 + backtest_results['High_Conf_Return'].fillna(0)).cumprod()
    axes[1, 1].plot(backtest_results['Date'], backtest_results['Cumulative_Return'], 
                   label='All Trades', linewidth=2)
    axes[1, 1].plot(backtest_results['Date'], backtest_results['Cumulative_HC_Return'], 
                   label='High Confidence Only', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Cumulative Strategy Returns', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Return')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{commodity_name}_advanced_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    backtest_results.to_csv(f'results/{commodity_name}_advanced_predictions.csv', index=False)
    pd.DataFrame([backtest_metrics]).to_csv(f'results/{commodity_name}_advanced_metrics.csv', index=False)
    
    print(f"   ✓ Results saved to results/ directory")
    
    return {
        'predictor': predictor,
        'metrics': backtest_metrics,
        'results': backtest_results
    }


def main():
    """Main execution"""
    print("="*70)
    print("ADVANCED COMMODITY PRICE PREDICTION")
    print("Ensemble Learning: XGBoost + LightGBM + CatBoost")
    print("Target: 55%+ Directional Accuracy")
    print("="*70)
    
    # Collect data
    print("\n[STEP 1] DATA COLLECTION")
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
    print("\n[STEP 2] ADVANCED ENSEMBLE TRAINING")
    print("-" * 70)
    
    results = {}
    
    if copper_data is not None and not copper_data.empty:
        results['copper'] = train_and_test_advanced('copper', copper_data)
    
    if aluminum_data is not None and not aluminum_data.empty:
        results['aluminum'] = train_and_test_advanced('aluminum', aluminum_data)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL ADVANCED ENSEMBLE RESULTS")
    print("="*70)
    
    for commodity, result in results.items():
        metrics = result['metrics']
        print(f"\n{commodity.upper()}:")
        print(f"  Directional Accuracy:     {metrics['Directional_Accuracy']:.2f}%")
        print(f"  High Conf Accuracy:       {metrics['High_Confidence_Accuracy']:.2f}%")
        print(f"  MAPE:                     {metrics['MAPE']:.2f}%")
        print(f"  Sharpe Ratio:             {metrics['Sharpe_Ratio']:.4f}")
        print(f"  Total Return:             {metrics['Total_Strategy_Return']:.2f}%")
    
    print("\n" + "="*70)
    print("✓ ADVANCED ENSEMBLE PROCESSING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
