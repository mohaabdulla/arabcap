"""
Main Execution Script for Production Volume Forecasting
Orchestrates data collection, model training, backtesting, and visualization
Forecasts monthly aluminum production volumes based on historical data
"""

import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import CommodityDataCollector
from price_predictor import CommodityPricePredictor
from backtesting import BacktestEngine, print_metrics_report

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


def train_and_backtest_commodity(commodity_name, data, retrain_freq=30):
    """
    Complete pipeline for training and backtesting production forecasting
    
    Parameters:
    -----------
    commodity_name : str
        Name of the commodity (aluminum)
    data : pd.DataFrame
        Historical production volume data
    retrain_freq : int
        How often to retrain during backtesting (in days)
    data : pd.DataFrame
        Historical price data
    retrain_freq : int
        How often to retrain during backtesting (in days)
    
    Returns:
    --------
    dict
        Results including metrics and predictions
    """
    print("\n" + "="*70)
    print(f"PROCESSING {commodity_name.upper()}")
    print("="*70)
    
    # Initialize predictor
    predictor = CommodityPricePredictor(commodity_name)
    
    # Create features
    print("\n1. Feature Engineering...")
    data_with_features = predictor.create_features(data)
    print(f"   ✓ Created {len([c for c in data_with_features.columns if c not in ['Target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Commodity']])} features")
    
    # Split data for initial training and testing
    print("\n2. Preparing Data...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = predictor.prepare_data(
        data_with_features, test_size=0.2
    )
    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Testing samples: {len(X_test)}")
    
    # Train model
    print("\n3. Training XGBoost Model...")
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate on test set
    print("\n4. Evaluating on Test Set...")
    metrics, predictions = predictor.evaluate(X_test, y_test)
    
    print("\n   Initial Test Set Metrics:")
    print(f"   MAE:  ${metrics['MAE']:.4f}")
    print(f"   RMSE: ${metrics['RMSE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")
    
    # Feature importance
    print("\n5. Top 10 Most Important Features:")
    importance = predictor.get_feature_importance()
    for idx, row in importance.head(10).iterrows():
        print(f"   {row['Feature']:30s} {row['Importance']:.4f}")
    
    # Save model
    predictor.save_model()
    
    # Backtesting
    print("\n6. Running Backtesting (Walk-Forward Validation)...")
    backtest = BacktestEngine(predictor, data)
    backtest_results = backtest.walk_forward_validation(
        initial_train_size=0.7,
        step_size=1,
        retrain_frequency=retrain_freq
    )
    
    # Calculate backtest metrics
    print("\n7. Calculating Backtest Metrics...")
    backtest_metrics = backtest.calculate_metrics()
    print_metrics_report(backtest_metrics)
    
    # Error statistics
    print("\n8. Error Statistics:")
    error_stats = backtest.get_error_statistics()
    for key, value in error_stats.items():
        if 'Error' in key:
            print(f"   {key:30s} ${value:.4f}")
        else:
            print(f"   {key:30s} {value:.4f}")
    
    # Create visualizations
    print("\n9. Creating Visualizations...")
    os.makedirs('results', exist_ok=True)
    
    fig = backtest.plot_results(
        save_path=f'results/{commodity_name}_backtest_results.png'
    )
    
    # Additional visualization - Feature Importance
    plt.figure(figsize=(12, 8))
    importance_top = importance.head(15)
    plt.barh(importance_top['Feature'], importance_top['Importance'])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'{commodity_name.upper()} - Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(f'results/{commodity_name}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export results
    backtest.export_results(f'results/{commodity_name}_predictions.csv')
    
    # Save metrics to file
    metrics_df = pd.DataFrame([backtest_metrics])
    metrics_df.to_csv(f'results/{commodity_name}_metrics.csv', index=False)
    
    print(f"\n✓ All results saved to 'results/' directory")
    
    return {
        'predictor': predictor,
        'backtest_metrics': backtest_metrics,
        'predictions': backtest_results,
        'feature_importance': importance,
        'test_metrics': metrics
    }


def compare_commodities(results_dict):
    """
    Create comparison visualization for multiple commodities
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with commodity names as keys and results as values
    """
    print("\n" + "="*70)
    print("COMMODITY COMPARISON")
    print("="*70)
    
    # Create comparison table
    comparison_data = []
    for commodity, results in results_dict.items():
        metrics = results['backtest_metrics']
        comparison_data.append({
            'Commodity': commodity.upper(),
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE (%)': metrics['MAPE'],
            'R² Score': metrics['R2 Score'],
            'Directional Accuracy (%)': metrics['Directional Accuracy (%)'],
            'Total Return (%)': metrics['Total Return (%)'],
            'Sharpe Ratio': metrics['Sharpe Ratio']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('results/commodity_comparison.csv', index=False)
    
    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAPE Comparison
    axes[0, 0].bar(comparison_df['Commodity'], comparison_df['MAPE (%)'], color='steelblue')
    axes[0, 0].set_title('MAPE Comparison (Lower is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('MAPE (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² Score Comparison
    axes[0, 1].bar(comparison_df['Commodity'], comparison_df['R² Score'], color='forestgreen')
    axes[0, 1].set_title('R² Score Comparison (Higher is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Directional Accuracy
    axes[1, 0].bar(comparison_df['Commodity'], comparison_df['Directional Accuracy (%)'], color='coral')
    axes[1, 0].set_title('Directional Accuracy (Higher is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total Return
    axes[1, 1].bar(comparison_df['Commodity'], comparison_df['Total Return (%)'], color='purple')
    axes[1, 1].set_title('Total Return (Higher is Better)', fontweight='bold')
    axes[1, 1].set_ylabel('Return (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/commodity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison chart saved to results/commodity_comparison.png")


def main():
    """Main execution function"""
    print("="*70)
    print("ALUMINUM PRODUCTION VOLUME FORECASTING WITH XGBOOST")
    print("Monthly Production Forecasting System")
    print("="*70)
    
    # Step 1: Data Collection
    print("\n[STEP 1] DATA COLLECTION FROM EXCEL")
    print("-" * 70)
    
    collector = CommodityDataCollector()
    
    # Check if data already exists
    copper_data = collector.load_data('copper')
    aluminum_data = collector.load_data('aluminum')
    
    if copper_data is None or aluminum_data is None:
        print("Loading data from Excel file...")
        all_data = collector.collect_all_data()
        copper_data = all_data.get('copper')
        aluminum_data = all_data.get('aluminum')
    else:
        print("Using existing data files...")
    
    # Step 2: Train and Backtest Models
    print("\n[STEP 2] MODEL TRAINING AND BACKTESTING")
    print("-" * 70)
    
    results = {}
    
    # Process Copper
    if copper_data is not None and not copper_data.empty:
        results['copper'] = train_and_backtest_commodity('copper', copper_data, retrain_freq=30)
    
    # Process Aluminum
    if aluminum_data is not None and not aluminum_data.empty:
        results['aluminum'] = train_and_backtest_commodity('aluminum', aluminum_data, retrain_freq=30)
    
    # Step 3: Compare Results
    if len(results) > 1:
        print("\n[STEP 3] COMPARING COMMODITIES")
        print("-" * 70)
        compare_commodities(results)
    
    # Final Summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated Files:")
    print("  📁 data/              - Historical price data")
    print("  📁 models/            - Trained XGBoost models")
    print("  📁 results/           - Predictions, metrics, and visualizations")
    print("\nKey Outputs:")
    print("  📊 Backtest plots showing actual vs predicted prices")
    print("  📈 Feature importance charts")
    print("  📉 Error distribution analysis")
    print("  📋 Detailed metrics CSV files")
    print("  🔄 Comparison charts across commodities")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
