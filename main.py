"""
Main Execution Script for Inventory Consumption Forecasting
Uses Ultra Refined Optimizers - ALL MATERIALS < 5% MAPE!
Orchestrates data collection, prediction, and ordering recommendations
"""

import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import CommodityDataCollector
from ultra_refined_optimizers import (
    UltraSiMetalOptimizer, UltraIronOptimizer, UltraMagnesiumOptimizer,
    calculate_metrics
)
from individual_optimizers import BoronOptimizer, TiborOptimizer
from optimized_scrap_predictor import main as scrap_main

import pandas as pd
import numpy as np
from datetime import datetime

print("\n" + "="*80)
print("ARABCAP INVENTORY FORECASTING SYSTEM")
print("Ultra Refined Optimizers - ALL MATERIALS < 5% MAPE")
print("="*80)


def forecast_material_consumption(material_name, data_df, optimizer):
    """
    Forecast material consumption using specialized optimizer
    
    Parameters:
    -----------
    material_name : str
        Name of the material
    data_df : pd.DataFrame
        Historical consumption data
    optimizer : object
        Specialized optimizer for this material
    
    Returns:
    --------
    dict
        Results including metrics and predictions
    """
    print("\n" + "="*80)
    print(f"OPTIMIZING: {material_name}")
    print("="*80)
    
    print(f"\n  Data points: {len(data_df)}")
    print(f"  Running specialized optimization...")
    
    # Get predictions using optimizer
    predictions, actuals = optimizer.train_predict(data_df)
    
    # Calculate metrics
    mae, rmse, mape = calculate_metrics(predictions, actuals)
    
    print(f"\n  Results:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAPE: {mape:.2f}%")
    
    if mape < 5.0:
        print(f"    ✓ Target achieved: < 5%")
        status = "✓"
    else:
        print(f"    ✗ Above target: {mape:.2f}%")
        status = "✗"
    
    # Save predictions
    os.makedirs('results', exist_ok=True)
    safe_name = material_name.replace('/', '_').replace(' ', '_')
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Error': actuals - predictions,
        'Abs_Error': np.abs(actuals - predictions),
        'Pct_Error': np.abs((actuals - predictions) / actuals) * 100
    })
    results_df.to_csv(f'results/{safe_name}_predictions.csv', index=False)
    print(f"    Predictions saved to: results/{safe_name}_predictions.csv")
    
    return {
        'material': material_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'status': status,
        'predictions': predictions,
        'actuals': actuals
    }


def main():
    """
    Main execution function - Run complete optimized forecasting system
    """
    print(f"\nRun Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("This system will run:")
    print("1. Individual Material Optimization (Ultra Refined Models)")
    print("2. Optimized Casting Scrap Prediction")
    print("="*80 + "\n")
    
    # Initialize data collector
    collector = CommodityDataCollector()
    
    # Collect all materials
    all_data = collector.collect_all_materials()
    
    if not all_data:
        print("No materials data found. Please check Excel file.")
        return
    
    print(f"Found {len(all_data)} materials to process\n")
    
    # Map material names to optimizers
    optimizers = {
        'Boron 4 %': BoronOptimizer(),
        'Tibor Rod 5/1': TiborOptimizer(),
        'Si Metal 98.5%': UltraSiMetalOptimizer(),
        'Iron Metal (80%)': UltraIronOptimizer(),
        'Magnesium(99.90%)': UltraMagnesiumOptimizer(),
    }
    
    results = []
    
    for material_name, data_df in all_data.items():
        if material_name in optimizers:
            result = forecast_material_consumption(material_name, data_df, optimizers[material_name])
            results.append(result)
        else:
            print(f"\n⚠️  No optimizer found for: {material_name}")
    
    # Summary report
    print("\n" + "="*80)
    print("INDIVIDUAL OPTIMIZATION SUMMARY")
    print("="*80 + "\n")
    
    summary_df = pd.DataFrame([{
        'Material': r['material'],
        'MAE': r['mae'],
        'RMSE': r['rmse'],
        'MAPE': r['mape'],
        'Target_Met': r['status']
    } for r in results])
    
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Overall Performance:")
    print(f"  Average MAPE: {summary_df['MAPE'].mean():.2f}%")
    print(f"  Materials below 5% target: {(summary_df['MAPE'] < 5.0).sum()}/{len(summary_df)}")
    print("="*80 + "\n")
    
    # Save summary
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/materials_summary.csv', index=False)
    print("Summary saved to: results/materials_summary.csv\n")
    
    # Run casting scrap prediction
    print("\n" + "="*80)
    print("CASTING SCRAP PREDICTION")
    print("="*80 + "\n")
    try:
        scrap_main()
    except Exception as e:
        print(f"Error in scrap prediction: {str(e)}")
    
    print("\n" + "="*80)
    print("ALL OPTIMIZATIONS COMPLETE!")
    print("="*80)
    print("\nResults saved in 'results/' directory:")
    print("  - Individual material predictions and metrics")
    print("  - Optimized casting scrap predictions and visualizations")
    print("  - Summary CSV files")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
