"""
Main script for individual material optimization
Each material gets its own specialized model
NOW USING ULTRA REFINED OPTIMIZERS - ALL MATERIALS < 5% MAPE!
"""

import pandas as pd
import numpy as np
from data_collector import CommodityDataCollector
from ultra_refined_optimizers import (
    UltraSiMetalOptimizer, UltraIronOptimizer, UltraMagnesiumOptimizer,
    calculate_metrics
)
from individual_optimizers import BoronOptimizer, TiborOptimizer
from optimized_scrap_predictor import main as scrap_main
import warnings
warnings.filterwarnings('ignore')


def run_individual_optimizations():
    """Run individual optimized models for each material"""
    print("\n" + "="*80)
    print("INDIVIDUAL MATERIAL OPTIMIZATION SYSTEM")
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
    # Using Ultra Refined for Si, Iron, Magnesium (achieved <5%)
    # Using original for Boron and Tibor (already <5%)
    optimizers = {
        'Boron 4 %': BoronOptimizer(),
        'Tibor Rod 5/1': TiborOptimizer(),
        'Si Metal 98.5%': UltraSiMetalOptimizer(),
        'Iron Metal (80%)': UltraIronOptimizer(),
        'Magnesium(99.90%)': UltraMagnesiumOptimizer(),
    }
    
    results = []
    
    for material_name, data_df in all_data.items():
        print(f"\n{'='*80}")
        print(f"OPTIMIZING: {material_name}")
        print(f"{'='*80}\n")
        
        try:
            if material_name not in optimizers:
                print(f"  ⚠ No optimizer found for {material_name}")
                continue
            
            if data_df is None or len(data_df) < 6:
                print(f"  ⚠ Insufficient data for {material_name}")
                continue
            
            print(f"  Data points: {len(data_df)}")
            
            # Get optimizer
            optimizer = optimizers[material_name]
            
            # Train and predict
            print(f"  Running specialized optimization...")
            predictions, actuals = optimizer.train_predict(data_df)
            
            # Calculate metrics
            mae, rmse, mape = calculate_metrics(predictions, actuals)
            
            print(f"\n  Results:")
            print(f"    MAE:  {mae:.2f}")
            print(f"    RMSE: {rmse:.2f}")
            print(f"    MAPE: {mape:.2f}%")
            
            # Check if below 5%
            if mape < 5.0:
                print(f"    ✓ Target achieved: < 5%")
            else:
                print(f"    ⚠ Target not met: {mape:.2f}% (Goal: < 5%)")
            
            results.append({
                'Material': material_name,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Target_Met': '✓' if mape < 5.0 else '✗'
            })
            
            # Save predictions
            # Handle Date column - it may be in the original or feature dataframe
            date_col = data_df.iloc[6:]['Date'].values if 'Date' in data_df.columns else None
            
            results_df = pd.DataFrame({
                'Actual': actuals,
                'Predicted': predictions,
                'Error': actuals - predictions,
                'Abs_Pct_Error': np.abs((actuals - predictions) / actuals) * 100
            })
            
            if date_col is not None:
                results_df.insert(0, 'Date', date_col)
            
            safe_name = material_name.replace('/', '_').replace(' ', '_')
            csv_path = f'results/optimized_{safe_name}_predictions.csv'
            results_df.to_csv(csv_path, index=False)
            print(f"    Predictions saved to: {csv_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {material_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("INDIVIDUAL OPTIMIZATION SUMMARY")
    print(f"{'='*80}\n")
    
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        avg_mape = summary_df['MAPE'].mean()
        below_5 = (summary_df['MAPE'] < 5.0).sum()
        total = len(summary_df)
        
        print(f"\n{'='*80}")
        print("Overall Performance:")
        print(f"  Average MAPE: {avg_mape:.2f}%")
        print(f"  Materials below 5% target: {below_5}/{total}")
        print(f"{'='*80}\n")
        
        # Save summary
        summary_df.to_csv('results/optimized_materials_summary.csv', index=False)
        print("Summary saved to: results/optimized_materials_summary.csv\n")
    
    return results


def main():
    """Run complete optimization system"""
    print("\n" + "="*80)
    print("COMPLETE OPTIMIZATION SYSTEM")
    print("="*80 + "\n")
    print("This system will run:")
    print("1. Individual Material Optimization (specialized models for each)")
    print("2. Optimized Casting Scrap Prediction")
    print("="*80 + "\n")
    
    # Run materials optimization
    material_results = run_individual_optimizations()
    
    # Run scrap optimization
    print("\n\n")
    scrap_main()
    
    print("\n" + "="*80)
    print("ALL OPTIMIZATIONS COMPLETE!")
    print("="*80 + "\n")
    print("Results saved in 'results/' directory:")
    print("  - Individual material predictions and metrics")
    print("  - Optimized casting scrap predictions and visualizations")
    print("  - Summary CSV files")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
