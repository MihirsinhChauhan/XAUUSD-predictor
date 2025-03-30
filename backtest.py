import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import glob
import re

# Import your prediction module
from model_predictor import ModelPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest_with_prediction')

class NumpyEncoder(json.JSONEncoder):
    """Handles numpy types for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

def backtest_with_predictor(
    data_path: str,
    model_dir: str = "models",
    model_name: str = "xauusd_model",
    processor_path: Optional[str] = None,
    model_version: Optional[str] = None,
    window_size: int = 500,
    step_size: int = 100,
    output_dir: Optional[str] = None,
    show_progress: bool = True
) -> Dict:
    """
    Run backtesting using the ModelPredictor for predictions.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        model_dir: Directory with model files
        model_name: Base name of the model
        processor_path: Path to processor state file
        model_version: Specific model version to use
        window_size: Number of candles to use in each training window
        step_size: Number of candles to step forward for each test
        output_dir: Directory to save results
        show_progress: Whether to show progress information
        
    Returns:
        Dictionary with backtesting results
    """
    # Set output directory
    if output_dir is None:
        output_dir = model_dir
    
    # Load data
    if show_progress:
        print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if len(df) < window_size + 100:
        raise ValueError(f"Not enough data for backtesting. Need at least {window_size + 100} rows.")
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows")
    
    # Initialize predictor
    if show_progress:
        print(f"Initializing predictor with model: {model_name}")
    
    predictor = ModelPredictor(
        model_dir=model_dir,
        model_name=model_name,
        processor_path=processor_path,
        model_version=model_version
    )
    
    # Check if predictor initialized successfully
    model_info = predictor.get_model_info()
    if "error" in model_info:
        raise ValueError(f"Failed to initialize predictor: {model_info['error']}")
    
    if show_progress:
        print(f"Using model version: {model_info.get('model_version', 'unknown')}")
        print(f"Using processor: {model_info.get('processor_path', 'unknown')}")
    
    # Initialize results structures
    window_results = []
    all_predictions = []
    
    # Calculate number of windows
    num_windows = (len(df) - window_size - 50) // step_size + 1
    
    if show_progress:
        print(f"Running backtest with {num_windows} windows...")
    
    # Loop through each window
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        test_end_idx = min(end_idx + 50, len(df))  # Test on next 50 candles or to end
        
        # Split data into train and test
        train_df = df.iloc[start_idx:end_idx].copy()
        test_df = df.iloc[end_idx:test_end_idx].copy()
        
        if show_progress:
            print(f"Window {i+1}/{num_windows}: rows {start_idx}-{end_idx}, testing on {len(test_df)} candles")
        
        # Test predictions on each candle one by one
        test_predictions = []
        
        # Use entire training data for initial state
        current_df = train_df.copy()
        
        # For each test point, predict the next candle
        for j in range(len(test_df) - 1):  # -1 because we need the next candle for validation
            try:
                # Add current test candle to the data
                current_df = pd.concat([current_df, test_df.iloc[j:j+1]])
                
                # Make prediction using the predictor
                pred_result = predictor.predict(current_df)
                
                # Skip this prediction if there was an error
                if "error" in pred_result:
                    logger.warning(f"Error in prediction at window {i}, test point {j}: {pred_result['error']}")
                    continue
                
                # Get actual next candle direction
                next_candle = test_df.iloc[j+1]
                actual = 1 if next_candle['close'] > next_candle['open'] else 0
                
                # Store result
                prediction = {
                    'window': i,
                    'test_index': end_idx + j + 1,
                    'prediction': pred_result['prediction'],
                    'probability': pred_result['bullish_probability'] if pred_result['prediction'] == 1 else pred_result['bearish_probability'],
                    'actual': actual,
                    'correct': pred_result['prediction'] == actual
                }
                
                test_predictions.append(prediction)
                all_predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error processing test point {j} in window {i}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Calculate window accuracy
        window_accuracy = sum(p['correct'] for p in test_predictions) / len(test_predictions) if test_predictions else 0
        
        # Store window results
        window_result = {
            'window': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'test_size': len(test_predictions),
            'accuracy': window_accuracy,
            'predictions': test_predictions
        }
        
        window_results.append(window_result)
        
        if show_progress:
            print(f"  Window accuracy: {window_accuracy:.4f}")
    
    # Calculate overall metrics
    overall_accuracy = sum(p['correct'] for p in all_predictions) / len(all_predictions) if all_predictions else 0
    
    # Calculate trading performance if price data available
    trading_performance = None
    if 'close' in df.columns and all_predictions:
        trading_performance = calculate_trading_performance(df, all_predictions)
        
        # Create a visualization
        visualize_backtest_results(df, all_predictions, trading_performance, 
                                  os.path.join(output_dir, f"{model_name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    # Prepare final results
    results = {
        'overall_accuracy': overall_accuracy,
        'windows_count': len(window_results),
        'predictions_count': len(all_predictions),
        'window_results': window_results, 
        'trading_performance': trading_performance,
        'model_info': model_info
    }
    
    # Save results to file
    results_path = os.path.join(output_dir, f"{model_name}_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    
    if show_progress:
        print("\n===== Backtesting Complete =====")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Windows tested: {len(window_results)}")
        print(f"Total predictions: {len(all_predictions)}")
        
        if trading_performance:
            print(f"Trading Return: {trading_performance['total_return']*100:.2f}%")
            print(f"Win Rate: {trading_performance['win_rate']*100:.2f}%")
            if 'profit_factor' in trading_performance:
                print(f"Profit Factor: {trading_performance['profit_factor']:.2f}")
        
        print(f"Results saved to: {results_path}")
        print("================================")
    
    return results


def calculate_trading_performance(df: pd.DataFrame, predictions: List[Dict]) -> Dict:
    """
    Calculate trading performance metrics based on predictions.
    
    Args:
        df: DataFrame with price data
        predictions: List of prediction dictionaries
        
    Returns:
        Dictionary with trading performance metrics
    """
    # Sort predictions by test_index
    predictions = sorted(predictions, key=lambda x: x['test_index'])
    
    # Create a DataFrame for performance calculations
    indices = [p['test_index'] for p in predictions]
    pred_signals = [p['prediction'] * 2 - 1 for p in predictions]  # Convert 0/1 to -1/1 for short/long
    
    # Ensure indices are valid
    valid_indices = [idx for idx in indices if 0 <= idx < len(df)]
    if not valid_indices:
        return {
            'total_return': 0,
            'win_rate': 0,
            'trades_count': 0,
            'profit_factor': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'cumulative_returns': []
        }
    
    # Create performance DataFrame
    perf_df = pd.DataFrame({
        'index': valid_indices,
        'signal': pred_signals[:len(valid_indices)]
    })
    
    # Join with price data
    perf_df = perf_df.merge(
        df[['close']].reset_index(),
        left_on='index',
        right_index=True,
        how='left'
    )
    
    # Calculate returns
    perf_df['price_change'] = perf_df['close'].pct_change()
    perf_df['strategy_return'] = perf_df['signal'].shift(1) * perf_df['price_change']
    
    # Remove NaN values
    perf_df = perf_df.dropna()
    
    # Exit if we don't have enough data
    if len(perf_df) < 2:
        return {
            'total_return': 0,
            'win_rate': 0,
            'trades_count': 0,
            'profit_factor': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'cumulative_returns': []
        }
    
    # Calculate cumulative returns
    perf_df['cumulative_return'] = (1 + perf_df['strategy_return']).cumprod() - 1
    
    # Calculate metrics
    total_return = perf_df['cumulative_return'].iloc[-1]
    win_rate = (perf_df['strategy_return'] > 0).mean()
    
    # Calculate profit factor safely
    profit_sum = perf_df.loc[perf_df['strategy_return'] > 0, 'strategy_return'].sum()
    loss_sum = abs(perf_df.loc[perf_df['strategy_return'] < 0, 'strategy_return'].sum())
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
    
    return {
        'total_return': float(total_return),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'trades_count': len(perf_df),
        'profitable_trades': int((perf_df['strategy_return'] > 0).sum()),
        'losing_trades': int((perf_df['strategy_return'] < 0).sum()),
        'cumulative_returns': perf_df['cumulative_return'].tolist()
    }


def visualize_backtest_results(df: pd.DataFrame, predictions: List[Dict], 
                             trading_perf: Dict, output_path: str):
    """
    Create a visualization of backtest results.
    
    Args:
        df: DataFrame with price data
        predictions: List of prediction dictionaries
        trading_perf: Trading performance metrics
        output_path: Path to save the visualization
    """
    # Create figure with multiple plots
    plt.figure(figsize=(12, 10))
    
    # 1. Plot accuracy over time
    plt.subplot(2, 1, 1)
    
    # Extract data for plotting
    indices = [p['test_index'] for p in predictions]
    correct = [1 if p['correct'] else 0 for p in predictions]
    
    # Calculate rolling accuracy (window of 20 predictions)
    window_size = min(20, len(correct))
    rolling_accuracy = pd.Series(correct).rolling(window_size).mean().values
    
    # Plot
    plt.plot(range(len(indices)), rolling_accuracy, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Prediction Accuracy (Rolling Window)', fontsize=14)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # 2. Plot cumulative returns
    plt.subplot(2, 1, 2)
    
    if trading_perf and 'cumulative_returns' in trading_perf and trading_perf['cumulative_returns']:
        cum_returns = trading_perf['cumulative_returns']
        plt.plot(range(len(cum_returns)), cum_returns, 'g-', linewidth=2)
        plt.title('Cumulative Strategy Returns', fontsize=14)
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Add summary text
    overall_accuracy = sum(correct) / len(correct) if correct else 0
    
    summary_text = f"Overall Accuracy: {overall_accuracy:.4f}\n"
    summary_text += f"Total Predictions: {len(predictions)}\n"
    
    if trading_perf:
        summary_text += f"Total Return: {trading_perf.get('total_return', 0)*100:.2f}%\n"
        summary_text += f"Win Rate: {trading_perf.get('win_rate', 0)*100:.2f}%\n"
        summary_text += f"Profit Factor: {trading_perf.get('profit_factor', 0):.2f}\n"
        summary_text += f"Trades: {trading_perf.get('trades_count', 0)}"
    
    plt.figtext(0.5, 0.01, summary_text, ha="center", fontsize=12, 
              bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest With ModelPredictor')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory with model files')
    parser.add_argument('--model-name', type=str, default='xauusd_model', help='Base name of the model')
    parser.add_argument('--processor-path', type=str, help='Path to processor state file')
    parser.add_argument('--model-version', type=str, help='Specific model version to use')
    parser.add_argument('--window-size', type=int, default=500, help='Number of candles per window')
    parser.add_argument('--step-size', type=int, default=100, help='Number of candles to step forward')
    parser.add_argument('--output-dir', type=str, help='Directory to save results')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress output')
    
    args = parser.parse_args()
    
    # Run backtesting with the ModelPredictor
    backtest_with_predictor(
        data_path=args.data,
        model_dir=args.model_dir,
        model_name=args.model_name,
        processor_path=args.processor_path,
        model_version=args.model_version,
        window_size=args.window_size,
        step_size=args.step_size,
        output_dir=args.output_dir,
        show_progress=not args.no_progress
    )