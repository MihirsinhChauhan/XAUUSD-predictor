import os
import time
import logging
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

from data_processor import DataProcessor
from ml_model import CandlePredictionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_trainer')

def train_xauusd_model(
    data_path: str,
    output_dir: str = "models",
    model_name: str = "xauusd_model",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    optimize_hyperparams: bool = True,
    handle_imbalance: bool = True,
    balance_method: str = 'smote',
    processor_scaling: bool = True,
    processor_feature_selection: bool = True,
    n_features: int = 30,
    handle_missing: str = 'fill_median',
    cv_folds: int = 5,
    random_state: int = 42,
    save_processor: bool = True,
    show_progress: bool = True
) -> Dict:
    """
    Complete pipeline for training an XAUUSD prediction model.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        output_dir: Directory to store model files and visualizations
        model_name: Base name for the model files
        model_type: Type of model ('random_forest' or 'gradient_boosting')
        test_size: Proportion of data to use for testing
        optimize_hyperparams: Whether to perform hyperparameter tuning
        handle_imbalance: Whether to address class imbalance
        balance_method: Method for handling class imbalance ('smote', 'undersample', 'oversample')
        processor_scaling: Whether to scale features in the DataProcessor
        processor_feature_selection: Whether to perform feature selection in the DataProcessor
        n_features: Number of top features to select if feature_selection is True
        handle_missing: Strategy for handling missing values
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        save_processor: Whether to save the DataProcessor state
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with training results and paths to saved artifacts
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if show_progress:
        print(f"Starting XAUUSD model training pipeline...")
        print(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print("Initializing data processor...")
    
    # Initialize data processor
    processor = DataProcessor(
        scale_features=processor_scaling,
        scaler_type='robust',
        feature_selection=processor_feature_selection,
        n_features=n_features,
        handle_missing=handle_missing,
        smooth_outliers=True
    )
    
    if show_progress:
        print("Preparing features and target...")
        prepare_start = time.time()
    
    # Prepare features and target
    X, y, feature_list = processor.prepare_ml_data(df)
    
    if show_progress:
        prepare_time = time.time() - prepare_start
        print(f"Features prepared in {prepare_time:.2f} seconds: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Save processor state
    if save_processor:
        processor_path = os.path.join(output_dir, f"{model_name}_processor.pkl")
        processor.save_processor_state(processor_path)
        if show_progress:
            print(f"Processor state saved to {processor_path}")
    
    # Initialize model
    if show_progress:
        print(f"Initializing {model_type} model...")
    
    model = CandlePredictionModel(
        model_dir=output_dir,
        model_name=model_name,
        model_type=model_type,
        scale_features=False,  # No scaling in model if already done in processor
        handle_imbalance=handle_imbalance
    )
    
    # Train model
    if show_progress:
        print("Training model...")
    
    results = model.train(
        X, y,
        feature_list=feature_list,
        test_size=test_size,
        cv_folds=cv_folds,
        optimize_hyperparams=optimize_hyperparams,
        balance_method=balance_method,
        show_progress=show_progress
    )
    
    # Prepare latest data for prediction example
    if show_progress:
        print("Testing prediction on latest data...")
    
    latest_features = processor.prepare_latest_for_prediction(df, feature_list)
    prediction = model.predict_single(latest_features)
    
    # Create summary
    training_time = time.time() - start_time
    
    summary = {
        "model_version": results["model_version"],
        "model_type": model_type,
        "features_count": len(feature_list),
        "accuracy": results["accuracy"],
        "f1_score": results["f1_score"],
        "precision": results["precision"],
        "recall": results["recall"],
        "roc_auc": results["roc_auc"],
        "cross_val_mean": results["cv_scores"]["mean"],
        "cross_val_std": results["cv_scores"]["std"],
        "example_prediction": prediction,
        "training_time": training_time,
        "model_path": model._get_model_path(),
        "processor_path": processor_path if save_processor else None,
        "metrics_path": model._get_metrics_path(),
        "feature_importance_path": model._get_feature_importance_path()
    }
    
    if show_progress:
        print("\n===== Training Complete =====")
        print(f"Model version: {summary['model_version']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"F1 Score: {summary['f1_score']:.4f}")
        print(f"ROC AUC: {summary['roc_auc']:.4f}")
        print(f"Cross-val F1: {summary['cross_val_mean']:.4f} ± {summary['cross_val_std']:.4f}")
        print(f"Example prediction: {prediction['prediction_label']} ({prediction['signal_strength']})")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model saved to: {summary['model_path']}")
        print(f"Processor saved to: {summary['processor_path']}")
        print(f"Visualizations saved to: {summary['metrics_path']}")
        print("=============================")
    
    return summary


def train_multiple_models(
    data_path: str,
    output_dir: str = "models",
    base_name: str = "xauusd_model",
    test_size: float = 0.2,
    cv_folds: int = 5,
    models_to_try: List[str] = None,
    balance_methods: List[str] = None,
    feature_configs: List[Dict] = None,
    metric: str = "f1_score",
    show_progress: bool = True
) -> Dict:
    """
    Train multiple models with different configurations and select the best one.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        output_dir: Directory to store model files
        base_name: Base name for model files
        test_size: Proportion of data to use for testing
        cv_folds: Number of cross-validation folds
        models_to_try: List of model types to try (e.g., ["random_forest", "gradient_boosting"])
        balance_methods: List of class balancing methods to try (e.g., ["none", "smote", "undersample"])
        feature_configs: List of feature processor configurations to try
        metric: Metric to use for model selection ("accuracy", "f1_score", "roc_auc", etc.)
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with results from all models and the best model
    """
    import json
    
    # Set defaults if not provided
    if models_to_try is None:
        # models_to_try = [
        #     "random_forest",
        #     "gradient_boosting",
        #     "extra_trees",
        #     "adaboost",
        #     "logistic_regression",
        #     "decision_tree",
        #     "knn",
        #     "neural_network",
        #     "qda"
        # ]
        models_to_try = ["adaboost", "gradient_boosting","qda"]
    
    if balance_methods is None:
        balance_methods = ["none", "smote","undersample"]
    
    if feature_configs is None:
        feature_configs = [
            {"scale_features": True, "feature_selection": True, "n_features": 30},
            {"scale_features": True, "feature_selection": True, "n_features": 50},
            {"scale_features": True, "feature_selection": False}
        ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if show_progress:
        print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Store results for all model configurations
    all_results = []
    
    # Create a unique timestamp for this model selection run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize counters
    total_models = len(models_to_try) * len(balance_methods) * len(feature_configs)
    completed_models = 0
    
    if show_progress:
        print(f"Training {total_models} different model configurations...")
        progress_bar = tqdm(total=total_models, desc="Overall Progress")
    
    # Iterate through all combinations
    for feature_idx, feature_config in enumerate(feature_configs):
        # Initialize data processor with current feature config
        processor = DataProcessor(
            scale_features=feature_config.get("scale_features", True),
            feature_selection=feature_config.get("feature_selection", True),
            n_features=feature_config.get("n_features", 30),
            handle_missing=feature_config.get("handle_missing", "fill_median"),
            smooth_outliers=feature_config.get("smooth_outliers", True)
        )
        
        # Prepare features (do this once per feature config)
        if show_progress:
            print(f"\nFeature config {feature_idx+1}/{len(feature_configs)}: {feature_config}")
            print("Preparing features...")
        
        X, y, feature_list = processor.prepare_ml_data(df)
        
        if show_progress:
            print(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Save processor for this configuration
        processor_path = os.path.join(output_dir, f"{base_name}_config{feature_idx+1}_{timestamp}_processor.pkl")
        processor.save_processor_state(processor_path)
        
        for model_type in models_to_try:
            for balance_method in balance_methods:
                # Generate model name based on configuration
                model_name = f"{base_name}_{model_type}_{balance_method}_config{feature_idx+1}_{timestamp}"
                
                if show_progress:
                    print(f"\nTraining {model_type} with {balance_method} balancing and feature config {feature_idx+1}...")
                
                # Initialize model
                model = CandlePredictionModel(
                    model_dir=output_dir,
                    model_name=model_name,
                    model_type=model_type,
                    scale_features=False,  # No scaling in model since done in processor
                    handle_imbalance=(balance_method != "none")
                )
                
                try:
                    # Train model
                    results = model.train(
                        X, y,
                        feature_list=feature_list,
                        test_size=test_size,
                        cv_folds=cv_folds,
                        optimize_hyperparams=True,
                        balance_method=balance_method,
                        show_progress=False  # Disable nested progress bars
                    )
                    
                    # Store results with configuration
                    model_results = {
                        "model_name": model_name,
                        "model_type": model_type,
                        "balance_method": balance_method,
                        "feature_config": feature_config,
                        "processor_path": processor_path,
                        "metrics": {
                            "accuracy": results["accuracy"],
                            "f1_score": results["f1_score"],
                            "precision": results["precision"],
                            "recall": results["recall"],
                            "roc_auc": results["roc_auc"],
                            "cv_mean": results["cv_scores"]["mean"],
                            "cv_std": results["cv_scores"]["std"]
                        },
                        "model_version": results["model_version"]
                    }
                    
                    all_results.append(model_results)
                    
                    if show_progress:
                        print(f"Trained successfully: {metric}={model_results['metrics'][metric]:.4f}")
                    
                except Exception as e:
                    if show_progress:
                        print(f"Error training {model_name}: {str(e)}")
                
                # Update progress
                completed_models += 1
                if show_progress and 'progress_bar' in locals():
                    progress_bar.update(1)
    
    if show_progress and 'progress_bar' in locals():
        progress_bar.close()
    
    # Find the best model based on the specified metric
    if all_results:
        best_model = max(all_results, key=lambda x: x["metrics"][metric])
        
        # Save summary of all results
        results_path = os.path.join(output_dir, f"{base_name}_selection_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump({
                "all_models": all_results,
                "best_model": best_model,
                "selection_metric": metric
            }, f, indent=4)
        
        if show_progress:
            print("\n===== Model Selection Complete =====")
            print(f"Best model: {best_model['model_name']}")
            print(f"Model type: {best_model['model_type']}")
            print(f"Balance method: {best_model['balance_method']}")
            print(f"Feature config: {best_model['feature_config']}")
            print(f"Performance ({metric}): {best_model['metrics'][metric]:.4f}")
            print(f"CV score: {best_model['metrics']['cv_mean']:.4f} ± {best_model['metrics']['cv_std']:.4f}")
            print(f"Results saved to: {results_path}")
            print("==================================")
        
        return {
            "all_results": all_results,
            "best_model": best_model,
            "results_path": results_path
        }
    else:
        if show_progress:
            print("No models were successfully trained.")
        return {"all_results": [], "best_model": None}


def train_with_backtesting(
    data_path: str,
    output_dir: str = "models",
    model_name: str = "xauusd_model",
    window_size: int = 1000,
    step_size: int = 200,
    test_periods: int = 100,
    show_progress: bool = True,
    **kwargs
) -> Dict:
    """
    Train model with walk-forward validation (backtesting).
    
    Args:
        data_path: Path to CSV file with OHLCV data
        output_dir: Directory to store model files and visualizations
        model_name: Base name for the model files
        window_size: Number of candles to use in each training window
        step_size: Number of candles to step forward for each test
        test_periods: Number of periods to use for testing after each window
        show_progress: Whether to show progress bars
        **kwargs: Additional arguments to pass to train_xauusd_model
        
    Returns:
        Dictionary with training and backtesting results
    """
    from backtest import backtest_model
    
    if show_progress:
        print(f"Starting walk-forward validation with window size {window_size}, step size {step_size}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if show_progress:
        print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check if enough data
    if len(df) < window_size + test_periods + 100:
        raise ValueError(f"Not enough data for walk-forward validation. Need at least {window_size + test_periods + 100} rows.")
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize processor first (we'll use the same processor config for all windows)
    processor = DataProcessor(
        scale_features=kwargs.get('processor_scaling', True),
        feature_selection=kwargs.get('processor_feature_selection', True),
        n_features=kwargs.get('n_features', 30),
        handle_missing=kwargs.get('handle_missing', 'fill_median'),
        smooth_outliers=True
    )
    
    # Save final full model (trained on all data)
    if show_progress:
        print("Training final model on all data...")
    
    final_model_name = f"{model_name}_final"
    final_results = train_xauusd_model(
        data_path=data_path,
        output_dir=output_dir,
        model_name=final_model_name,
        show_progress=show_progress,
        **kwargs
    )
    
    # Initialize backtest model
    if show_progress:
        print("Setting up backtesting...")
    
    backtest_model_obj = CandlePredictionModel(
        model_dir=output_dir,
        model_name=model_name,
        model_type=kwargs.get('model_type', 'random_forest'),
        scale_features=False,  # No scaling in model
        handle_imbalance=kwargs.get('handle_imbalance', True)
    )
    
    # Load the model for backtesting
    backtest_model_obj.load_model_version(final_results['model_version'])
    
    # Perform backtesting
    if show_progress:
        print("Starting backtesting...")
    
    backtest_results = backtest_model_obj.backtest(
        df=df,
        feature_processor=processor,
        window_size=window_size,
        step_size=step_size,
        show_progress=show_progress
    )
    
    # Create summary
    summary = {
        "final_model": final_results,
        "backtest_results": {
            "overall_accuracy": backtest_results["overall_accuracy"],
            "windows_count": backtest_results["windows_count"],
            "predictions_count": backtest_results["predictions_count"],
        }
    }
    
    if show_progress:
        print("\n===== Backtesting Complete =====")
        print(f"Overall accuracy: {backtest_results['overall_accuracy']:.4f}")
        print(f"Windows tested: {backtest_results['windows_count']}")
        print(f"Total predictions: {backtest_results['predictions_count']}")
        print(f"Final model version: {final_results['model_version']}")
        print(f"Final model accuracy: {final_results['accuracy']:.4f}")
        print(f"Final model F1 score: {final_results['f1_score']:.4f}")
        print("================================")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='XAUUSD Prediction Model Trainer')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--output', type=str, default='models', help='Directory to store model files')
    parser.add_argument('--name', type=str, default='xauusd_model', help='Base name for model files')
    parser.add_argument('--model-type', type=str, choices=['random_forest', 'gradient_boosting'], 
                      default='random_forest', help='Type of model to train')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--balance', action='store_true', help='Handle class imbalance')
    parser.add_argument('--balance-method', type=str, choices=['smote', 'undersample', 'oversample'], 
                      default='smote', help='Method for handling class imbalance')
    parser.add_argument('--features', type=int, default=30, help='Number of features to select')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    parser.add_argument('--compare-all', action='store_true', help='Train and compare multiple model configurations')
    parser.add_argument('--metric', type=str, default='f1_score', 
                      choices=['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'cv_mean'],
                      help='Metric to use for model selection')
    
    args = parser.parse_args()
    
    show_progress = not args.no_progress
    
    if args.compare_all:
        # Train multiple models with different configurations
        train_multiple_models(
            data_path=args.data,
            output_dir=args.output,
            base_name=args.name,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            metric=args.metric,
            show_progress=show_progress
        )
    else:
        # Train a single model with the specified configuration
        train_xauusd_model(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.name,
            model_type=args.model_type,
            test_size=args.test_size,
            optimize_hyperparams=args.optimize,
            handle_imbalance=args.balance,
            balance_method=args.balance_method,
            n_features=args.features,
            cv_folds=args.cv_folds,
            show_progress=show_progress
        )