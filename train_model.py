from data_processor import DataProcessor
from ml_model import CandlePredictionModel
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                           classification_report, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator
import logging
import time
import numpy as np


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_trainning.log')

# Modify the train method in CandlePredictionModel to include tqdm
def train(self, 
          X: pd.DataFrame, 
          y: pd.Series, 
          feature_list: Optional[List[str]] = None,
          test_size: float = 0.2,
          cv_folds: int = 5,
          optimize_hyperparams: bool = True,
          balance_method: str = 'smote',
          show_progress: bool = True) -> Dict:
    """
    Train the machine learning model with advanced options.
    
    Args:
        X: Feature matrix
        y: Target vector (binary: 0 or 1)
        feature_list: List of feature names (optional)
        test_size: Size of test set for evaluation
        cv_folds: Number of cross-validation folds
        optimize_hyperparams: Whether to perform hyperparameter tuning
        balance_method: Method to handle class imbalance ('none', 'smote', 'undersample', 'oversample')
        show_progress: Whether to show progress bars
        
    Returns:
        dict: Training results with comprehensive metrics
    """
    # Start time for performance tracking
    start_time = datetime.datetime.now()
    
    # Validate input data
    if not self._validate_input_data(X, y):
        raise ValueError("Input data validation failed")
    
    # Store feature list
    if feature_list is not None:
        self.feature_list = feature_list
    else:
        self.feature_list = X.columns.tolist()
    
    # Update metadata
    self.metadata["feature_list"] = self.feature_list
    self.metadata["training_date"] = start_time.isoformat()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    logger.info(f"Class distribution in training set: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Handle class imbalance if needed
    if self.handle_imbalance and balance_method != 'none':
        X_train, y_train = self._balance_classes(X_train, y_train, method=balance_method)
        logger.info(f"After balancing - Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Feature preprocessing
    pipeline_steps = []
    
    # Add scaling if enabled
    if self.scale_features:
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        pipeline_steps.append(('scaler', self.scaler))
    
    # Create base model
    base_model = self._create_base_model()
    
    # Hyperparameter optimization if enabled
    if optimize_hyperparams:
        logger.info("Starting hyperparameter optimization...")
        param_grid = self._get_param_grid()
        
        # Add the model to the pipeline
        pipeline_steps.append(('model', base_model))
        pipeline = Pipeline(pipeline_steps)
        
        # Grid search with cross-validation and progress bar if enabled
        if show_progress:
            print("Performing hyperparameter optimization...")
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid={"model__" + key: value for key, value in param_grid.items()},
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=1 if show_progress else 0
        )
        
        if show_progress:
            # Count total number of fits
            n_candidates = 1
            for key, values in param_grid.items():
                n_candidates *= len(values)
            total_fits = n_candidates * cv_folds
            
            print(f"Fitting {n_candidates} candidates with {cv_folds}-fold cross-validation ({total_fits} fits)")
            
            # Custom verbose output with tqdm
            with tqdm(total=total_fits, desc="Grid Search Progress") as pbar:
                # We can't directly track GridSearchCV progress, so we'll update based on time
                last_update = time.time()
                while not hasattr(grid_search, 'cv_results_'):
                    grid_search.fit(X_train, y_train)
                    # Update progress bar based on time estimate
                    if time.time() - last_update > 0.5:  # Update every half second
                        pbar.update(5)  # Approximate update
                        last_update = time.time()
                        if pbar.n >= total_fits:
                            break
                pbar.n = total_fits  # Ensure we reach 100%
                pbar.refresh()
        else:
            grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        self.model = grid_search.best_estimator_
        best_params = {key.replace("model__", ""): value for key, value in grid_search.best_params_.items()}
        logger.info(f"Best parameters: {best_params}")
        self.metadata["model_params"] = best_params
        
    else:
        # Without hyperparameter optimization, just fit the pipeline
        pipeline_steps.append(('model', base_model))
        self.model = Pipeline(pipeline_steps)
        
        if show_progress:
            print("Training model...")
            with tqdm(total=100, desc="Training Progress") as pbar:
                self.model.fit(X_train, y_train)
                pbar.update(100)
        else:
            self.model.fit(X_train, y_train)
            
        self.metadata["model_params"] = base_model.get_params()
    
    # Track training metrics
    train_pred = self.model.predict(X_train)
    self.training_history['train_accuracy'].append(accuracy_score(y_train, train_pred))
    self.training_history['train_f1'].append(f1_score(y_train, train_pred))
    
    # Evaluate the model on test data
    if show_progress:
        print("Evaluating model...")
    
    y_pred = self.model.predict(X_test)
    y_prob = self.model.predict_proba(X_test)[:, 1]
    
    # Comprehensive evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store validation metrics
    self.training_history['val_accuracy'].append(accuracy)
    self.training_history['val_f1'].append(f1)
    self.training_history['val_precision'].append(precision)
    self.training_history['val_recall'].append(recall)
    self.training_history['val_roc_auc'].append(roc_auc)
    
    # Store test set predictions for later visualization
    self.predictions = []
    for i in range(len(X_test)):
        self.predictions.append({
            'actual': int(y_test.iloc[i]) if isinstance(y_test, pd.Series) else int(y_test[i]),
            'predicted': int(y_pred[i]),
            'probability': float(y_prob[i])
        })
    
    # Cross-validation score
    if show_progress:
        print("Performing cross-validation...")
        cv_scores = []
        cv_folds_iter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42).split(X, y)
        
        for i, (train_idx, val_idx) in enumerate(tqdm(cv_folds_iter, total=cv_folds, desc="Cross-validation")):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # Try to create a clone of the model
                if isinstance(self.model, BaseEstimator):
                    cv_model = clone(self.model)
                    cv_model.fit(X_cv_train, y_cv_train)
                    y_cv_pred = cv_model.predict(X_cv_val)
                else:
                    # Fall back to using the existing model (may overfit slightly)
                    y_cv_pred = self.model.predict(X_cv_val)
                    
                cv_scores.append(f1_score(y_cv_val, y_cv_pred))
            except Exception as e:
                logger.warning(f"Error in CV fold {i}: {str(e)}. Using existing model.")
                # Fall back to using the existing model
                y_cv_pred = self.model.predict(X_cv_val)
                cv_scores.append(f1_score(y_cv_val, y_cv_pred))
    else:
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1'
        )
    
    self.training_history['cross_val_scores'] = cv_scores if isinstance(cv_scores, list) else cv_scores.tolist()
    
    # Feature importance analysis
    feature_imp = self._get_feature_importance()
    
    # Store metrics in metadata
    self.metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "cv_f1_mean": np.mean(cv_scores),
        "cv_f1_std": np.std(cv_scores),
        "training_time": (datetime.datetime.now() - start_time).total_seconds()
    }
    
    self.metadata["performance_metrics"] = self.metrics
    
    # Generate and save visualizations
    if show_progress:
        print("Generating visualizations...")
    
    self._visualize_metrics(X_test, y_test, y_pred, y_prob)
    self._visualize_feature_importance(feature_imp)
    
    # Save the model
    if show_progress:
        print("Saving model...")
    
    self._save_model()
    
    # Return comprehensive results
    results = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "cv_scores": {
            "mean": np.mean(cv_scores),
            "std": np.std(cv_scores),
            "all": cv_scores if isinstance(cv_scores, list) else cv_scores.tolist()
        },
        "feature_importances": feature_imp,
        "training_time": (datetime.datetime.now() - start_time).total_seconds(),
        "model_version": self.metadata["model_version"]
    }
    
    logger.info(f"Model trained successfully with accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    if show_progress:
        print(f"Training complete! Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
    return results

# Replace the existing backtest method with this improved version
def backtest(self, df: pd.DataFrame, feature_processor, window_size: int = 500, 
           step_size: int = 100, show_progress: bool = True) -> Dict:
    """
    Perform backtesting by creating a series of predictions on historical data.
    
    Args:
        df: Historical OHLCV data
        feature_processor: DataProcessor instance to prepare features
        window_size: Number of candles to use in each training window
        step_size: Number of candles to step forward for each test
        show_progress: Whether to show progress bars
        
    Returns:
        dict: Backtesting results
    """
    if len(df) < window_size + 100:
        raise ValueError(f"Not enough data for backtesting. Need at least {window_size + 100} rows.")
        
    logger.info(f"Starting backtesting with window size {window_size}, step size {step_size}")
    
    results = []
    
    # Get all start indices for windows
    start_indices = list(range(0, len(df) - window_size - 50, step_size))
    
    # Use tqdm for progress tracking if enabled
    if show_progress:
        start_indices_iter = tqdm(start_indices, desc="Backtesting Windows")
    else:
        start_indices_iter = start_indices
    
    # Loop through the data in steps
    for start_idx in start_indices_iter:
        end_idx = start_idx + window_size
        test_end_idx = end_idx + 50  # Test on next 50 candles
        
        # Ensure we don't go beyond the data
        if test_end_idx > len(df):
            test_end_idx = len(df)
        
        # Get the window data
        train_df = df.iloc[start_idx:end_idx].copy()
        test_df = df.iloc[end_idx:test_end_idx].copy()
        
        try:
            # Prepare features for training
            X_train, y_train, feature_list = feature_processor.prepare_ml_data(train_df)
            
            # Train a model on this window
            window_model = CandlePredictionModel(
                model_name=f"{self.model_name}_backtest",
                scale_features=self.scale_features,
                handle_imbalance=self.handle_imbalance
            )
            
            # Hide training progress within the loop
            training_result = window_model.train(
                X_train, y_train,
                feature_list=feature_list,
                optimize_hyperparams=False,  # Skip optimization for speed
                test_size=0.2,
                show_progress=False  # No nested progress bars
            )
            
            # Create features for test data (one at a time to simulate real-time prediction)
            test_predictions = []
            
            # Show progress for test predictions if enabled
            test_range = range(len(test_df) - 1)  # -1 because we need the next candle for validation
            if show_progress:
                # Only show this progress for large test sets
                if len(test_range) > 10:
                    test_range = tqdm(test_range, desc=f"Window {start_idx}-{end_idx} Tests", leave=False)
            
            for i in test_range:
                # Get data up to this point
                current_df = pd.concat([train_df, test_df.iloc[:i+1]])
                
                # Prepare latest data for prediction
                latest_features = feature_processor.prepare_latest_for_prediction(current_df, feature_list)
                
                # Make prediction
                pred_result = window_model.predict_single(latest_features)
                
                # Get actual next candle direction
                next_candle = test_df.iloc[i+1]
                actual = 1 if next_candle['close'] > next_candle['open'] else 0
                
                # Store result
                test_predictions.append({
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'test_index': end_idx + i + 1,
                    'prediction': pred_result['prediction'],
                    'probability': pred_result.get('bullish_probability', 0.5),
                    'actual': actual,
                    'correct': pred_result['prediction'] == actual
                })
            
            # Calculate metrics for this window
            window_accuracy = sum(p['correct'] for p in test_predictions) / len(test_predictions) if test_predictions else 0
            
            # Store window results
            window_result = {
                'window_start': start_idx,
                'window_end': end_idx,
                'test_start': end_idx,
                'test_end': test_end_idx,
                'train_accuracy': training_result['accuracy'],
                'train_f1': training_result['f1_score'],
                'test_accuracy': window_accuracy,
                'test_predictions': test_predictions
            }
            
            results.append(window_result)
            
            if show_progress:
                # Update description with accuracy
                if isinstance(start_indices_iter, tqdm):
                    start_indices_iter.set_postfix({"Window Accuracy": f"{window_accuracy:.4f}"})
            
            logger.info(f"Completed backtest window {start_idx}-{end_idx}, test accuracy: {window_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error in backtest window {start_idx}-{end_idx}: {str(e)}")
            continue
    
    # Calculate overall metrics
    all_predictions = []
    for window in results:
        all_predictions.extend(window['test_predictions'])
        
    overall_accuracy = sum(p['correct'] for p in all_predictions) / len(all_predictions) if all_predictions else 0
    
    # Create visualization of backtest results
    self._visualize_backtest_results(results)
    
    return {
        'overall_accuracy': overall_accuracy,
        'windows_count': len(results),
        'predictions_count': len(all_predictions),
        'window_results': results
    }

# Add these functions at the end of your ml_model.py file

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
    import os
    import time
    import pandas as pd
    from data_processor import DataProcessor
    
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


def backtest_only(
    data_path: str,
    model_dir: str = "models",
    model_name: str = "xauusd_model",
    processor_path: Optional[str] = None,
    model_version: Optional[str] = None,
    window_size: int = 500,
    step_size: int = 100,
    show_progress: bool = True
) -> Dict:
    """
    Perform backtesting without training a new model.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        model_dir: Directory with model files
        model_name: Base name of the model
        processor_path: Path to processor state file (if None, will look for default path)
        model_version: Specific model version to use (if None, will use latest)
        window_size: Number of candles to use in each training window
        step_size: Number of candles to step forward for each test
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with backtesting results
    """
    import os
    import pandas as pd
    from data_processor import DataProcessor
    
    if show_progress:
        print(f"Starting backtesting with window size {window_size}, step size {step_size}")
    
    # Load data
    if show_progress:
        print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows")
    
    # Determine processor path if not provided
    if processor_path is None:
        processor_path = os.path.join(model_dir, f"{model_name}_processor.pkl")
    
    # Load processor
    if show_progress:
        print(f"Loading processor from {processor_path}")
    
    processor = DataProcessor()
    if not processor.load_processor_state(processor_path):
        raise ValueError(f"Failed to load processor from {processor_path}")
    
    # Initialize model
    if show_progress:
        print(f"Initializing model")
    
    model = CandlePredictionModel(
        model_dir=model_dir,
        model_name=model_name
    )
    
    # Load specific version if provided
    if model_version:
        if not model.load_model_version(model_version):
            raise ValueError(f"Failed to load model version {model_version}")
        if show_progress:
            print(f"Loaded model version {model_version}")
    else:
        # Already loads latest version in __init__
        if show_progress:
            print(f"Using latest model version: {model.metadata.get('model_version', 'unknown')}")
    
    # Perform backtesting
    if show_progress:
        print("Starting backtesting...")
    
    backtest_results = model.backtest(
        df=df,
        feature_processor=processor,
        window_size=window_size,
        step_size=step_size,
    )
    
    # Generate additional visualizations
    if show_progress:
        print("Generating backtest visualizations...")
    
    # Create time-based visualization with dates
    if 'time' in df.columns:
        _visualize_backtest_with_dates(df, backtest_results, os.path.join(model_dir, f"{model_name}_backtest_timeline.png"))
    
    if show_progress:
        print("\n===== Backtesting Complete =====")
        print(f"Overall accuracy: {backtest_results['overall_accuracy']:.4f}")
        print(f"Windows tested: {backtest_results['windows_count']}")
        print(f"Total predictions: {backtest_results['predictions_count']}")
        print(f"Model version: {model.metadata.get('model_version', 'unknown')}")
        print("================================")
    
    return backtest_results


def test_model(
    data_path: str,
    model_dir: str = "models",
    model_name: str = "xauusd_model",
    processor_path: Optional[str] = None,
    model_version: Optional[str] = None,
    test_size: float = 0.3,
    threshold: float = 0.5,
    show_progress: bool = True
) -> Dict:
    """
    Test a trained model on new data.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        model_dir: Directory with model files
        model_name: Base name of the model
        processor_path: Path to processor state file (if None, will look for default path)
        model_version: Specific model version to use (if None, will use latest)
        test_size: Proportion of data to use for testing (from the end)
        threshold: Probability threshold for classification
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with test results
    """
    import os
    import pandas as pd
    import numpy as np
    from data_processor import DataProcessor
    
    if show_progress:
        print(f"Starting model testing with {test_size*100:.0f}% of data")
    
    # Load data
    if show_progress:
        print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows")
    
    # Determine processor path if not provided
    if processor_path is None:
        processor_path = os.path.join(model_dir, f"{model_name}_processor.pkl")
    
    # Load processor
    if show_progress:
        print(f"Loading processor from {processor_path}")
    
    processor = DataProcessor()
    if not processor.load_processor_state(processor_path):
        raise ValueError(f"Failed to load processor from {processor_path}")
    
    # Initialize model
    if show_progress:
        print(f"Initializing model")
    
    model = CandlePredictionModel(
        model_dir=model_dir,
        model_name=model_name
    )
    
    # Load specific version if provided
    if model_version:
        if not model.load_model_version(model_version):
            raise ValueError(f"Failed to load model version {model_version}")
        if show_progress:
            print(f"Loaded model version {model_version}")
    else:
        # Already loads latest version in __init__
        if show_progress:
            print(f"Using latest model version: {model.metadata.get('model_version', 'unknown')}")
    
    # Split data for testing
    test_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:test_index]
    test_df = df.iloc[test_index:]
    
    if show_progress:
        print(f"Split data: {len(train_df)} training rows, {len(test_df)} testing rows")
    
    # Process full dataset to create targets for testing
    if show_progress:
        print("Processing data...")
    
    # Get features and targets
    full_X, full_y, feature_list = processor.prepare_ml_data(df)
    
    # Extract test portion
    X_test = full_X.iloc[test_index:]
    y_test = full_y.iloc[test_index:]
    
    if show_progress:
        print(f"Testing model on {len(X_test)} samples...")
    
    # Evaluate model
    test_results = model.evaluate_on_data(X_test, y_test, threshold=threshold)
    
    # Calculate cumulative returns
    if show_progress:
        print("Calculating trading performance...")
    
    # Get returns from test_df (assuming we have close prices)
    if 'close' in test_df.columns:
        price_changes = test_df['close'].pct_change().fillna(0)
        
        # Create predictions array aligned with price_changes
        predictions = np.zeros(len(price_changes))
        for p in test_results['predictions']:
            if p['actual'] is not None and p['predicted'] is not None:
                idx = p.get('index', 0)
                if idx < len(predictions):
                    predictions[idx] = p['predicted'] * 2 - 1  # Convert 0/1 to -1/1
        
        # Calculate strategy returns (assuming we trade based on predictions)
        strategy_returns = predictions[:-1] * price_changes[1:].values
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod() - 1
        
        # Calculate metrics
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 288) if strategy_returns.std() > 0 else 0
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # Add to results
        test_results['trading_performance'] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'cumulative_returns': cumulative_returns.tolist()
        }
        
        # Create trading performance visualization
        _visualize_trading_performance(
            test_df, 
            test_results,
            output_path=os.path.join(model_dir, f"{model_name}_trading_performance.png")
        )
    
    if show_progress:
        print("\n===== Testing Complete =====")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"F1 Score: {test_results['f1_score']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"ROC AUC: {test_results['roc_auc']:.4f}")
        
        if 'trading_performance' in test_results:
            perf = test_results['trading_performance']
            print(f"Trading Return: {perf['total_return']*100:.2f}%")
            print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            print(f"Win Rate: {perf['win_rate']*100:.2f}%")
        
        print("============================")
    
    return test_results


def _visualize_backtest_with_dates(df: pd.DataFrame, backtest_results: Dict, output_path: str):
    """
    Create a visualization of backtest results with actual dates on the x-axis.
    
    Args:
        df: Original dataframe with time information
        backtest_results: Results from backtesting
        output_path: Path to save the visualization
    """
    # Check if we have datetime information
    if 'time' not in df.columns:
        return
    
    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Extract all predictions
    all_predictions = []
    for window in backtest_results.get('window_results', []):
        all_predictions.extend(window.get('test_predictions', []))
    
    # Sort by test_index
    all_predictions.sort(key=lambda x: x.get('test_index', 0))
    
    # Prepare data for plotting
    indices = [p.get('test_index', 0) for p in all_predictions]
    if not indices:
        return
    
    # Map indices to dates
    dates = []
    for idx in indices:
        if 0 <= idx < len(df):
            dates.append(df['time'].iloc[idx])
        else:
            # Use a placeholder date if index is out of bounds
            dates.append(pd.NaT)
    
    # Only use valid dates
    valid_mask = ~pd.isnull(dates)
    dates = [d for i, d in enumerate(dates) if valid_mask[i]]
    
    # Filter predictions to match valid dates
    filtered_predictions = [p for i, p in enumerate(all_predictions) if valid_mask[i]]
    
    if not filtered_predictions:
        return
    
    # Extract data for plotting
    actuals = [p.get('actual', 0) for p in filtered_predictions]
    predictions = [p.get('prediction', 0) for p in filtered_predictions]
    correct = [p.get('correct', False) for p in filtered_predictions]
    
    # Calculate cumulative accuracy
    correct_count = 0
    cumulative_accuracies = []
    
    for c in correct:
        if c:
            correct_count += 1
        cumulative_accuracies.append(correct_count / (len(cumulative_accuracies) + 1))
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Predictions vs Actuals
    plt.subplot(2, 1, 1)
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Every week
    
    # Plot actual and predicted values
    plt.plot(dates, actuals, 'o-', label='Actual', alpha=0.5)
    plt.plot(dates, predictions, 'x-', label='Predicted', alpha=0.5)
    
    plt.title('Actual vs Predicted Values Over Time', fontsize=14)
    plt.ylabel('Class (0=Bearish, 1=Bullish)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 2: Cumulative Accuracy
    plt.subplot(2, 1, 2)
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Every week
    
    plt.plot(dates, cumulative_accuracies, 'o-', color='purple')
    plt.title('Cumulative Prediction Accuracy Over Time', fontsize=14)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add overall metrics as text
    plt.figtext(0.5, 0.01, 
              f"Overall Accuracy: {sum(correct) / len(correct):.4f} | "
              f"Total Predictions: {len(correct)}", 
              ha="center", fontsize=12, 
              bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path)
    plt.close()


def _visualize_trading_performance(test_df: pd.DataFrame, test_results: Dict, output_path: str):
    """
    Create a visualization of trading performance.
    
    Args:
        test_df: Test dataframe with price information
        test_results: Results from testing
        output_path: Path to save the visualization
    """
    # Check if we have trading performance data
    if 'trading_performance' not in test_results:
        return
    
    # Extract performance data
    perf = test_results['trading_performance']
    cumulative_returns = perf.get('cumulative_returns', [])
    
    if not cumulative_returns:
        return
    
    # Check if we have time information
    has_time = 'time' in test_df.columns
    
    if has_time:
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(test_df['time']):
            test_df['time'] = pd.to_datetime(test_df['time'])
        
        # Create a date range for plotting
        dates = test_df['time'].iloc[1:len(cumulative_returns)+1].values
    else:
        # Use indices if no time information
        dates = range(1, len(cumulative_returns)+1)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Cumulative Returns
    plt.subplot(2, 1, 1)
    
    if has_time:
        # Format x-axis with dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Every week
    
    plt.plot(dates, cumulative_returns, 'b-', linewidth=2)
    plt.title('Strategy Cumulative Returns', fontsize=14)
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    if has_time:
        plt.xticks(rotation=45)
    
    # Plot 2: Drawdowns
    plt.subplot(2, 1, 2)
    
    # Calculate drawdowns
    running_max = np.maximum.accumulate(np.array([1.0] + cumulative_returns))
    drawdowns = (np.array([1.0] + cumulative_returns)) / running_max - 1.0
    
    if has_time:
        # Format x-axis with dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Every week
    
    plt.fill_between(dates, 0, drawdowns[1:], color='red', alpha=0.3)
    plt.plot(dates, drawdowns[1:], 'r-', linewidth=1)
    plt.title('Strategy Drawdowns', fontsize=14)
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    if has_time:
        plt.xticks(rotation=45)
    
    # Add performance metrics as text
    metrics_text = f"Total Return: {perf['total_return']*100:.2f}%\n"
    metrics_text += f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}\n"
    metrics_text += f"Win Rate: {perf['win_rate']*100:.2f}%\n"
    metrics_text += f"Max Drawdown: {min(drawdowns)*100:.2f}%"
    
    plt.figtext(0.15, 0.01, metrics_text, fontsize=12, 
              bbox={"facecolor":"lightgreen", "alpha":0.2, "pad":5})
    
    # Add model metrics
    model_text = f"Model Accuracy: {test_results['accuracy']:.4f}\n"
    model_text += f"F1 Score: {test_results['f1_score']:.4f}\n"
    model_text += f"Precision: {test_results['precision']:.4f}\n"
    model_text += f"Recall: {test_results['recall']:.4f}"
    
    plt.figtext(0.85, 0.01, model_text, fontsize=12, ha='right',
              bbox={"facecolor":"lightblue", "alpha":0.2, "pad":5})
    
    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(output_path)
    plt.close()


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
    import os
    import pandas as pd
    from data_processor import DataProcessor
    
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
        verbose=show_progress,
        **kwargs
    )
    
    # Initialize backtest model
    if show_progress:
        print("Setting up backtesting...")
    
    backtest_model = CandlePredictionModel(
        model_dir=output_dir,
        model_name=model_name,
        model_type=kwargs.get('model_type', 'random_forest'),
        scale_features=False,  # No scaling in model
        handle_imbalance=kwargs.get('handle_imbalance', True)
    )
    
    # Load the model for backtesting
    backtest_model.load_model_version(final_results['model_version'])
    
    # Perform backtesting
    if show_progress:
        print("Starting backtesting...")
    
    backtest_results = backtest_model.backtest(
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


def predict_next_candle(
    data_path: str,
    model_dir: str = "models",
    model_name: str = "xauusd_model",
    processor_path: Optional[str] = None,
    model_version: Optional[str] = None,
    threshold: float = 0.5,
    show_progress: bool = True
) -> Dict:
    """
    Make a prediction for the next candle using the trained model.
    
    Args:
        data_path: Path to the CSV file with OHLCV data
        model_dir: Directory with model files
        model_name: Base name of the model
        processor_path: Path to processor state file (if None, will look for default path)
        model_version: Specific model version to use (if None, will use latest)
        threshold: Probability threshold for classification
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with prediction details
    """
    import os
    import pandas as pd
    from data_processor import DataProcessor
    
    # Load data
    if show_progress:
        print(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if show_progress:
        print(f"Data loaded: {len(df)} rows")
    
    # Determine processor path if not provided
    if processor_path is None:
        processor_path = os.path.join(model_dir, f"{model_name}_processor.pkl")
    
    # Load processor
    if show_progress:
        print(f"Loading processor from {processor_path}")
    
    processor = DataProcessor()
    if not processor.load_processor_state(processor_path):
        raise ValueError(f"Failed to load processor from {processor_path}")
    
    # Initialize model
    if show_progress:
        print(f"Initializing model")
    
    model = CandlePredictionModel(
        model_dir=model_dir,
        model_name=model_name
    )
    
    # Load specific version if provided
    if model_version:
        if not model.load_model_version(model_version):
            raise ValueError(f"Failed to load model version {model_version}")
        if show_progress:
            print(f"Loaded model version {model_version}")
    else:
        # Already loads latest version in __init__
        if show_progress:
            print(f"Using latest model version: {model.metadata.get('model_version', 'unknown')}")
    
    # Get feature list
    feature_list = model.feature_list
    if not feature_list:
        raise ValueError("Model does not have a feature list")
    
    # Prepare latest data for prediction
    if show_progress:
        print("Preparing features for prediction...")
    
    latest_features = processor.prepare_latest_for_prediction(df, feature_list)
    
    # Make prediction
    if show_progress:
        print("Making prediction...")
    
    prediction = model.predict_single(latest_features, threshold=threshold)
    
    # Add timestamp
    prediction['timestamp'] = pd.Timestamp.now().isoformat()
    prediction['model_version'] = model.metadata.get('model_version', 'unknown')
    
    if show_progress:
        print("\n===== Prediction =====")
        print(f"Direction: {prediction['prediction_label']}")
        print(f"Signal strength: {prediction['signal_strength']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print(f"Bullish probability: {prediction['bullish_probability']:.4f}")
        print(f"Bearish probability: {prediction['bearish_probability']:.4f}")
        print("=======================")
    
    return prediction

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
    import os
    import pandas as pd
    import numpy as np
    import json
    from tqdm.auto import tqdm
    from data_processor import DataProcessor
    from datetime import datetime
    
    # Set defaults if not provided
    if models_to_try is None:
        models_to_try = ["random_forest", "gradient_boosting"]
    
    if balance_methods is None:
        balance_methods = ["none", "smote"]
    
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
                if show_progress:
                    progress_bar.update(1)
    
    if show_progress:
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


def compare_models_visually(
    results_path: str,
    output_dir: str = "models",
    metrics: List[str] = None,
    show_progress: bool = True
) -> str:
    """
    Generate visualizations comparing all trained models.
    
    Args:
        results_path: Path to the JSON file with model selection results
        output_dir: Directory to store visualizations
        metrics: List of metrics to visualize (default: accuracy, f1_score, roc_auc, cv_mean)
        show_progress: Whether to show progress information
        
    Returns:
        Path to the saved visualization file
    """
    import os
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set default metrics if not provided
    if metrics is None:
        metrics = ["accuracy", "f1_score", "roc_auc", "cv_mean"]
    
    # Load results
    if show_progress:
        print(f"Loading results from {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    all_models = results.get("all_models", [])
    best_model = results.get("best_model", {})
    selection_metric = results.get("selection_metric", "f1_score")
    
    if not all_models:
        if show_progress:
            print("No models found in results file.")
        return None
    
    # Create a DataFrame from results for easier visualization
    models_data = []
    for model in all_models:
        model_data = {
            "Model Name": model["model_name"],
            "Model Type": model["model_type"],
            "Balance Method": model["balance_method"],
            "Feature Config": str(model["feature_config"]),
            # Extract metrics
            **{metric: model["metrics"].get(metric, 0) for metric in metrics}
        }
        models_data.append(model_data)
    
    df_models = pd.DataFrame(models_data)
    
    # Create visualizations
    timestamp = os.path.basename(results_path).split("_selection_results_")[1].split(".")[0]
    viz_path = os.path.join(output_dir, f"model_comparison_{timestamp}.png")
    
    plt.figure(figsize=(15, 12))
    
    # 1. Overall comparison of selected metric
    plt.subplot(2, 2, 1)
    
    # Sort by the selection metric
    df_sorted = df_models.sort_values(by=selection_metric, ascending=False)
    
    # Create bar colors - highlight the best model
    colors = ['lightblue'] * len(df_sorted)
    best_idx = df_sorted["Model Name"].tolist().index(best_model["model_name"]) if best_model else -1
    if best_idx >= 0:
        colors[best_idx] = 'orange'
    
    # Create the bar chart
    bars = plt.barh(range(len(df_sorted)), df_sorted[selection_metric], color=colors)
    plt.yticks(range(len(df_sorted)), [f"{i+1}" for i in range(len(df_sorted))])
    plt.xlabel(f"{selection_metric.replace('_', ' ').title()}")
    plt.title(f"Models Ranked by {selection_metric.replace('_', ' ').title()}")
    
    # Annotate bars with model names
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{df_sorted.iloc[i]['Model Type']}/{df_sorted.iloc[i]['Balance Method']}",
            va='center'
        )
    
    # 2. Metrics comparison by model type
    plt.subplot(2, 2, 2)
    model_types = df_models["Model Type"].unique()
    
    metric_by_type = {}
    for metric in metrics:
        metric_by_type[metric] = [df_models[df_models["Model Type"] == mt][metric].mean() for mt in model_types]
    
    x = range(len(model_types))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        plt.bar([pos + i * width for pos in x], metric_by_type[metric], width=width, label=metric.replace('_', ' ').title())
    
    plt.xlabel("Model Type")
    plt.ylabel("Average Score")
    plt.title("Performance by Model Type")
    plt.xticks([pos + width * (len(metrics) - 1) / 2 for pos in x], model_types)
    plt.legend()
    
    # 3. Metrics comparison by balance method
    plt.subplot(2, 2, 3)
    balance_methods = df_models["Balance Method"].unique()
    
    metric_by_balance = {}
    for metric in metrics:
        metric_by_balance[metric] = [df_models[df_models["Balance Method"] == bm][metric].mean() for bm in balance_methods]
    
    x = range(len(balance_methods))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        plt.bar([pos + i * width for pos in x], metric_by_balance[metric], width=width, label=metric.replace('_', ' ').title())
    
    plt.xlabel("Balance Method")
    plt.ylabel("Average Score")
    plt.title("Performance by Balance Method")
    plt.xticks([pos + width * (len(metrics) - 1) / 2 for pos in x], balance_methods)
    plt.legend()
    
    # 4. Top models details
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Show details of top 5 models
    top_models = df_sorted.head(5)
    text_content = "Top 5 Models:\n\n"
    
    for i, (_, model) in enumerate(top_models.iterrows()):
        text_content += f"{i+1}. {model['Model Type']}/{model['Balance Method']}\n"
        text_content += f"   Config: {model['Feature Config']}\n"
        
        # Add metrics
        for metric in metrics:
            text_content += f"   {metric.replace('_', ' ').title()}: {model[metric]:.4f}\n"
        
        text_content += "\n"
    
    plt.text(0.05, 0.95, text_content, va='top', fontsize=10)
    
    # Finish and save
    plt.tight_layout()
    plt.savefig(viz_path)
    plt.close()
    
    if show_progress:
        print(f"Model comparison visualization saved to {viz_path}")
    
    return viz_path


def get_best_model(
    data_path: str,
    results_path: str,
    processor_path: Optional[str] = None,
    model_version: Optional[str] = None,
    show_progress: bool = True
) -> Dict:
    """
    Load the best model from model selection results and make a prediction.
    
    Args:
        data_path: Path to CSV file with OHLCV data
        results_path: Path to the JSON file with model selection results
        processor_path: Path to processor state file (override the one in results)
        model_version: Specific model version to use (override the one in results)
        show_progress: Whether to show progress information
        
    Returns:
        Dictionary with model info and a sample prediction
    """
    import os
    import json
    import pandas as pd
    from data_processor import DataProcessor
    
    # Load results
    if show_progress:
        print(f"Loading results from {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    best_model = results.get("best_model", {})
    
    if not best_model:
        if show_progress:
            print("No best model found in results file.")
        return None
    
    # Determine paths
    model_dir = os.path.dirname(results_path)
    model_name = best_model.get("model_name")
    model_version = model_version or best_model.get("model_version")
    processor_path = processor_path or best_model.get("processor_path")
    
    if show_progress:
        print(f"Loading best model: {model_name} (version {model_version})")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Load processor
    if show_progress:
        print(f"Loading processor from {processor_path}")
    
    processor = DataProcessor()
    if not processor.load_processor_state(processor_path):
        raise ValueError(f"Failed to load processor from {processor_path}")
    
    # Initialize model
    model = CandlePredictionModel(
        model_dir=model_dir,
        model_name=model_name
    )
    
    # Load specific version if provided
    if model_version:
        if not model.load_model_version(model_version):
            raise ValueError(f"Failed to load model version {model_version}")
    
    # Get feature list
    feature_list = model.feature_list
    if not feature_list:
        raise ValueError("Model does not have a feature list")
    
    # Prepare latest data for prediction
    latest_features = processor.prepare_latest_for_prediction(df, feature_list)
    
    # Make prediction
    prediction = model.predict_single(latest_features)
    
    result = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": best_model.get("model_type"),
        "balance_method": best_model.get("balance_method"),
        "feature_config": best_model.get("feature_config"),
        "metrics": best_model.get("metrics", {}),
        "sample_prediction": prediction
    }
    
    if show_progress:
        print("\n===== Best Model Information =====")
        print(f"Model name: {result['model_name']}")
        print(f"Model type: {result['model_type']}")
        print(f"Balance method: {result['balance_method']}")
        print(f"Performance metrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("\nSample prediction:")
        print(f"Direction: {prediction['prediction_label']}")
        print(f"Signal strength: {prediction['signal_strength']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print("=================================")
    
    return result

    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='XAUUSD Prediction Model Trainer')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--output', type=str, default='models', help='Directory to store model files')
    parser.add_argument('--name', type=str, default='xauusd_model', help='Base name for model files')
    parser.add_argument('--model-type', type=str, choices=['random_forest', 'gradient_boosting'], 
                      default='random_forest', help='Type of model to train')
    parser.add_argument('--metric', type=str, default='f1_score', 
                      choices=['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'cv_mean'],
                      help='Metric to use for model selection')
    parser.add_argument('--compare-only', action='store_true', help='Only compare existing models')
    parser.add_argument('--results-file', type=str, help='Path to existing results JSON file (for --compare-only)')
    parser.add_argument('--get-best', action='store_true', help='Load the best model and make a prediction')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--balance', action='store_true', help='Handle class imbalance')
    parser.add_argument('--balance-method', type=str, choices=['smote', 'undersample', 'oversample'], 
                      default='smote', help='Method for handling class imbalance')
    parser.add_argument('--features', type=int, default=30, help='Number of features to select')
    parser.add_argument('--backtest', action='store_true', help='Perform backtesting')
    parser.add_argument('--backtest-only', action='store_true', help='Perform backtesting without training')
    parser.add_argument('--window-size', type=int, default=1000, help='Number of candles per training window')
    parser.add_argument('--step-size', type=int, default=200, help='Number of candles to step forward')
    parser.add_argument('--predict', action='store_true', help='Make prediction for next candle')
    parser.add_argument('--test', action='store_true', help='Test the model on the latest data')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proportion of data to use for testing')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction probability threshold')
    parser.add_argument('--model-version', type=str, help='Specific model version to use')
    parser.add_argument('--processor-path', type=str, help='Path to processor state file')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    
    args = parser.parse_args()
    
    show_progress = not args.no_progress
    
    if args.backtest_only:
        # Backtest without training
        backtest_only(
            data_path=args.data,
            model_name=args.name,
            processor_path=args.processor_path,
            model_version=args.model_version,
            window_size=args.window_size,
            step_size=args.step_size,
        )
    elif args.test:
        # Test the model
        test_model(
            data_path=args.data,
            model_dir=args.output,
            model_name=args.name,
            processor_path=args.processor_path,
            model_version=args.model_version,
            test_size=args.test_size,
            threshold=args.threshold,
            show_progress=show_progress
        )
    elif args.backtest:
        # Train with backtesting
        train_with_backtesting(
            data_path=args.data,
            output_dir=args.output,
            model_name=args.name,
            model_type=args.model_type,
            optimize_hyperparams=args.optimize,
            handle_imbalance=args.balance,
            balance_method=args.balance_method,
            n_features=args.features,
            window_size=args.window_size,
            step_size=args.step_size,
            show_progress=show_progress
        )
    elif args.predict:
        # Make prediction
        predict_next_candle(
            data_path=args.data,
            model_dir=args.output,
            model_name=args.name,
            processor_path=args.processor_path,
            model_version=args.model_version,
            threshold=args.threshold,
            show_progress=show_progress
        )
    
    elif args.compare_only and args.results_file:
        # Only generate comparison visualizations
        compare_models_visually(
            results_path=args.results_file,
            output_dir=args.output,
            show_progress=show_progress
        )
    elif args.get_best and args.results_file:
        # Load best model and make prediction
        get_best_model(
            data_path=args.data,
            results_path=args.results_file,
            show_progress=show_progress
        )
    else:
        # Train multiple models
        # Define model configurations to try
        models_to_try = [
            "random_forest",
            "gradient_boosting",
            "extra_trees",
            "adaboost",
            "logistic_regression",
            "decision_tree",
            "knn",
            "neural_network",
            "qda"
        ]
        balance_methods = ["none", "smote", "undersample"]
        
        # Define feature configurations to try
        feature_configs = [
            {"scale_features": True, "feature_selection": True, "n_features": 20},
            {"scale_features": True, "feature_selection": True, "n_features": 40},
            {"scale_features": True, "feature_selection": False}
        ]
        
        # Run model selection
        results = train_multiple_models(
            data_path=args.data,
            output_dir=args.output,
            base_name=args.name,
            models_to_try=models_to_try,
            balance_methods=balance_methods,
            feature_configs=feature_configs,
            metric=args.metric,
            show_progress=show_progress
        )
        
        # Generate comparison visualizations
        if results.get("results_path"):
            compare_models_visually(
                results_path=results["results_path"],
                output_dir=args.output,
                show_progress=show_progress
            )
            
            # Load best model
            get_best_model(
                data_path=args.data,
                results_path=results["results_path"],
                show_progress=show_progress
            )

    