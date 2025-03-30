import os
import time
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import gc

from data_processor import DataProcessor
from ml_model import CandlePredictionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_predictor')

class ModelPredictor:
    """Class for making predictions with trained models"""
    
    def __init__(
        self,
        model_dir: str = "models",
        model_name: str = "xauusd_model",
        processor_path: Optional[str] = None,
        model_version: Optional[str] = None,
        threshold: float = 0.5,
        cache_size: int = 100
    ):
        """
        Initialize the model predictor.
        
        Args:
            model_dir: Directory containing model files
            model_name: Base name of the model
            processor_path: Path to processor state file (if None, will try default path)
            model_version: Specific model version to use (if None, will use latest)
            threshold: Probability threshold for classification
            cache_size: Size of prediction cache
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_version = model_version
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.processor_path = processor_path
        self.feature_list = None
        self.prediction_cache = {}
        self.cache_size = cache_size
        
        # Initialize immediately
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize the model and processor"""
        try:
            # Determine processor path if not provided
            if self.processor_path is None:
                self.processor_path = os.path.join(self.model_dir, f"{self.model_name}_processor.pkl")
            
            logger.info(f"Loading processor from {self.processor_path}")
            
            # Load processor
            self.processor = DataProcessor()
            if not self.processor.load_processor_state(self.processor_path):
                logger.error(f"Failed to load processor from {self.processor_path}")
                return False
            
            logger.info(f"Initializing model from {self.model_dir}/{self.model_name}")
            
            # Initialize model
            self.model = CandlePredictionModel(
                model_dir=self.model_dir,
                model_name=self.model_name
            )
            
            # Load specific version if provided
            if self.model_version:
                if not self.model.load_model_version(self.model_version):
                    logger.error(f"Failed to load model version {self.model_version}")
                    return False
                logger.info(f"Loaded model version {self.model_version}")
            else:
                # Already loads latest version in __init__
                logger.info(f"Using latest model version: {self.model.metadata.get('model_version', 'unknown')}")
                self.model_version = self.model.metadata.get('model_version', 'unknown')
            
            # Get feature list
            self.feature_list = self.model.feature_list
            if not self.feature_list:
                logger.error("Model does not have a feature list")
                return False
            
            logger.info(f"Model initialized successfully with {len(self.feature_list)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make a prediction for the next candle using the trained model.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with prediction details
        """
        try:
            # Create a hash of the most recent data to use as cache key
            # We'll use the last 20 rows to make sure we have a unique identifier
            if len(df) >= 20:
                recent_data = df.iloc[-20:].to_json()
                cache_key = hash(recent_data)
                
                # Check if we have this prediction cached
                if cache_key in self.prediction_cache:
                    logger.info(f"Using cached prediction")
                    return self.prediction_cache[cache_key]
            
            # Prepare latest data for prediction
            logger.info("Preparing features for prediction...")
            
            # Use memory-efficient approach for large dataframes
            if len(df) > 10000:
                # Only keep necessary rows for feature engineering
                # Most features only need the last N rows
                # Consult your feature engineering code to determine minimum rows needed
                df = df.iloc[-5000:].copy()
                
                # Force garbage collection
                gc.collect()
            
            start_time = time.time()
            latest_features = self.processor.prepare_latest_for_prediction(df, self.feature_list)
            feature_time = time.time() - start_time
            logger.info(f"Features prepared in {feature_time:.2f} seconds")
            
            # Make prediction
            logger.info("Making prediction...")
            start_time = time.time()
            prediction = self.model.predict_single(latest_features, threshold=self.threshold)
            predict_time = time.time() - start_time
            
            # Add additional information
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['model_version'] = self.model_version
            prediction['prediction_time_ms'] = int(predict_time * 1000)
            prediction['feature_prep_time_ms'] = int(feature_time * 1000)
            
            logger.info(f"Prediction completed in {predict_time:.2f} seconds")
            
            # Cache the prediction if we have a cache key
            if 'cache_key' in locals():
                # Manage cache size
                if len(self.prediction_cache) >= self.cache_size:
                    # Remove oldest item (simple approach - can be improved)
                    self.prediction_cache.pop(next(iter(self.prediction_cache)))
                
                self.prediction_cache[cache_key] = prediction
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def predict_from_file(self, data_path: str) -> Dict:
        """
        Load data from file and make a prediction.
        
        Args:
            data_path: Path to CSV file with OHLCV data
            
        Returns:
            Dictionary with prediction details
        """
        try:
            # Load data
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded: {len(df)} rows")
            
            return self.predict(df)
        except Exception as e:
            logger.error(f"Error predicting from file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model details
        """
        if not self.model:
            return {"error": "Model not initialized"}
        
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model.metadata.get("model_type", "unknown"),
            "model_path": self.model._get_model_path(),
            "processor_path": self.processor_path,
            "feature_count": len(self.feature_list) if self.feature_list else 0,
            "threshold": self.threshold,
            "metadata": self.model.metadata
        }

# Function to streamline prediction
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
        show_progress: Whether to show progress information
        
    Returns:
        Dictionary with prediction details
    """
    # Initialize predictor
    predictor = ModelPredictor(
        model_dir=model_dir,
        model_name=model_name,
        processor_path=processor_path,
        model_version=model_version,
        threshold=threshold
    )
    
    # Make prediction
    if show_progress:
        print("Making prediction...")
    
    prediction = predictor.predict_from_file(data_path)
    
    if show_progress and "error" not in prediction:
        print("\n===== Prediction =====")
        print(f"Direction: {prediction['prediction_label']}")
        print(f"Signal strength: {prediction['signal_strength']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print(f"Bullish probability: {prediction['bullish_probability']:.4f}")
        print(f"Bearish probability: {prediction['bearish_probability']:.4f}")
        print(f"Model version: {prediction['model_version']}")
        print("=======================")
    elif show_progress and "error" in prediction:
        print(f"Error making prediction: {prediction['error']}")
    
    return prediction


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
    best_model_version = model_version or best_model.get("model_version")
    best_processor_path = processor_path or best_model.get("processor_path")
    
    if show_progress:
        print(f"Loading best model: {model_name} (version {best_model_version})")
    
    # Create predictor
    predictor = ModelPredictor(
        model_dir=model_dir,
        model_name=model_name,
        processor_path=best_processor_path,
        model_version=best_model_version
    )
    
    # Get model info
    model_info = predictor.get_model_info()
    
    # Make a sample prediction
    prediction = predictor.predict_from_file(data_path)
    
    result = {
        "model_name": model_name,
        "model_version": best_model_version,
        "model_type": best_model.get("model_type"),
        "balance_method": best_model.get("balance_method"),
        "feature_config": best_model.get("feature_config"),
        "metrics": best_model.get("metrics", {}),
        "model_info": model_info,
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
    
    parser = argparse.ArgumentParser(description='XAUUSD Prediction Model Predictor')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory with model files')
    parser.add_argument('--model-name', type=str, default='xauusd_model', help='Base name of the model')
    parser.add_argument('--processor-path', type=str, help='Path to processor state file')
    parser.add_argument('--model-version', type=str, help='Specific model version to use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction probability threshold')
    parser.add_argument('--results-file', type=str, help='Path to model selection results JSON file')
    parser.add_argument('--use-best', action='store_true', help='Use the best model from selection results')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress output')
    
    args = parser.parse_args()
    
    show_progress = not args.no_progress
    
    if args.use_best and args.results_file:
        # Use the best model from selection results
        get_best_model(
            data_path=args.data,
            results_path=args.results_file,
            processor_path=args.processor_path,
            model_version=args.model_version,
            show_progress=show_progress
        )
    else:
        # Use specified model
        predict_next_candle(
            data_path=args.data,
            model_dir=args.model_dir,
            model_name=args.model_name,
            processor_path=args.processor_path,
            model_version=args.model_version,
            threshold=args.threshold,
            show_progress=show_progress
        )