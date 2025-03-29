import numpy as np
import pandas as pd
import os
import joblib
import json
import logging
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                           classification_report, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CandlePredictionModel')

class CandlePredictionModel:
    """
    Enhanced model for predicting XAUUSD price movements using machine learning.
    
    Features:
    - Hyperparameter optimization
    - Advanced model evaluation metrics
    - Feature importance analysis
    - Model versioning
    - Data preprocessing pipeline
    - Handling of class imbalance
    - Proper error handling and validation
    - Visualization of metrics and performance
    """
    
    def __init__(self, 
                 model_dir: str = "models",
                 model_name: str = "xauusd_prediction_model",
                 model_type: str = "random_forest",
                 scale_features: bool = True,
                 handle_imbalance: bool = True):
        """
        Initialize the model with configuration parameters.
        
        Args:
            model_dir: Directory to store model files
            model_name: Base name for the model files
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            scale_features: Whether to scale features (use False if already scaled in DataProcessor)
            handle_imbalance: Whether to address class imbalance
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_type = model_type
        self.scale_features = scale_features
        self.handle_imbalance = handle_imbalance
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model components
        self.model = None
        self.feature_list = None
        self.scaler = None
        self.feature_selector = None
        self.smote = None
        
        # Training history
        self.training_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_roc_auc': [],
            'cross_val_scores': []
        }
        
        # Performance metrics
        self.metrics = {}
        self.predictions = []
        
        # Model metadata
        self.metadata = {
            "model_version": self._generate_version(),
            "training_date": None,
            "model_type": model_type,
            "model_params": None,
            "feature_list": None,
            "performance_metrics": None,
            "preprocessing": {
                "scaling": scale_features,
                "imbalance_handling": handle_imbalance
            }
        }
        
        # Try to load the latest model
        self._load_latest_model()
    
    def _generate_version(self) -> str:
        """Generate a version string for model versioning."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_model_path(self, version: Optional[str] = None) -> str:
        """Get the full path for model file with version."""
        if version is None:
            version = self.metadata["model_version"]
        return os.path.join(self.model_dir, f"{self.model_name}_{version}.joblib")
    
    def _get_metadata_path(self, version: Optional[str] = None) -> str:
        """Get the full path for metadata file with version."""
        if version is None:
            version = self.metadata["model_version"]
        return os.path.join(self.model_dir, f"{self.model_name}_{version}_metadata.json")
    
    def _get_metrics_path(self, version: Optional[str] = None) -> str:
        """Get the full path for metrics visualization."""
        if version is None:
            version = self.metadata["model_version"]
        return os.path.join(self.model_dir, f"{self.model_name}_{version}_metrics.png")
    
    def _get_feature_importance_path(self, version: Optional[str] = None) -> str:
        """Get the full path for feature importance visualization."""
        if version is None:
            version = self.metadata["model_version"]
        return os.path.join(self.model_dir, f"{self.model_name}_{version}_features.png")
    
    def _load_latest_model(self) -> bool:
        """
        Load the latest model version from the model directory.
        
        Returns:
            bool: True if a model was loaded successfully, False otherwise
        """
        try:
            # Find all model files
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.startswith(self.model_name) and f.endswith('.joblib')]
            
            if not model_files:
                logger.info("No existing model found.")
                return False
            
            # Sort by version (which is based on datetime)
            model_files.sort(reverse=True)
            latest_model = model_files[0]
            
            # Extract version from filename
            version = latest_model.replace(f"{self.model_name}_", "").replace(".joblib", "")
            
            # Load model and metadata
            model_path = self._get_model_path(version)
            metadata_path = self._get_metadata_path(version)
            
            self.model = joblib.load(model_path)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_list = self.metadata.get("feature_list")
            
            logger.info(f"Loaded model version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_model(self) -> None:
        """Save the model and its metadata."""
        if self.model is None:
            logger.error("Cannot save model - model not trained")
            return
        
        try:
            # Save model
            model_path = self._get_model_path()
            joblib.dump(self.model, model_path)
            
            # Save metadata
            metadata_path = self._get_metadata_path()
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def _validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
        """
        Validate input data before training or prediction.
        
        Args:
            X: Feature dataframe
            y: Target series (optional, for training)
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        # Check if X is a dataframe
        if not isinstance(X, pd.DataFrame):
            logger.error("X must be a pandas DataFrame")
            return False
        
        # Check for NaN values
        if X.isnull().any().any():
            logger.error("X contains NaN values")
            return False
        
        # Check for infinite values
        if np.isinf(X.values).any():
            logger.error("X contains infinite values")
            return False
        
        # Check y if provided (for training)
        if y is not None:
            # Check if y is a series
            if not isinstance(y, (pd.Series, np.ndarray)):
                logger.error("y must be a pandas Series or numpy array")
                return False
            
            # Check if y has the same length as X
            if len(y) != len(X):
                logger.error(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples")
                return False
            
            # Check for unique values in y (should be binary classification)
            unique_values = np.unique(y)
            if len(unique_values) != 2:
                logger.error(f"y should have exactly 2 classes, found {len(unique_values)}")
                return False
        
        return True
    
    def _create_base_model(self) -> Any:
        """Create the base model based on model_type."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        elif self.model_type == "adaboost":
            return AdaBoostClassifier(random_state=42)
        elif self.model_type == "extra_trees":
            return ExtraTreesClassifier(random_state=42)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "svm":
            return SVC(probability=True, random_state=42)
        elif self.model_type == "knn":
            return KNeighborsClassifier(n_jobs=-1)
        elif self.model_type == "neural_network":
            return MLPClassifier(random_state=42, max_iter=500)
        elif self.model_type == "qda":
            return QuadraticDiscriminantAnalysis()
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(random_state=42)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, defaulting to RandomForest")
            return RandomForestClassifier(random_state=42)

    
    def _get_param_grid(self) -> Dict:
        """Get the hyperparameter grid for model tuning based on model_type."""
        if self.model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
        elif self.model_type == "gradient_boosting":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 1.0]
            }
        elif self.model_type == "adaboost":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        elif self.model_type == "extra_trees":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == "logistic_regression":
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            }
        elif self.model_type == "svm":
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'class_weight': [None, 'balanced']
            }
        elif self.model_type == "knn":
            return {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        elif self.model_type == "neural_network":
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        elif self.model_type == "qda":
            return {
                'reg_param': [0.0, 0.1, 0.5]
            }
        elif self.model_type == "decision_tree":
            return {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        else:
            return {
                'n_estimators': [100],
                'max_depth': [None]
            }
    
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
                try:
                    # If tqdm is available, use it
                    from tqdm.auto import tqdm
                    import time
                    
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
                        grid_search.fit(X_train, y_train)
                        pbar.update(total_fits)  # Ensure we reach 100%
                except ImportError:
                    # If tqdm is not available, proceed without progress bar
                    grid_search.fit(X_train, y_train)
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
                try:
                    from tqdm.auto import tqdm
                    print("Training model...")
                    with tqdm(total=1, desc="Training Progress") as pbar:
                        self.model.fit(X_train, y_train)
                        pbar.update(1)
                except ImportError:
                    self.model.fit(X_train, y_train)
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
            try:
                from tqdm.auto import tqdm
                from sklearn.base import clone, BaseEstimator
                
                print("Performing cross-validation...")
                cv_scores = []
                cv_folds_iter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42).split(X, y)
                
                for i, (train_idx, val_idx) in enumerate(tqdm(list(cv_folds_iter), desc="Cross-validation")):
                    X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Use existing model to predict - safer than trying to clone
                    y_cv_pred = self.model.predict(X_cv_val)
                    cv_scores.append(f1_score(y_cv_val, y_cv_pred))
            except ImportError:
                # Fall back to standard cross_val_score if tqdm not available
                cv_scores = cross_val_score(
                    self.model, X, y, 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='f1'
                )
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
    
    def _balance_classes(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes using various methods.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Method to use ('smote', 'undersample', 'oversample')
            
        Returns:
            Tuple of balanced X and y
        """
        # Get class distribution
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        if method == 'smote':
            # SMOTE: Synthetic Minority Over-sampling Technique
            try:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                return X_balanced, y_balanced
            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}. Falling back to random oversampling.")
                method = 'oversample'
        
        if method == 'oversample':
            # Random oversampling of minority class
            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()
            
            # Separate by class
            X_majority = X[y == majority_class]
            y_majority = y[y == majority_class]
            X_minority = X[y == minority_class]
            y_minority = y[y == minority_class]
            
            # Oversample minority class
            X_minority_resampled, y_minority_resampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=len(X_majority),
                random_state=42
            )
            
            # Combine majority and oversampled minority
            X_balanced = pd.concat([X_majority, X_minority_resampled])
            y_balanced = pd.concat([y_majority, y_minority_resampled])
            
            return X_balanced, y_balanced
            
        elif method == 'undersample':
            # Random undersampling of majority class
            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()
            
            # Separate by class
            X_majority = X[y == majority_class]
            y_majority = y[y == majority_class]
            X_minority = X[y == minority_class]
            y_minority = y[y == minority_class]
            
            # Undersample majority class
            X_majority_resampled, y_majority_resampled = resample(
                X_majority, y_majority,
                replace=False,
                n_samples=len(X_minority),
                random_state=42
            )
            
            # Combine undersampled majority and minority
            X_balanced = pd.concat([X_majority_resampled, X_minority])
            y_balanced = pd.concat([y_majority_resampled, y_minority])
            
            return X_balanced, y_balanced
        
        # If no method matched or 'none' was specified, return original data
        return X, y
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from the model."""
        if self.model is None:
            logger.error("Model not trained yet.")
            return {}
        
        try:
            # Different ways to get feature importance depending on pipeline and model type
            if isinstance(self.model, Pipeline):
                model_step = self.model.named_steps['model']
                importances = model_step.feature_importances_
            else:
                importances = self.model.feature_importances_
            
            # Map importance to feature names
            feature_list = self.feature_list or [f"feature_{i}" for i in range(len(importances))]
            importance_dict = dict(zip(feature_list, importances))
            
            # Sort by importance (descending)
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return {}
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5, return_proba: bool = True) -> Dict:
        """
        Make predictions with the trained model.
        
        Args:
            X: Features to predict on
            threshold: Probability threshold for positive class
            return_proba: Whether to return probabilities
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please train or load a model first.")
        
        # Validate input
        if not self._validate_input_data(X):
            raise ValueError("Invalid input data for prediction")
        
        # Check for expected features
        if self.feature_list is not None:
            missing_features = set(self.feature_list) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
            
            # Ensure correct order of features
            X = X[self.feature_list]
        
        try:
            # Get probabilities
            probabilities = self.model.predict_proba(X)
            
            # Create prediction based on threshold
            prediction = (probabilities[:, 1] >= threshold).astype(int)
            
            result = {
                "prediction": prediction.tolist(),
                "prediction_label": ["Bearish" if p == 0 else "Bullish" for p in prediction],
            }
            
            if return_proba:
                result["probability"] = probabilities.tolist()
                result["bullish_probability"] = probabilities[:, 1].tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_single(self, X: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """
        Make a prediction for a single sample.
        
        Args:
            X: Single row of features
            threshold: Probability threshold for positive class
            
        Returns:
            dict: Prediction details
        """
        if len(X) != 1:
            logger.warning(f"Expected single sample, got {len(X)} samples.")
        
        # Make prediction
        result = self.predict(X, threshold)
        
        # Extract single values for cleaner output
        single_result = {
            "prediction": result["prediction"][0],
            "prediction_label": result["prediction_label"][0],
            "bullish_probability": result["bullish_probability"][0] if "bullish_probability" in result else None,
            "bearish_probability": result["probability"][0][0] if "probability" in result else None,
        }
        
        # Add trading signal strength based on probability
        if "bullish_probability" in result:
            prob = result["bullish_probability"][0]
            if prob > 0.75:
                signal_strength = "Strong Bullish"
            elif prob > 0.60:
                signal_strength = "Moderate Bullish"
            elif prob > 0.5:
                signal_strength = "Weak Bullish"
            elif prob < 0.25:
                signal_strength = "Strong Bearish"
            elif prob < 0.4:
                signal_strength = "Moderate Bearish"
            else:
                signal_strength = "Weak Bearish"
                
            single_result["signal_strength"] = signal_strength
            single_result["confidence"] = max(prob, 1-prob)
        
        return single_result
    
    def _visualize_metrics(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Generate and save visualizations of model metrics.
        
        Args:
            X_test: Test feature data
            y_test: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
        """
        # Create a figure with subplots
        plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix", fontsize=14)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        # 2. ROC Curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision)
        plt.axhline(y=np.sum(y_test) / len(y_test), color='r', linestyle='--', 
                  label=f'Baseline ({np.sum(y_test) / len(y_test):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 4. Probability Distribution
        plt.subplot(2, 3, 4)
        for i, label in enumerate(['Bearish', 'Bullish']):
            sns.kdeplot(y_prob[y_test == i], label=f'Actual {label}', shade=True)
        plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution by Class', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Metrics Summary
        plt.subplot(2, 3, 5)
        metrics = [
            f"Accuracy: {self.metrics['accuracy']:.4f}",
            f"F1 Score: {self.metrics['f1_score']:.4f}",
            f"Precision: {self.metrics['precision']:.4f}",
            f"Recall: {self.metrics['recall']:.4f}",
            f"ROC AUC: {self.metrics['roc_auc']:.4f}",
            f"CV F1 (mean): {self.metrics['cv_f1_mean']:.4f}",
            f"CV F1 (std): {self.metrics['cv_f1_std']:.4f}",
            f"Training Time: {self.metrics['training_time']:.2f} sec"
        ]
        plt.axis('off')
        y_pos = 0.9
        plt.text(0.1, 0.95, "Model Performance Metrics", fontsize=16, fontweight='bold')
        for metric in metrics:
            plt.text(0.1, y_pos, metric, fontsize=14)
            y_pos -= 0.1
            
        # 6. Class Distribution
        plt.subplot(2, 3, 6)
        class_dist = pd.Series(y_test).value_counts()
        class_dist.index = ['Bearish' if i == 0 else 'Bullish' for i in class_dist.index]
        ax = sns.barplot(x=class_dist.index, y=class_dist.values)
        plt.title('Class Distribution in Test Set', fontsize=14)
        plt.xlabel('Class')
        plt.ylabel('Count')
        for i, v in enumerate(class_dist.values):
            ax.text(i, v + 1, str(v), ha='center')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self._get_metrics_path())
        plt.close()
        
        logger.info(f"Saved metrics visualization to {self._get_metrics_path()}")
    
    def _visualize_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """
        Generate and save visualization of feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
        """
        if not feature_importance:
            logger.warning("No feature importance data to visualize")
            return
            
        # Get top 20 features
        top_features = dict(sorted(feature_importance.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:20])
        
        plt.figure(figsize=(12, 10))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(top_features)), list(top_features.values()), align='center')
        plt.yticks(range(len(top_features)), list(top_features.keys()))
        
        # Add data labels
        for i, v in enumerate(top_features.values()):
            plt.text(v + 0.001, i, f"{v:.4f}", va='center')
        
        # Add details
        plt.title('Top 20 Feature Importance', fontsize=16)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self._get_feature_importance_path())
        plt.close()
        
        logger.info(f"Saved feature importance visualization to {self._get_feature_importance_path()}")
    
    def visualize_prediction_history(self, predictions: List[Dict], filename: str = "prediction_history.png"):
        """
        Visualize prediction history over time.
        
        Args:
            predictions: List of prediction dictionaries with actual, predicted and probability
            filename: Path to save the visualization
        """
        if not predictions:
            logger.warning("No prediction history to visualize")
            return
            
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(predictions)
        
        # Calculate accuracy over time
        df['correct'] = df['actual'] == df['predicted']
        df['cumulative_accuracy'] = df['correct'].cumsum() / (df.index + 1)
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['actual'], 'o-', label='Actual', alpha=0.7)
        plt.plot(df.index, df['predicted'], 'x-', label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Values', fontsize=14)
        plt.ylabel('Class (0=Bearish, 1=Bullish)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Probability
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['probability'], 'o-', color='green')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.title('Prediction Probability', fontsize=14)
        plt.ylabel('Probability of Bullish')
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Accuracy
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['cumulative_accuracy'], 'o-', color='purple')
        plt.title('Cumulative Prediction Accuracy', fontsize=14)
        plt.ylabel('Accuracy')
        plt.xlabel('Prediction Number')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Saved prediction history visualization to {filename}")
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_type": self.model_type,
            "model_version": self.metadata.get("model_version", "Unknown"),
            "training_date": self.metadata.get("training_date", "Unknown"),
            "performance": self.metadata.get("performance_metrics", {}),
            "features": {
                "count": len(self.feature_list) if self.feature_list else 0,
                "list": self.feature_list
            },
            "preprocessing": self.metadata.get("preprocessing", {}),
            "parameters": self.metadata.get("model_params", {})
        }
    
    def list_available_models(self) -> List[Dict]:
        """List all available model versions in the model directory."""
        try:
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.startswith(self.model_name) and f.endswith('.joblib')]
            
            models = []
            for model_file in model_files:
                # Extract version
                version = model_file.replace(f"{self.model_name}_", "").replace(".joblib", "")
                
                # Check for metadata
                metadata_path = self._get_metadata_path(version)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    models.append({
                        "version": version,
                        "training_date": metadata.get("training_date", "Unknown"),
                        "accuracy": metadata.get("performance_metrics", {}).get("accuracy", 0),
                        "f1_score": metadata.get("performance_metrics", {}).get("f1_score", 0)
                    })
                else:
                    models.append({
                        "version": version,
                        "training_date": "Unknown",
                        "accuracy": 0,
                        "f1_score": 0
                    })
            
            # Sort by version (newest first)
            return sorted(models, key=lambda x: x["version"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def load_model_version(self, version: str) -> bool:
        """
        Load a specific model version.
        
        Args:
            version: Version string of the model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_path = self._get_model_path(version)
            metadata_path = self._get_metadata_path(version)
            
            if not os.path.exists(model_path):
                logger.error(f"Model version {version} not found")
                return False
            
            self.model = joblib.load(model_path)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_list = self.metadata.get("feature_list")
            
            logger.info(f"Loaded model version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model version {version}: {str(e)}")
            return False
    
    def evaluate_on_data(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
        """
        Evaluate the model on new data.
        
        Args:
            X: Feature data
            y: True labels
            threshold: Probability threshold for positive class
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please train or load a model first.")
            
        # Validate input
        if not self._validate_input_data(X, y):
            raise ValueError("Invalid input data for evaluation")
            
        # Make predictions
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        conf_matrix = confusion_matrix(y, y_pred).tolist()
        
        # Store predictions for visualization
        predictions = []
        for i in range(len(X)):
            predictions.append({
                'actual': int(y.iloc[i]) if isinstance(y, pd.Series) else int(y[i]),
                'predicted': int(y_pred[i]),
                'probability': float(y_prob[i])
            })
            
        # Generate visualization
        self.visualize_prediction_history(predictions, f"{self.model_name}_evaluation.png")
        
        # Return results
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix,
            "predictions": predictions
        }
    
    def backtest(self, df: pd.DataFrame, feature_processor, window_size: int = 500, step_size: int = 100) -> Dict:
        """
        Perform backtesting by creating a series of predictions on historical data.
        
        Args:
            df: Historical OHLCV data
            feature_processor: DataProcessor instance to prepare features
            window_size: Number of candles to use in each training window
            step_size: Number of candles to step forward for each test
            
        Returns:
            dict: Backtesting results
        """
        if len(df) < window_size + 100:
            raise ValueError(f"Not enough data for backtesting. Need at least {window_size + 100} rows.")
            
        logger.info(f"Starting backtesting with window size {window_size}, step size {step_size}")
        
        results = []
        
        # Loop through the data in steps
        for start_idx in range(0, len(df) - window_size - 50, step_size):
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
                
                training_result = window_model.train(
                    X_train, y_train,
                    feature_list=feature_list,
                    optimize_hyperparams=False,  # Skip optimization for speed
                    test_size=0.2
                )
                
                # Create features for test data (one at a time to simulate real-time prediction)
                test_predictions = []
                for i in range(len(test_df) - 1):  # -1 because we need the next candle for validation
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
                window_accuracy = sum(p['correct'] for p in test_predictions) / len(test_predictions)
                
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
                logger.info(f"Completed backtest window {start_idx}-{end_idx}, test accuracy: {window_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error in backtest window {start_idx}-{end_idx}: {str(e)}")
                continue
        
        # Calculate overall metrics
        all_predictions = []
        for window in results:
            all_predictions.extend(window['test_predictions'])
            
        overall_accuracy = sum(p['correct'] for p in all_predictions) / len(all_predictions)
        
        # Create visualization of backtest results
        self._visualize_backtest_results(results)
        
        return {
            'overall_accuracy': overall_accuracy,
            'windows_count': len(results),
            'predictions_count': len(all_predictions),
            'window_results': results
        }
    
    def _visualize_backtest_results(self, results: List[Dict], filename: str = "backtest_results.png"):
        """
        Visualize backtest results.
        
        Args:
            results: List of window results
            filename: Path to save the visualization
        """
        if not results:
            logger.warning("No backtest results to visualize")
            return
            
        # Extract data for plotting
        window_starts = [r['window_start'] for r in results]
        train_accuracies = [r['train_accuracy'] for r in results]
        test_accuracies = [r['test_accuracy'] for r in results]
        
        # Flatten predictions for overall view
        all_predictions = []
        for window in results:
            all_predictions.extend(window['test_predictions'])
            
        # Sort by test_index
        all_predictions.sort(key=lambda x: x['test_index'])
        
        # Calculate cumulative accuracy
        correct_count = 0
        cumulative_accuracies = []
        
        for i, pred in enumerate(all_predictions):
            if pred['correct']:
                correct_count += 1
            cumulative_accuracies.append(correct_count / (i + 1))
        
        # Create visualization
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Training vs Testing accuracy by window
        plt.subplot(3, 1, 1)
        plt.plot(window_starts, train_accuracies, 'o-', label='Training Accuracy')
        plt.plot(window_starts, test_accuracies, 'x-', label='Testing Accuracy')
        plt.title('Training vs Testing Accuracy by Window', fontsize=14)
        plt.xlabel('Window Start Index')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Predictions vs Actuals
        plt.subplot(3, 1, 2)
        indices = [p['test_index'] for p in all_predictions]
        actuals = [p['actual'] for p in all_predictions]
        predictions = [p['prediction'] for p in all_predictions]
        
        plt.plot(indices, actuals, 'o-', label='Actual', alpha=0.5)
        plt.plot(indices, predictions, 'x-', label='Predicted', alpha=0.5)
        plt.title('Actual vs Predicted Values Across All Windows', fontsize=14)
        plt.xlabel('Data Index')
        plt.ylabel('Class (0=Bearish, 1=Bullish)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Accuracy
        plt.subplot(3, 1, 3)
        plt.plot(indices, cumulative_accuracies, 'o-', color='purple')
        plt.title('Cumulative Prediction Accuracy', fontsize=14)
        plt.xlabel('Data Index')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        
        # Add overall metrics as text
        plt.figtext(0.5, 0.01, 
                  f"Overall Accuracy: {sum(p['correct'] for p in all_predictions) / len(all_predictions):.4f} | "
                  f"Total Windows: {len(results)} | "
                  f"Total Predictions: {len(all_predictions)}", 
                  ha="center", fontsize=12, 
                  bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Saved backtest results visualization to {filename}")
