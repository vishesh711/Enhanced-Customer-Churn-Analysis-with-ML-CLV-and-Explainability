"""
Model training and evaluation module for Customer Churn ML Pipeline
Handles model training, hyperparameter tuning, calibration, and evaluation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
import joblib
import json

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

# Optional imports for advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

from config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for trained models"""
    model_id: str
    model_type: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    data_hash: str
    cv_scores: Dict[str, List[float]]
    calibrated: bool = False
    calibration_method: Optional[str] = None


@dataclass
class CrossValidationResults:
    """Results from cross-validation"""
    cv_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_predictions: List[np.ndarray]
    fold_probabilities: List[np.ndarray]


class ModelTrainer:
    """
    Handles ML model training including baseline models, ensemble models,
    and hyperparameter tuning with cross-validation
    """
    
    def __init__(self, random_state: Optional[int] = None, n_jobs: int = -1):
        """
        Initialize ModelTrainer
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for training
        """
        self.random_state = random_state or config.RANDOM_SEED
        self.n_jobs = n_jobs
        self.trained_models = {}
        self.model_metadata = {}
        self.cv_results = {}
        
        logger.info(f"ModelTrainer initialized with random_state={self.random_state}")
    
    def train_baseline_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, BaseEstimator]:
        """
        Train baseline models including majority class and logistic regression
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of trained baseline models
        """
        logger.info("Training baseline models")
        
        baseline_models = {}
        
        # 1. Majority class classifier
        majority_clf = DummyClassifier(
            strategy="most_frequent",
            random_state=self.random_state
        )
        majority_clf.fit(X, y)
        baseline_models['majority_class'] = majority_clf
        
        # 2. Stratified classifier (random with class distribution)
        stratified_clf = DummyClassifier(
            strategy="stratified",
            random_state=self.random_state
        )
        stratified_clf.fit(X, y)
        baseline_models['stratified'] = stratified_clf
        
        # 3. Logistic regression with balanced class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y), 
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        logistic_clf = LogisticRegression(
            class_weight=class_weight_dict,
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'
        )
        logistic_clf.fit(X, y)
        baseline_models['logistic_balanced'] = logistic_clf
        
        # 4. Simple logistic regression (no class balancing)
        simple_logistic_clf = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'
        )
        simple_logistic_clf.fit(X, y)
        baseline_models['logistic_simple'] = simple_logistic_clf
        
        # Store models
        self.trained_models.update(baseline_models)
        
        # Evaluate baseline models
        for name, model in baseline_models.items():
            cv_results = self._evaluate_model_cv(model, X, y, cv_folds=3)
            self.cv_results[name] = cv_results
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type=name,
                training_date=datetime.now(),
                performance_metrics=cv_results.mean_scores,
                hyperparameters=model.get_params(),
                feature_names=X.columns.tolist() if hasattr(X, 'columns') else [],
                data_hash=self._compute_data_hash(X, y),
                cv_scores=cv_results.cv_scores
            )
            self.model_metadata[name] = metadata
        
        logger.info(f"Trained {len(baseline_models)} baseline models")
        return baseline_models
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                            param_grids: Optional[Dict[str, Dict]] = None) -> Dict[str, BaseEstimator]:
        """
        Train ensemble models (RandomForest, XGBoost, LightGBM) with hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grids: Custom parameter grids for hyperparameter tuning
            
        Returns:
            Dictionary of trained ensemble models
        """
        logger.info("Training ensemble models with hyperparameter tuning")
        
        ensemble_models = {}
        
        # Default parameter grids
        default_param_grids = self._get_default_param_grids()
        if param_grids:
            default_param_grids.update(param_grids)
        
        # 1. Random Forest
        rf_model = self._train_random_forest(X, y, default_param_grids.get('random_forest', {}))
        if rf_model:
            ensemble_models['random_forest'] = rf_model
        
        # 2. XGBoost (if available)
        if HAS_XGBOOST:
            xgb_model = self._train_xgboost(X, y, default_param_grids.get('xgboost', {}))
            if xgb_model:
                ensemble_models['xgboost'] = xgb_model
        else:
            logger.warning("XGBoost not available, skipping")
        
        # 3. LightGBM (if available)
        if HAS_LIGHTGBM:
            lgb_model = self._train_lightgbm(X, y, default_param_grids.get('lightgbm', {}))
            if lgb_model:
                ensemble_models['lightgbm'] = lgb_model
        else:
            logger.warning("LightGBM not available, skipping")
        
        # Store models
        self.trained_models.update(ensemble_models)
        
        logger.info(f"Trained {len(ensemble_models)} ensemble models")
        return ensemble_models
    
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           param_grid: Dict[str, Any]) -> Optional[RandomForestClassifier]:
        """Train Random Forest with hyperparameter tuning"""
        logger.info("Training Random Forest")
        
        # Default parameters if not provided
        if not param_grid:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            }
        
        # Base model
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Hyperparameter tuning
        best_rf = self._tune_hyperparameters(rf, param_grid, X, y, 'random_forest')
        
        return best_rf
    
    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series,
                      param_grid: Dict[str, Any]) -> Optional[BaseEstimator]:
        """Train XGBoost with hyperparameter tuning"""
        logger.info("Training XGBoost")
        
        # Default parameters if not provided
        if not param_grid:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        
        # Base model
        xgb_clf = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )
        
        # Hyperparameter tuning
        best_xgb = self._tune_hyperparameters(xgb_clf, param_grid, X, y, 'xgboost')
        
        return best_xgb
    
    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                       param_grid: Dict[str, Any]) -> Optional[BaseEstimator]:
        """Train LightGBM with hyperparameter tuning"""
        logger.info("Training LightGBM")
        
        # Default parameters if not provided
        if not param_grid:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # Base model
        lgb_clf = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight='balanced',
            verbose=-1
        )
        
        # Hyperparameter tuning
        best_lgb = self._tune_hyperparameters(lgb_clf, param_grid, X, y, 'lightgbm')
        
        return best_lgb
    
    def _tune_hyperparameters(self, base_model: BaseEstimator, param_grid: Dict[str, Any],
                            X: pd.DataFrame, y: pd.Series, model_name: str) -> BaseEstimator:
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        
        Args:
            base_model: Base model to tune
            param_grid: Parameter grid for tuning
            X: Feature matrix
            y: Target variable
            model_name: Name of the model for logging
            
        Returns:
            Best model after tuning
        """
        logger.info(f"Tuning hyperparameters for {model_name}")
        
        # Use RandomizedSearchCV for efficiency
        cv_folds = StratifiedKFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=self.random_state
        )
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=min(50, len(param_grid)),  # Limit iterations
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        search.fit(X, y)
        
        best_model = search.best_estimator_
        
        logger.info(f"Best {model_name} parameters: {search.best_params_}")
        logger.info(f"Best {model_name} CV score: {search.best_score_:.4f}")
        
        # Evaluate best model with cross-validation
        cv_results = self._evaluate_model_cv(best_model, X, y)
        self.cv_results[model_name] = cv_results
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=model_name,
            training_date=datetime.now(),
            performance_metrics=cv_results.mean_scores,
            hyperparameters=best_model.get_params(),
            feature_names=X.columns.tolist() if hasattr(X, 'columns') else [],
            data_hash=self._compute_data_hash(X, y),
            cv_scores=cv_results.cv_scores
        )
        self.model_metadata[model_name] = metadata
        
        return best_model
    
    def _evaluate_model_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                          cv_folds: int = None) -> CrossValidationResults:
        """
        Evaluate model using cross-validation
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            
        Returns:
            CrossValidationResults object
        """
        cv_folds = cv_folds or config.CV_FOLDS
        
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Define scoring metrics
        scoring = {
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=self.n_jobs
        )
        
        # Extract scores
        cv_scores = {}
        mean_scores = {}
        std_scores = {}
        
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            cv_scores[metric] = scores.tolist()
            mean_scores[metric] = scores.mean()
            std_scores[metric] = scores.std()
        
        # Get fold predictions for additional analysis
        fold_predictions = []
        fold_probabilities = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and fit model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Get predictions
            predictions = fold_model.predict(X_val)
            probabilities = fold_model.predict_proba(X_val)[:, 1]
            
            fold_predictions.append(predictions)
            fold_probabilities.append(probabilities)
        
        return CrossValidationResults(
            cv_scores=cv_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            fold_predictions=fold_predictions,
            fold_probabilities=fold_probabilities
        )
    
    def _get_default_param_grids(self) -> Dict[str, Dict[str, Any]]:
        """Get default parameter grids for hyperparameter tuning"""
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
    
    def _compute_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Compute hash of the training data for reproducibility tracking"""
        import hashlib
        
        # Combine X and y for hashing
        data_str = f"{X.shape}_{y.shape}_{X.sum().sum()}_{y.sum()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all trained models
        
        Returns:
            DataFrame with model performance comparison
        """
        if not self.cv_results:
            logger.warning("No models have been trained yet")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, cv_results in self.cv_results.items():
            row = {'model': model_name}
            row.update(cv_results.mean_scores)
            
            # Add standard deviations
            for metric, std_val in cv_results.std_scores.items():
                row[f'{metric}_std'] = std_val
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by ROC-AUC (descending)
        if 'roc_auc' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, BaseEstimator]:
        """
        Get the best model based on specified metric
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_name, model)
        """
        if not self.cv_results:
            raise ValueError("No models have been trained yet")
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name, cv_results in self.cv_results.items():
            if metric in cv_results.mean_scores:
                score = cv_results.mean_scores[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Metric '{metric}' not found in CV results")
        
        best_model = self.trained_models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with {metric}={best_score:.4f}")
        
        return best_model_name, best_model
    
    def save_models(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save all trained models to disk
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary mapping model names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for model_name, model in self.trained_models.items():
            # Save model
            model_path = output_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            saved_paths[f"{model_name}_model"] = str(model_path)
            
            # Save metadata
            if model_name in self.model_metadata:
                metadata_path = output_dir / f"{model_name}_metadata.json"
                metadata = self.model_metadata[model_name]
                
                # Convert to serializable format
                metadata_dict = {
                    'model_id': metadata.model_id,
                    'model_type': metadata.model_type,
                    'training_date': metadata.training_date.isoformat(),
                    'performance_metrics': metadata.performance_metrics,
                    'hyperparameters': metadata.hyperparameters,
                    'feature_names': metadata.feature_names,
                    'data_hash': metadata.data_hash,
                    'cv_scores': metadata.cv_scores,
                    'calibrated': metadata.calibrated,
                    'calibration_method': metadata.calibration_method
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
                
                saved_paths[f"{model_name}_metadata"] = str(metadata_path)
        
        logger.info(f"Saved {len(self.trained_models)} models to {output_dir}")
        
        return saved_paths
    
    @classmethod
    def load_models(cls, model_dir: Union[str, Path]) -> 'ModelTrainer':
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            ModelTrainer instance with loaded models
        """
        model_dir = Path(model_dir)
        
        trainer = cls()
        
        # Load models
        for model_file in model_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace('_model', '')
            model = joblib.load(model_file)
            trainer.trained_models[model_name] = model
            
            # Load metadata if available
            metadata_file = model_dir / f"{model_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = ModelMetadata(
                    model_id=metadata_dict['model_id'],
                    model_type=metadata_dict['model_type'],
                    training_date=datetime.fromisoformat(metadata_dict['training_date']),
                    performance_metrics=metadata_dict['performance_metrics'],
                    hyperparameters=metadata_dict['hyperparameters'],
                    feature_names=metadata_dict['feature_names'],
                    data_hash=metadata_dict['data_hash'],
                    cv_scores=metadata_dict['cv_scores'],
                    calibrated=metadata_dict.get('calibrated', False),
                    calibration_method=metadata_dict.get('calibration_method')
                )
                
                trainer.model_metadata[model_name] = metadata
        
        logger.info(f"Loaded {len(trainer.trained_models)} models from {model_dir}")
        
        return trainer

class ProbabilityCalibrator:
    """
    Handles probability calibration using CalibratedClassifierCV
    Implements isotonic and sigmoid calibration methods with validation
    """
    
    def __init__(self, method: str = 'isotonic', cv: int = 3):
        """
        Initialize ProbabilityCalibrator
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of cross-validation folds for calibration
        """
        self.method = method
        self.cv = cv
        self.calibrated_models = {}
        self.calibration_curves = {}
        
        logger.info(f"ProbabilityCalibrator initialized with method={method}, cv={cv}")
    
    def calibrate_model(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                       model_name: str) -> CalibratedClassifierCV:
        """
        Calibrate a trained model's probabilities
        
        Args:
            model: Trained model to calibrate
            X: Feature matrix for calibration
            y: Target variable for calibration
            model_name: Name of the model for tracking
            
        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model {model_name} using {self.method} method")
        
        # Create calibrated classifier
        calibrated_clf = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=self.cv
        )
        
        # Fit calibrated classifier
        calibrated_clf.fit(X, y)
        
        # Store calibrated model
        self.calibrated_models[model_name] = calibrated_clf
        
        # Generate calibration curve for analysis
        self._generate_calibration_curve(model, calibrated_clf, X, y, model_name)
        
        logger.info(f"Model {model_name} calibrated successfully")
        
        return calibrated_clf
    
    def calibrate_all_models(self, models: Dict[str, BaseEstimator], 
                           X: pd.DataFrame, y: pd.Series) -> Dict[str, CalibratedClassifierCV]:
        """
        Calibrate multiple models
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix for calibration
            y: Target variable for calibration
            
        Returns:
            Dictionary of calibrated models
        """
        logger.info(f"Calibrating {len(models)} models")
        
        calibrated_models = {}
        
        for model_name, model in models.items():
            try:
                calibrated_model = self.calibrate_model(model, X, y, model_name)
                calibrated_models[model_name] = calibrated_model
            except Exception as e:
                logger.error(f"Failed to calibrate model {model_name}: {str(e)}")
        
        return calibrated_models
    
    def _generate_calibration_curve(self, original_model: BaseEstimator, 
                                  calibrated_model: CalibratedClassifierCV,
                                  X: pd.DataFrame, y: pd.Series, model_name: str) -> None:
        """
        Generate calibration curve data for analysis
        
        Args:
            original_model: Original uncalibrated model
            calibrated_model: Calibrated model
            X: Feature matrix
            y: Target variable
            model_name: Name of the model
        """
        # Get probabilities from both models
        original_probs = original_model.predict_proba(X)[:, 1]
        calibrated_probs = calibrated_model.predict_proba(X)[:, 1]
        
        # Generate calibration curves
        original_fraction_pos, original_mean_pred = calibration_curve(
            y, original_probs, n_bins=10
        )
        calibrated_fraction_pos, calibrated_mean_pred = calibration_curve(
            y, calibrated_probs, n_bins=10
        )
        
        # Store calibration curve data
        self.calibration_curves[model_name] = {
            'original': {
                'fraction_positive': original_fraction_pos,
                'mean_predicted': original_mean_pred,
                'probabilities': original_probs
            },
            'calibrated': {
                'fraction_positive': calibrated_fraction_pos,
                'mean_predicted': calibrated_mean_pred,
                'probabilities': calibrated_probs
            }
        }
        
        # Calculate calibration metrics
        original_brier = brier_score_loss(y, original_probs)
        calibrated_brier = brier_score_loss(y, calibrated_probs)
        
        logger.info(f"Calibration metrics for {model_name}:")
        logger.info(f"  Original Brier Score: {original_brier:.4f}")
        logger.info(f"  Calibrated Brier Score: {calibrated_brier:.4f}")
        logger.info(f"  Improvement: {original_brier - calibrated_brier:.4f}")
    
    def plot_calibration_curves(self, model_names: Optional[List[str]] = None,
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot calibration curves for specified models
        
        Args:
            model_names: List of model names to plot (if None, plot all)
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if model_names is None:
            model_names = list(self.calibration_curves.keys())
        
        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.calibration_curves:
                logger.warning(f"No calibration curve data for {model_name}")
                continue
            
            ax = axes[i]
            curve_data = self.calibration_curves[model_name]
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            # Plot original model
            ax.plot(
                curve_data['original']['mean_predicted'],
                curve_data['original']['fraction_positive'],
                marker='o', label='Original model'
            )
            
            # Plot calibrated model
            ax.plot(
                curve_data['calibrated']['mean_predicted'],
                curve_data['calibrated']['fraction_positive'],
                marker='s', label='Calibrated model'
            )
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        
        plt.show()
    
    def evaluate_calibration_quality(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Evaluate calibration quality using various metrics
        
        Args:
            model_names: List of model names to evaluate (if None, evaluate all)
            
        Returns:
            DataFrame with calibration quality metrics
        """
        if model_names is None:
            model_names = list(self.calibration_curves.keys())
        
        results = []
        
        for model_name in model_names:
            if model_name not in self.calibration_curves:
                continue
            
            curve_data = self.calibration_curves[model_name]
            
            # Calculate Expected Calibration Error (ECE)
            original_ece = self._calculate_ece(
                curve_data['original']['fraction_positive'],
                curve_data['original']['mean_predicted']
            )
            calibrated_ece = self._calculate_ece(
                curve_data['calibrated']['fraction_positive'],
                curve_data['calibrated']['mean_predicted']
            )
            
            # Calculate Maximum Calibration Error (MCE)
            original_mce = self._calculate_mce(
                curve_data['original']['fraction_positive'],
                curve_data['original']['mean_predicted']
            )
            calibrated_mce = self._calculate_mce(
                curve_data['calibrated']['fraction_positive'],
                curve_data['calibrated']['mean_predicted']
            )
            
            results.append({
                'model': model_name,
                'original_ece': original_ece,
                'calibrated_ece': calibrated_ece,
                'ece_improvement': original_ece - calibrated_ece,
                'original_mce': original_mce,
                'calibrated_mce': calibrated_mce,
                'mce_improvement': original_mce - calibrated_mce
            })
        
        return pd.DataFrame(results)
    
    def _calculate_ece(self, fraction_positive: np.ndarray, 
                      mean_predicted: np.ndarray) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            fraction_positive: Actual fraction of positive samples in each bin
            mean_predicted: Mean predicted probability in each bin
            
        Returns:
            Expected Calibration Error
        """
        # Assume equal bin sizes for simplicity
        bin_weights = np.ones(len(fraction_positive)) / len(fraction_positive)
        ece = np.sum(bin_weights * np.abs(fraction_positive - mean_predicted))
        return ece
    
    def _calculate_mce(self, fraction_positive: np.ndarray,
                      mean_predicted: np.ndarray) -> float:
        """
        Calculate Maximum Calibration Error (MCE)
        
        Args:
            fraction_positive: Actual fraction of positive samples in each bin
            mean_predicted: Mean predicted probability in each bin
            
        Returns:
            Maximum Calibration Error
        """
        mce = np.max(np.abs(fraction_positive - mean_predicted))
        return mce
    
    def compare_calibration_methods(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                  model_name: str) -> Dict[str, Any]:
        """
        Compare isotonic and sigmoid calibration methods
        
        Args:
            model: Model to calibrate
            X: Feature matrix
            y: Target variable
            model_name: Name of the model
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing calibration methods for {model_name}")
        
        results = {}
        
        for method in ['isotonic', 'sigmoid']:
            # Create calibrated classifier
            calibrated_clf = CalibratedClassifierCV(
                estimator=clone(model),
                method=method,
                cv=self.cv
            )
            
            # Fit and get probabilities
            calibrated_clf.fit(X, y)
            calibrated_probs = calibrated_clf.predict_proba(X)[:, 1]
            
            # Calculate metrics
            brier_score = brier_score_loss(y, calibrated_probs)
            log_loss_score = log_loss(y, calibrated_probs)
            
            # Calculate calibration curve
            fraction_pos, mean_pred = calibration_curve(y, calibrated_probs, n_bins=10)
            ece = self._calculate_ece(fraction_pos, mean_pred)
            mce = self._calculate_mce(fraction_pos, mean_pred)
            
            results[method] = {
                'brier_score': brier_score,
                'log_loss': log_loss_score,
                'ece': ece,
                'mce': mce,
                'calibrated_model': calibrated_clf
            }
        
        # Determine best method
        best_method = min(results.keys(), key=lambda x: results[x]['brier_score'])
        results['best_method'] = best_method
        
        logger.info(f"Best calibration method for {model_name}: {best_method}")
        
        return results
    
    def get_calibrated_models(self) -> Dict[str, CalibratedClassifierCV]:
        """
        Get all calibrated models
        
        Returns:
            Dictionary of calibrated models
        """
        return self.calibrated_models.copy()
    
    def save_calibrated_models(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save calibrated models to disk
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary mapping model names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for model_name, calibrated_model in self.calibrated_models.items():
            model_path = output_dir / f"{model_name}_calibrated.joblib"
            joblib.dump(calibrated_model, model_path)
            saved_paths[model_name] = str(model_path)
            
            logger.info(f"Saved calibrated model {model_name} to {model_path}")
        
        # Save calibration curve data
        curves_path = output_dir / "calibration_curves.json"
        with open(curves_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_curves = {}
            for model_name, curve_data in self.calibration_curves.items():
                serializable_curves[model_name] = {
                    'original': {
                        'fraction_positive': curve_data['original']['fraction_positive'].tolist(),
                        'mean_predicted': curve_data['original']['mean_predicted'].tolist()
                    },
                    'calibrated': {
                        'fraction_positive': curve_data['calibrated']['fraction_positive'].tolist(),
                        'mean_predicted': curve_data['calibrated']['mean_predicted'].tolist()
                    }
                }
            
            json.dump(serializable_curves, f, indent=2)
        
        saved_paths['calibration_curves'] = str(curves_path)
        
        logger.info(f"Saved {len(self.calibrated_models)} calibrated models to {output_dir}")
        
        return saved_paths
class ThresholdOptimizer:
    """
    Implements cost-sensitive threshold optimization for business-aware decision making
    Creates profit curve analysis and sensitivity analysis for business parameters
    """
    
    def __init__(self, business_config: Optional[Dict[str, float]] = None):
        """
        Initialize ThresholdOptimizer
        
        Args:
            business_config: Dictionary with business parameters
                - retention_value: Value of retaining a customer
                - contact_cost: Cost of contacting a customer
                - churn_cost: Cost of losing a customer (optional)
        """
        self.business_config = business_config or config.get_business_config()
        
        self.retention_value = self.business_config.get('retention_value', 1000.0)
        self.contact_cost = self.business_config.get('contact_cost', 50.0)
        self.churn_cost = self.business_config.get('churn_cost', 500.0)
        
        self.optimal_thresholds = {}
        self.profit_curves = {}
        
        logger.info(f"ThresholdOptimizer initialized with retention_value={self.retention_value}, "
                   f"contact_cost={self.contact_cost}, churn_cost={self.churn_cost}")
    
    def optimize_business_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                                  model_name: str = 'model') -> Tuple[float, Dict[str, float]]:
        """
        Find optimal probability threshold based on business cost considerations
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model for tracking
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        logger.info(f"Optimizing business threshold for {model_name}")
        
        # Generate range of thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        
        profits = []
        metrics_list = []
        
        for threshold in thresholds:
            # Make predictions at this threshold
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
            fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
            tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
            fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
            
            # Calculate business metrics
            profit = self._calculate_profit(tp, fp, tn, fn)
            profits.append(profit)
            
            # Calculate standard metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'threshold': threshold,
                'profit': profit,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
            metrics_list.append(metrics)
        
        # Find optimal threshold (maximum profit)
        optimal_idx = np.argmax(profits)
        optimal_threshold = thresholds[optimal_idx]
        optimal_metrics = metrics_list[optimal_idx]
        
        # Store results
        self.optimal_thresholds[model_name] = optimal_threshold
        self.profit_curves[model_name] = {
            'thresholds': thresholds,
            'profits': profits,
            'metrics': metrics_list
        }
        
        logger.info(f"Optimal threshold for {model_name}: {optimal_threshold:.3f}")
        logger.info(f"Expected profit: ${optimal_metrics['profit']:,.2f}")
        
        return optimal_threshold, optimal_metrics
    
    def _calculate_profit(self, tp: int, fp: int, tn: int, fn: int) -> float:
        """
        Calculate expected profit based on confusion matrix and business parameters
        
        Args:
            tp: True Positives
            fp: False Positives  
            tn: True Negatives
            fn: False Negatives
            
        Returns:
            Expected profit
        """
        # Revenue from successfully retaining customers who would churn
        retention_revenue = tp * self.retention_value
        
        # Cost of contacting customers (both TP and FP)
        contact_costs = (tp + fp) * self.contact_cost
        
        # Cost of customers who churn despite not being contacted (FN)
        # This is an opportunity cost
        churn_costs = fn * self.churn_cost
        
        # Total profit = Revenue - Costs
        total_profit = retention_revenue - contact_costs - churn_costs
        
        return total_profit
    
    def plot_profit_curves(self, model_names: Optional[List[str]] = None,
                          save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot profit curves for specified models
        
        Args:
            model_names: List of model names to plot (if None, plot all)
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if model_names is None:
            model_names = list(self.profit_curves.keys())
        
        plt.figure(figsize=(12, 8))
        
        for model_name in model_names:
            if model_name not in self.profit_curves:
                logger.warning(f"No profit curve data for {model_name}")
                continue
            
            curve_data = self.profit_curves[model_name]
            thresholds = curve_data['thresholds']
            profits = curve_data['profits']
            
            plt.plot(thresholds, profits, marker='o', label=model_name, linewidth=2)
            
            # Mark optimal threshold
            if model_name in self.optimal_thresholds:
                optimal_threshold = self.optimal_thresholds[model_name]
                optimal_profit = max(profits)
                plt.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
                plt.annotate(f'Optimal: {optimal_threshold:.3f}', 
                           xy=(optimal_threshold, optimal_profit),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Probability Threshold')
        plt.ylabel('Expected Profit ($)')
        plt.title('Profit Curves by Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Profit curves saved to {save_path}")
        
        plt.show()
    
    def sensitivity_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                           parameter_ranges: Optional[Dict[str, List[float]]] = None,
                           model_name: str = 'model') -> pd.DataFrame:
        """
        Perform sensitivity analysis for business parameters
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            parameter_ranges: Dictionary with parameter ranges to test
            model_name: Name of the model
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        logger.info(f"Performing sensitivity analysis for {model_name}")
        
        # Default parameter ranges
        if parameter_ranges is None:
            parameter_ranges = {
                'retention_value': [500, 750, 1000, 1250, 1500],
                'contact_cost': [25, 37.5, 50, 62.5, 75],
                'churn_cost': [250, 375, 500, 625, 750]
            }
        
        results = []
        
        # Store original values
        original_retention = self.retention_value
        original_contact = self.contact_cost
        original_churn = self.churn_cost
        
        # Test each parameter
        for param_name, param_values in parameter_ranges.items():
            for param_value in param_values:
                # Set parameter value
                if param_name == 'retention_value':
                    self.retention_value = param_value
                elif param_name == 'contact_cost':
                    self.contact_cost = param_value
                elif param_name == 'churn_cost':
                    self.churn_cost = param_value
                
                # Find optimal threshold with this parameter value
                optimal_threshold, optimal_metrics = self.optimize_business_threshold(
                    y_true, y_prob, f"{model_name}_sensitivity"
                )
                
                results.append({
                    'parameter': param_name,
                    'parameter_value': param_value,
                    'optimal_threshold': optimal_threshold,
                    'optimal_profit': optimal_metrics['profit'],
                    'precision': optimal_metrics['precision'],
                    'recall': optimal_metrics['recall'],
                    'f1': optimal_metrics['f1']
                })
        
        # Restore original values
        self.retention_value = original_retention
        self.contact_cost = original_contact
        self.churn_cost = original_churn
        
        sensitivity_df = pd.DataFrame(results)
        
        logger.info("Sensitivity analysis completed")
        
        return sensitivity_df
    
    def plot_sensitivity_analysis(self, sensitivity_df: pd.DataFrame,
                                save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot sensitivity analysis results
        
        Args:
            sensitivity_df: DataFrame from sensitivity_analysis method
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        parameters = sensitivity_df['parameter'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(parameters):
            if i >= len(axes):
                break
            
            param_data = sensitivity_df[sensitivity_df['parameter'] == param]
            
            ax = axes[i]
            
            # Plot optimal threshold
            ax.plot(param_data['parameter_value'], param_data['optimal_threshold'], 
                   'bo-', label='Optimal Threshold')
            ax.set_xlabel(f'{param.replace("_", " ").title()}')
            ax.set_ylabel('Optimal Threshold')
            ax.set_title(f'Threshold Sensitivity to {param.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Add secondary y-axis for profit
            ax2 = ax.twinx()
            ax2.plot(param_data['parameter_value'], param_data['optimal_profit'], 
                    'ro-', label='Optimal Profit')
            ax2.set_ylabel('Optimal Profit ($)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Remove unused subplots
        for j in range(len(parameters), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sensitivity analysis plot saved to {save_path}")
        
        plt.show()
    
    def calculate_expected_value(self, y_true: np.ndarray, y_prob: np.ndarray,
                               threshold: Optional[float] = None,
                               model_name: str = 'model') -> Dict[str, float]:
        """
        Calculate expected value metrics for a given threshold
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            threshold: Probability threshold (if None, use optimal)
            model_name: Name of the model
            
        Returns:
            Dictionary with expected value metrics
        """
        if threshold is None:
            if model_name in self.optimal_thresholds:
                threshold = self.optimal_thresholds[model_name]
            else:
                threshold, _ = self.optimize_business_threshold(y_true, y_prob, model_name)
        
        # Make predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        total_customers = len(y_true)
        
        # Calculate per-customer metrics
        profit_per_customer = self._calculate_profit(tp, fp, tn, fn) / total_customers
        
        # Calculate retention rate and contact rate
        contact_rate = (tp + fp) / total_customers
        retention_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate cost breakdown
        total_retention_revenue = tp * self.retention_value
        total_contact_costs = (tp + fp) * self.contact_cost
        total_churn_costs = fn * self.churn_cost
        
        return {
            'threshold': threshold,
            'total_profit': self._calculate_profit(tp, fp, tn, fn),
            'profit_per_customer': profit_per_customer,
            'contact_rate': contact_rate,
            'retention_rate': retention_rate,
            'total_retention_revenue': total_retention_revenue,
            'total_contact_costs': total_contact_costs,
            'total_churn_costs': total_churn_costs,
            'customers_contacted': tp + fp,
            'customers_retained': tp,
            'customers_lost': fn,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def compare_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray,
                          thresholds: List[float], model_name: str = 'model') -> pd.DataFrame:
        """
        Compare multiple thresholds and their business impact
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to compare
            model_name: Name of the model
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(thresholds)} thresholds for {model_name}")
        
        results = []
        
        for threshold in thresholds:
            metrics = self.calculate_expected_value(y_true, y_prob, threshold, model_name)
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        
        # Sort by profit (descending)
        comparison_df = comparison_df.sort_values('total_profit', ascending=False)
        
        return comparison_df
    
    def get_optimal_thresholds(self) -> Dict[str, float]:
        """
        Get optimal thresholds for all models
        
        Returns:
            Dictionary mapping model names to optimal thresholds
        """
        return self.optimal_thresholds.copy()
    
    def save_optimization_results(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save threshold optimization results to disk
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping result types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save optimal thresholds
        thresholds_path = output_dir / "optimal_thresholds.json"
        with open(thresholds_path, 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        saved_paths['optimal_thresholds'] = str(thresholds_path)
        
        # Save profit curves
        curves_path = output_dir / "profit_curves.json"
        with open(curves_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_curves = {}
            for model_name, curve_data in self.profit_curves.items():
                serializable_curves[model_name] = {
                    'thresholds': curve_data['thresholds'].tolist(),
                    'profits': curve_data['profits'],
                    'metrics': curve_data['metrics']
                }
            
            json.dump(serializable_curves, f, indent=2)
        saved_paths['profit_curves'] = str(curves_path)
        
        # Save business configuration
        config_path = output_dir / "business_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.business_config, f, indent=2)
        saved_paths['business_config'] = str(config_path)
        
        logger.info(f"Saved threshold optimization results to {output_dir}")
        
        return saved_paths
class ModelRegistry:
    """
    Handles model versioning, metadata tracking, and experiment logging
    Provides MLflow-compatible experiment tracking and model comparison
    """
    
    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        """
        Initialize ModelRegistry
        
        Args:
            registry_path: Path to store model registry (defaults to config.MODEL_PATH)
        """
        self.registry_path = Path(registry_path) if registry_path else config.MODEL_PATH
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_db_path = self.registry_path / "models_registry.json"
        self.experiments_db_path = self.registry_path / "experiments_registry.json"
        
        # Load existing registries
        self.models_db = self._load_models_db()
        self.experiments_db = self._load_experiments_db()
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def register_model(self, model: BaseEstimator, model_name: str, 
                      model_type: str, performance_metrics: Dict[str, float],
                      feature_names: List[str], hyperparameters: Dict[str, Any],
                      data_hash: str, experiment_id: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None,
                      description: Optional[str] = None) -> str:
        """
        Register a trained model with metadata
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type/algorithm of the model
            performance_metrics: Dictionary of performance metrics
            feature_names: List of feature names used for training
            hyperparameters: Model hyperparameters
            data_hash: Hash of the training data
            experiment_id: ID of the experiment (optional)
            tags: Additional tags for the model
            description: Model description
            
        Returns:
            Model ID (unique identifier)
        """
        # Generate unique model ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{model_name}_{model_type}_{timestamp}"
        
        # Create model directory
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model artifact
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Create model metadata
        metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'registration_date': datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'hyperparameters': hyperparameters,
            'feature_names': feature_names,
            'data_hash': data_hash,
            'experiment_id': experiment_id,
            'tags': tags or {},
            'description': description,
            'model_path': str(model_path),
            'status': 'registered',
            'version': self._get_next_version(model_name)
        }
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update models database
        self.models_db[model_id] = metadata
        self._save_models_db()
        
        logger.info(f"Registered model {model_id} (version {metadata['version']})")
        
        return model_id
    
    def register_experiment(self, experiment_name: str, models: Dict[str, str],
                          dataset_info: Dict[str, Any], 
                          experiment_config: Dict[str, Any],
                          results_summary: Dict[str, Any],
                          tags: Optional[Dict[str, str]] = None,
                          description: Optional[str] = None) -> str:
        """
        Register an experiment with multiple models
        
        Args:
            experiment_name: Name of the experiment
            models: Dictionary mapping model names to model IDs
            dataset_info: Information about the dataset used
            experiment_config: Configuration used for the experiment
            results_summary: Summary of experiment results
            tags: Additional tags for the experiment
            description: Experiment description
            
        Returns:
            Experiment ID
        """
        # Generate unique experiment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment metadata
        experiment_metadata = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'creation_date': datetime.now().isoformat(),
            'models': models,
            'dataset_info': dataset_info,
            'experiment_config': experiment_config,
            'results_summary': results_summary,
            'tags': tags or {},
            'description': description,
            'status': 'completed'
        }
        
        # Update experiments database
        self.experiments_db[experiment_id] = experiment_metadata
        self._save_experiments_db()
        
        logger.info(f"Registered experiment {experiment_id}")
        
        return experiment_id
    
    def load_model(self, model_id: str) -> BaseEstimator:
        """
        Load a registered model
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model object
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_metadata = self.models_db[model_id]
        model_path = model_metadata['model_path']
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        logger.info(f"Loaded model {model_id}")
        
        return model
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model metadata dictionary
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        return self.models_db[model_id].copy()
    
    def list_models(self, model_name: Optional[str] = None,
                   model_type: Optional[str] = None,
                   tags: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        List registered models with optional filtering
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            tags: Filter by tags
            
        Returns:
            DataFrame with model information
        """
        models_list = []
        
        for model_id, metadata in self.models_db.items():
            # Apply filters
            if model_name and metadata['model_name'] != model_name:
                continue
            if model_type and metadata['model_type'] != model_type:
                continue
            if tags:
                model_tags = metadata.get('tags', {})
                if not all(model_tags.get(k) == v for k, v in tags.items()):
                    continue
            
            # Extract key information
            model_info = {
                'model_id': model_id,
                'model_name': metadata['model_name'],
                'model_type': metadata['model_type'],
                'version': metadata['version'],
                'registration_date': metadata['registration_date'],
                'status': metadata['status']
            }
            
            # Add performance metrics
            for metric, value in metadata['performance_metrics'].items():
                model_info[f'metric_{metric}'] = value
            
            models_list.append(model_info)
        
        return pd.DataFrame(models_list)
    
    def compare_models(self, model_ids: List[str], 
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models across specified metrics
        
        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to compare (if None, use all available)
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_id in model_ids:
            if model_id not in self.models_db:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            
            metadata = self.models_db[model_id]
            
            row = {
                'model_id': model_id,
                'model_name': metadata['model_name'],
                'model_type': metadata['model_type'],
                'version': metadata['version']
            }
            
            # Add performance metrics
            performance_metrics = metadata['performance_metrics']
            if metrics:
                # Only include specified metrics
                for metric in metrics:
                    if metric in performance_metrics:
                        row[metric] = performance_metrics[metric]
            else:
                # Include all metrics
                row.update(performance_metrics)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def get_best_model(self, model_name: Optional[str] = None,
                      metric: str = 'roc_auc') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best model based on a specific metric
        
        Args:
            model_name: Filter by model name (optional)
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_id, metadata)
        """
        best_score = -np.inf
        best_model_id = None
        best_metadata = None
        
        for model_id, metadata in self.models_db.items():
            # Apply model name filter
            if model_name and metadata['model_name'] != model_name:
                continue
            
            # Check if metric exists
            if metric not in metadata['performance_metrics']:
                continue
            
            score = metadata['performance_metrics'][metric]
            if score > best_score:
                best_score = score
                best_model_id = model_id
                best_metadata = metadata
        
        if best_model_id is None:
            raise ValueError(f"No models found with metric '{metric}'")
        
        logger.info(f"Best model: {best_model_id} with {metric}={best_score:.4f}")
        
        return best_model_id, best_metadata
    
    def promote_model(self, model_id: str, stage: str = 'production') -> None:
        """
        Promote a model to a specific stage (e.g., staging, production)
        
        Args:
            model_id: ID of the model to promote
            stage: Stage to promote to
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Demote other models of the same name from this stage
        model_name = self.models_db[model_id]['model_name']
        for mid, metadata in self.models_db.items():
            if (metadata['model_name'] == model_name and 
                metadata.get('stage') == stage and mid != model_id):
                metadata['stage'] = 'archived'
                metadata['stage_change_date'] = datetime.now().isoformat()
        
        # Promote the specified model
        self.models_db[model_id]['stage'] = stage
        self.models_db[model_id]['stage_change_date'] = datetime.now().isoformat()
        
        self._save_models_db()
        
        logger.info(f"Promoted model {model_id} to {stage}")
    
    def archive_model(self, model_id: str) -> None:
        """
        Archive a model (mark as inactive)
        
        Args:
            model_id: ID of the model to archive
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self.models_db[model_id]['status'] = 'archived'
        self.models_db[model_id]['archive_date'] = datetime.now().isoformat()
        
        self._save_models_db()
        
        logger.info(f"Archived model {model_id}")
    
    def delete_model(self, model_id: str, force: bool = False) -> None:
        """
        Delete a model from the registry
        
        Args:
            model_id: ID of the model to delete
            force: Force deletion even if model is in production
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.models_db[model_id]
        
        # Check if model is in production
        if metadata.get('stage') == 'production' and not force:
            raise ValueError(f"Cannot delete production model {model_id}. Use force=True to override.")
        
        # Delete model files
        model_dir = Path(metadata['model_path']).parent
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Remove from database
        del self.models_db[model_id]
        self._save_models_db()
        
        logger.info(f"Deleted model {model_id}")
    
    def export_model_info(self, model_id: str, output_path: Union[str, Path]) -> None:
        """
        Export model information to a file
        
        Args:
            model_id: ID of the model to export
            output_path: Path to save the export
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.models_db[model_id]
        
        # Create export data
        export_data = {
            'model_metadata': metadata,
            'export_date': datetime.now().isoformat(),
            'registry_version': '1.0'
        }
        
        # Save to file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported model {model_id} info to {output_path}")
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get lineage information for a model (experiment, data, etc.)
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with lineage information
        """
        if model_id not in self.models_db:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.models_db[model_id]
        
        lineage = {
            'model_id': model_id,
            'model_name': metadata['model_name'],
            'model_type': metadata['model_type'],
            'data_hash': metadata['data_hash'],
            'experiment_id': metadata.get('experiment_id'),
            'feature_names': metadata['feature_names'],
            'hyperparameters': metadata['hyperparameters'],
            'registration_date': metadata['registration_date']
        }
        
        # Add experiment information if available
        experiment_id = metadata.get('experiment_id')
        if experiment_id and experiment_id in self.experiments_db:
            lineage['experiment_info'] = self.experiments_db[experiment_id]
        
        return lineage
    
    def _load_models_db(self) -> Dict[str, Any]:
        """Load models database from disk"""
        if self.models_db_path.exists():
            with open(self.models_db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_models_db(self) -> None:
        """Save models database to disk"""
        with open(self.models_db_path, 'w') as f:
            json.dump(self.models_db, f, indent=2)
    
    def _load_experiments_db(self) -> Dict[str, Any]:
        """Load experiments database from disk"""
        if self.experiments_db_path.exists():
            with open(self.experiments_db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiments_db(self) -> None:
        """Save experiments database to disk"""
        with open(self.experiments_db_path, 'w') as f:
            json.dump(self.experiments_db, f, indent=2)
    
    def _get_next_version(self, model_name: str) -> int:
        """Get the next version number for a model name"""
        max_version = 0
        for metadata in self.models_db.values():
            if metadata['model_name'] == model_name:
                max_version = max(max_version, metadata.get('version', 0))
        return max_version + 1
    
    def cleanup_old_models(self, keep_versions: int = 5) -> None:
        """
        Clean up old model versions, keeping only the most recent ones
        
        Args:
            keep_versions: Number of versions to keep per model name
        """
        # Group models by name
        models_by_name = {}
        for model_id, metadata in self.models_db.items():
            model_name = metadata['model_name']
            if model_name not in models_by_name:
                models_by_name[model_name] = []
            models_by_name[model_name].append((model_id, metadata))
        
        # Clean up each model name
        for model_name, models in models_by_name.items():
            # Sort by version (descending)
            models.sort(key=lambda x: x[1].get('version', 0), reverse=True)
            
            # Keep only the most recent versions
            models_to_delete = models[keep_versions:]
            
            for model_id, metadata in models_to_delete:
                # Don't delete production models
                if metadata.get('stage') == 'production':
                    continue
                
                try:
                    self.delete_model(model_id, force=False)
                    logger.info(f"Cleaned up old model version: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to clean up model {model_id}: {str(e)}")
        
        logger.info("Model cleanup completed")