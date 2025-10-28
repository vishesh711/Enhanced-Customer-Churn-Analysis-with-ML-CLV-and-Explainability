"""
Tests for model training and evaluation module
Tests model training reproducibility, cross-validation, and serialization consistency
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('src')

from train import (
    ModelTrainer, ProbabilityCalibrator, ThresholdOptimizer, 
    ModelRegistry, ModelMetadata, CrossValidationResults
)
from config import config


class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification dataset"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance"""
        return ModelTrainer(random_state=42, n_jobs=1)
    
    def test_trainer_initialization(self, trainer):
        """Test ModelTrainer initialization"""
        assert trainer.random_state == 42
        assert trainer.n_jobs == 1
        assert isinstance(trainer.trained_models, dict)
        assert isinstance(trainer.model_metadata, dict)
        assert isinstance(trainer.cv_results, dict)
    
    def test_train_baseline_models(self, trainer, sample_data):
        """Test baseline model training"""
        X, y = sample_data
        
        baseline_models = trainer.train_baseline_models(X, y)
        
        # Check that all expected baseline models are trained
        expected_models = ['majority_class', 'stratified', 'logistic_balanced', 'logistic_simple']
        assert all(model_name in baseline_models for model_name in expected_models)
        
        # Check that models are stored in trainer
        assert all(model_name in trainer.trained_models for model_name in expected_models)
        
        # Check that CV results are generated
        assert all(model_name in trainer.cv_results for model_name in expected_models)
        
        # Check that metadata is created
        assert all(model_name in trainer.model_metadata for model_name in expected_models)
    
    def test_train_ensemble_models(self, trainer, sample_data):
        """Test ensemble model training"""
        X, y = sample_data
        
        # Use small parameter grids for faster testing
        param_grids = {
            'random_forest': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5]
            }
        }
        
        ensemble_models = trainer.train_ensemble_models(X, y, param_grids)
        
        # Check that Random Forest is trained
        assert 'random_forest' in ensemble_models
        assert 'random_forest' in trainer.trained_models
        
        # Check that the model is actually a RandomForestClassifier
        assert isinstance(ensemble_models['random_forest'], RandomForestClassifier)
    
    def test_model_reproducibility(self, sample_data):
        """Test that model training is reproducible with fixed seeds"""
        X, y = sample_data
        
        # Train models with same random state
        trainer1 = ModelTrainer(random_state=42)
        trainer2 = ModelTrainer(random_state=42)
        
        models1 = trainer1.train_baseline_models(X, y)
        models2 = trainer2.train_baseline_models(X, y)
        
        # Check that predictions are identical
        for model_name in models1.keys():
            pred1 = models1[model_name].predict_proba(X)
            pred2 = models2[model_name].predict_proba(X)
            
            np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
    
    def test_cross_validation_implementation(self, trainer, sample_data):
        """Test cross-validation implementation"""
        X, y = sample_data
        
        # Create a simple model for testing
        model = LogisticRegression(random_state=42)
        
        cv_results = trainer._evaluate_model_cv(model, X, y, cv_folds=3)
        
        # Check CV results structure
        assert isinstance(cv_results, CrossValidationResults)
        assert 'roc_auc' in cv_results.cv_scores
        assert 'accuracy' in cv_results.cv_scores
        assert len(cv_results.fold_predictions) == 3
        assert len(cv_results.fold_probabilities) == 3
        
        # Check that scores are reasonable
        assert 0 <= cv_results.mean_scores['roc_auc'] <= 1
        assert 0 <= cv_results.mean_scores['accuracy'] <= 1
    
    def test_model_comparison(self, trainer, sample_data):
        """Test model comparison functionality"""
        X, y = sample_data
        
        # Train some models
        trainer.train_baseline_models(X, y)
        
        comparison_df = trainer.get_model_comparison()
        
        # Check comparison DataFrame structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'model' in comparison_df.columns
        assert 'roc_auc' in comparison_df.columns
        assert len(comparison_df) > 0
        
        # Check that it's sorted by ROC-AUC
        roc_auc_values = comparison_df['roc_auc'].values
        assert all(roc_auc_values[i] >= roc_auc_values[i+1] for i in range(len(roc_auc_values)-1))
    
    def test_best_model_selection(self, trainer, sample_data):
        """Test best model selection"""
        X, y = sample_data
        
        # Train some models
        trainer.train_baseline_models(X, y)
        
        best_name, best_model = trainer.get_best_model(metric='roc_auc')
        
        # Check that we get a valid model
        assert isinstance(best_name, str)
        assert hasattr(best_model, 'predict')
        assert hasattr(best_model, 'predict_proba')
    
    def test_model_serialization(self, trainer, sample_data):
        """Test model saving and loading"""
        X, y = sample_data
        
        # Train models
        trainer.train_baseline_models(X, y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            saved_paths = trainer.save_models(temp_dir)
            
            # Check that files are created
            assert len(saved_paths) > 0
            for path in saved_paths.values():
                assert Path(path).exists()
            
            # Load models
            loaded_trainer = ModelTrainer.load_models(temp_dir)
            
            # Check that models are loaded correctly
            assert len(loaded_trainer.trained_models) == len(trainer.trained_models)
            
            # Test that loaded models make same predictions
            for model_name in trainer.trained_models.keys():
                original_pred = trainer.trained_models[model_name].predict_proba(X)
                loaded_pred = loaded_trainer.trained_models[model_name].predict_proba(X)
                
                np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestProbabilityCalibrator:
    """Test cases for ProbabilityCalibrator class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification dataset"""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for calibration testing"""
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def calibrator(self):
        """Create ProbabilityCalibrator instance"""
        return ProbabilityCalibrator(method='isotonic', cv=3)
    
    def test_calibrator_initialization(self, calibrator):
        """Test ProbabilityCalibrator initialization"""
        assert calibrator.method == 'isotonic'
        assert calibrator.cv == 3
        assert isinstance(calibrator.calibrated_models, dict)
        assert isinstance(calibrator.calibration_curves, dict)
    
    def test_model_calibration(self, calibrator, trained_model, sample_data):
        """Test model calibration"""
        X, y = sample_data
        
        calibrated_model = calibrator.calibrate_model(trained_model, X, y, 'test_model')
        
        # Check that calibrated model is created
        assert 'test_model' in calibrator.calibrated_models
        assert hasattr(calibrated_model, 'predict_proba')
        
        # Check that calibration curve data is generated
        assert 'test_model' in calibrator.calibration_curves
        
        # Test that calibrated probabilities are different from original
        original_probs = trained_model.predict_proba(X)[:, 1]
        calibrated_probs = calibrated_model.predict_proba(X)[:, 1]
        
        # They should be different (unless perfectly calibrated already)
        assert not np.allclose(original_probs, calibrated_probs, rtol=1e-3)
    
    def test_calibration_methods_comparison(self, calibrator, trained_model, sample_data):
        """Test comparison of calibration methods"""
        X, y = sample_data
        
        comparison_results = calibrator.compare_calibration_methods(
            trained_model, X, y, 'test_model'
        )
        
        # Check that both methods are compared
        assert 'isotonic' in comparison_results
        assert 'sigmoid' in comparison_results
        assert 'best_method' in comparison_results
        
        # Check that metrics are calculated
        for method in ['isotonic', 'sigmoid']:
            assert 'brier_score' in comparison_results[method]
            assert 'ece' in comparison_results[method]
    
    def test_calibration_quality_evaluation(self, calibrator, trained_model, sample_data):
        """Test calibration quality evaluation"""
        X, y = sample_data
        
        # Calibrate model first
        calibrator.calibrate_model(trained_model, X, y, 'test_model')
        
        quality_df = calibrator.evaluate_calibration_quality(['test_model'])
        
        # Check evaluation results
        assert isinstance(quality_df, pd.DataFrame)
        assert 'model' in quality_df.columns
        assert 'original_ece' in quality_df.columns
        assert 'calibrated_ece' in quality_df.columns
        assert len(quality_df) == 1


class TestThresholdOptimizer:
    """Test cases for ThresholdOptimizer class"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for threshold optimization"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic probability distribution
        y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
        
        # Create probabilities that are somewhat predictive
        y_prob = np.random.beta(2, 5, n_samples)  # Skewed towards lower probabilities
        y_prob[y_true == 1] += 0.3  # Boost probabilities for positive class
        y_prob = np.clip(y_prob, 0.01, 0.99)  # Ensure valid probability range
        
        return y_true, y_prob
    
    @pytest.fixture
    def optimizer(self):
        """Create ThresholdOptimizer instance"""
        business_config = {
            'retention_value': 1000.0,
            'contact_cost': 50.0,
            'churn_cost': 500.0
        }
        return ThresholdOptimizer(business_config)
    
    def test_optimizer_initialization(self, optimizer):
        """Test ThresholdOptimizer initialization"""
        assert optimizer.retention_value == 1000.0
        assert optimizer.contact_cost == 50.0
        assert optimizer.churn_cost == 500.0
        assert isinstance(optimizer.optimal_thresholds, dict)
        assert isinstance(optimizer.profit_curves, dict)
    
    def test_business_threshold_optimization(self, optimizer, sample_predictions):
        """Test business threshold optimization"""
        y_true, y_prob = sample_predictions
        
        optimal_threshold, metrics = optimizer.optimize_business_threshold(
            y_true, y_prob, 'test_model'
        )
        
        # Check that we get a valid threshold
        assert 0 < optimal_threshold < 1
        assert isinstance(metrics, dict)
        assert 'profit' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Check that threshold is stored
        assert 'test_model' in optimizer.optimal_thresholds
        assert optimizer.optimal_thresholds['test_model'] == optimal_threshold
    
    def test_profit_calculation(self, optimizer):
        """Test profit calculation logic"""
        # Test with known values
        tp, fp, tn, fn = 100, 50, 800, 50
        
        expected_profit = (
            tp * optimizer.retention_value -  # Revenue from retained customers
            (tp + fp) * optimizer.contact_cost -  # Contact costs
            fn * optimizer.churn_cost  # Churn costs
        )
        
        calculated_profit = optimizer._calculate_profit(tp, fp, tn, fn)
        
        assert calculated_profit == expected_profit
    
    def test_sensitivity_analysis(self, optimizer, sample_predictions):
        """Test sensitivity analysis"""
        y_true, y_prob = sample_predictions
        
        parameter_ranges = {
            'retention_value': [800, 1000, 1200],
            'contact_cost': [40, 50, 60]
        }
        
        sensitivity_df = optimizer.sensitivity_analysis(
            y_true, y_prob, parameter_ranges, 'test_model'
        )
        
        # Check sensitivity analysis results
        assert isinstance(sensitivity_df, pd.DataFrame)
        assert 'parameter' in sensitivity_df.columns
        assert 'parameter_value' in sensitivity_df.columns
        assert 'optimal_threshold' in sensitivity_df.columns
        assert 'optimal_profit' in sensitivity_df.columns
        
        # Check that we have results for all parameter combinations
        expected_rows = sum(len(values) for values in parameter_ranges.values())
        assert len(sensitivity_df) == expected_rows
    
    def test_expected_value_calculation(self, optimizer, sample_predictions):
        """Test expected value calculation"""
        y_true, y_prob = sample_predictions
        
        # First optimize threshold
        threshold, _ = optimizer.optimize_business_threshold(y_true, y_prob, 'test_model')
        
        # Calculate expected value
        ev_metrics = optimizer.calculate_expected_value(y_true, y_prob, threshold, 'test_model')
        
        # Check expected value metrics
        assert isinstance(ev_metrics, dict)
        assert 'total_profit' in ev_metrics
        assert 'profit_per_customer' in ev_metrics
        assert 'contact_rate' in ev_metrics
        assert 'retention_rate' in ev_metrics
        
        # Check that rates are valid percentages
        assert 0 <= ev_metrics['contact_rate'] <= 1
        assert 0 <= ev_metrics['retention_rate'] <= 1
    
    def test_threshold_comparison(self, optimizer, sample_predictions):
        """Test threshold comparison"""
        y_true, y_prob = sample_predictions
        
        thresholds = [0.1, 0.3, 0.5, 0.7]
        
        comparison_df = optimizer.compare_thresholds(y_true, y_prob, thresholds, 'test_model')
        
        # Check comparison results
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(thresholds)
        assert 'threshold' in comparison_df.columns
        assert 'total_profit' in comparison_df.columns
        
        # Check that it's sorted by profit (descending)
        profits = comparison_df['total_profit'].values
        assert all(profits[i] >= profits[i+1] for i in range(len(profits)-1))


class TestModelRegistry:
    """Test cases for ModelRegistry class"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model, X, y
    
    @pytest.fixture
    def registry(self):
        """Create ModelRegistry instance with temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ModelRegistry(registry_path=temp_dir)
    
    def test_registry_initialization(self, registry):
        """Test ModelRegistry initialization"""
        assert registry.registry_path.exists()
        assert isinstance(registry.models_db, dict)
        assert isinstance(registry.experiments_db, dict)
    
    def test_model_registration(self, registry, sample_model):
        """Test model registration"""
        model, X, y = sample_model
        
        model_id = registry.register_model(
            model=model,
            model_name='test_model',
            model_type='logistic_regression',
            performance_metrics={'accuracy': 0.85, 'roc_auc': 0.90},
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            hyperparameters={'C': 1.0, 'random_state': 42},
            data_hash='test_hash_123',
            description='Test model for unit testing'
        )
        
        # Check that model is registered
        assert model_id in registry.models_db
        
        # Check model metadata
        metadata = registry.get_model_metadata(model_id)
        assert metadata['model_name'] == 'test_model'
        assert metadata['model_type'] == 'logistic_regression'
        assert metadata['performance_metrics']['accuracy'] == 0.85
        assert metadata['version'] == 1
    
    def test_model_loading(self, registry, sample_model):
        """Test model loading"""
        model, X, y = sample_model
        
        # Register model
        model_id = registry.register_model(
            model=model,
            model_name='test_model',
            model_type='logistic_regression',
            performance_metrics={'accuracy': 0.85},
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            hyperparameters={'C': 1.0},
            data_hash='test_hash'
        )
        
        # Load model
        loaded_model = registry.load_model(model_id)
        
        # Test that loaded model makes same predictions
        original_pred = model.predict_proba(X)
        loaded_pred = loaded_model.predict_proba(X)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_model_comparison(self, registry, sample_model):
        """Test model comparison"""
        model, X, y = sample_model
        
        # Register multiple models
        model_ids = []
        for i in range(3):
            model_id = registry.register_model(
                model=model,
                model_name=f'test_model_{i}',
                model_type='logistic_regression',
                performance_metrics={'accuracy': 0.8 + i * 0.05, 'roc_auc': 0.85 + i * 0.03},
                feature_names=[f'feature_{j}' for j in range(X.shape[1])],
                hyperparameters={'C': 1.0},
                data_hash=f'test_hash_{i}'
            )
            model_ids.append(model_id)
        
        # Compare models
        comparison_df = registry.compare_models(model_ids, metrics=['accuracy', 'roc_auc'])
        
        # Check comparison results
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert 'accuracy' in comparison_df.columns
        assert 'roc_auc' in comparison_df.columns
    
    def test_best_model_selection(self, registry, sample_model):
        """Test best model selection"""
        model, X, y = sample_model
        
        # Register models with different performance
        performances = [0.80, 0.85, 0.90]
        for i, perf in enumerate(performances):
            registry.register_model(
                model=model,
                model_name='test_model',
                model_type='logistic_regression',
                performance_metrics={'roc_auc': perf},
                feature_names=[f'feature_{j}' for j in range(X.shape[1])],
                hyperparameters={'C': 1.0},
                data_hash=f'test_hash_{i}'
            )
        
        # Get best model
        best_model_id, best_metadata = registry.get_best_model(
            model_name='test_model', metric='roc_auc'
        )
        
        # Check that we get the best performing model
        assert best_metadata['performance_metrics']['roc_auc'] == 0.90
    
    def test_model_promotion(self, registry, sample_model):
        """Test model promotion to different stages"""
        model, X, y = sample_model
        
        # Register model
        model_id = registry.register_model(
            model=model,
            model_name='test_model',
            model_type='logistic_regression',
            performance_metrics={'accuracy': 0.85},
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            hyperparameters={'C': 1.0},
            data_hash='test_hash'
        )
        
        # Promote to production
        registry.promote_model(model_id, 'production')
        
        # Check promotion
        metadata = registry.get_model_metadata(model_id)
        assert metadata['stage'] == 'production'
        assert 'stage_change_date' in metadata
    
    def test_experiment_registration(self, registry, sample_model):
        """Test experiment registration"""
        model, X, y = sample_model
        
        # Register some models first
        model_ids = {}
        for i in range(2):
            model_id = registry.register_model(
                model=model,
                model_name=f'model_{i}',
                model_type='logistic_regression',
                performance_metrics={'accuracy': 0.8 + i * 0.05},
                feature_names=[f'feature_{j}' for j in range(X.shape[1])],
                hyperparameters={'C': 1.0},
                data_hash=f'hash_{i}'
            )
            model_ids[f'model_{i}'] = model_id
        
        # Register experiment
        experiment_id = registry.register_experiment(
            experiment_name='test_experiment',
            models=model_ids,
            dataset_info={'n_samples': 100, 'n_features': 5},
            experiment_config={'cv_folds': 5, 'random_state': 42},
            results_summary={'best_model': 'model_1', 'best_score': 0.85},
            description='Test experiment'
        )
        
        # Check experiment registration
        assert experiment_id in registry.experiments_db
        experiment_metadata = registry.experiments_db[experiment_id]
        assert experiment_metadata['experiment_name'] == 'test_experiment'
        assert len(experiment_metadata['models']) == 2


if __name__ == '__main__':
    pytest.main([__file__])