"""
Unit tests for feature engineering components
Tests feature transformations, temporal features, and pipeline orchestration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.features import (
    FeatureTransformer, TimeFeatureGenerator, FeaturePipeline,
    WinsorizerTransformer, FeatureConfig
)
from config import config


class TestFeatureTransformer:
    """Test cases for FeatureTransformer class"""
    
    @pytest.fixture
    def sample_mixed_data(self):
        """Create sample data with mixed feature types"""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.uniform(0, 100, 100),
            'numeric_3': np.random.exponential(2, 100),  # Skewed data
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice(['X', 'Y'], 100),
            'binary_feature': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values"""
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.uniform(0, 100, 100),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice(['X', 'Y'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Introduce missing values
        df.loc[0:10, 'numeric_1'] = np.nan
        df.loc[5:15, 'categorical_1'] = None
        
        return df
    
    def test_feature_transformer_initialization(self):
        """Test FeatureTransformer initialization"""
        transformer = FeatureTransformer()
        assert transformer.feature_config is not None
        assert isinstance(transformer.feature_config, FeatureConfig)
        assert transformer.numeric_pipeline is None
        assert transformer.categorical_pipeline is None
    
    def test_create_numeric_pipeline(self, sample_mixed_data):
        """Test numeric pipeline creation"""
        transformer = FeatureTransformer()
        numeric_features = ['numeric_1', 'numeric_2', 'numeric_3']
        
        pipeline = transformer.create_numeric_pipeline(numeric_features)
        
        assert pipeline is not None
        assert len(pipeline.steps) >= 2  # At least imputer and scaler
        
        # Test pipeline can fit and transform
        X_numeric = sample_mixed_data[numeric_features]
        pipeline.fit(X_numeric)
        X_transformed = pipeline.transform(X_numeric)
        
        assert X_transformed.shape == X_numeric.shape
        assert not np.isnan(X_transformed).any()  # No missing values after transformation
    
    def test_create_categorical_pipeline(self, sample_mixed_data):
        """Test categorical pipeline creation"""
        transformer = FeatureTransformer()
        categorical_features = ['categorical_1', 'categorical_2']
        
        pipeline = transformer.create_categorical_pipeline(categorical_features)
        
        assert pipeline is not None
        assert len(pipeline.steps) >= 2  # At least imputer and encoder
        
        # Test pipeline can fit and transform
        X_categorical = sample_mixed_data[categorical_features]
        pipeline.fit(X_categorical)
        X_transformed = pipeline.transform(X_categorical)
        
        assert X_transformed.shape[0] == X_categorical.shape[0]
        assert X_transformed.shape[1] >= X_categorical.shape[1]  # One-hot encoding increases features
    
    def test_create_interaction_features(self, sample_mixed_data):
        """Test interaction feature creation"""
        transformer = FeatureTransformer()
        
        # Select only numeric features for interactions
        numeric_data = sample_mixed_data[['numeric_1', 'numeric_2', 'numeric_3']]
        
        # Create specific interactions
        interactions = [('numeric_1', 'numeric_2'), ('numeric_2', 'numeric_3')]
        result = transformer.create_interaction_features(numeric_data, interactions)
        
        # Should have original features plus interaction features
        assert result.shape[0] == numeric_data.shape[0]
        assert result.shape[1] > numeric_data.shape[1]
        
        # Check that interaction features were created
        expected_interactions = ['numeric_1_x_numeric_2', 'numeric_2_x_numeric_3']
        for interaction in expected_interactions:
            assert interaction in result.columns
        
        # Verify interaction calculations
        assert np.allclose(
            result['numeric_1_x_numeric_2'], 
            numeric_data['numeric_1'] * numeric_data['numeric_2']
        )
    
    def test_fit_transform_complete_pipeline(self, sample_mixed_data):
        """Test complete fit and transform process"""
        transformer = FeatureTransformer()
        
        X = sample_mixed_data.drop('target', axis=1)
        y = sample_mixed_data['target']
        
        # Fit and transform
        X_transformed = transformer.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] >= X.shape[1]  # Should have at least original features
        
        # Check that numeric and categorical features were identified
        assert len(transformer.numeric_features_) > 0
        assert len(transformer.categorical_features_) > 0
    
    def test_handle_missing_values(self, sample_data_with_missing):
        """Test handling of missing values"""
        transformer = FeatureTransformer()
        
        X = sample_data_with_missing.drop('target', axis=1)
        y = sample_data_with_missing['target']
        
        # Fit and transform
        X_transformed = transformer.fit_transform(X, y)
        
        # Should not have any missing values after transformation
        assert not X_transformed.isnull().any().any()
    
    def test_transform_consistency(self, sample_mixed_data):
        """Test that transform produces consistent results"""
        transformer = FeatureTransformer()
        
        X = sample_mixed_data.drop('target', axis=1)
        y = sample_mixed_data['target']
        
        # Fit on full data
        transformer.fit(X, y)
        
        # Transform in parts
        X_part1 = transformer.transform(X.iloc[:50])
        X_part2 = transformer.transform(X.iloc[50:])
        
        # Transform all at once
        X_full = transformer.transform(X)
        
        # Results should be consistent
        X_combined = pd.concat([X_part1, X_part2], axis=0, ignore_index=True)
        
        # Check shapes match
        assert X_combined.shape == X_full.shape
        
        # Check values are close (allowing for small numerical differences)
        numeric_cols = X_combined.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert np.allclose(X_combined[col], X_full[col], rtol=1e-10)


class TestWinsorizerTransformer:
    """Test cases for WinsorizerTransformer"""
    
    def test_winsorizer_initialization(self):
        """Test WinsorizerTransformer initialization"""
        winsorizer = WinsorizerTransformer()
        assert winsorizer.limits == (0.01, 0.99)
        assert winsorizer.lower_bounds_ is None
        assert winsorizer.upper_bounds_ is None
    
    def test_winsorizer_fit_transform(self):
        """Test winsorizer fit and transform"""
        # Create data with outliers
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        X[0, 0] = 100  # Extreme outlier
        X[1, 1] = -100  # Extreme outlier
        
        winsorizer = WinsorizerTransformer(limits=(0.05, 0.95))
        X_winsorized = winsorizer.fit_transform(X)
        
        # Check that outliers were clipped
        assert X_winsorized[0, 0] < 100
        assert X_winsorized[1, 1] > -100
        
        # Check that bounds were computed
        assert winsorizer.lower_bounds_ is not None
        assert winsorizer.upper_bounds_ is not None
        assert len(winsorizer.lower_bounds_) == X.shape[1]
        assert len(winsorizer.upper_bounds_) == X.shape[1]


class TestTimeFeatureGenerator:
    """Test cases for TimeFeatureGenerator"""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data"""
        np.random.seed(42)
        base_date = datetime(2023, 1, 1)
        
        # Generate transactions for 3 customers over 6 months
        data = []
        for customer_id in ['C001', 'C002', 'C003']:
            n_transactions = np.random.randint(5, 20)
            for i in range(n_transactions):
                date = base_date + timedelta(days=np.random.randint(0, 180))
                value = np.random.uniform(10, 200)
                data.append({
                    'customer_id': customer_id,
                    'date': date,
                    'value': value
                })
        
        return pd.DataFrame(data)
    
    def test_time_feature_generator_initialization(self):
        """Test TimeFeatureGenerator initialization"""
        generator = TimeFeatureGenerator()
        assert generator.customer_col == 'customer_id'
        assert generator.date_col == 'date'
        assert generator.value_col == 'value'
        assert generator.reference_date is None
    
    def test_generate_rfm_features(self, sample_transaction_data):
        """Test RFM feature generation"""
        generator = TimeFeatureGenerator()
        
        rfm_features = generator.generate_rfm_features(sample_transaction_data)
        
        # Check that RFM features were created
        assert 'recency_days' in rfm_features.columns
        assert 'frequency' in rfm_features.columns
        assert 'monetary_total' in rfm_features.columns
        assert 'monetary_avg' in rfm_features.columns
        
        # Check that we have one row per customer
        unique_customers = sample_transaction_data['customer_id'].nunique()
        assert len(rfm_features) == unique_customers
        
        # Check that RFM scores were created
        assert 'rfm_score' in rfm_features.columns
        
        # Verify calculations make sense
        assert (rfm_features['recency_days'] >= 0).all()
        assert (rfm_features['frequency'] > 0).all()
        assert (rfm_features['monetary_total'] > 0).all()
    
    def test_calculate_recency(self, sample_transaction_data):
        """Test recency calculation"""
        generator = TimeFeatureGenerator()
        
        recency = generator.calculate_recency(sample_transaction_data)
        
        # Should have one value per customer
        unique_customers = sample_transaction_data['customer_id'].nunique()
        assert len(recency) == unique_customers
        
        # All recency values should be non-negative
        assert (recency >= 0).all()
    
    def test_calculate_frequency(self, sample_transaction_data):
        """Test frequency calculation"""
        generator = TimeFeatureGenerator()
        
        frequency = generator.calculate_frequency(sample_transaction_data)
        
        # Should have one value per customer
        unique_customers = sample_transaction_data['customer_id'].nunique()
        assert len(frequency) == unique_customers
        
        # All frequency values should be positive
        assert (frequency > 0).all()
    
    def test_calculate_monetary(self, sample_transaction_data):
        """Test monetary value calculation"""
        generator = TimeFeatureGenerator()
        
        monetary = generator.calculate_monetary(sample_transaction_data)
        
        # Should have one value per customer
        unique_customers = sample_transaction_data['customer_id'].nunique()
        assert len(monetary) == unique_customers
        
        # All monetary values should be positive
        assert (monetary > 0).all()
    
    def test_generate_rolling_features(self, sample_transaction_data):
        """Test rolling feature generation"""
        generator = TimeFeatureGenerator()
        
        rolling_features = generator.generate_rolling_features(
            sample_transaction_data, 
            windows=[7, 30]
        )
        
        # Check that rolling features were added
        expected_features = [
            'rolling_7d_sum', 'rolling_7d_count', 'rolling_7d_mean',
            'rolling_30d_sum', 'rolling_30d_count', 'rolling_30d_mean'
        ]
        
        for feature in expected_features:
            assert feature in rolling_features.columns
        
        # Check that data shape is preserved
        assert rolling_features.shape[0] == sample_transaction_data.shape[0]
    
    def test_generate_session_features(self, sample_transaction_data):
        """Test session feature generation"""
        generator = TimeFeatureGenerator()
        
        session_features = generator.generate_session_features(
            sample_transaction_data,
            session_gap_hours=24
        )
        
        # Should have one row per customer
        unique_customers = sample_transaction_data['customer_id'].nunique()
        assert len(session_features) == unique_customers
        
        # Check that session features were created
        session_cols = [col for col in session_features.columns if 'session' in col]
        assert len(session_cols) > 0
    
    def test_generate_behavioral_patterns(self, sample_transaction_data):
        """Test behavioral pattern feature generation"""
        generator = TimeFeatureGenerator()
        
        behavioral_features = generator.generate_behavioral_patterns(sample_transaction_data)
        
        # Should have one row per customer
        unique_customers = sample_transaction_data['customer_id'].nunique()
        assert len(behavioral_features) == unique_customers
        
        # Check that behavioral features were created
        behavioral_cols = [col for col in behavioral_features.columns if 'behavior' in col]
        assert len(behavioral_cols) > 0
    
    def test_fit_transform(self, sample_transaction_data):
        """Test complete fit and transform process"""
        generator = TimeFeatureGenerator()
        
        # Fit and transform
        result = generator.fit_transform(sample_transaction_data)
        
        # Should have more features than original
        assert result.shape[1] > sample_transaction_data.shape[1]
        
        # Should preserve number of rows
        assert result.shape[0] == sample_transaction_data.shape[0]


class TestFeaturePipeline:
    """Test cases for FeaturePipeline orchestration"""
    
    @pytest.fixture
    def sample_pipeline_data(self):
        """Create sample data for pipeline testing"""
        np.random.seed(42)
        base_date = datetime(2023, 1, 1)
        
        data = []
        for customer_id in ['C001', 'C002', 'C003', 'C004', 'C005']:
            # Create multiple transactions per customer
            n_transactions = np.random.randint(3, 8)
            for i in range(n_transactions):
                data.append({
                    'customer_id': customer_id,
                    'date': base_date + timedelta(days=np.random.randint(0, 90)),
                    'amount': np.random.uniform(10, 200),
                    'category': np.random.choice(['A', 'B', 'C']),
                    'channel': np.random.choice(['online', 'store']),
                    'numeric_feature': np.random.randn(),
                    'target': np.random.choice([0, 1])
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    @pytest.fixture
    def simple_classification_data(self):
        """Create simple classification data without temporal features"""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.uniform(0, 100, 100),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_2': np.random.choice(['X', 'Y'], 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    def test_feature_pipeline_initialization(self):
        """Test FeaturePipeline initialization"""
        pipeline = FeaturePipeline()
        
        assert pipeline.include_temporal == True
        assert pipeline.include_interactions == True
        assert pipeline.pipeline is None
        assert pipeline.feature_names_in_ == []
        assert pipeline.feature_names_out_ == []
    
    def test_pipeline_without_temporal_features(self, simple_classification_data):
        """Test pipeline with simple data (no temporal features)"""
        pipeline = FeaturePipeline(include_temporal=False)
        
        X = simple_classification_data.drop('target', axis=1)
        y = simple_classification_data['target']
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X, y)
        
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > 0
        
        # Check that feature names were tracked
        assert len(pipeline.feature_names_in_) == X.shape[1]
        assert len(pipeline.feature_names_out_) > 0
    
    def test_pipeline_with_feature_selection(self, simple_classification_data):
        """Test pipeline with feature selection"""
        pipeline = FeaturePipeline(
            include_temporal=False,
            feature_selection_method='f_classif',
            max_features=5
        )
        
        X = simple_classification_data.drop('target', axis=1)
        y = simple_classification_data['target']
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X, y)
        
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == X.shape[0]
        
        # Should have selected features
        assert pipeline.feature_selector is not None
        
        # Get feature importance
        importance_df = pipeline.get_feature_importance()
        if importance_df is not None:
            assert isinstance(importance_df, pd.DataFrame)
            assert 'feature' in importance_df.columns
            assert 'importance' in importance_df.columns
    
    def test_pipeline_with_dimensionality_reduction(self, simple_classification_data):
        """Test pipeline with PCA dimensionality reduction"""
        pipeline = FeaturePipeline(
            include_temporal=False,
            dimensionality_reduction='pca',
            n_components=3
        )
        
        X = simple_classification_data.drop('target', axis=1)
        y = simple_classification_data['target']
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X, y)
        
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 3  # Should have exactly 3 components
        
        # Check that PCA was applied
        assert pipeline.dimensionality_reducer is not None
    
    def test_pipeline_cross_validation_compatibility(self, simple_classification_data):
        """Test that pipeline is compatible with cross-validation"""
        pipeline = FeaturePipeline(include_temporal=False)
        
        X = simple_classification_data.drop('target', axis=1)
        y = simple_classification_data['target']
        
        # Create a simple classifier pipeline
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.ensemble import RandomForestClassifier
        
        full_pipeline = SklearnPipeline([
            ('features', pipeline),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Test cross-validation (should not raise errors)
        try:
            scores = cross_val_score(full_pipeline, X, y, cv=3, scoring='accuracy')
            assert len(scores) == 3
            assert all(score >= 0 for score in scores)  # Scores should be reasonable
        except Exception as e:
            pytest.fail(f"Cross-validation failed: {str(e)}")
    
    def test_pipeline_serialization(self, simple_classification_data):
        """Test pipeline serialization and deserialization"""
        pipeline = FeaturePipeline(include_temporal=False)
        
        X = simple_classification_data.drop('target', axis=1)
        y = simple_classification_data['target']
        
        # Fit pipeline
        pipeline.fit(X, y)
        
        # Test save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_pipeline.pkl"
            
            # Save pipeline
            pipeline.save_pipeline(filepath)
            assert filepath.exists()
            
            # Load pipeline
            loaded_pipeline = FeaturePipeline.load_pipeline(filepath)
            
            # Test that loaded pipeline works
            X_original = pipeline.transform(X)
            X_loaded = loaded_pipeline.transform(X)
            
            # Results should be identical
            assert np.allclose(X_original, X_loaded)
            
            # Check that metadata was preserved
            assert loaded_pipeline.feature_names_in_ == pipeline.feature_names_in_
            assert loaded_pipeline.feature_names_out_ == pipeline.feature_names_out_
    
    def test_transform_consistency_across_cv_folds(self, simple_classification_data):
        """Test that transformations are consistent across CV folds"""
        from sklearn.model_selection import KFold
        
        pipeline = FeaturePipeline(include_temporal=False)
        
        X = simple_classification_data.drop('target', axis=1)
        y = simple_classification_data['target']
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        fold_results = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit on training data
            pipeline_fold = FeaturePipeline(include_temporal=False)
            pipeline_fold.fit(X_train, y_train)
            
            # Transform validation data
            X_val_transformed = pipeline_fold.transform(X_val)
            fold_results.append(X_val_transformed.shape[1])
        
        # All folds should produce the same number of features
        assert len(set(fold_results)) == 1, "Feature counts differ across CV folds"
    
    def test_business_logic_validation(self, sample_pipeline_data):
        """Test that feature engineering follows business logic"""
        pipeline = FeaturePipeline(include_temporal=True)
        
        X = sample_pipeline_data.drop('target', axis=1)
        y = sample_pipeline_data['target']
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X, y)
        
        # Basic validation - should not have NaN values
        assert not np.isnan(X_transformed).any()
        
        # Should have reasonable number of features (not too many, not too few)
        assert X_transformed.shape[1] >= X.shape[1]  # At least original features
        assert X_transformed.shape[1] <= X.shape[1] * 10  # Not excessive feature explosion


if __name__ == '__main__':
    pytest.main([__file__])