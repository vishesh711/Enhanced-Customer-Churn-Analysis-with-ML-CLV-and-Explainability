"""
Unit tests for data preparation components
Tests data loading, cleaning, and splitting functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_prep import DataLoader, DataCleaner, DataSplitter, DataQualityReport
from config import config


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    @pytest.fixture
    def sample_telco_data(self):
        """Create sample Telco data for testing"""
        return pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [12, 24, 6, 36, 48],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'No internet service', 'Yes', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'],
            'MonthlyCharges': [29.85, 56.95, 20.05, 78.70, 99.65],
            'TotalCharges': ['359.4', '1367.8', '120.3', ' ', '4784.4'],  # Note: space for missing value
            'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
        })
    
    @pytest.fixture
    def sample_olist_customers(self):
        """Create sample Olist customers data"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'customer_unique_id': ['U001', 'U002', 'U003'],
            'customer_zip_code_prefix': [12345, 23456, 34567],
            'customer_city': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte'],
            'customer_state': ['SP', 'RJ', 'MG']
        })
    
    @pytest.fixture
    def sample_olist_orders(self):
        """Create sample Olist orders data"""
        base_date = datetime(2023, 1, 1)
        return pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003', 'O004'],
            'customer_id': ['C001', 'C002', 'C001', 'C003'],
            'order_status': ['delivered', 'delivered', 'delivered', 'delivered'],
            'order_purchase_timestamp': [
                base_date,
                base_date + timedelta(days=30),
                base_date + timedelta(days=60),
                base_date + timedelta(days=90)
            ],
            'order_approved_at': [
                base_date + timedelta(hours=1),
                base_date + timedelta(days=30, hours=2),
                base_date + timedelta(days=60, hours=1),
                base_date + timedelta(days=90, hours=3)
            ],
            'order_delivered_customer_date': [
                base_date + timedelta(days=7),
                base_date + timedelta(days=37),
                base_date + timedelta(days=67),
                base_date + timedelta(days=97)
            ]
        })
    
    @pytest.fixture
    def sample_olist_items(self):
        """Create sample Olist order items data"""
        return pd.DataFrame({
            'order_id': ['O001', 'O001', 'O002', 'O003', 'O004'],
            'order_item_id': [1, 2, 1, 1, 1],
            'product_id': ['P001', 'P002', 'P003', 'P001', 'P004'],
            'seller_id': ['S001', 'S002', 'S001', 'S003', 'S001'],
            'price': [29.99, 15.50, 45.00, 29.99, 120.00],
            'freight_value': [5.99, 3.50, 8.00, 5.99, 15.00]
        })
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.data_path == config.RAW_DATA_PATH
        assert isinstance(loader.telco_schema, dict)
        assert isinstance(loader.olist_required_tables, list)
    
    def test_validate_telco_schema_valid(self, sample_telco_data):
        """Test Telco schema validation with valid data"""
        loader = DataLoader()
        # Convert TotalCharges to numeric as would happen in load_telco_data
        sample_telco_data['TotalCharges'] = pd.to_numeric(sample_telco_data['TotalCharges'], errors='coerce')
        
        assert loader.validate_telco_schema(sample_telco_data) == True
    
    def test_validate_telco_schema_missing_columns(self, sample_telco_data):
        """Test Telco schema validation with missing columns"""
        loader = DataLoader()
        # Remove required column
        sample_telco_data_incomplete = sample_telco_data.drop('Churn', axis=1)
        
        assert loader.validate_telco_schema(sample_telco_data_incomplete) == False
    
    def test_validate_telco_schema_duplicate_customers(self, sample_telco_data):
        """Test Telco schema validation with duplicate customer IDs"""
        loader = DataLoader()
        # Create duplicate customer ID
        sample_telco_data.loc[1, 'customerID'] = sample_telco_data.loc[0, 'customerID']
        
        assert loader.validate_telco_schema(sample_telco_data) == False
    
    def test_validate_olist_schema_valid(self, sample_olist_customers, sample_olist_orders, sample_olist_items):
        """Test Olist schema validation with valid joined data"""
        loader = DataLoader()
        
        # Simulate joined data structure
        joined_data = sample_olist_customers.merge(sample_olist_orders, on='customer_id', how='left')
        
        assert loader.validate_olist_schema(joined_data) == True
    
    def test_generate_quality_report(self, sample_telco_data):
        """Test data quality report generation"""
        loader = DataLoader()
        
        # Add some missing values for testing
        sample_telco_data.loc[0, 'Partner'] = None
        sample_telco_data.loc[1, 'MonthlyCharges'] = None
        
        report = loader.generate_quality_report(sample_telco_data, "test_dataset")
        
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == len(sample_telco_data)
        assert report.total_columns == len(sample_telco_data.columns)
        assert len(report.missing_values) > 0  # Should detect missing values
        assert isinstance(report.data_types, dict)
    
    @patch('kagglehub.dataset_download')
    def test_download_telco_data(self, mock_download):
        """Test Telco data download"""
        mock_download.return_value = '/fake/path/to/dataset'
        
        loader = DataLoader()
        result = loader.download_telco_data()
        
        mock_download.assert_called_once_with(config.TELCO_DATASET_ID)
        assert result == '/fake/path/to/dataset'
    
    @patch('kagglehub.dataset_download')
    def test_download_olist_data(self, mock_download):
        """Test Olist data download"""
        mock_download.return_value = '/fake/path/to/dataset'
        
        loader = DataLoader()
        result = loader.download_olist_data()
        
        mock_download.assert_called_once_with(config.OLIST_DATASET_ID)
        assert result == '/fake/path/to/dataset'


class TestDataCleaner:
    """Test cases for DataCleaner class"""
    
    @pytest.fixture
    def sample_dirty_telco_data(self):
        """Create sample Telco data with quality issues"""
        return pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003', 'C004'],
            'gender': ['Male', 'Female', ' Male ', 'Female'],  # Whitespace
            'SeniorCitizen': [0, 1, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'Yes', 'No', 'Yes'],
            'tenure': [0, 24, 6, 36],  # Zero tenure
            'PhoneService': ['No', 'Yes', 'Yes', 'Yes'],
            'MultipleLines': ['Yes', 'No', 'No', 'Yes'],  # Business rule violation
            'InternetService': ['No', 'DSL', 'Fiber optic', 'DSL'],
            'OnlineSecurity': ['Yes', 'No', 'Yes', 'No'],  # Business rule violation
            'MonthlyCharges': [29.85, 156.95, 20.05, 78.70],  # Outlier
            'TotalCharges': [0.0, np.nan, 120.3, 2836.2],  # Missing value
            'Churn': ['No', 'Yes', 'No', 'Yes']
        })
    
    @pytest.fixture
    def sample_dirty_olist_data(self):
        """Create sample Olist data with quality issues"""
        base_date = datetime(2023, 1, 1)
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'order_id': ['O001', 'O002', 'O003'],
            'order_purchase_timestamp': [base_date, base_date + timedelta(days=30), base_date + timedelta(days=60)],
            'order_delivered_customer_date': [
                base_date - timedelta(days=1),  # Delivered before purchase (invalid)
                base_date + timedelta(days=37),
                base_date + timedelta(days=67)
            ],
            'total_price': [29.99, -15.50, 45.00],  # Negative price
            'total_items': [2, 0, 1],  # Zero items
            'total_freight': [5.99, 3.50, np.nan]  # Missing freight
        })
    
    def test_data_cleaner_initialization(self):
        """Test DataCleaner initialization"""
        cleaner = DataCleaner()
        assert cleaner.outlier_method == 'iqr'
        assert cleaner.outlier_threshold == 1.5
        assert isinstance(cleaner.winsorization_limits, tuple)
    
    def test_clean_telco_data(self, sample_dirty_telco_data):
        """Test Telco data cleaning"""
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_telco_data(sample_dirty_telco_data)
        
        # Check that missing TotalCharges was handled
        assert not cleaned_data['TotalCharges'].isnull().any()
        
        # Check that zero tenure customer has zero TotalCharges
        zero_tenure_mask = cleaned_data['tenure'] == 0
        if zero_tenure_mask.any():
            assert cleaned_data.loc[zero_tenure_mask, 'TotalCharges'].iloc[0] == 0.0
        
        # Check that binary columns were converted
        assert cleaned_data['Partner'].dtype in [np.int64, np.float64]
        assert cleaned_data['Churn'].dtype in [np.int64, np.float64]
        
        # Check business rule fixes
        no_phone_mask = cleaned_data['PhoneService'] == 0
        if no_phone_mask.any():
            # Should not have multiple lines if no phone service
            assert not (cleaned_data.loc[no_phone_mask, 'MultipleLines'] == 1).any()
    
    def test_clean_olist_data(self, sample_dirty_olist_data):
        """Test Olist data cleaning"""
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_olist_data(sample_dirty_olist_data)
        
        # Check that negative prices were fixed
        assert (cleaned_data['total_price'] >= 0).all()
        
        # Check that zero items orders were removed
        assert (cleaned_data['total_items'] > 0).all()
        
        # Check that missing freight was filled
        assert not cleaned_data['total_freight'].isnull().any()
    
    def test_handle_missing_values(self):
        """Test missing value handling with different strategies"""
        cleaner = DataCleaner()
        
        # Create test data with missing values
        df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'B']
        })
        
        strategy = {
            'numeric_col': 'median',
            'categorical_col': 'mode'
        }
        
        result = cleaner.handle_missing_values(df, strategy)
        
        # Check that missing values were handled
        assert not result['numeric_col'].isnull().any()
        assert not result['categorical_col'].isnull().any()
        
        # Check imputation values
        assert result['numeric_col'].iloc[2] == df['numeric_col'].median()
        assert result['categorical_col'].iloc[2] in ['A', 'B']  # Should be mode
    
    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection"""
        cleaner = DataCleaner()
        
        # Create data with clear outliers
        df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'outlier_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        })
        
        outliers = cleaner.detect_outliers(df, ['normal_col', 'outlier_col'], method='iqr')
        
        # Should detect the outlier in outlier_col
        assert outliers.any()
        assert outliers.iloc[-1] == True  # Last row has outlier
    
    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection"""
        cleaner = DataCleaner()
        
        # Create data with clear outliers
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]  # 50 is outlier
        })
        
        outliers = cleaner.detect_outliers(df, ['col1'], method='zscore')
        
        # Should detect the outlier
        assert outliers.any()
        assert outliers.iloc[-1] == True  # Last row has outlier


class TestDataSplitter:
    """Test cases for DataSplitter class"""
    
    @pytest.fixture
    def sample_data_with_dates(self):
        """Create sample data with date column"""
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=i*10) for i in range(100)]
        
        return pd.DataFrame({
            'date_col': dates,
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # Imbalanced
        })
    
    def test_data_splitter_initialization(self):
        """Test DataSplitter initialization"""
        splitter = DataSplitter()
        assert splitter.random_state == config.RANDOM_SEED
        
        # Test with custom random state
        splitter_custom = DataSplitter(random_state=123)
        assert splitter_custom.random_state == 123
    
    def test_temporal_split(self, sample_data_with_dates):
        """Test temporal splitting"""
        splitter = DataSplitter()
        
        train_df, val_df, test_df = splitter.temporal_split(
            sample_data_with_dates, 
            'date_col', 
            test_size=0.2, 
            val_size=0.2
        )
        
        # Check sizes
        total_size = len(sample_data_with_dates)
        assert len(test_df) == int(total_size * 0.2)
        assert len(val_df) == int((total_size - len(test_df)) * 0.2)
        assert len(train_df) == total_size - len(val_df) - len(test_df)
        
        # Check temporal ordering (no leakage)
        train_max_date = train_df['date_col'].max()
        val_min_date = val_df['date_col'].min()
        val_max_date = val_df['date_col'].max()
        test_min_date = test_df['date_col'].min()
        
        assert train_max_date <= val_min_date
        assert val_max_date <= test_min_date
    
    def test_stratified_split(self, sample_classification_data):
        """Test stratified splitting"""
        splitter = DataSplitter()
        
        train_df, val_df, test_df = splitter.stratified_split(
            sample_classification_data,
            'target',
            test_size=0.2,
            val_size=0.2
        )
        
        # Check sizes
        total_size = len(sample_classification_data)
        expected_test_size = int(total_size * 0.2)
        expected_val_size = int((total_size - expected_test_size) * 0.2)
        
        assert abs(len(test_df) - expected_test_size) <= 1  # Allow for rounding
        assert abs(len(val_df) - expected_val_size) <= 1
        
        # Check stratification (target distribution should be similar)
        original_dist = sample_classification_data['target'].value_counts(normalize=True)
        train_dist = train_df['target'].value_counts(normalize=True)
        val_dist = val_df['target'].value_counts(normalize=True)
        test_dist = test_df['target'].value_counts(normalize=True)
        
        # Distributions should be similar (within 10% tolerance)
        for class_val in original_dist.index:
            assert abs(original_dist[class_val] - train_dist[class_val]) < 0.1
            assert abs(original_dist[class_val] - val_dist[class_val]) < 0.1
            assert abs(original_dist[class_val] - test_dist[class_val]) < 0.1
    
    def test_random_split(self, sample_classification_data):
        """Test random splitting"""
        splitter = DataSplitter()
        
        train_df, val_df, test_df = splitter.random_split(
            sample_classification_data,
            test_size=0.2,
            val_size=0.2
        )
        
        # Check that all data is accounted for
        total_original = len(sample_classification_data)
        total_split = len(train_df) + len(val_df) + len(test_df)
        assert total_original == total_split
        
        # Check no overlap in indices
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0
    
    def test_create_cv_splits(self, sample_classification_data):
        """Test cross-validation splits creation"""
        splitter = DataSplitter()
        
        cv_splits = splitter.create_cv_splits(sample_classification_data, 'target', n_splits=5)
        
        # Check number of splits
        assert len(cv_splits) == 5
        
        # Check that each split has train and validation indices
        for train_idx, val_idx in cv_splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(set(train_idx) & set(val_idx)) == 0  # No overlap
    
    def test_get_split_summary(self, sample_classification_data):
        """Test split summary generation"""
        splitter = DataSplitter()
        
        train_df, val_df, test_df = splitter.stratified_split(
            sample_classification_data, 'target'
        )
        
        summary = splitter.get_split_summary(train_df, val_df, test_df, 'target')
        
        # Check summary structure
        assert 'total_samples' in summary
        assert 'train_samples' in summary
        assert 'val_samples' in summary
        assert 'test_samples' in summary
        assert 'target_distribution' in summary
        
        # Check calculations
        assert summary['total_samples'] == len(sample_classification_data)
        assert summary['train_samples'] == len(train_df)
        assert summary['val_samples'] == len(val_df)
        assert summary['test_samples'] == len(test_df)
    
    def test_temporal_split_invalid_column(self, sample_data_with_dates):
        """Test temporal split with invalid date column"""
        splitter = DataSplitter()
        
        with pytest.raises(ValueError, match="Date column 'invalid_col' not found"):
            splitter.temporal_split(sample_data_with_dates, 'invalid_col')
    
    def test_stratified_split_invalid_column(self, sample_classification_data):
        """Test stratified split with invalid target column"""
        splitter = DataSplitter()
        
        with pytest.raises(ValueError, match="Target column 'invalid_col' not found"):
            splitter.stratified_split(sample_classification_data, 'invalid_col')


if __name__ == '__main__':
    pytest.main([__file__])