"""
Data preparation module for Customer Churn ML Pipeline
Handles data loading, cleaning, and preprocessing operations
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import kagglehub
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import warnings

from config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality assessment results"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    memory_usage: str


class DataLoader:
    """
    Handles loading and initial validation of customer data from multiple sources
    Supports Telco and Olist datasets with schema validation and quality checks
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader with data path
        
        Args:
            data_path: Path to store downloaded data (defaults to config.RAW_DATA_PATH)
        """
        self.data_path = data_path or config.RAW_DATA_PATH
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Expected schemas for validation
        self.telco_schema = {
            'customerID': 'object',
            'gender': 'object',
            'SeniorCitizen': 'int64',
            'Partner': 'object',
            'Dependents': 'object',
            'tenure': 'int64',
            'PhoneService': 'object',
            'MultipleLines': 'object',
            'InternetService': 'object',
            'OnlineSecurity': 'object',
            'OnlineBackup': 'object',
            'DeviceProtection': 'object',
            'TechSupport': 'object',
            'StreamingTV': 'object',
            'StreamingMovies': 'object',
            'Contract': 'object',
            'PaperlessBilling': 'object',
            'PaymentMethod': 'object',
            'MonthlyCharges': 'float64',
            'TotalCharges': 'object',  # Will be converted to float
            'Churn': 'object'
        }
        
        self.olist_required_tables = [
            'olist_customers_dataset.csv',
            'olist_orders_dataset.csv',
            'olist_order_items_dataset.csv',
            'olist_order_payments_dataset.csv',
            'olist_products_dataset.csv'
        ]
    
    def download_telco_data(self) -> str:
        """
        Download Telco customer churn dataset using kagglehub
        
        Returns:
            str: Path to downloaded dataset directory
        """
        try:
            logger.info(f"Downloading Telco dataset: {config.TELCO_DATASET_ID}")
            dataset_path = kagglehub.dataset_download(config.TELCO_DATASET_ID)
            logger.info(f"Telco dataset downloaded to: {dataset_path}")
            return dataset_path
        except Exception as e:
            logger.error(f"Failed to download Telco dataset: {str(e)}")
            raise
    
    def download_olist_data(self) -> str:
        """
        Download Olist Brazilian e-commerce dataset using kagglehub
        
        Returns:
            str: Path to downloaded dataset directory
        """
        try:
            logger.info(f"Downloading Olist dataset: {config.OLIST_DATASET_ID}")
            dataset_path = kagglehub.dataset_download(config.OLIST_DATASET_ID)
            logger.info(f"Olist dataset downloaded to: {dataset_path}")
            return dataset_path
        except Exception as e:
            logger.error(f"Failed to download Olist dataset: {str(e)}")
            raise
    
    def load_telco_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load Telco customer churn data with explicit dtype handling
        
        Args:
            filepath: Path to CSV file (if None, downloads dataset)
            
        Returns:
            pd.DataFrame: Loaded and validated Telco dataset
        """
        try:
            if filepath is None:
                dataset_dir = self.download_telco_data()
                # Find the CSV file in the downloaded directory
                csv_files = list(Path(dataset_dir).glob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in downloaded Telco dataset")
                filepath = str(csv_files[0])
            
            logger.info(f"Loading Telco data from: {filepath}")
            
            # Load with initial dtypes
            df = pd.read_csv(filepath, dtype=self.telco_schema)
            
            # Handle TotalCharges conversion (contains spaces for missing values)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Validate schema
            if not self.validate_telco_schema(df):
                raise ValueError("Telco data schema validation failed")
            
            logger.info(f"Successfully loaded Telco data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Telco data: {str(e)}")
            raise
    
    def load_olist_data(self, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load Olist e-commerce data with multi-table joins
        
        Args:
            data_dir: Directory containing Olist CSV files (if None, downloads dataset)
            
        Returns:
            pd.DataFrame: Joined and processed Olist dataset for churn analysis
        """
        try:
            if data_dir is None:
                data_dir = self.download_olist_data()
            
            logger.info(f"Loading Olist data from: {data_dir}")
            data_path = Path(data_dir)
            
            # Check for required tables
            missing_tables = []
            for table in self.olist_required_tables:
                if not (data_path / table).exists():
                    missing_tables.append(table)
            
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
            
            # Load core tables
            customers = pd.read_csv(data_path / 'olist_customers_dataset.csv')
            orders = pd.read_csv(data_path / 'olist_orders_dataset.csv')
            order_items = pd.read_csv(data_path / 'olist_order_items_dataset.csv')
            
            # Convert date columns
            date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                          'order_delivered_carrier_date', 'order_delivered_customer_date',
                          'order_estimated_delivery_date']
            
            for col in date_columns:
                if col in orders.columns:
                    orders[col] = pd.to_datetime(orders[col], errors='coerce')
            
            # Join tables to create customer-centric view
            # Start with customers and orders
            customer_orders = customers.merge(orders, on='customer_id', how='left')
            
            # Aggregate order items by order
            order_summary = order_items.groupby('order_id').agg({
                'order_item_id': 'count',
                'product_id': 'nunique',
                'price': ['sum', 'mean'],
                'freight_value': 'sum'
            }).round(2)
            
            # Flatten column names
            order_summary.columns = [
                'total_items', 'unique_products', 'total_price', 
                'avg_item_price', 'total_freight'
            ]
            order_summary = order_summary.reset_index()
            
            # Join with order summary
            df = customer_orders.merge(order_summary, on='order_id', how='left')
            
            # Create churn label based on recency (customers who haven't ordered in last 6 months)
            if 'order_purchase_timestamp' in df.columns:
                max_date = df['order_purchase_timestamp'].max()
                df['days_since_last_order'] = (max_date - df['order_purchase_timestamp']).dt.days
                df['churn'] = (df['days_since_last_order'] > 180).astype(int)
            
            # Validate the joined dataset
            if not self.validate_olist_schema(df):
                logger.warning("Olist data schema validation had issues")
            
            logger.info(f"Successfully loaded and joined Olist data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Olist data: {str(e)}")
            raise
    
    def validate_telco_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate Telco dataset schema and required columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if schema is valid
        """
        try:
            required_columns = set(self.telco_schema.keys())
            actual_columns = set(df.columns)
            
            missing_columns = required_columns - actual_columns
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for customer ID uniqueness
            if df['customerID'].duplicated().any():
                logger.error("Duplicate customer IDs found")
                return False
            
            # Check target variable
            if 'Churn' not in df.columns:
                logger.error("Target variable 'Churn' not found")
                return False
            
            churn_values = df['Churn'].unique()
            expected_churn_values = {'Yes', 'No'}
            if not set(churn_values).issubset(expected_churn_values):
                logger.error(f"Invalid churn values: {churn_values}")
                return False
            
            logger.info("Telco schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return False
    
    def validate_olist_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate Olist dataset schema and required columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if schema is valid
        """
        try:
            required_columns = {'customer_id', 'order_id', 'customer_city', 'customer_state'}
            actual_columns = set(df.columns)
            
            missing_columns = required_columns - actual_columns
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for reasonable data ranges
            if 'total_price' in df.columns:
                if (df['total_price'] < 0).any():
                    logger.warning("Negative prices found in data")
            
            logger.info("Olist schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return False
    
    def generate_quality_report(self, df: pd.DataFrame, dataset_name: str) -> DataQualityReport:
        """
        Generate comprehensive data quality report
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            DataQualityReport: Comprehensive quality assessment
        """
        logger.info(f"Generating quality report for {dataset_name}")
        
        # Calculate missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}
        
        # Count duplicates
        duplicate_rows = df.duplicated().sum()
        
        # Detect outliers for numeric columns
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        # Get data types
        data_types = df.dtypes.astype(str).to_dict()
        
        # Calculate memory usage
        memory_usage = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        
        report = DataQualityReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            outliers=outliers,
            data_types=data_types,
            memory_usage=memory_usage
        )
        
        logger.info(f"Quality report completed for {dataset_name}")
        return report


class DataCleaner:
    """
    Handles data cleaning and preprocessing operations
    Implements missing value handling, outlier detection, and data validation
    """
    
    def __init__(self, config_params: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner with configuration parameters
        
        Args:
            config_params: Dictionary of configuration parameters
        """
        self.config_params = config_params or config.get_feature_config()
        self.outlier_method = self.config_params.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config_params.get('outlier_threshold', 1.5)
        self.winsorization_limits = self.config_params.get('winsorization_limits', (0.01, 0.99))
    
    def clean_telco_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Telco customer churn dataset
        
        Args:
            df: Raw Telco DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Starting Telco data cleaning")
        df_clean = df.copy()
        
        # Handle TotalCharges missing values (already converted to numeric in loader)
        if df_clean['TotalCharges'].isnull().any():
            logger.info("Handling missing TotalCharges values")
            # For customers with 0 tenure, set TotalCharges to 0
            mask_zero_tenure = (df_clean['tenure'] == 0) & df_clean['TotalCharges'].isnull()
            df_clean.loc[mask_zero_tenure, 'TotalCharges'] = 0.0
            
            # For others, use median imputation
            median_charges = df_clean['TotalCharges'].median()
            df_clean['TotalCharges'].fillna(median_charges, inplace=True)
        
        # Standardize categorical values
        df_clean = self._standardize_categorical_values(df_clean)
        
        # Convert binary categorical to numeric
        binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
        # Convert target variable
        if 'Churn' in df_clean.columns:
            df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
        # Handle impossible value combinations
        df_clean = self._validate_telco_business_rules(df_clean)
        
        # Detect and handle outliers
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_clean = self._handle_outliers(df_clean, numeric_columns)
        
        logger.info(f"Telco data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def clean_olist_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Olist e-commerce dataset
        
        Args:
            df: Raw Olist DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Starting Olist data cleaning")
        df_clean = df.copy()
        
        # Handle missing values in key columns
        if 'total_price' in df_clean.columns:
            df_clean['total_price'].fillna(0, inplace=True)
        
        if 'total_freight' in df_clean.columns:
            df_clean['total_freight'].fillna(0, inplace=True)
        
        # Remove orders with invalid dates
        date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date']
        for col in date_columns:
            if col in df_clean.columns:
                invalid_dates = df_clean[col].isnull()
                if invalid_dates.any():
                    logger.info(f"Removing {invalid_dates.sum()} rows with invalid {col}")
                    df_clean = df_clean[~invalid_dates]
        
        # Handle impossible value combinations for Olist
        df_clean = self._validate_olist_business_rules(df_clean)
        
        # Detect and handle outliers in monetary columns
        monetary_columns = ['total_price', 'avg_item_price', 'total_freight']
        existing_monetary = [col for col in monetary_columns if col in df_clean.columns]
        df_clean = self._handle_outliers(df_clean, existing_monetary)
        
        logger.info(f"Olist data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values using specified strategies
        
        Args:
            df: DataFrame with missing values
            strategy: Dictionary mapping column names to imputation strategies
                     ('mean', 'median', 'mode', 'forward_fill', 'drop')
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        df_imputed = df.copy()
        
        for column, method in strategy.items():
            if column not in df_imputed.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
            
            missing_count = df_imputed[column].isnull().sum()
            if missing_count == 0:
                continue
            
            logger.info(f"Imputing {missing_count} missing values in {column} using {method}")
            
            if method == 'mean':
                df_imputed[column].fillna(df_imputed[column].mean(), inplace=True)
            elif method == 'median':
                df_imputed[column].fillna(df_imputed[column].median(), inplace=True)
            elif method == 'mode':
                mode_value = df_imputed[column].mode()
                if len(mode_value) > 0:
                    df_imputed[column].fillna(mode_value[0], inplace=True)
            elif method == 'forward_fill':
                df_imputed[column].fillna(method='ffill', inplace=True)
            elif method == 'drop':
                df_imputed.dropna(subset=[column], inplace=True)
            else:
                logger.warning(f"Unknown imputation method: {method}")
        
        return df_imputed
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.Series:
        """
        Detect outliers using specified method
        
        Args:
            df: DataFrame to analyze
            columns: List of numeric columns to check
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        
        Returns:
            pd.Series: Boolean series indicating outlier rows
        """
        logger.info(f"Detecting outliers using {method} method")
        
        if method == 'iqr':
            return self._detect_outliers_iqr(df, columns)
        elif method == 'zscore':
            return self._detect_outliers_zscore(df, columns)
        elif method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(df, columns)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Detect outliers using Interquartile Range method"""
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask |= col_outliers
            
            logger.info(f"Found {col_outliers.sum()} outliers in {col}")
        
        return outlier_mask
    
    def _detect_outliers_zscore(self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > threshold
            outlier_mask |= col_outliers
            
            logger.info(f"Found {col_outliers.sum()} outliers in {col}")
        
        return outlier_mask
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Detect outliers using Isolation Forest"""
        from sklearn.ensemble import IsolationForest
        
        # Select numeric columns that exist
        valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not valid_columns:
            return pd.Series(False, index=df.index)
        
        # Prepare data
        X = df[valid_columns].fillna(df[valid_columns].median())
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=config.RANDOM_SEED)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Convert to boolean mask (Isolation Forest returns -1 for outliers)
        outlier_mask = pd.Series(outlier_labels == -1, index=df.index)
        
        logger.info(f"Found {outlier_mask.sum()} outliers using Isolation Forest")
        return outlier_mask
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle outliers using winsorization"""
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            # Apply winsorization
            lower_limit, upper_limit = self.winsorization_limits
            lower_val = df_clean[col].quantile(lower_limit)
            upper_val = df_clean[col].quantile(upper_limit)
            
            # Count outliers before winsorization
            outliers_before = ((df_clean[col] < lower_val) | (df_clean[col] > upper_val)).sum()
            
            # Apply winsorization
            df_clean[col] = np.clip(df_clean[col], lower_val, upper_val)
            
            if outliers_before > 0:
                logger.info(f"Winsorized {outliers_before} outliers in {col}")
        
        return df_clean
    
    def _standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values (trim whitespace, handle case)"""
        df_clean = df.copy()
        
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col == 'customerID':  # Skip ID columns
                continue
            
            # Strip whitespace and standardize case
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # Handle common variations
            if col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']:
                df_clean[col] = df_clean[col].replace('No internet service', 'No')
                df_clean[col] = df_clean[col].replace('No phone service', 'No')
        
        return df_clean
    
    def _validate_telco_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix business rule violations in Telco data"""
        df_clean = df.copy()
        
        # Rule 1: Customers with no phone service shouldn't have multiple lines
        if 'PhoneService' in df_clean.columns and 'MultipleLines' in df_clean.columns:
            mask = (df_clean['PhoneService'] == 'No') & (df_clean['MultipleLines'] == 'Yes')
            if mask.any():
                logger.info(f"Fixing {mask.sum()} business rule violations: No phone service but multiple lines")
                df_clean.loc[mask, 'MultipleLines'] = 'No'
        
        # Rule 2: Customers with no internet service shouldn't have internet-dependent services
        internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        if 'InternetService' in df_clean.columns:
            no_internet_mask = df_clean['InternetService'] == 'No'
            for service in internet_services:
                if service in df_clean.columns:
                    violation_mask = no_internet_mask & (df_clean[service] == 'Yes')
                    if violation_mask.any():
                        logger.info(f"Fixing {violation_mask.sum()} violations: No internet but {service} = Yes")
                        df_clean.loc[violation_mask, service] = 'No'
        
        # Rule 3: TotalCharges should be reasonable given tenure and MonthlyCharges
        if all(col in df_clean.columns for col in ['tenure', 'MonthlyCharges', 'TotalCharges']):
            expected_total = df_clean['tenure'] * df_clean['MonthlyCharges']
            # Allow for reasonable variation (setup fees, prorations, etc.)
            unreasonable_mask = (df_clean['TotalCharges'] > expected_total * 1.5) | \
                              (df_clean['TotalCharges'] < expected_total * 0.5)
            
            # Only flag cases where the difference is very large
            large_diff_mask = unreasonable_mask & (df_clean['tenure'] > 0)
            
            if large_diff_mask.any():
                logger.warning(f"Found {large_diff_mask.sum()} customers with unreasonable TotalCharges")
                # Could implement correction logic here if needed
        
        return df_clean
    
    def _validate_olist_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix business rule violations in Olist data"""
        df_clean = df.copy()
        
        # Rule 1: Total price should be positive
        if 'total_price' in df_clean.columns:
            negative_price_mask = df_clean['total_price'] < 0
            if negative_price_mask.any():
                logger.info(f"Fixing {negative_price_mask.sum()} negative prices")
                df_clean.loc[negative_price_mask, 'total_price'] = 0
        
        # Rule 2: Delivery date should be after purchase date
        if all(col in df_clean.columns for col in ['order_purchase_timestamp', 'order_delivered_customer_date']):
            invalid_delivery_mask = (df_clean['order_delivered_customer_date'] < 
                                   df_clean['order_purchase_timestamp'])
            if invalid_delivery_mask.any():
                logger.warning(f"Found {invalid_delivery_mask.sum()} orders with delivery before purchase")
                # Set delivery date to null for these cases
                df_clean.loc[invalid_delivery_mask, 'order_delivered_customer_date'] = pd.NaT
        
        # Rule 3: Number of items should be positive
        if 'total_items' in df_clean.columns:
            zero_items_mask = df_clean['total_items'] <= 0
            if zero_items_mask.any():
                logger.info(f"Removing {zero_items_mask.sum()} orders with zero items")
                df_clean = df_clean[~zero_items_mask]
        
        return df_clean


class DataSplitter:
    """
    Handles train/validation/test splits with support for temporal and stratified splitting
    Ensures no data leakage between splits
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize DataSplitter
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state or config.RANDOM_SEED
    
    def temporal_split(self, df: pd.DataFrame, date_col: str, 
                      test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally to prevent data leakage in time-series scenarios
        
        Args:
            df: DataFrame to split
            date_col: Column name containing dates
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Performing temporal split on {date_col}")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        # Calculate split indices
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_val = int((n_total - n_test) * val_size)
        
        # Split indices
        test_start_idx = n_total - n_test
        val_start_idx = test_start_idx - n_val
        
        # Create splits
        train_df = df_sorted.iloc[:val_start_idx].copy()
        val_df = df_sorted.iloc[val_start_idx:test_start_idx].copy()
        test_df = df_sorted.iloc[test_start_idx:].copy()
        
        logger.info(f"Temporal split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Validate no temporal leakage
        self._validate_temporal_split(train_df, val_df, test_df, date_col)
        
        return train_df, val_df, test_df
    
    def stratified_split(self, df: pd.DataFrame, target_col: str,
                        test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data using stratified sampling to maintain target distribution
        
        Args:
            df: DataFrame to split
            target_col: Target column name for stratification
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Performing stratified split on {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df[target_col],
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        adjusted_val_size = val_size / (1 - test_size)  # Adjust for remaining data
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df[target_col],
            random_state=self.random_state
        )
        
        logger.info(f"Stratified split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Validate stratification
        self._validate_stratified_split(train_df, val_df, test_df, target_col)
        
        return train_df, val_df, test_df
    
    def random_split(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly (for cases where stratification is not needed)
        
        Args:
            df: DataFrame to split
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing random split")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        adjusted_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=self.random_state
        )
        
        logger.info(f"Random split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_cv_splits(self, df: pd.DataFrame, target_col: str, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits with stratification
        
        Args:
            df: DataFrame to split
            target_col: Target column for stratification
            n_splits: Number of CV folds
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        logger.info(f"Creating {n_splits}-fold stratified CV splits")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        sss = StratifiedShuffleSplit(
            n_splits=n_splits, 
            test_size=1/n_splits, 
            random_state=self.random_state
        )
        
        splits = list(sss.split(df, df[target_col]))
        
        logger.info(f"Created {len(splits)} CV splits")
        return splits
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   dataset_name: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Save train/validation/test splits to files
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            dataset_name: Name for the dataset (used in filenames)
            output_dir: Directory to save files (defaults to config.PROCESSED_DATA_PATH)
        
        Returns:
            Dictionary mapping split names to file paths
        """
        output_dir = output_dir or config.PROCESSED_DATA_PATH
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        # Save each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            filename = f"{dataset_name}_{split_name}.parquet"
            filepath = output_dir / filename
            
            split_df.to_parquet(filepath, index=False)
            file_paths[split_name] = str(filepath)
            
            logger.info(f"Saved {split_name} split to {filepath} ({len(split_df)} rows)")
        
        return file_paths
    
    def _validate_temporal_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               test_df: pd.DataFrame, date_col: str) -> None:
        """Validate that temporal split has no leakage"""
        train_max_date = train_df[date_col].max()
        val_min_date = val_df[date_col].min()
        val_max_date = val_df[date_col].max()
        test_min_date = test_df[date_col].min()
        
        if train_max_date >= val_min_date:
            logger.warning("Potential temporal leakage: train max date >= validation min date")
        
        if val_max_date >= test_min_date:
            logger.warning("Potential temporal leakage: validation max date >= test min date")
        
        logger.info(f"Temporal validation: Train ends {train_max_date}, Val: {val_min_date} to {val_max_date}, Test starts {test_min_date}")
    
    def _validate_stratified_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                 test_df: pd.DataFrame, target_col: str) -> None:
        """Validate that stratified split maintains target distribution"""
        train_dist = train_df[target_col].value_counts(normalize=True).sort_index()
        val_dist = val_df[target_col].value_counts(normalize=True).sort_index()
        test_dist = test_df[target_col].value_counts(normalize=True).sort_index()
        
        logger.info("Target distribution validation:")
        logger.info(f"Train: {train_dist.to_dict()}")
        logger.info(f"Val: {val_dist.to_dict()}")
        logger.info(f"Test: {test_dist.to_dict()}")
        
        # Check if distributions are reasonably similar (within 5% tolerance)
        tolerance = 0.05
        for class_val in train_dist.index:
            train_prop = train_dist[class_val]
            val_prop = val_dist.get(class_val, 0)
            test_prop = test_dist.get(class_val, 0)
            
            if abs(train_prop - val_prop) > tolerance or abs(train_prop - test_prop) > tolerance:
                logger.warning(f"Class {class_val} distribution varies significantly across splits")
    
    def get_split_summary(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                         test_df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for the splits
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            target_col: Optional target column for distribution analysis
        
        Returns:
            Dictionary with split summary statistics
        """
        summary = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_proportion': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
            'val_proportion': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
            'test_proportion': len(test_df) / (len(train_df) + len(val_df) + len(test_df))
        }
        
        if target_col and target_col in train_df.columns:
            summary['target_distribution'] = {
                'train': train_df[target_col].value_counts(normalize=True).to_dict(),
                'val': val_df[target_col].value_counts(normalize=True).to_dict(),
                'test': test_df[target_col].value_counts(normalize=True).to_dict()
            }
        
        return summary