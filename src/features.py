"""
Feature engineering module for Customer Churn ML Pipeline
Handles feature transformations, temporal features, and pipeline orchestration
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder, TargetEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import chi2_contingency

from config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering operations"""
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    scaling_method: str = "standard"  # standard, minmax, robust
    encoding_method: str = "onehot"   # onehot, target, label
    outlier_treatment: str = "winsorize"  # winsorize, clip, remove
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 1.5
    winsorization_limits: Tuple[float, float] = (0.01, 0.99)
    interaction_degree: int = 2
    max_categorical_cardinality: int = 10  # Threshold for target encoding
    max_features: int = 100
    feature_selection_method: str = "mutual_info"


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Handles basic feature transformations including numeric scaling, 
    categorical encoding, and interaction feature generation
    """
    
    def __init__(self, config_params: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureTransformer
        
        Args:
            config_params: Dictionary of configuration parameters
        """
        self.config_params = config_params or config.get_feature_config()
        
        # Filter config params to only include valid FeatureConfig fields
        valid_fields = {field.name for field in FeatureConfig.__dataclass_fields__.values()}
        filtered_params = {k: v for k, v in self.config_params.items() if k in valid_fields}
        self.feature_config = FeatureConfig(**filtered_params)
        
        # Initialize transformers
        self.numeric_pipeline = None
        self.categorical_pipeline = None
        self.interaction_features = []
        self.feature_names_ = []
        self.numeric_features_ = []
        self.categorical_features_ = []
        
        logger.info("FeatureTransformer initialized")
    
    def create_numeric_pipeline(self, numeric_features: List[str]) -> Pipeline:
        """
        Create preprocessing pipeline for numeric features
        
        Args:
            numeric_features: List of numeric feature names
            
        Returns:
            sklearn Pipeline for numeric preprocessing
        """
        logger.info(f"Creating numeric pipeline for {len(numeric_features)} features")
        
        steps = []
        
        # Imputation
        if self.feature_config.numeric_imputation == "median":
            imputer = SimpleImputer(strategy="median")
        elif self.feature_config.numeric_imputation == "mean":
            imputer = SimpleImputer(strategy="mean")
        else:
            imputer = SimpleImputer(strategy="constant", fill_value=0)
        
        steps.append(("imputer", imputer))
        
        # Outlier treatment (winsorization)
        if self.feature_config.outlier_treatment == "winsorize":
            steps.append(("winsorizer", WinsorizerTransformer(
                limits=self.feature_config.winsorization_limits
            )))
        
        # Scaling
        if self.feature_config.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.feature_config.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif self.feature_config.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()  # Default
        
        steps.append(("scaler", scaler))
        
        pipeline = Pipeline(steps)
        logger.info(f"Numeric pipeline created with steps: {[step[0] for step in steps]}")
        
        return pipeline
    
    def create_categorical_pipeline(self, categorical_features: List[str], 
                                  target: Optional[pd.Series] = None) -> Pipeline:
        """
        Create preprocessing pipeline for categorical features
        
        Args:
            categorical_features: List of categorical feature names
            target: Target variable for target encoding (optional)
            
        Returns:
            sklearn Pipeline for categorical preprocessing
        """
        logger.info(f"Creating categorical pipeline for {len(categorical_features)} features")
        
        steps = []
        
        # Imputation
        imputer = SimpleImputer(strategy=self.feature_config.categorical_imputation)
        steps.append(("imputer", imputer))
        
        # Encoding
        if self.feature_config.encoding_method == "onehot":
            encoder = OneHotEncoder(
                drop="first",  # Avoid multicollinearity
                sparse_output=False,
                handle_unknown="ignore"
            )
        elif self.feature_config.encoding_method == "target" and target is not None:
            encoder = TargetEncoder(
                smooth="auto",
                target_type="binary"
            )
        else:
            # Fallback to label encoding
            encoder = LabelEncoder()
        
        steps.append(("encoder", encoder))
        
        pipeline = Pipeline(steps)
        logger.info(f"Categorical pipeline created with steps: {[step[0] for step in steps]}")
        
        return pipeline
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  interactions: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Create interaction features between specified columns
        
        Args:
            df: Input DataFrame
            interactions: List of column pairs for interactions (if None, auto-generate)
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features")
        
        df_interactions = df.copy()
        
        if interactions is None:
            # Auto-generate interactions for numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            interactions = []
            
            # Create interactions between key numeric features
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    interactions.append((col1, col2))
            
            # Limit number of interactions to avoid feature explosion
            interactions = interactions[:20]  # Top 20 interactions
        
        interaction_names = []
        
        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                # Multiplicative interaction
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
                interaction_names.append(interaction_name)
                
                # Ratio interaction (if col2 is not zero)
                if (df[col2] != 0).all():
                    ratio_name = f"{col1}_div_{col2}"
                    df_interactions[ratio_name] = df[col1] / df[col2]
                    interaction_names.append(ratio_name)
        
        self.interaction_features = interaction_names
        logger.info(f"Created {len(interaction_names)} interaction features")
        
        return df_interactions
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureTransformer':
        """
        Fit the feature transformer on training data
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional, needed for target encoding)
            
        Returns:
            self
        """
        logger.info(f"Fitting FeatureTransformer on data with shape {X.shape}")
        
        # Identify feature types
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numeric_features_)} numeric and {len(self.categorical_features_)} categorical features")
        
        # Create and fit pipelines
        if self.numeric_features_:
            self.numeric_pipeline = self.create_numeric_pipeline(self.numeric_features_)
            self.numeric_pipeline.fit(X[self.numeric_features_])
        
        if self.categorical_features_:
            self.categorical_pipeline = self.create_categorical_pipeline(self.categorical_features_, y)
            if self.feature_config.encoding_method == "target" and y is not None:
                self.categorical_pipeline.fit(X[self.categorical_features_], y)
            else:
                self.categorical_pipeline.fit(X[self.categorical_features_])
        
        # Store feature names for later use
        self._update_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted transformers
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info(f"Transforming data with shape {X.shape}")
        
        transformed_parts = []
        
        # Transform numeric features
        if self.numeric_features_ and self.numeric_pipeline:
            numeric_transformed = self.numeric_pipeline.transform(X[self.numeric_features_])
            numeric_df = pd.DataFrame(
                numeric_transformed, 
                columns=self.numeric_features_,
                index=X.index
            )
            transformed_parts.append(numeric_df)
        
        # Transform categorical features
        if self.categorical_features_ and self.categorical_pipeline:
            categorical_transformed = self.categorical_pipeline.transform(X[self.categorical_features_])
            
            # Handle different encoder outputs
            if hasattr(self.categorical_pipeline.named_steps['encoder'], 'get_feature_names_out'):
                # OneHotEncoder case
                cat_feature_names = self.categorical_pipeline.named_steps['encoder'].get_feature_names_out(
                    self.categorical_features_
                )
            else:
                # TargetEncoder or LabelEncoder case
                cat_feature_names = self.categorical_features_
            
            categorical_df = pd.DataFrame(
                categorical_transformed,
                columns=cat_feature_names,
                index=X.index
            )
            transformed_parts.append(categorical_df)
        
        # Combine all transformed features
        if transformed_parts:
            X_transformed = pd.concat(transformed_parts, axis=1)
        else:
            X_transformed = X.copy()
        
        # Add interaction features
        X_transformed = self.create_interaction_features(X_transformed)
        
        logger.info(f"Transformation completed. Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit transformer and transform data in one step
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names after transformation
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def _update_feature_names(self, X: pd.DataFrame) -> None:
        """Update feature names after fitting"""
        feature_names = []
        
        # Add numeric feature names
        if self.numeric_features_:
            feature_names.extend(self.numeric_features_)
        
        # Add categorical feature names
        if self.categorical_features_ and self.categorical_pipeline:
            if hasattr(self.categorical_pipeline.named_steps['encoder'], 'get_feature_names_out'):
                cat_names = self.categorical_pipeline.named_steps['encoder'].get_feature_names_out(
                    self.categorical_features_
                )
                feature_names.extend(cat_names)
            else:
                feature_names.extend(self.categorical_features_)
        
        self.feature_names_ = feature_names


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for winsorizing outliers in numeric features
    """
    
    def __init__(self, limits: Tuple[float, float] = (0.01, 0.99)):
        """
        Initialize winsorizer
        
        Args:
            limits: Lower and upper percentile limits for winsorization
        """
        self.limits = limits
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'WinsorizerTransformer':
        """
        Fit winsorizer by computing percentile bounds
        
        Args:
            X: Input data
            y: Target (ignored)
            
        Returns:
            self
        """
        X_array = np.asarray(X)
        self.lower_bounds_ = np.percentile(X_array, self.limits[0] * 100, axis=0)
        self.upper_bounds_ = np.percentile(X_array, self.limits[1] * 100, axis=0)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply winsorization to input data
        
        Args:
            X: Input data
            
        Returns:
            Winsorized data
        """
        X_array = np.asarray(X)
        X_winsorized = np.clip(X_array, self.lower_bounds_, self.upper_bounds_)
        
        return X_winsorized


class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates temporal features including RFM analysis and behavioral patterns
    """
    
    def __init__(self, customer_col: str = 'customer_id', 
                 date_col: str = 'date', 
                 value_col: str = 'value',
                 reference_date: Optional[datetime] = None):
        """
        Initialize TimeFeatureGenerator
        
        Args:
            customer_col: Column name for customer identifier
            date_col: Column name for date/timestamp
            value_col: Column name for monetary value
            reference_date: Reference date for recency calculation (defaults to max date)
        """
        self.customer_col = customer_col
        self.date_col = date_col
        self.value_col = value_col
        self.reference_date = reference_date
        self.rfm_features_ = []
        
        logger.info("TimeFeatureGenerator initialized")
    
    def generate_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RFM (Recency, Frequency, Monetary) features
        
        Args:
            df: Input DataFrame with customer transactions
            
        Returns:
            DataFrame with RFM features aggregated by customer
        """
        logger.info("Generating RFM features")
        
        # Ensure date column is datetime
        if df[self.date_col].dtype == 'object':
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Set reference date if not provided
        if self.reference_date is None:
            self.reference_date = df[self.date_col].max()
        
        # Calculate RFM metrics
        rfm_df = df.groupby(self.customer_col).agg({
            self.date_col: ['max', 'min', 'count'],
            self.value_col: ['sum', 'mean', 'std', 'count']
        }).round(2)
        
        # Flatten column names
        rfm_df.columns = [f"{col[1]}_{col[0]}" for col in rfm_df.columns]
        
        # Calculate recency (days since last transaction)
        rfm_df['recency_days'] = (self.reference_date - rfm_df[f'max_{self.date_col}']).dt.days
        
        # Calculate frequency (number of transactions)
        rfm_df['frequency'] = rfm_df[f'count_{self.date_col}']
        
        # Calculate monetary value (total and average)
        rfm_df['monetary_total'] = rfm_df[f'sum_{self.value_col}']
        rfm_df['monetary_avg'] = rfm_df[f'mean_{self.value_col}']
        
        # Calculate customer lifetime (days between first and last transaction)
        rfm_df['customer_lifetime_days'] = (
            rfm_df[f'max_{self.date_col}'] - rfm_df[f'min_{self.date_col}']
        ).dt.days
        
        # Calculate transaction frequency rate (transactions per day)
        rfm_df['transaction_rate'] = rfm_df['frequency'] / (rfm_df['customer_lifetime_days'] + 1)
        
        # Calculate coefficient of variation for monetary values
        rfm_df['monetary_cv'] = rfm_df[f'std_{self.value_col}'] / (rfm_df['monetary_avg'] + 1e-8)
        
        # Create RFM scores (quintiles)
        rfm_df['recency_score'] = pd.qcut(rfm_df['recency_days'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary_total'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Create combined RFM score
        rfm_df['rfm_score'] = (
            rfm_df['recency_score'].astype(float) * 100 +
            rfm_df['frequency_score'].astype(float) * 10 +
            rfm_df['monetary_score'].astype(float)
        )
        
        # Reset index to make customer_col a regular column
        rfm_df = rfm_df.reset_index()
        
        # Store feature names
        self.rfm_features_ = [col for col in rfm_df.columns if col != self.customer_col]
        
        logger.info(f"Generated {len(self.rfm_features_)} RFM features")
        
        return rfm_df
    
    def calculate_recency(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate recency (days since last transaction) for each customer
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with recency values
        """
        if self.reference_date is None:
            self.reference_date = df[self.date_col].max()
        
        last_transaction = df.groupby(self.customer_col)[self.date_col].max()
        recency = (self.reference_date - last_transaction).dt.days
        
        return recency
    
    def calculate_frequency(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate frequency (number of transactions) for each customer
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with frequency values
        """
        frequency = df.groupby(self.customer_col)[self.date_col].count()
        
        return frequency
    
    def calculate_monetary(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate monetary value (total spending) for each customer
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with monetary values
        """
        monetary = df.groupby(self.customer_col)[self.value_col].sum()
        
        return monetary
    
    def generate_rolling_features(self, df: pd.DataFrame, 
                                windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Generate rolling window aggregations for behavioral patterns
        
        Args:
            df: Input DataFrame sorted by customer and date
            windows: List of window sizes in days
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Generating rolling features for windows: {windows}")
        
        # Ensure data is sorted
        df_sorted = df.sort_values([self.customer_col, self.date_col])
        
        rolling_features = []
        
        for window in windows:
            # Rolling sum of values
            rolling_sum = df_sorted.groupby(self.customer_col)[self.value_col].rolling(
                window=f'{window}D', on=self.date_col
            ).sum().reset_index(level=0, drop=True)
            
            # Rolling count of transactions
            rolling_count = df_sorted.groupby(self.customer_col)[self.value_col].rolling(
                window=f'{window}D', on=self.date_col
            ).count().reset_index(level=0, drop=True)
            
            # Rolling mean
            rolling_mean = df_sorted.groupby(self.customer_col)[self.value_col].rolling(
                window=f'{window}D', on=self.date_col
            ).mean().reset_index(level=0, drop=True)
            
            # Add to DataFrame
            df_sorted[f'rolling_{window}d_sum'] = rolling_sum
            df_sorted[f'rolling_{window}d_count'] = rolling_count
            df_sorted[f'rolling_{window}d_mean'] = rolling_mean
            
            rolling_features.extend([
                f'rolling_{window}d_sum',
                f'rolling_{window}d_count', 
                f'rolling_{window}d_mean'
            ])
        
        logger.info(f"Generated {len(rolling_features)} rolling features")
        
        return df_sorted
    
    def generate_session_features(self, df: pd.DataFrame, 
                                session_gap_hours: int = 24) -> pd.DataFrame:
        """
        Generate session-based features using gap analysis
        
        Args:
            df: Input DataFrame
            session_gap_hours: Hours gap to define new session
            
        Returns:
            DataFrame with session features
        """
        logger.info(f"Generating session features with {session_gap_hours}h gap")
        
        df_sessions = df.sort_values([self.customer_col, self.date_col]).copy()
        
        # Calculate time gaps between consecutive transactions
        df_sessions['time_gap'] = df_sessions.groupby(self.customer_col)[self.date_col].diff()
        
        # Identify session starts (gap > threshold or first transaction)
        session_threshold = pd.Timedelta(hours=session_gap_hours)
        df_sessions['session_start'] = (
            (df_sessions['time_gap'] > session_threshold) | 
            (df_sessions['time_gap'].isna())
        )
        
        # Assign session IDs
        df_sessions['session_id'] = df_sessions.groupby(self.customer_col)['session_start'].cumsum()
        
        # Calculate session-level features
        session_features = df_sessions.groupby([self.customer_col, 'session_id']).agg({
            self.date_col: ['min', 'max', 'count'],
            self.value_col: ['sum', 'mean'],
            'time_gap': 'sum'
        })
        
        # Flatten column names
        session_features.columns = [f"session_{col[1]}_{col[0]}" for col in session_features.columns]
        
        # Calculate session duration
        session_features['session_duration_hours'] = (
            session_features[f'session_max_{self.date_col}'] - 
            session_features[f'session_min_{self.date_col}']
        ).dt.total_seconds() / 3600
        
        # Aggregate to customer level
        customer_session_features = session_features.groupby(self.customer_col).agg({
            f'session_count_{self.date_col}': ['mean', 'std', 'sum'],
            f'session_sum_{self.value_col}': ['mean', 'std'],
            'session_duration_hours': ['mean', 'std', 'max']
        })
        
        # Flatten column names
        customer_session_features.columns = [
            f"{col[1]}_{col[0]}" for col in customer_session_features.columns
        ]
        
        customer_session_features = customer_session_features.reset_index()
        
        logger.info(f"Generated session features for {len(customer_session_features)} customers")
        
        return customer_session_features
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeFeatureGenerator':
        """
        Fit the time feature generator (mainly for storing reference date)
        
        Args:
            X: Input DataFrame
            y: Target (ignored)
            
        Returns:
            self
        """
        if self.reference_date is None and self.date_col in X.columns:
            self.reference_date = pd.to_datetime(X[self.date_col]).max()
            logger.info(f"Set reference date to {self.reference_date}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding temporal features
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Transforming data with temporal features")
        
        # Generate RFM features if we have the required columns
        if all(col in X.columns for col in [self.customer_col, self.date_col, self.value_col]):
            rfm_features = self.generate_rfm_features(X)
            
            # Merge with original data
            X_transformed = X.merge(rfm_features, on=self.customer_col, how='left')
        else:
            logger.warning("Required columns for RFM features not found, skipping temporal features")
            X_transformed = X.copy()
        
        return X_transformed
    
    def generate_behavioral_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate customer behavior pattern features
        
        Args:
            df: Input DataFrame with customer transactions
            
        Returns:
            DataFrame with behavioral pattern features
        """
        logger.info("Generating behavioral pattern features")
        
        # Ensure date column is datetime
        if df[self.date_col].dtype == 'object':
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        behavioral_features = []
        
        # Group by customer for aggregations
        customer_groups = df.groupby(self.customer_col)
        
        # Time-based patterns
        df['hour'] = df[self.date_col].dt.hour
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['month'] = df[self.date_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Customer-level behavioral aggregations
        behavioral_df = customer_groups.agg({
            'hour': ['mean', 'std'],  # Preferred transaction times
            'day_of_week': ['mean', 'std'],  # Preferred days
            'month': ['nunique'],  # Seasonal activity
            'is_weekend': ['mean'],  # Weekend vs weekday preference
            self.value_col: ['std', 'min', 'max']  # Spending variability
        }).round(2)
        
        # Flatten column names
        behavioral_df.columns = [f"behavior_{col[1]}_{col[0]}" for col in behavioral_df.columns]
        
        # Calculate spending patterns
        behavioral_df['spending_volatility'] = (
            behavioral_df[f'behavior_std_{self.value_col}'] / 
            (behavioral_df[f'behavior_mean_{self.value_col}'] + 1e-8)
        )
        
        # Calculate transaction timing consistency
        behavioral_df['time_consistency'] = 1 / (1 + behavioral_df['behavior_std_hour'])
        behavioral_df['day_consistency'] = 1 / (1 + behavioral_df['behavior_std_day_of_week'])
        
        # Calculate value range (max - min spending)
        behavioral_df['spending_range'] = (
            behavioral_df[f'behavior_max_{self.value_col}'] - 
            behavioral_df[f'behavior_min_{self.value_col}']
        )
        
        # Reset index
        behavioral_df = behavioral_df.reset_index()
        
        logger.info(f"Generated behavioral pattern features for {len(behavioral_df)} customers")
        
        return behavioral_df
    
    def generate_trend_features(self, df: pd.DataFrame, 
                              periods: List[int] = [30, 90, 180]) -> pd.DataFrame:
        """
        Generate trend features showing changes over time periods
        
        Args:
            df: Input DataFrame with customer transactions
            periods: List of periods in days to calculate trends
            
        Returns:
            DataFrame with trend features
        """
        logger.info(f"Generating trend features for periods: {periods}")
        
        # Ensure data is sorted
        df_sorted = df.sort_values([self.customer_col, self.date_col])
        
        trend_features = []
        
        if self.reference_date is None:
            self.reference_date = df_sorted[self.date_col].max()
        
        for period in periods:
            # Define period boundaries
            period_start = self.reference_date - pd.Timedelta(days=period)
            
            # Filter data for this period
            period_data = df_sorted[df_sorted[self.date_col] >= period_start]
            
            if len(period_data) == 0:
                continue
            
            # Calculate period metrics
            period_metrics = period_data.groupby(self.customer_col).agg({
                self.date_col: 'count',
                self.value_col: ['sum', 'mean']
            })
            
            # Flatten column names
            period_metrics.columns = [f"trend_{period}d_{col[1]}_{col[0]}" for col in period_metrics.columns]
            
            trend_features.append(period_metrics)
        
        # Combine all trend features
        if trend_features:
            combined_trends = trend_features[0]
            for trend_df in trend_features[1:]:
                combined_trends = combined_trends.join(trend_df, how='outer')
            
            # Calculate trend ratios (recent vs older periods)
            if len(periods) >= 2:
                # Compare most recent period to previous period
                recent_period = periods[0]
                older_period = periods[1]
                
                recent_freq_col = f"trend_{recent_period}d_count_{self.date_col}"
                older_freq_col = f"trend_{older_period}d_count_{self.date_col}"
                
                if recent_freq_col in combined_trends.columns and older_freq_col in combined_trends.columns:
                    combined_trends['frequency_trend_ratio'] = (
                        combined_trends[recent_freq_col] / 
                        (combined_trends[older_freq_col] + 1e-8)
                    )
                
                recent_value_col = f"trend_{recent_period}d_sum_{self.value_col}"
                older_value_col = f"trend_{older_period}d_sum_{self.value_col}"
                
                if recent_value_col in combined_trends.columns and older_value_col in combined_trends.columns:
                    combined_trends['spending_trend_ratio'] = (
                        combined_trends[recent_value_col] / 
                        (combined_trends[older_value_col] + 1e-8)
                    )
            
            combined_trends = combined_trends.reset_index()
            
            logger.info(f"Generated trend features for {len(combined_trends)} customers")
            
            return combined_trends
        else:
            logger.warning("No trend features generated - insufficient data")
            return pd.DataFrame()
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            X: Input DataFrame
            y: Target (ignored)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Orchestrates the complete feature engineering pipeline with sklearn compatibility
    Includes preprocessing, feature selection, and dimensionality reduction
    """
    
    def __init__(self, 
                 feature_config: Optional[Dict[str, Any]] = None,
                 include_temporal: bool = True,
                 include_interactions: bool = True,
                 feature_selection_method: Optional[str] = None,
                 max_features: Optional[int] = None,
                 dimensionality_reduction: Optional[str] = None,
                 n_components: Optional[int] = None):
        """
        Initialize FeaturePipeline
        
        Args:
            feature_config: Configuration for feature transformations
            include_temporal: Whether to include temporal features
            include_interactions: Whether to include interaction features
            feature_selection_method: Method for feature selection ('mutual_info', 'f_classif', 'variance')
            max_features: Maximum number of features to select
            dimensionality_reduction: Method for dimensionality reduction ('pca')
            n_components: Number of components for dimensionality reduction
        """
        self.feature_config = feature_config or config.get_feature_config()
        self.include_temporal = include_temporal
        self.include_interactions = include_interactions
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features or self.feature_config.get('max_features', 100)
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        
        # Initialize components
        self.feature_transformer = None
        self.time_feature_generator = None
        self.column_transformer = None
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.pipeline = None
        
        # Feature tracking
        self.feature_names_in_ = []
        self.feature_names_out_ = []
        self.selected_features_ = []
        
        logger.info("FeaturePipeline initialized")
    
    def _create_column_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create sklearn ColumnTransformer for preprocessing
        
        Args:
            X: Input DataFrame
            
        Returns:
            Configured ColumnTransformer
        """
        # Identify feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        # Numeric transformer
        if numeric_features:
            numeric_steps = [
                ('imputer', SimpleImputer(strategy=self.feature_config.get('numeric_imputation', 'median'))),
                ('winsorizer', WinsorizerTransformer(
                    limits=self.feature_config.get('winsorization_limits', (0.01, 0.99))
                )),
                ('scaler', StandardScaler())
            ]
            numeric_transformer = Pipeline(numeric_steps)
            transformers.append(('numeric', numeric_transformer, numeric_features))
        
        # Categorical transformer
        if categorical_features:
            categorical_steps = [
                ('imputer', SimpleImputer(strategy=self.feature_config.get('categorical_imputation', 'most_frequent'))),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]
            categorical_transformer = Pipeline(categorical_steps)
            transformers.append(('categorical', categorical_transformer, categorical_features))
        
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            sparse_threshold=0
        )
        
        return column_transformer
    
    def _create_feature_selector(self, method: str, max_features: int) -> BaseEstimator:
        """
        Create feature selector based on specified method
        
        Args:
            method: Feature selection method
            max_features: Maximum number of features to select
            
        Returns:
            Configured feature selector
        """
        if method == 'mutual_info':
            return SelectKBest(score_func=mutual_info_classif, k=max_features)
        elif method == 'f_classif':
            return SelectKBest(score_func=f_classif, k=max_features)
        elif method == 'variance':
            return VarianceThreshold(threshold=0.01)
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return None
    
    def _create_dimensionality_reducer(self, method: str, n_components: int) -> BaseEstimator:
        """
        Create dimensionality reducer
        
        Args:
            method: Dimensionality reduction method
            n_components: Number of components
            
        Returns:
            Configured dimensionality reducer
        """
        if method == 'pca':
            return PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        else:
            logger.warning(f"Unknown dimensionality reduction method: {method}")
            return None
    
    def _build_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Pipeline:
        """
        Build the complete feature engineering pipeline
        
        Args:
            X: Input DataFrame
            y: Target variable (optional)
            
        Returns:
            Configured Pipeline
        """
        steps = []
        
        # Step 1: Column transformation (preprocessing)
        self.column_transformer = self._create_column_transformer(X)
        steps.append(('preprocessing', self.column_transformer))
        
        # Step 2: Feature selection (if specified)
        if self.feature_selection_method:
            self.feature_selector = self._create_feature_selector(
                self.feature_selection_method, 
                self.max_features
            )
            if self.feature_selector:
                steps.append(('feature_selection', self.feature_selector))
        
        # Step 3: Dimensionality reduction (if specified)
        if self.dimensionality_reduction and self.n_components:
            self.dimensionality_reducer = self._create_dimensionality_reducer(
                self.dimensionality_reduction,
                self.n_components
            )
            if self.dimensionality_reducer:
                steps.append(('dimensionality_reduction', self.dimensionality_reducer))
        
        pipeline = Pipeline(steps)
        
        return pipeline
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeaturePipeline':
        """
        Fit the feature pipeline
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            
        Returns:
            self
        """
        logger.info(f"Fitting FeaturePipeline on data with shape {X.shape}")
        
        # Store input feature names
        self.feature_names_in_ = X.columns.tolist()
        
        # Handle temporal features if requested
        X_processed = X.copy()
        
        if self.include_temporal:
            # Check if we have the required columns for temporal features
            temporal_cols = ['customer_id', 'date', 'value']  # Default column names
            
            # Try to infer temporal columns from data
            date_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
            id_cols = [col for col in X.columns if 'id' in col.lower() or 'customer' in col.lower()]
            value_cols = [col for col in X.columns if 'amount' in col.lower() or 'value' in col.lower() or 'price' in col.lower()]
            
            if date_cols and id_cols and (value_cols or X.select_dtypes(include=[np.number]).columns.tolist()):
                # Use inferred columns
                customer_col = id_cols[0]
                date_col = date_cols[0]
                value_col = value_cols[0] if value_cols else X.select_dtypes(include=[np.number]).columns[0]
                
                self.time_feature_generator = TimeFeatureGenerator(
                    customer_col=customer_col,
                    date_col=date_col,
                    value_col=value_col
                )
                
                try:
                    X_processed = self.time_feature_generator.fit_transform(X_processed, y)
                    logger.info("Temporal features added successfully")
                except Exception as e:
                    logger.warning(f"Could not generate temporal features: {str(e)}")
                    self.include_temporal = False
            else:
                logger.warning("Required columns for temporal features not found, skipping")
                self.include_temporal = False
        
        # Build and fit the main pipeline
        self.pipeline = self._build_pipeline(X_processed, y)
        self.pipeline.fit(X_processed, y)
        
        # Update output feature names
        self._update_output_feature_names(X_processed)
        
        logger.info(f"FeaturePipeline fitted. Output features: {len(self.feature_names_out_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using the fitted pipeline
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed feature array
        """
        logger.info(f"Transforming data with shape {X.shape}")
        
        # Handle temporal features if they were included during fit
        X_processed = X.copy()
        
        if self.include_temporal and self.time_feature_generator:
            try:
                X_processed = self.time_feature_generator.transform(X_processed)
            except Exception as e:
                logger.warning(f"Could not apply temporal features during transform: {str(e)}")
        
        # Apply main pipeline
        X_transformed = self.pipeline.transform(X_processed)
        
        logger.info(f"Transformation completed. Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit pipeline and transform data in one step
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            
        Returns:
            Transformed feature array
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names after transformation
        
        Returns:
            List of output feature names
        """
        return self.feature_names_out_
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores if feature selection was used
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_selector and hasattr(self.feature_selector, 'scores_'):
            # Get feature names before selection
            if hasattr(self.column_transformer, 'get_feature_names_out'):
                feature_names = self.column_transformer.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(len(self.feature_selector.scores_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_selector.scores_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def get_selected_features(self) -> List[str]:
        """
        Get names of selected features after feature selection
        
        Returns:
            List of selected feature names
        """
        return self.selected_features_
    
    def _update_output_feature_names(self, X: pd.DataFrame) -> None:
        """
        Update output feature names based on pipeline transformations
        
        Args:
            X: Input DataFrame after temporal feature generation
        """
        try:
            # Get feature names after preprocessing
            if hasattr(self.column_transformer, 'get_feature_names_out'):
                preprocessed_names = self.column_transformer.get_feature_names_out()
            else:
                # Fallback: generate generic names
                n_features = self.column_transformer.transform(X).shape[1]
                preprocessed_names = [f"feature_{i}" for i in range(n_features)]
            
            current_names = list(preprocessed_names)
            
            # Update names after feature selection
            if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                selected_mask = self.feature_selector.get_support()
                current_names = [name for name, selected in zip(current_names, selected_mask) if selected]
                self.selected_features_ = current_names.copy()
            
            # Update names after dimensionality reduction
            if self.dimensionality_reducer:
                n_components = self.dimensionality_reducer.n_components_
                current_names = [f"component_{i}" for i in range(n_components)]
            
            self.feature_names_out_ = current_names
            
        except Exception as e:
            logger.warning(f"Could not determine output feature names: {str(e)}")
            # Fallback to generic names
            try:
                n_features = self.pipeline.transform(X).shape[1]
                self.feature_names_out_ = [f"feature_{i}" for i in range(n_features)]
            except:
                self.feature_names_out_ = []
    
    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted pipeline to disk
        
        Args:
            filepath: Path to save the pipeline
        """
        import joblib
        
        pipeline_data = {
            'pipeline': self.pipeline,
            'time_feature_generator': self.time_feature_generator,
            'feature_names_in_': self.feature_names_in_,
            'feature_names_out_': self.feature_names_out_,
            'selected_features_': self.selected_features_,
            'config': {
                'include_temporal': self.include_temporal,
                'include_interactions': self.include_interactions,
                'feature_selection_method': self.feature_selection_method,
                'max_features': self.max_features,
                'dimensionality_reduction': self.dimensionality_reduction,
                'n_components': self.n_components
            }
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: Union[str, Path]) -> 'FeaturePipeline':
        """
        Load a fitted pipeline from disk
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            Loaded FeaturePipeline instance
        """
        import joblib
        
        pipeline_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            include_temporal=pipeline_data['config']['include_temporal'],
            include_interactions=pipeline_data['config']['include_interactions'],
            feature_selection_method=pipeline_data['config']['feature_selection_method'],
            max_features=pipeline_data['config']['max_features'],
            dimensionality_reduction=pipeline_data['config']['dimensionality_reduction'],
            n_components=pipeline_data['config']['n_components']
        )
        
        # Restore fitted components
        instance.pipeline = pipeline_data['pipeline']
        instance.time_feature_generator = pipeline_data['time_feature_generator']
        instance.feature_names_in_ = pipeline_data['feature_names_in_']
        instance.feature_names_out_ = pipeline_data['feature_names_out_']
        instance.selected_features_ = pipeline_data['selected_features_']
        
        logger.info(f"Pipeline loaded from {filepath}")
        
        return instance