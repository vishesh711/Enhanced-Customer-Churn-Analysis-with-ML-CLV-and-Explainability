"""
Configuration settings for Customer Churn ML Pipeline
Contains environment settings, dataset IDs, and business parameters
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class Config:
    """Main configuration class for the churn prediction system"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_PATH: Path = PROJECT_ROOT / "data"
    RAW_DATA_PATH: Path = DATA_PATH / "raw"
    PROCESSED_DATA_PATH: Path = DATA_PATH / "processed"
    INTERIM_DATA_PATH: Path = DATA_PATH / "interim"
    
    # Model and artifact paths
    MODEL_PATH: Path = PROJECT_ROOT / "models"
    REPORTS_PATH: Path = PROJECT_ROOT / "reports"
    FIGURES_PATH: Path = REPORTS_PATH / "figures"
    TABLES_PATH: Path = REPORTS_PATH / "tables"
    
    # Dataset configuration
    TELCO_DATASET_ID: str = "blastchar/telco-customer-churn"
    OLIST_DATASET_ID: str = "olistbr/brazilian-ecommerce"
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Data splitting parameters
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.2
    
    # Cross-validation settings
    CV_FOLDS: int = 5
    CV_SCORING: str = "roc_auc"
    
    # Business parameters for cost-sensitive evaluation
    RETENTION_VALUE: float = 1000.0  # Average value of retaining a customer
    CONTACT_COST: float = 50.0       # Cost of contacting a customer for retention
    CHURN_COST: float = 500.0        # Cost of losing a customer
    
    # Feature engineering parameters
    NUMERIC_IMPUTATION: str = "median"
    CATEGORICAL_IMPUTATION: str = "most_frequent"
    OUTLIER_METHOD: str = "iqr"
    OUTLIER_THRESHOLD: float = 1.5
    WINSORIZATION_LIMITS: tuple = (0.01, 0.99)
    
    # Model training parameters
    CALIBRATION_METHOD: str = "isotonic"
    HYPERPARAMETER_TRIALS: int = 100
    EARLY_STOPPING_ROUNDS: int = 50
    
    # Clustering parameters
    MAX_CLUSTERS: int = 10
    CLUSTERING_FEATURES: list = None  # Will be set during feature engineering
    
    # CLV calculation parameters
    CLV_TIME_HORIZON: int = 12  # months
    DISCOUNT_RATE: float = 0.01  # monthly discount rate
    
    # Dashboard configuration
    DASHBOARD_HOST: str = "localhost"
    DASHBOARD_PORT: int = 8501
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model performance thresholds
    MIN_ROC_AUC: float = 0.7
    MIN_PR_AUC: float = 0.3
    MAX_BRIER_SCORE: float = 0.25
    
    # Feature selection parameters
    MAX_FEATURES: int = 100
    FEATURE_SELECTION_METHOD: str = "mutual_info"
    
    # Explanation parameters
    SHAP_SAMPLE_SIZE: int = 1000
    LIME_NUM_FEATURES: int = 10
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        directories = [
            self.DATA_PATH,
            self.RAW_DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.INTERIM_DATA_PATH,
            self.MODEL_PATH,
            self.REPORTS_PATH,
            self.FIGURES_PATH,
            self.TABLES_PATH
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if they exist
        if os.getenv('RETENTION_VALUE'):
            config.RETENTION_VALUE = float(os.getenv('RETENTION_VALUE'))
        
        if os.getenv('CONTACT_COST'):
            config.CONTACT_COST = float(os.getenv('CONTACT_COST'))
        
        if os.getenv('RANDOM_SEED'):
            config.RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
        
        if os.getenv('CV_FOLDS'):
            config.CV_FOLDS = int(os.getenv('CV_FOLDS'))
        
        if os.getenv('LOG_LEVEL'):
            config.LOG_LEVEL = os.getenv('LOG_LEVEL')
        
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'random_state': self.RANDOM_SEED,
            'cv_folds': self.CV_FOLDS,
            'scoring': self.CV_SCORING,
            'calibration_method': self.CALIBRATION_METHOD,
            'hyperparameter_trials': self.HYPERPARAMETER_TRIALS
        }
    
    def get_business_config(self) -> Dict[str, Any]:
        """Get business evaluation configuration"""
        return {
            'retention_value': self.RETENTION_VALUE,
            'contact_cost': self.CONTACT_COST,
            'churn_cost': self.CHURN_COST,
            'clv_time_horizon': self.CLV_TIME_HORIZON,
            'discount_rate': self.DISCOUNT_RATE
        }
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return {
            'numeric_imputation': self.NUMERIC_IMPUTATION,
            'categorical_imputation': self.CATEGORICAL_IMPUTATION,
            'outlier_method': self.OUTLIER_METHOD,
            'outlier_threshold': self.OUTLIER_THRESHOLD,
            'winsorization_limits': self.WINSORIZATION_LIMITS,
            'max_features': self.MAX_FEATURES,
            'feature_selection_method': self.FEATURE_SELECTION_METHOD
        }


# Global configuration instance
config = Config.from_env()