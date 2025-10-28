"""
Exploratory Data Analysis (EDA) module for Customer Churn ML Pipeline
Provides comprehensive analysis functions for univariate, bivariate, and correlation analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr, spearmanr
from scipy.stats.contingency import association
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configure matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class UnivariateAnalysisResult:
    """Results from univariate analysis"""
    column_name: str
    data_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    
    # Numeric statistics
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical statistics
    mode: Optional[str] = None
    mode_frequency: Optional[int] = None
    mode_percentage: Optional[float] = None
    top_categories: Optional[Dict[str, int]] = None


@dataclass
class BivariateAnalysisResult:
    """Results from bivariate analysis"""
    feature_name: str
    target_name: str
    feature_type: str
    
    # Statistical test results
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    test_name: Optional[str] = None
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    
    # Descriptive statistics
    churn_rates_by_category: Optional[Dict[str, float]] = None
    mean_by_churn: Optional[Dict[str, float]] = None
    correlation_coefficient: Optional[float] = None


@dataclass
class CorrelationAnalysisResult:
    """Results from correlation analysis"""
    correlation_matrix: pd.DataFrame
    correlation_method: str
    significant_correlations: List[Tuple[str, str, float, float]]  # (var1, var2, correlation, p_value)
    high_correlations: List[Tuple[str, str, float]]  # (var1, var2, correlation) where |correlation| > threshold


class EDAAnalyzer:
    """
    Comprehensive exploratory data analysis framework
    Handles univariate, bivariate, and correlation analysis for churn prediction
    """
    
    def __init__(self, significance_level: float = 0.05, correlation_threshold: float = 0.7):
        """
        Initialize EDA analyzer
        
        Args:
            significance_level: Alpha level for statistical tests
            correlation_threshold: Threshold for identifying high correlations
        """
        self.significance_level = significance_level
        self.correlation_threshold = correlation_threshold
        self.results = {}
        
        # Configure plotting
        self.figure_size = (10, 6)
        self.color_palette = sns.color_palette("husl", 8)
        
    def perform_univariate_analysis(self, df: pd.DataFrame, 
                                  columns: Optional[List[str]] = None) -> Dict[str, UnivariateAnalysisResult]:
        """
        Perform comprehensive univariate analysis on specified columns
        
        Args:
            df: DataFrame to analyze
            columns: List of columns to analyze (if None, analyzes all columns)
            
        Returns:
            Dictionary mapping column names to UnivariateAnalysisResult objects
        """
        logger.info("Starting univariate analysis")
        
        if columns is None:
            columns = df.columns.tolist()
        
        results = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
                
            logger.debug(f"Analyzing column: {col}")
            
            # Basic statistics
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(df[col]):
                data_type = "numeric"
                result = self._analyze_numeric_column(df[col], col, missing_count, 
                                                   missing_percentage, unique_count)
            else:
                data_type = "categorical"
                result = self._analyze_categorical_column(df[col], col, missing_count,
                                                        missing_percentage, unique_count)
            
            results[col] = result
            
        logger.info(f"Completed univariate analysis for {len(results)} columns")
        self.results['univariate'] = results
        return results
    
    def perform_bivariate_analysis(self, df: pd.DataFrame, target_col: str,
                                 feature_cols: Optional[List[str]] = None) -> Dict[str, BivariateAnalysisResult]:
        """
        Perform bivariate analysis between features and target variable
        
        Args:
            df: DataFrame to analyze
            target_col: Name of target column (churn indicator)
            feature_cols: List of feature columns (if None, uses all except target)
            
        Returns:
            Dictionary mapping feature names to BivariateAnalysisResult objects
        """
        logger.info(f"Starting bivariate analysis with target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        results = {}
        
        for feature_col in feature_cols:
            if feature_col not in df.columns:
                logger.warning(f"Feature column '{feature_col}' not found in DataFrame")
                continue
                
            logger.debug(f"Analyzing feature: {feature_col} vs target: {target_col}")
            
            # Determine feature type and perform appropriate analysis
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                result = self._analyze_numeric_vs_target(df, feature_col, target_col)
            else:
                result = self._analyze_categorical_vs_target(df, feature_col, target_col)
            
            results[feature_col] = result
        
        logger.info(f"Completed bivariate analysis for {len(results)} features")
        self.results['bivariate'] = results
        return results
    
    def perform_correlation_analysis(self, df: pd.DataFrame, 
                                   method: str = 'auto',
                                   include_categorical: bool = True) -> CorrelationAnalysisResult:
        """
        Perform correlation analysis using appropriate measures for different data types
        
        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', 'cramers_v', 'auto')
            include_categorical: Whether to include categorical variables
            
        Returns:
            CorrelationAnalysisResult object
        """
        logger.info(f"Starting correlation analysis with method: {method}")
        
        # Prepare data for correlation analysis
        df_corr = self._prepare_correlation_data(df, include_categorical)
        
        if method == 'auto':
            # Use mixed correlation matrix with appropriate measures
            correlation_matrix = self._calculate_mixed_correlation_matrix(df_corr)
            correlation_method = 'mixed'
        elif method == 'cramers_v':
            # Use Cramér's V for categorical associations
            correlation_matrix = self._calculate_cramers_v_matrix(df_corr)
            correlation_method = 'cramers_v'
        elif method in ['pearson', 'spearman']:
            # Use standard correlation for numeric data
            numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
            correlation_matrix = df_corr[numeric_cols].corr(method=method)
            correlation_method = method
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Find significant and high correlations
        significant_correlations = self._find_significant_correlations(df_corr, correlation_matrix)
        high_correlations = self._find_high_correlations(correlation_matrix)
        
        result = CorrelationAnalysisResult(
            correlation_matrix=correlation_matrix,
            correlation_method=correlation_method,
            significant_correlations=significant_correlations,
            high_correlations=high_correlations
        )
        
        logger.info(f"Completed correlation analysis. Found {len(high_correlations)} high correlations")
        self.results['correlation'] = result
        return result
    
    def _analyze_numeric_column(self, series: pd.Series, col_name: str,
                              missing_count: int, missing_percentage: float,
                              unique_count: int) -> UnivariateAnalysisResult:
        """Analyze numeric column and return statistics"""
        
        # Calculate descriptive statistics
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            logger.warning(f"No valid data for numeric column: {col_name}")
            return UnivariateAnalysisResult(
                column_name=col_name,
                data_type="numeric",
                missing_count=missing_count,
                missing_percentage=missing_percentage,
                unique_count=unique_count
            )
        
        stats_dict = {
            'mean': valid_data.mean(),
            'median': valid_data.median(),
            'std': valid_data.std(),
            'min_val': valid_data.min(),
            'max_val': valid_data.max(),
            'q25': valid_data.quantile(0.25),
            'q75': valid_data.quantile(0.75),
            'skewness': valid_data.skew(),
            'kurtosis': valid_data.kurtosis()
        }
        
        return UnivariateAnalysisResult(
            column_name=col_name,
            data_type="numeric",
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            **stats_dict
        )
    
    def _analyze_categorical_column(self, series: pd.Series, col_name: str,
                                  missing_count: int, missing_percentage: float,
                                  unique_count: int) -> UnivariateAnalysisResult:
        """Analyze categorical column and return statistics"""
        
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            logger.warning(f"No valid data for categorical column: {col_name}")
            return UnivariateAnalysisResult(
                column_name=col_name,
                data_type="categorical",
                missing_count=missing_count,
                missing_percentage=missing_percentage,
                unique_count=unique_count
            )
        
        # Calculate mode and frequency statistics
        value_counts = valid_data.value_counts()
        mode = value_counts.index[0] if len(value_counts) > 0 else None
        mode_frequency = value_counts.iloc[0] if len(value_counts) > 0 else None
        mode_percentage = (mode_frequency / len(valid_data)) * 100 if mode_frequency else None
        
        # Get top categories (up to 10)
        top_categories = value_counts.head(10).to_dict()
        
        return UnivariateAnalysisResult(
            column_name=col_name,
            data_type="categorical",
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            mode=str(mode),
            mode_frequency=mode_frequency,
            mode_percentage=mode_percentage,
            top_categories=top_categories
        )
    
    def _analyze_numeric_vs_target(self, df: pd.DataFrame, feature_col: str, 
                                 target_col: str) -> BivariateAnalysisResult:
        """Analyze relationship between numeric feature and binary target"""
        
        # Remove missing values
        clean_data = df[[feature_col, target_col]].dropna()
        
        if len(clean_data) < 10:
            logger.warning(f"Insufficient data for analysis: {feature_col} vs {target_col}")
            return BivariateAnalysisResult(
                feature_name=feature_col,
                target_name=target_col,
                feature_type="numeric"
            )
        
        # Separate groups
        group_0 = clean_data[clean_data[target_col] == 0][feature_col]
        group_1 = clean_data[clean_data[target_col] == 1][feature_col]
        
        # Perform Mann-Whitney U test (non-parametric)
        try:
            statistic, p_value = mannwhitneyu(group_1, group_0, alternative='two-sided')
            test_name = "Mann-Whitney U"
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(group_1), len(group_0)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            effect_size_interpretation = self._interpret_effect_size(abs(effect_size), "rank_biserial")
            
        except Exception as e:
            logger.warning(f"Statistical test failed for {feature_col}: {str(e)}")
            statistic, p_value, test_name, effect_size, effect_size_interpretation = None, None, None, None, None
        
        # Calculate descriptive statistics
        mean_by_churn = {
            0: group_0.mean() if len(group_0) > 0 else None,
            1: group_1.mean() if len(group_1) > 0 else None
        }
        
        # Calculate correlation coefficient
        correlation_coefficient, _ = spearmanr(clean_data[feature_col], clean_data[target_col])
        
        return BivariateAnalysisResult(
            feature_name=feature_col,
            target_name=target_col,
            feature_type="numeric",
            test_statistic=statistic,
            p_value=p_value,
            test_name=test_name,
            effect_size=effect_size,
            effect_size_interpretation=effect_size_interpretation,
            mean_by_churn=mean_by_churn,
            correlation_coefficient=correlation_coefficient
        )
    
    def _analyze_categorical_vs_target(self, df: pd.DataFrame, feature_col: str,
                                     target_col: str) -> BivariateAnalysisResult:
        """Analyze relationship between categorical feature and binary target"""
        
        # Remove missing values
        clean_data = df[[feature_col, target_col]].dropna()
        
        if len(clean_data) < 10:
            logger.warning(f"Insufficient data for analysis: {feature_col} vs {target_col}")
            return BivariateAnalysisResult(
                feature_name=feature_col,
                target_name=target_col,
                feature_type="categorical"
            )
        
        # Create contingency table
        contingency_table = pd.crosstab(clean_data[feature_col], clean_data[target_col])
        
        # Perform Chi-square test
        try:
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square"
            
            # Calculate Cramér's V as effect size
            n = contingency_table.sum().sum()
            effect_size = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            effect_size_interpretation = self._interpret_effect_size(effect_size, "cramers_v")
            
        except Exception as e:
            logger.warning(f"Chi-square test failed for {feature_col}: {str(e)}")
            chi2_stat, p_value, test_name, effect_size, effect_size_interpretation = None, None, None, None, None
        
        # Calculate churn rates by category
        churn_rates = clean_data.groupby(feature_col)[target_col].mean().to_dict()
        
        return BivariateAnalysisResult(
            feature_name=feature_col,
            target_name=target_col,
            feature_type="categorical",
            test_statistic=chi2_stat,
            p_value=p_value,
            test_name=test_name,
            effect_size=effect_size,
            effect_size_interpretation=effect_size_interpretation,
            churn_rates_by_category=churn_rates
        )
    
    def _prepare_correlation_data(self, df: pd.DataFrame, include_categorical: bool) -> pd.DataFrame:
        """Prepare data for correlation analysis"""
        
        df_corr = df.copy()
        
        if include_categorical:
            # Encode categorical variables for correlation analysis
            categorical_cols = df_corr.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                if df_corr[col].nunique() <= 10:  # Only encode low-cardinality categoricals
                    # Use label encoding for ordinal-like categories or binary
                    if df_corr[col].nunique() == 2:
                        df_corr[col] = pd.Categorical(df_corr[col]).codes
                    else:
                        # Use one-hot encoding for nominal categories
                        dummies = pd.get_dummies(df_corr[col], prefix=col, drop_first=True)
                        df_corr = pd.concat([df_corr.drop(col, axis=1), dummies], axis=1)
                else:
                    # Drop high-cardinality categorical columns
                    df_corr = df_corr.drop(col, axis=1)
        else:
            # Keep only numeric columns
            df_corr = df_corr.select_dtypes(include=[np.number])
        
        return df_corr
    
    def _calculate_mixed_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mixed correlation matrix using appropriate measures"""
        
        # For simplicity, use Spearman correlation which works for both numeric and ordinal data
        correlation_matrix = df.corr(method='spearman')
        
        return correlation_matrix
    
    def _calculate_cramers_v_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Cramér's V correlation matrix for categorical variables"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            logger.warning("No categorical columns found for Cramér's V calculation")
            return pd.DataFrame()
        
        n_cols = len(categorical_cols)
        cramers_v_matrix = np.zeros((n_cols, n_cols))
        
        for i, col1 in enumerate(categorical_cols):
            for j, col2 in enumerate(categorical_cols):
                if i == j:
                    cramers_v_matrix[i, j] = 1.0
                else:
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        chi2_stat, _, _, _ = chi2_contingency(contingency_table)
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                        cramers_v_matrix[i, j] = cramers_v
                    except:
                        cramers_v_matrix[i, j] = 0.0
        
        return pd.DataFrame(cramers_v_matrix, index=categorical_cols, columns=categorical_cols)
    
    def _find_significant_correlations(self, df: pd.DataFrame, 
                                     correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float, float]]:
        """Find statistically significant correlations"""
        
        significant_correlations = []
        
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    correlation = correlation_matrix.loc[col1, col2]
                    
                    if pd.isna(correlation):
                        continue
                    
                    # Calculate p-value for correlation
                    try:
                        if col1 in df.columns and col2 in df.columns:
                            clean_data = df[[col1, col2]].dropna()
                            if len(clean_data) > 10:
                                _, p_value = spearmanr(clean_data[col1], clean_data[col2])
                                
                                if p_value < self.significance_level:
                                    significant_correlations.append((col1, col2, correlation, p_value))
                    except:
                        continue
        
        return significant_correlations
    
    def _find_high_correlations(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find correlations above threshold"""
        
        high_correlations = []
        
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    correlation = correlation_matrix.loc[col1, col2]
                    
                    if pd.isna(correlation):
                        continue
                    
                    if abs(correlation) > self.correlation_threshold:
                        high_correlations.append((col1, col2, correlation))
        
        return high_correlations
    
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """Interpret effect size magnitude"""
        
        if effect_type == "cramers_v":
            if effect_size < 0.1:
                return "negligible"
            elif effect_size < 0.3:
                return "small"
            elif effect_size < 0.5:
                return "medium"
            else:
                return "large"
        
        elif effect_type == "rank_biserial":
            if effect_size < 0.1:
                return "negligible"
            elif effect_size < 0.3:
                return "small"
            elif effect_size < 0.5:
                return "medium"
            else:
                return "large"
        
        else:
            return "unknown"
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all performed analyses"""
        
        summary = {
            'analyses_performed': list(self.results.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if 'univariate' in self.results:
            univariate_results = self.results['univariate']
            summary['univariate_summary'] = {
                'total_columns': len(univariate_results),
                'numeric_columns': len([r for r in univariate_results.values() if r.data_type == 'numeric']),
                'categorical_columns': len([r for r in univariate_results.values() if r.data_type == 'categorical']),
                'columns_with_missing': len([r for r in univariate_results.values() if r.missing_count > 0])
            }
        
        if 'bivariate' in self.results:
            bivariate_results = self.results['bivariate']
            significant_features = [r for r in bivariate_results.values() 
                                  if r.p_value is not None and r.p_value < self.significance_level]
            summary['bivariate_summary'] = {
                'total_features': len(bivariate_results),
                'significant_features': len(significant_features),
                'significant_feature_names': [r.feature_name for r in significant_features]
            }
        
        if 'correlation' in self.results:
            correlation_result = self.results['correlation']
            summary['correlation_summary'] = {
                'correlation_method': correlation_result.correlation_method,
                'high_correlations_count': len(correlation_result.high_correlations),
                'significant_correlations_count': len(correlation_result.significant_correlations)
            }
        
        return summary


class StatisticalTester:
    """
    Statistical testing framework for churn analysis
    Provides comprehensive statistical tests and effect size calculations
    """
    
    def __init__(self, significance_level: float = 0.05, 
                 multiple_testing_correction: str = 'bonferroni'):
        """
        Initialize statistical tester
        
        Args:
            significance_level: Alpha level for statistical tests
            multiple_testing_correction: Method for multiple testing correction
        """
        self.significance_level = significance_level
        self.multiple_testing_correction = multiple_testing_correction
        
    def chi_square_test(self, df: pd.DataFrame, categorical_col: str, 
                       target_col: str) -> Dict[str, Any]:
        """
        Perform Chi-square test for categorical variable vs binary target
        
        Args:
            df: DataFrame containing the data
            categorical_col: Name of categorical column
            target_col: Name of binary target column
            
        Returns:
            Dictionary with test results including statistic, p-value, effect size
        """
        logger.debug(f"Performing Chi-square test: {categorical_col} vs {target_col}")
        
        # Remove missing values
        clean_data = df[[categorical_col, target_col]].dropna()
        
        if len(clean_data) < 5:
            logger.warning(f"Insufficient data for Chi-square test: {categorical_col}")
            return {
                'test_name': 'Chi-square',
                'statistic': None,
                'p_value': None,
                'degrees_of_freedom': None,
                'effect_size': None,
                'effect_size_name': 'Cramér\'s V',
                'interpretation': 'insufficient_data',
                'contingency_table': None,
                'expected_frequencies': None
            }
        
        # Create contingency table
        contingency_table = pd.crosstab(clean_data[categorical_col], clean_data[target_col])
        
        # Check minimum expected frequency requirement
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        if (expected < 5).any():
            logger.warning(f"Chi-square test assumption violated: expected frequencies < 5 for {categorical_col}")
            interpretation = 'assumption_violated'
        else:
            interpretation = 'valid'
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        # Interpret effect size
        if cramers_v < 0.1:
            effect_interpretation = 'negligible'
        elif cramers_v < 0.3:
            effect_interpretation = 'small'
        elif cramers_v < 0.5:
            effect_interpretation = 'medium'
        else:
            effect_interpretation = 'large'
        
        return {
            'test_name': 'Chi-square',
            'statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size': cramers_v,
            'effect_size_name': 'Cramér\'s V',
            'effect_interpretation': effect_interpretation,
            'interpretation': interpretation,
            'contingency_table': contingency_table,
            'expected_frequencies': pd.DataFrame(expected, 
                                               index=contingency_table.index,
                                               columns=contingency_table.columns),
            'is_significant': p_value < self.significance_level if p_value is not None else False
        }
    
    def mann_whitney_u_test(self, df: pd.DataFrame, numeric_col: str, 
                           target_col: str) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test for numeric variable vs binary target
        
        Args:
            df: DataFrame containing the data
            numeric_col: Name of numeric column
            target_col: Name of binary target column
            
        Returns:
            Dictionary with test results including statistic, p-value, effect size
        """
        logger.debug(f"Performing Mann-Whitney U test: {numeric_col} vs {target_col}")
        
        # Remove missing values
        clean_data = df[[numeric_col, target_col]].dropna()
        
        if len(clean_data) < 10:
            logger.warning(f"Insufficient data for Mann-Whitney U test: {numeric_col}")
            return {
                'test_name': 'Mann-Whitney U',
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'effect_size_name': 'rank-biserial correlation',
                'interpretation': 'insufficient_data',
                'group_statistics': None
            }
        
        # Separate groups
        group_0 = clean_data[clean_data[target_col] == 0][numeric_col]
        group_1 = clean_data[clean_data[target_col] == 1][numeric_col]
        
        if len(group_0) < 3 or len(group_1) < 3:
            logger.warning(f"Insufficient group sizes for Mann-Whitney U test: {numeric_col}")
            return {
                'test_name': 'Mann-Whitney U',
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'effect_size_name': 'rank-biserial correlation',
                'interpretation': 'insufficient_group_size',
                'group_statistics': None
            }
        
        # Perform test
        try:
            statistic, p_value = mannwhitneyu(group_1, group_0, alternative='two-sided')
            
            # Calculate rank-biserial correlation (effect size)
            n1, n2 = len(group_1), len(group_0)
            rank_biserial = 1 - (2 * statistic) / (n1 * n2)
            
            # Interpret effect size
            abs_effect = abs(rank_biserial)
            if abs_effect < 0.1:
                effect_interpretation = 'negligible'
            elif abs_effect < 0.3:
                effect_interpretation = 'small'
            elif abs_effect < 0.5:
                effect_interpretation = 'medium'
            else:
                effect_interpretation = 'large'
            
            # Calculate group statistics
            group_stats = {
                'group_0': {
                    'n': len(group_0),
                    'median': group_0.median(),
                    'mean': group_0.mean(),
                    'std': group_0.std(),
                    'q25': group_0.quantile(0.25),
                    'q75': group_0.quantile(0.75)
                },
                'group_1': {
                    'n': len(group_1),
                    'median': group_1.median(),
                    'mean': group_1.mean(),
                    'std': group_1.std(),
                    'q25': group_1.quantile(0.25),
                    'q75': group_1.quantile(0.75)
                }
            }
            
            return {
                'test_name': 'Mann-Whitney U',
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': rank_biserial,
                'effect_size_name': 'rank-biserial correlation',
                'effect_interpretation': effect_interpretation,
                'interpretation': 'valid',
                'group_statistics': group_stats,
                'is_significant': p_value < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"Mann-Whitney U test failed for {numeric_col}: {str(e)}")
            return {
                'test_name': 'Mann-Whitney U',
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'effect_size_name': 'rank-biserial correlation',
                'interpretation': 'test_failed',
                'error': str(e),
                'group_statistics': None
            }
    
    def fishers_exact_test(self, df: pd.DataFrame, categorical_col: str,
                          target_col: str) -> Dict[str, Any]:
        """
        Perform Fisher's exact test for 2x2 contingency tables
        
        Args:
            df: DataFrame containing the data
            categorical_col: Name of categorical column (must be binary)
            target_col: Name of binary target column
            
        Returns:
            Dictionary with test results
        """
        from scipy.stats import fisher_exact
        
        logger.debug(f"Performing Fisher's exact test: {categorical_col} vs {target_col}")
        
        # Remove missing values
        clean_data = df[[categorical_col, target_col]].dropna()
        
        # Check if both variables are binary
        if clean_data[categorical_col].nunique() != 2 or clean_data[target_col].nunique() != 2:
            logger.warning(f"Fisher's exact test requires binary variables: {categorical_col}")
            return {
                'test_name': 'Fisher\'s exact',
                'statistic': None,
                'p_value': None,
                'interpretation': 'not_binary',
                'odds_ratio': None,
                'confidence_interval': None
            }
        
        # Create 2x2 contingency table
        contingency_table = pd.crosstab(clean_data[categorical_col], clean_data[target_col])
        
        if contingency_table.shape != (2, 2):
            logger.warning(f"Invalid contingency table shape for Fisher's exact test: {categorical_col}")
            return {
                'test_name': 'Fisher\'s exact',
                'statistic': None,
                'p_value': None,
                'interpretation': 'invalid_table',
                'odds_ratio': None,
                'confidence_interval': None
            }
        
        try:
            odds_ratio, p_value = fisher_exact(contingency_table.values)
            
            return {
                'test_name': 'Fisher\'s exact',
                'statistic': odds_ratio,
                'p_value': p_value,
                'interpretation': 'valid',
                'odds_ratio': odds_ratio,
                'contingency_table': contingency_table,
                'is_significant': p_value < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"Fisher's exact test failed for {categorical_col}: {str(e)}")
            return {
                'test_name': 'Fisher\'s exact',
                'statistic': None,
                'p_value': None,
                'interpretation': 'test_failed',
                'error': str(e),
                'odds_ratio': None
            }
    
    def kruskal_wallis_test(self, df: pd.DataFrame, numeric_col: str,
                           grouping_col: str) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis test for numeric variable across multiple groups
        
        Args:
            df: DataFrame containing the data
            numeric_col: Name of numeric column
            grouping_col: Name of grouping column (categorical with >2 categories)
            
        Returns:
            Dictionary with test results
        """
        from scipy.stats import kruskal
        
        logger.debug(f"Performing Kruskal-Wallis test: {numeric_col} across {grouping_col}")
        
        # Remove missing values
        clean_data = df[[numeric_col, grouping_col]].dropna()
        
        # Get groups
        groups = []
        group_names = []
        
        for group_name in clean_data[grouping_col].unique():
            group_data = clean_data[clean_data[grouping_col] == group_name][numeric_col]
            if len(group_data) >= 3:  # Minimum group size
                groups.append(group_data.values)
                group_names.append(group_name)
        
        if len(groups) < 2:
            logger.warning(f"Insufficient groups for Kruskal-Wallis test: {grouping_col}")
            return {
                'test_name': 'Kruskal-Wallis',
                'statistic': None,
                'p_value': None,
                'interpretation': 'insufficient_groups',
                'group_statistics': None
            }
        
        try:
            statistic, p_value = kruskal(*groups)
            
            # Calculate group statistics
            group_stats = {}
            for i, group_name in enumerate(group_names):
                group_data = groups[i]
                group_stats[str(group_name)] = {
                    'n': len(group_data),
                    'median': np.median(group_data),
                    'mean': np.mean(group_data),
                    'std': np.std(group_data),
                    'q25': np.percentile(group_data, 25),
                    'q75': np.percentile(group_data, 75)
                }
            
            return {
                'test_name': 'Kruskal-Wallis',
                'statistic': statistic,
                'p_value': p_value,
                'degrees_of_freedom': len(groups) - 1,
                'interpretation': 'valid',
                'group_statistics': group_stats,
                'group_names': group_names,
                'is_significant': p_value < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"Kruskal-Wallis test failed for {numeric_col}: {str(e)}")
            return {
                'test_name': 'Kruskal-Wallis',
                'statistic': None,
                'p_value': None,
                'interpretation': 'test_failed',
                'error': str(e),
                'group_statistics': None
            }
    
    def perform_comprehensive_testing(self, df: pd.DataFrame, target_col: str,
                                    feature_cols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Perform comprehensive statistical testing for all features vs target
        
        Args:
            df: DataFrame containing the data
            target_col: Name of target column
            feature_cols: List of feature columns (if None, uses all except target)
            
        Returns:
            Dictionary mapping feature names to test results
        """
        logger.info(f"Performing comprehensive statistical testing with target: {target_col}")
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        results = {}
        
        for feature_col in feature_cols:
            if feature_col not in df.columns:
                logger.warning(f"Feature column '{feature_col}' not found in DataFrame")
                continue
            
            logger.debug(f"Testing feature: {feature_col}")
            
            # Determine appropriate test based on data types
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                # Numeric feature - use Mann-Whitney U test
                test_result = self.mann_whitney_u_test(df, feature_col, target_col)
            else:
                # Categorical feature
                unique_values = df[feature_col].nunique()
                
                if unique_values == 2 and df[target_col].nunique() == 2:
                    # Both binary - can use Fisher's exact test for small samples
                    clean_data = df[[feature_col, target_col]].dropna()
                    if len(clean_data) < 100:  # Use Fisher's for small samples
                        test_result = self.fishers_exact_test(df, feature_col, target_col)
                    else:
                        test_result = self.chi_square_test(df, feature_col, target_col)
                else:
                    # Use Chi-square test
                    test_result = self.chi_square_test(df, feature_col, target_col)
            
            results[feature_col] = test_result
        
        # Apply multiple testing correction if requested
        if self.multiple_testing_correction and len(results) > 1:
            results = self._apply_multiple_testing_correction(results)
        
        logger.info(f"Completed comprehensive testing for {len(results)} features")
        return results
    
    def _apply_multiple_testing_correction(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply multiple testing correction to p-values"""
        
        # Extract p-values
        p_values = []
        feature_names = []
        
        for feature_name, result in results.items():
            if result.get('p_value') is not None:
                p_values.append(result['p_value'])
                feature_names.append(feature_name)
        
        if len(p_values) == 0:
            return results
        
        # Apply correction
        if self.multiple_testing_correction == 'bonferroni':
            corrected_p_values = [p * len(p_values) for p in p_values]
            corrected_p_values = [min(p, 1.0) for p in corrected_p_values]  # Cap at 1.0
        elif self.multiple_testing_correction == 'holm':
            from scipy.stats import false_discovery_control
            corrected_p_values = false_discovery_control(p_values, method='holm')
        else:
            logger.warning(f"Unknown correction method: {self.multiple_testing_correction}")
            return results
        
        # Update results with corrected p-values
        corrected_results = results.copy()
        for i, feature_name in enumerate(feature_names):
            corrected_results[feature_name]['p_value_corrected'] = corrected_p_values[i]
            corrected_results[feature_name]['is_significant_corrected'] = corrected_p_values[i] < self.significance_level
            corrected_results[feature_name]['correction_method'] = self.multiple_testing_correction
        
        return corrected_results
    
    def calculate_effect_sizes(self, df: pd.DataFrame, target_col: str,
                             feature_cols: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate effect sizes for all features vs target
        
        Args:
            df: DataFrame containing the data
            target_col: Name of target column
            feature_cols: List of feature columns (if None, uses all except target)
            
        Returns:
            Dictionary mapping feature names to effect size metrics
        """
        logger.info("Calculating effect sizes for all features")
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        effect_sizes = {}
        
        for feature_col in feature_cols:
            if feature_col not in df.columns:
                continue
            
            clean_data = df[[feature_col, target_col]].dropna()
            
            if len(clean_data) < 10:
                continue
            
            feature_effects = {}
            
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                # Numeric feature - calculate multiple effect sizes
                group_0 = clean_data[clean_data[target_col] == 0][feature_col]
                group_1 = clean_data[clean_data[target_col] == 1][feature_col]
                
                if len(group_0) > 0 and len(group_1) > 0:
                    # Cohen's d
                    pooled_std = np.sqrt(((len(group_0) - 1) * group_0.var() + 
                                        (len(group_1) - 1) * group_1.var()) / 
                                       (len(group_0) + len(group_1) - 2))
                    if pooled_std > 0:
                        cohens_d = (group_1.mean() - group_0.mean()) / pooled_std
                        feature_effects['cohens_d'] = cohens_d
                    
                    # Point-biserial correlation
                    try:
                        point_biserial, _ = pearsonr(clean_data[feature_col], clean_data[target_col])
                        feature_effects['point_biserial_correlation'] = point_biserial
                    except:
                        pass
                    
                    # Spearman correlation
                    try:
                        spearman_corr, _ = spearmanr(clean_data[feature_col], clean_data[target_col])
                        feature_effects['spearman_correlation'] = spearman_corr
                    except:
                        pass
            
            else:
                # Categorical feature - calculate Cramér's V
                try:
                    contingency_table = pd.crosstab(clean_data[feature_col], clean_data[target_col])
                    chi2_stat, _, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                    feature_effects['cramers_v'] = cramers_v
                except:
                    pass
            
            if feature_effects:
                effect_sizes[feature_col] = feature_effects
        
        logger.info(f"Calculated effect sizes for {len(effect_sizes)} features")
        return effect_sizes
    
    def generate_testing_report(self, test_results: Dict[str, Dict[str, Any]],
                              effect_sizes: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Generate a comprehensive testing report
        
        Args:
            test_results: Results from comprehensive testing
            effect_sizes: Optional effect size calculations
            
        Returns:
            DataFrame with testing summary
        """
        logger.info("Generating statistical testing report")
        
        report_data = []
        
        for feature_name, result in test_results.items():
            row = {
                'feature': feature_name,
                'test_name': result.get('test_name', 'Unknown'),
                'statistic': result.get('statistic'),
                'p_value': result.get('p_value'),
                'is_significant': result.get('is_significant', False),
                'effect_size': result.get('effect_size'),
                'effect_size_name': result.get('effect_size_name', ''),
                'effect_interpretation': result.get('effect_interpretation', ''),
                'interpretation': result.get('interpretation', '')
            }
            
            # Add corrected p-values if available
            if 'p_value_corrected' in result:
                row['p_value_corrected'] = result['p_value_corrected']
                row['is_significant_corrected'] = result['is_significant_corrected']
                row['correction_method'] = result['correction_method']
            
            # Add additional effect sizes if available
            if effect_sizes and feature_name in effect_sizes:
                for effect_name, effect_value in effect_sizes[feature_name].items():
                    row[f'effect_{effect_name}'] = effect_value
            
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by significance and effect size
        if 'p_value' in report_df.columns:
            report_df = report_df.sort_values(['is_significant', 'p_value'], 
                                            ascending=[False, True])
        
        logger.info(f"Generated testing report with {len(report_df)} features")
        return report_df


class EDAVisualizer:
    """
    Visualization generation system for exploratory data analysis
    Creates publication-ready plots and automatically saves them to reports directory
    """
    
    def __init__(self, output_dir: Optional[Path] = None, 
                 figure_size: Tuple[int, int] = (10, 6),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8'):
        """
        Initialize EDA visualizer
        
        Args:
            output_dir: Directory to save figures (defaults to config.FIGURES_PATH)
            figure_size: Default figure size (width, height)
            dpi: Resolution for saved figures
            style: Matplotlib style to use
        """
        self.output_dir = output_dir or config.FIGURES_PATH
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            logger.warning(f"Style '{style}' not available, using default")
        
        # Color palettes
        self.categorical_palette = sns.color_palette("Set2", 8)
        self.diverging_palette = sns.color_palette("RdYlBu_r", 11)
        self.sequential_palette = sns.color_palette("viridis", 8)
        
    def plot_churn_rate_by_segments(self, df: pd.DataFrame, categorical_col: str,
                                   target_col: str, title: Optional[str] = None,
                                   save_name: Optional[str] = None) -> plt.Figure:
        """
        Create churn rate plot by categorical segments
        
        Args:
            df: DataFrame containing the data
            categorical_col: Name of categorical column for segmentation
            target_col: Name of binary target column (churn)
            title: Optional plot title
            save_name: Optional filename for saving (without extension)
            
        Returns:
            matplotlib Figure object
        """
        logger.debug(f"Creating churn rate plot for {categorical_col}")
        
        # Calculate churn rates
        clean_data = df[[categorical_col, target_col]].dropna()
        churn_rates = clean_data.groupby(categorical_col)[target_col].agg(['mean', 'count']).reset_index()
        churn_rates.columns = [categorical_col, 'churn_rate', 'count']
        churn_rates = churn_rates.sort_values('churn_rate', ascending=False)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Churn rate by category
        bars = ax1.bar(range(len(churn_rates)), churn_rates['churn_rate'], 
                      color=self.categorical_palette[0], alpha=0.7)
        
        ax1.set_xlabel(categorical_col.replace('_', ' ').title())
        ax1.set_ylabel('Churn Rate')
        ax1.set_title(title or f'Churn Rate by {categorical_col.replace("_", " ").title()}')
        ax1.set_xticks(range(len(churn_rates)))
        ax1.set_xticklabels(churn_rates[categorical_col], rotation=45, ha='right')
        ax1.set_ylim(0, max(churn_rates['churn_rate']) * 1.1)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        
        # Plot 2: Sample size by category
        bars2 = ax2.bar(range(len(churn_rates)), churn_rates['count'], 
                       color=self.categorical_palette[1], alpha=0.7)
        
        ax2.set_xlabel(categorical_col.replace('_', ' ').title())
        ax2.set_ylabel('Sample Size')
        ax2.set_title(f'Sample Size by {categorical_col.replace("_", " ").title()}')
        ax2.set_xticks(range(len(churn_rates)))
        ax2.set_xticklabels(churn_rates[categorical_col], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(churn_rates['count']) * 0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_numeric_distribution_by_churn(self, df: pd.DataFrame, numeric_col: str,
                                         target_col: str, title: Optional[str] = None,
                                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Create distribution plots for numeric variables by churn status
        
        Args:
            df: DataFrame containing the data
            numeric_col: Name of numeric column
            target_col: Name of binary target column (churn)
            title: Optional plot title
            save_name: Optional filename for saving (without extension)
            
        Returns:
            matplotlib Figure object
        """
        logger.debug(f"Creating distribution plot for {numeric_col}")
        
        clean_data = df[[numeric_col, target_col]].dropna()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Separate data by churn status
        no_churn = clean_data[clean_data[target_col] == 0][numeric_col]
        churn = clean_data[clean_data[target_col] == 1][numeric_col]
        
        # Plot 1: Overlapping histograms
        ax1.hist(no_churn, bins=30, alpha=0.7, label='No Churn', 
                color=self.categorical_palette[0], density=True)
        ax1.hist(churn, bins=30, alpha=0.7, label='Churn', 
                color=self.categorical_palette[1], density=True)
        ax1.set_xlabel(numeric_col.replace('_', ' ').title())
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plots
        box_data = [no_churn, churn]
        box_plot = ax2.boxplot(box_data, labels=['No Churn', 'Churn'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor(self.categorical_palette[0])
        box_plot['boxes'][1].set_facecolor(self.categorical_palette[1])
        ax2.set_ylabel(numeric_col.replace('_', ' ').title())
        ax2.set_title('Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Violin plots
        violin_data = pd.DataFrame({
            'value': pd.concat([no_churn, churn]),
            'churn': ['No Churn'] * len(no_churn) + ['Churn'] * len(churn)
        })
        sns.violinplot(data=violin_data, x='churn', y='value', ax=ax3, palette=self.categorical_palette[:2])
        ax3.set_ylabel(numeric_col.replace('_', ' ').title())
        ax3.set_xlabel('Churn Status')
        ax3.set_title('Violin Plot Comparison')
        
        # Plot 4: Cumulative distribution
        ax4.hist(no_churn, bins=50, alpha=0.7, label='No Churn', 
                color=self.categorical_palette[0], density=True, cumulative=True, histtype='step')
        ax4.hist(churn, bins=50, alpha=0.7, label='Churn', 
                color=self.categorical_palette[1], density=True, cumulative=True, histtype='step')
        ax4.set_xlabel(numeric_col.replace('_', ' ').title())
        ax4.set_ylabel('Cumulative Density')
        ax4.set_title('Cumulative Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            fig.suptitle(f'{numeric_col.replace("_", " ").title()} Distribution by Churn Status', 
                        fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                               title: Optional[str] = None,
                               save_name: Optional[str] = None,
                               annot: bool = True,
                               mask_upper: bool = True) -> plt.Figure:
        """
        Create correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix to plot
            title: Optional plot title
            save_name: Optional filename for saving (without extension)
            annot: Whether to annotate cells with correlation values
            mask_upper: Whether to mask upper triangle
            
        Returns:
            matplotlib Figure object
        """
        logger.debug("Creating correlation heatmap")
        
        # Create mask for upper triangle if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=annot, 
                   cmap=self.diverging_palette,
                   center=0,
                   square=True,
                   fmt='.2f' if annot else '',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title(title or 'Correlation Matrix', fontsize=16, pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_univariate_summary(self, univariate_results: Dict[str, UnivariateAnalysisResult],
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Create summary plot of univariate analysis results
        
        Args:
            univariate_results: Results from univariate analysis
            save_name: Optional filename for saving (without extension)
            
        Returns:
            matplotlib Figure object
        """
        logger.debug("Creating univariate summary plot")
        
        # Prepare data for plotting
        numeric_results = [r for r in univariate_results.values() if r.data_type == 'numeric']
        categorical_results = [r for r in univariate_results.values() if r.data_type == 'categorical']
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Missing values percentage
        all_results = list(univariate_results.values())
        missing_data = [(r.column_name, r.missing_percentage) for r in all_results if r.missing_percentage > 0]
        
        if missing_data:
            missing_data.sort(key=lambda x: x[1], reverse=True)
            columns, percentages = zip(*missing_data)
            
            bars = ax1.barh(range(len(columns)), percentages, color=self.categorical_palette[0])
            ax1.set_yticks(range(len(columns)))
            ax1.set_yticklabels([col.replace('_', ' ') for col in columns])
            ax1.set_xlabel('Missing Percentage (%)')
            ax1.set_title('Missing Values by Column')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%', ha='left', va='center')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Missing Values by Column')
        
        # Plot 2: Unique values count
        unique_data = [(r.column_name, r.unique_count) for r in all_results]
        unique_data.sort(key=lambda x: x[1], reverse=True)
        columns, unique_counts = zip(*unique_data)
        
        bars = ax2.bar(range(len(columns)), unique_counts, color=self.categorical_palette[1])
        ax2.set_xticks(range(len(columns)))
        ax2.set_xticklabels([col.replace('_', ' ') for col in columns], rotation=45, ha='right')
        ax2.set_ylabel('Unique Values Count')
        ax2.set_title('Unique Values by Column')
        ax2.set_yscale('log')
        
        # Plot 3: Numeric variables skewness
        if numeric_results:
            skewness_data = [(r.column_name, r.skewness) for r in numeric_results if r.skewness is not None]
            
            if skewness_data:
                columns, skewness_values = zip(*skewness_data)
                colors = [self.categorical_palette[2] if abs(s) < 1 else self.categorical_palette[3] for s in skewness_values]
                
                bars = ax3.bar(range(len(columns)), skewness_values, color=colors)
                ax3.set_xticks(range(len(columns)))
                ax3.set_xticklabels([col.replace('_', ' ') for col in columns], rotation=45, ha='right')
                ax3.set_ylabel('Skewness')
                ax3.set_title('Skewness of Numeric Variables')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='|Skewness| = 1')
                ax3.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No Numeric Variables', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Skewness of Numeric Variables')
        else:
            ax3.text(0.5, 0.5, 'No Numeric Variables', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Skewness of Numeric Variables')
        
        # Plot 4: Categorical variables mode percentage
        if categorical_results:
            mode_data = [(r.column_name, r.mode_percentage) for r in categorical_results if r.mode_percentage is not None]
            
            if mode_data:
                columns, mode_percentages = zip(*mode_data)
                
                bars = ax4.bar(range(len(columns)), mode_percentages, color=self.categorical_palette[4])
                ax4.set_xticks(range(len(columns)))
                ax4.set_xticklabels([col.replace('_', ' ') for col in columns], rotation=45, ha='right')
                ax4.set_ylabel('Mode Percentage (%)')
                ax4.set_title('Mode Dominance in Categorical Variables')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom')
            else:
                ax4.text(0.5, 0.5, 'No Categorical Variables', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Mode Dominance in Categorical Variables')
        else:
            ax4.text(0.5, 0.5, 'No Categorical Variables', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Mode Dominance in Categorical Variables')
        
        plt.tight_layout()
        
        # Save figure
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_bivariate_summary(self, bivariate_results: Dict[str, BivariateAnalysisResult],
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Create summary plot of bivariate analysis results
        
        Args:
            bivariate_results: Results from bivariate analysis
            save_name: Optional filename for saving (without extension)
            
        Returns:
            matplotlib Figure object
        """
        logger.debug("Creating bivariate summary plot")
        
        # Prepare data
        results_list = list(bivariate_results.values())
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: P-values (significance)
        p_value_data = [(r.feature_name, r.p_value) for r in results_list if r.p_value is not None]
        
        if p_value_data:
            p_value_data.sort(key=lambda x: x[1])
            features, p_values = zip(*p_value_data)
            
            colors = [self.categorical_palette[0] if p < 0.05 else self.categorical_palette[1] for p in p_values]
            
            bars = ax1.barh(range(len(features)), [-np.log10(p) for p in p_values], color=colors)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels([f.replace('_', ' ') for f in features])
            ax1.set_xlabel('-log10(p-value)')
            ax1.set_title('Statistical Significance of Features')
            ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No P-values Available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Statistical Significance of Features')
        
        # Plot 2: Effect sizes
        effect_size_data = [(r.feature_name, r.effect_size) for r in results_list if r.effect_size is not None]
        
        if effect_size_data:
            effect_size_data.sort(key=lambda x: abs(x[1]), reverse=True)
            features, effect_sizes = zip(*effect_size_data)
            
            colors = [self.categorical_palette[2] if abs(e) > 0.3 else self.categorical_palette[3] for e in effect_sizes]
            
            bars = ax2.barh(range(len(features)), effect_sizes, color=colors)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels([f.replace('_', ' ') for f in features])
            ax2.set_xlabel('Effect Size')
            ax2.set_title('Effect Sizes of Features')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Effect Sizes Available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Effect Sizes of Features')
        
        # Plot 3: Feature types distribution
        feature_types = [r.feature_type for r in results_list]
        type_counts = pd.Series(feature_types).value_counts()
        
        wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index, 
                                          autopct='%1.1f%%', colors=self.categorical_palette[:len(type_counts)])
        ax3.set_title('Distribution of Feature Types')
        
        # Plot 4: Significance by feature type
        significance_by_type = {}
        for result in results_list:
            if result.p_value is not None:
                if result.feature_type not in significance_by_type:
                    significance_by_type[result.feature_type] = {'significant': 0, 'total': 0}
                
                significance_by_type[result.feature_type]['total'] += 1
                if result.p_value < 0.05:
                    significance_by_type[result.feature_type]['significant'] += 1
        
        if significance_by_type:
            types = list(significance_by_type.keys())
            sig_rates = [significance_by_type[t]['significant'] / significance_by_type[t]['total'] * 100 
                        for t in types]
            
            bars = ax4.bar(types, sig_rates, color=self.categorical_palette[:len(types)])
            ax4.set_ylabel('Significant Features (%)')
            ax4.set_title('Significance Rate by Feature Type')
            ax4.set_ylim(0, 100)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}%', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No Significance Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Significance Rate by Feature Type')
        
        plt.tight_layout()
        
        # Save figure
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def create_eda_dashboard(self, df: pd.DataFrame, target_col: str,
                           univariate_results: Dict[str, UnivariateAnalysisResult],
                           bivariate_results: Dict[str, BivariateAnalysisResult],
                           correlation_result: CorrelationAnalysisResult,
                           save_name: Optional[str] = None) -> List[plt.Figure]:
        """
        Create comprehensive EDA dashboard with multiple plots
        
        Args:
            df: Original DataFrame
            target_col: Target column name
            univariate_results: Univariate analysis results
            bivariate_results: Bivariate analysis results
            correlation_result: Correlation analysis results
            save_name: Base name for saving figures
            
        Returns:
            List of matplotlib Figure objects
        """
        logger.info("Creating comprehensive EDA dashboard")
        
        figures = []
        
        # 1. Univariate summary
        fig1 = self.plot_univariate_summary(univariate_results, 
                                           f"{save_name}_univariate" if save_name else None)
        figures.append(fig1)
        
        # 2. Bivariate summary
        fig2 = self.plot_bivariate_summary(bivariate_results,
                                         f"{save_name}_bivariate" if save_name else None)
        figures.append(fig2)
        
        # 3. Correlation heatmap
        fig3 = self.plot_correlation_heatmap(correlation_result.correlation_matrix,
                                           f"Correlation Matrix ({correlation_result.correlation_method})",
                                           f"{save_name}_correlation" if save_name else None)
        figures.append(fig3)
        
        # 4. Top significant features - churn rate plots
        significant_features = [r for r in bivariate_results.values() 
                              if r.p_value is not None and r.p_value < 0.05]
        significant_features.sort(key=lambda x: x.p_value)
        
        # Plot top 6 most significant categorical features
        categorical_features = [r for r in significant_features if r.feature_type == 'categorical'][:6]
        
        if categorical_features:
            fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, result in enumerate(categorical_features):
                if i < 6:
                    clean_data = df[[result.feature_name, target_col]].dropna()
                    churn_rates = clean_data.groupby(result.feature_name)[target_col].mean()
                    
                    bars = axes[i].bar(range(len(churn_rates)), churn_rates.values, 
                                     color=self.categorical_palette[i % len(self.categorical_palette)])
                    axes[i].set_xticks(range(len(churn_rates)))
                    axes[i].set_xticklabels(churn_rates.index, rotation=45, ha='right')
                    axes[i].set_ylabel('Churn Rate')
                    axes[i].set_title(f'{result.feature_name.replace("_", " ").title()}\n(p={result.p_value:.3f})')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.2%}', ha='center', va='bottom', fontsize=8)
            
            # Hide unused subplots
            for i in range(len(categorical_features), 6):
                axes[i].set_visible(False)
            
            plt.suptitle('Top Significant Categorical Features - Churn Rates', fontsize=16)
            plt.tight_layout()
            
            if save_name:
                self._save_figure(fig4, f"{save_name}_top_categorical")
            
            figures.append(fig4)
        
        # 5. Top significant numeric features - distributions
        numeric_features = [r for r in significant_features if r.feature_type == 'numeric'][:4]
        
        if numeric_features:
            fig5, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, result in enumerate(numeric_features):
                if i < 4:
                    clean_data = df[[result.feature_name, target_col]].dropna()
                    no_churn = clean_data[clean_data[target_col] == 0][result.feature_name]
                    churn = clean_data[clean_data[target_col] == 1][result.feature_name]
                    
                    axes[i].hist(no_churn, bins=20, alpha=0.7, label='No Churn', 
                               color=self.categorical_palette[0], density=True)
                    axes[i].hist(churn, bins=20, alpha=0.7, label='Churn', 
                               color=self.categorical_palette[1], density=True)
                    
                    axes[i].set_xlabel(result.feature_name.replace('_', ' ').title())
                    axes[i].set_ylabel('Density')
                    axes[i].set_title(f'{result.feature_name.replace("_", " ").title()}\n(p={result.p_value:.3f})')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(numeric_features), 4):
                axes[i].set_visible(False)
            
            plt.suptitle('Top Significant Numeric Features - Distributions', fontsize=16)
            plt.tight_layout()
            
            if save_name:
                self._save_figure(fig5, f"{save_name}_top_numeric")
            
            figures.append(fig5)
        
        logger.info(f"Created EDA dashboard with {len(figures)} figures")
        return figures
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to output directory"""
        
        # Ensure filename has extension
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename += '.png'
        
        filepath = self.output_dir / filename
        
        try:
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved figure: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save figure {filepath}: {str(e)}")
    
    def save_all_figures(self, figures: List[plt.Figure], base_name: str) -> List[str]:
        """
        Save all figures with sequential naming
        
        Args:
            figures: List of matplotlib figures
            base_name: Base name for files
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for i, fig in enumerate(figures):
            filename = f"{base_name}_{i+1:02d}.png"
            filepath = self.output_dir / filename
            
            try:
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                saved_paths.append(str(filepath))
                logger.debug(f"Saved figure: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save figure {filepath}: {str(e)}")
        
        logger.info(f"Saved {len(saved_paths)} figures with base name: {base_name}")
        return saved_paths