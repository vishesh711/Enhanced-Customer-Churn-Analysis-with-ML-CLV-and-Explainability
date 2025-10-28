"""
Explainability and interpretability module for Customer Churn ML Pipeline
Implements SHAP and LIME explanations for global and local model interpretability
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# SHAP imports
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# LIME imports
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    warnings.warn("LIME not available. Install with: pip install lime")

from config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class GlobalExplanation:
    """Container for global model explanations"""
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray]
    shap_summary_data: Dict[str, Any]
    feature_interactions: Optional[pd.DataFrame]
    partial_dependence_data: Dict[str, Dict[str, Any]]
    model_name: str
    explanation_date: datetime


@dataclass
class LocalExplanation:
    """Container for local (instance-level) explanations"""
    instance_id: Union[str, int]
    prediction: float
    shap_values: Optional[Dict[str, float]]
    lime_explanation: Optional[Dict[str, Any]]
    feature_contributions: Dict[str, float]
    top_features: List[Tuple[str, float]]
    explanation_date: datetime


class GlobalExplainer:
    """
    Creates global model-level insights using SHAP summary plots, feature importance,
    feature interaction analysis, and partial dependence plots
    """
    
    def __init__(self, model: BaseEstimator, feature_names: List[str], 
                 model_name: str = 'model'):
        """
        Initialize GlobalExplainer
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            model_name: Name of the model for tracking
        """
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
        logger.info(f"GlobalExplainer initialized for model {model_name} with {len(feature_names)} features")
    
    def fit_explainer(self, X_background: pd.DataFrame, 
                     explainer_type: str = 'auto',
                     max_background_samples: int = 100) -> None:
        """
        Fit SHAP explainer on background data
        
        Args:
            X_background: Background dataset for SHAP explainer
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel')
            max_background_samples: Maximum number of background samples to use
        """
        if not HAS_SHAP:
            logger.error("SHAP not available. Cannot create explainer.")
            return
        
        logger.info(f"Fitting SHAP explainer with type '{explainer_type}'")
        
        # Sample background data if too large
        if len(X_background) > max_background_samples:
            X_background = X_background.sample(n=max_background_samples, random_state=config.RANDOM_SEED)
        
        self.background_data = X_background
        
        # Choose explainer type
        if explainer_type == 'auto':
            # Auto-detect based on model type
            model_type = type(self.model).__name__.lower()
            
            if any(tree_type in model_type for tree_type in ['forest', 'tree', 'xgb', 'lgb', 'gradient']):
                explainer_type = 'tree'
            elif any(linear_type in model_type for linear_type in ['linear', 'logistic']):
                explainer_type = 'linear'
            else:
                explainer_type = 'kernel'
        
        # Create appropriate explainer
        try:
            if explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Created TreeExplainer")
            elif explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, X_background)
                logger.info("Created LinearExplainer")
            elif explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    X_background,
                    link="logit"
                )
                logger.info("Created KernelExplainer")
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
                
        except Exception as e:
            logger.warning(f"Failed to create {explainer_type} explainer: {str(e)}")
            logger.info("Falling back to KernelExplainer")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background,
                link="logit"
            )
    
    def generate_shap_summary(self, X: pd.DataFrame, 
                            max_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate SHAP summary plots and feature importance
        
        Args:
            X: Dataset to explain
            max_samples: Maximum number of samples to use for SHAP calculation
            
        Returns:
            Dictionary with SHAP summary data
        """
        if not HAS_SHAP or self.explainer is None:
            logger.error("SHAP explainer not available or not fitted")
            return {}
        
        logger.info("Generating SHAP summary analysis")
        
        # Sample data if too large
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=config.RANDOM_SEED)
        else:
            X_sample = X.copy()
        
        # Calculate SHAP values
        try:
            if hasattr(self.explainer, 'shap_values'):
                # For tree explainers
                shap_values = self.explainer.shap_values(X_sample)
                # For binary classification, take positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:
                # For other explainers
                shap_values = self.explainer(X_sample)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                    # For binary classification, take positive class
                    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
                        shap_values = shap_values[:, :, 1]
            
            self.shap_values = shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return {}
        
        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_importance_dict = dict(zip(self.feature_names, feature_importance))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Calculate summary statistics
        shap_summary_stats = {
            'mean_abs_shap': feature_importance_dict,
            'feature_ranking': [feat for feat, _ in sorted_features],
            'top_10_features': sorted_features[:10],
            'shap_values_shape': shap_values.shape,
            'samples_explained': len(X_sample)
        }
        
        # Create summary plots
        self._create_shap_summary_plots(X_sample, shap_values)
        
        logger.info(f"SHAP summary completed for {len(X_sample)} samples")
        
        return shap_summary_stats
    
    def calculate_feature_interactions(self, X: pd.DataFrame,
                                     max_samples: int = 500,
                                     top_features: int = 10) -> pd.DataFrame:
        """
        Calculate feature interactions using SHAP interaction values
        
        Args:
            X: Dataset to analyze
            max_samples: Maximum number of samples for interaction calculation
            top_features: Number of top features to analyze for interactions
            
        Returns:
            DataFrame with feature interaction strengths
        """
        if not HAS_SHAP or self.explainer is None:
            logger.error("SHAP explainer not available")
            return pd.DataFrame()
        
        logger.info("Calculating feature interactions")
        
        # Sample data if too large
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=config.RANDOM_SEED)
        else:
            X_sample = X.copy()
        
        try:
            # Calculate interaction values (only works with TreeExplainer)
            if hasattr(self.explainer, 'shap_interaction_values'):
                interaction_values = self.explainer.shap_interaction_values(X_sample)
                
                # For binary classification, take positive class
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[1]
                
                # Calculate interaction strengths
                n_features = len(self.feature_names)
                interaction_matrix = np.zeros((n_features, n_features))
                
                for i in range(n_features):
                    for j in range(n_features):
                        if i != j:
                            # Mean absolute interaction effect
                            interaction_matrix[i, j] = np.abs(interaction_values[:, i, j]).mean()
                
                # Create DataFrame
                interaction_df = pd.DataFrame(
                    interaction_matrix,
                    index=self.feature_names,
                    columns=self.feature_names
                )
                
                # Get top interactions
                interactions_list = []
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        interactions_list.append({
                            'feature_1': self.feature_names[i],
                            'feature_2': self.feature_names[j],
                            'interaction_strength': interaction_matrix[i, j] + interaction_matrix[j, i]
                        })
                
                interactions_df = pd.DataFrame(interactions_list)
                interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
                
                # Create interaction heatmap
                self._create_interaction_heatmap(interaction_df, top_features)
                
                logger.info(f"Feature interactions calculated for {len(X_sample)} samples")
                
                return interactions_df.head(20)  # Return top 20 interactions
                
            else:
                logger.warning("Model does not support SHAP interaction values")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating feature interactions: {str(e)}")
            return pd.DataFrame()
    
    def create_partial_dependence_plots(self, X: pd.DataFrame, 
                                      features: Optional[List[str]] = None,
                                      n_features: int = 6) -> Dict[str, Dict[str, Any]]:
        """
        Create partial dependence plots for key features
        
        Args:
            X: Dataset for partial dependence calculation
            features: List of features to plot (if None, use top features)
            n_features: Number of features to plot if features not specified
            
        Returns:
            Dictionary with partial dependence data
        """
        logger.info("Creating partial dependence plots")
        
        # Select features to plot
        if features is None:
            if self.shap_values is not None:
                # Use top SHAP features
                feature_importance = np.abs(self.shap_values).mean(axis=0)
                top_indices = np.argsort(feature_importance)[-n_features:]
                features = [self.feature_names[i] for i in top_indices]
            else:
                # Use first n features
                features = self.feature_names[:n_features]
        
        pd_data = {}
        
        try:
            # Calculate partial dependence for each feature
            for feature in features:
                if feature not in X.columns:
                    logger.warning(f"Feature {feature} not found in dataset")
                    continue
                
                feature_idx = list(X.columns).index(feature)
                
                # Calculate partial dependence
                pd_result = partial_dependence(
                    self.model, 
                    X, 
                    features=[feature_idx],
                    kind='average',
                    grid_resolution=50
                )
                
                pd_data[feature] = {
                    'values': pd_result['values'][0],
                    'average': pd_result['average'][0],
                    'feature_type': 'numeric' if X[feature].dtype in ['int64', 'float64'] else 'categorical'
                }
            
            # Create partial dependence plots
            self._create_partial_dependence_plots(pd_data, X)
            
            logger.info(f"Partial dependence plots created for {len(features)} features")
            
        except Exception as e:
            logger.error(f"Error creating partial dependence plots: {str(e)}")
        
        return pd_data
    
    def generate_global_explanation(self, X: pd.DataFrame) -> GlobalExplanation:
        """
        Generate comprehensive global explanation
        
        Args:
            X: Dataset to explain
            
        Returns:
            GlobalExplanation object with all global insights
        """
        logger.info("Generating comprehensive global explanation")
        
        # Generate SHAP summary
        shap_summary = self.generate_shap_summary(X)
        
        # Calculate feature interactions
        interactions = self.calculate_feature_interactions(X)
        
        # Create partial dependence plots
        pd_data = self.create_partial_dependence_plots(X)
        
        # Extract feature importance
        feature_importance = shap_summary.get('mean_abs_shap', {})
        
        global_explanation = GlobalExplanation(
            feature_importance=feature_importance,
            shap_values=self.shap_values,
            shap_summary_data=shap_summary,
            feature_interactions=interactions,
            partial_dependence_data=pd_data,
            model_name=self.model_name,
            explanation_date=datetime.now()
        )
        
        logger.info("Global explanation generation completed")
        
        return global_explanation
    
    def _create_shap_summary_plots(self, X: pd.DataFrame, shap_values: np.ndarray) -> None:
        """Create SHAP summary plots"""
        try:
            # Summary plot (beeswarm)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {self.model_name}')
            plt.tight_layout()
            
            # Save plot
            plot_path = config.REPORTS_PATH / "figures" / f"shap_summary_{self.model_name.lower().replace(' ', '_')}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance bar plot
            feature_importance = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(feature_importance)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'Feature Importance - {self.model_name}')
            plt.tight_layout()
            
            # Save plot
            importance_path = config.REPORTS_PATH / "figures" / f"feature_importance_{self.model_name.lower().replace(' ', '_')}.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP summary plots saved")
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plots: {str(e)}")
    
    def _create_interaction_heatmap(self, interaction_df: pd.DataFrame, 
                                  top_features: int) -> None:
        """Create feature interaction heatmap"""
        try:
            # Select top features for visualization
            top_feature_names = interaction_df.index[:top_features]
            interaction_subset = interaction_df.loc[top_feature_names, top_feature_names]
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(interaction_subset, annot=True, cmap='RdYlBu_r', 
                       center=0, fmt='.3f', square=True)
            plt.title(f'Feature Interactions Heatmap - {self.model_name}')
            plt.tight_layout()
            
            # Save plot
            heatmap_path = config.REPORTS_PATH / "figures" / f"interaction_heatmap_{self.model_name.lower().replace(' ', '_')}.png"
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Interaction heatmap saved")
            
        except Exception as e:
            logger.error(f"Error creating interaction heatmap: {str(e)}")
    
    def _create_partial_dependence_plots(self, pd_data: Dict[str, Dict[str, Any]], 
                                       X: pd.DataFrame) -> None:
        """Create partial dependence plots"""
        try:
            n_features = len(pd_data)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (feature, data) in enumerate(pd_data.items()):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                ax.plot(data['values'], data['average'], linewidth=2)
                ax.set_xlabel(feature)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence - {feature}')
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.remove()
            
            plt.suptitle(f'Partial Dependence Plots - {self.model_name}')
            plt.tight_layout()
            
            # Save plot
            pd_path = config.REPORTS_PATH / "figures" / f"partial_dependence_{self.model_name.lower().replace(' ', '_')}.png"
            pd_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(pd_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Partial dependence plots saved")
            
        except Exception as e:
            logger.error(f"Error creating partial dependence plots: {str(e)}")
    
    def save_explanation(self, explanation: GlobalExplanation, 
                        output_path: Union[str, Path]) -> None:
        """
        Save global explanation to file
        
        Args:
            explanation: GlobalExplanation object to save
            output_path: Path to save the explanation
        """
        import joblib
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save explanation object
        joblib.dump(explanation, output_path)
        
        logger.info(f"Global explanation saved to {output_path}")
    
    @classmethod
    def load_explanation(cls, explanation_path: Union[str, Path]) -> GlobalExplanation:
        """
        Load global explanation from file
        
        Args:
            explanation_path: Path to the saved explanation
            
        Returns:
            GlobalExplanation object
        """
        import joblib
        
        explanation = joblib.load(explanation_path)
        
        logger.info(f"Global explanation loaded from {explanation_path}")
        
        return explanation


class LocalExplainer:
    """
    Generates individual prediction explanations using SHAP force plots and LIME
    Implements explanation consistency validation and local surrogate explanations
    """
    
    def __init__(self, model: BaseEstimator, feature_names: List[str],
                 global_explainer: Optional[GlobalExplainer] = None):
        """
        Initialize LocalExplainer
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            global_explainer: Optional GlobalExplainer for consistency checks
        """
        self.model = model
        self.feature_names = feature_names
        self.global_explainer = global_explainer
        self.lime_explainer = None
        self.background_data = None
        
        logger.info(f"LocalExplainer initialized with {len(feature_names)} features")
    
    def fit_lime_explainer(self, X_background: pd.DataFrame,
                          mode: str = 'classification',
                          max_background_samples: int = 1000) -> None:
        """
        Fit LIME explainer on background data
        
        Args:
            X_background: Background dataset for LIME explainer
            mode: LIME mode ('classification' or 'regression')
            max_background_samples: Maximum number of background samples
        """
        if not HAS_LIME:
            logger.error("LIME not available. Cannot create explainer.")
            return
        
        logger.info("Fitting LIME explainer")
        
        # Sample background data if too large
        if len(X_background) > max_background_samples:
            X_background = X_background.sample(n=max_background_samples, random_state=config.RANDOM_SEED)
        
        self.background_data = X_background
        
        try:
            # Determine feature types
            categorical_features = []
            for i, feature in enumerate(self.feature_names):
                if feature in X_background.columns:
                    if X_background[feature].dtype == 'object' or X_background[feature].nunique() < 10:
                        categorical_features.append(i)
            
            # Create LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background.values,
                feature_names=self.feature_names,
                categorical_features=categorical_features,
                mode=mode,
                random_state=config.RANDOM_SEED
            )
            
            logger.info(f"LIME explainer fitted with {len(X_background)} background samples")
            
        except Exception as e:
            logger.error(f"Error fitting LIME explainer: {str(e)}")
    
    def explain_prediction_shap(self, X_instance: Union[pd.Series, np.ndarray],
                               instance_id: Union[str, int] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanation for individual prediction
        
        Args:
            X_instance: Single instance to explain
            instance_id: Identifier for the instance
            
        Returns:
            Dictionary with SHAP explanation data
        """
        if self.global_explainer is None or self.global_explainer.explainer is None:
            logger.error("Global SHAP explainer not available")
            return {}
        
        logger.info(f"Generating SHAP explanation for instance {instance_id}")
        
        try:
            # Ensure instance is in correct format
            if isinstance(X_instance, pd.Series):
                X_array = X_instance.values.reshape(1, -1)
                X_df = X_instance.to_frame().T
            else:
                X_array = X_instance.reshape(1, -1)
                X_df = pd.DataFrame(X_array, columns=self.feature_names)
            
            # Get prediction
            prediction = self.model.predict_proba(X_array)[0, 1]
            
            # Calculate SHAP values
            if hasattr(self.global_explainer.explainer, 'shap_values'):
                # Tree explainer
                shap_values = self.global_explainer.explainer.shap_values(X_array)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                shap_values = shap_values[0]  # Single instance
            else:
                # Other explainers
                shap_values = self.global_explainer.explainer(X_array)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values[0]
                    if shap_values.ndim == 2:
                        shap_values = shap_values[:, 1]  # Positive class
            
            # Create feature contributions dictionary
            feature_contributions = dict(zip(self.feature_names, shap_values))
            
            # Get top contributing features
            top_features = sorted(feature_contributions.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:10]
            
            # Create SHAP force plot data
            force_plot_data = self._create_shap_force_plot_data(
                X_df.iloc[0], shap_values, prediction, instance_id
            )
            
            shap_explanation = {
                'instance_id': instance_id,
                'prediction': prediction,
                'shap_values': feature_contributions,
                'top_features': top_features,
                'force_plot_data': force_plot_data,
                'base_value': self.global_explainer.explainer.expected_value if hasattr(self.global_explainer.explainer, 'expected_value') else 0.5,
                'feature_values': dict(zip(self.feature_names, X_array[0]))
            }
            
            logger.info(f"SHAP explanation generated for instance {instance_id}")
            
            return shap_explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return {}
    
    def explain_prediction_lime(self, X_instance: Union[pd.Series, np.ndarray],
                               instance_id: Union[str, int] = None,
                               num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for individual prediction
        
        Args:
            X_instance: Single instance to explain
            instance_id: Identifier for the instance
            num_features: Number of features to include in explanation
            
        Returns:
            Dictionary with LIME explanation data
        """
        if self.lime_explainer is None:
            logger.error("LIME explainer not fitted")
            return {}
        
        logger.info(f"Generating LIME explanation for instance {instance_id}")
        
        try:
            # Ensure instance is in correct format
            if isinstance(X_instance, pd.Series):
                X_array = X_instance.values
            else:
                X_array = X_instance
            
            # Get prediction
            prediction = self.model.predict_proba(X_array.reshape(1, -1))[0, 1]
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_array,
                self.model.predict_proba,
                num_features=num_features,
                labels=[1]  # Explain positive class
            )
            
            # Extract explanation data
            lime_data = explanation.as_list(label=1)
            
            # Parse feature contributions
            feature_contributions = {}
            for feature_desc, contribution in lime_data:
                # Extract feature name from description (e.g., "feature_name <= 5.0")
                feature_name = feature_desc.split(' ')[0]
                if feature_name in self.feature_names:
                    feature_contributions[feature_name] = contribution
            
            # Get top contributing features
            top_features = sorted(feature_contributions.items(),
                                key=lambda x: abs(x[1]), reverse=True)
            
            lime_explanation = {
                'instance_id': instance_id,
                'prediction': prediction,
                'lime_contributions': feature_contributions,
                'top_features': top_features,
                'explanation_fit': explanation.score,
                'local_pred': explanation.local_pred[1] if len(explanation.local_pred) > 1 else explanation.local_pred[0],
                'feature_values': dict(zip(self.feature_names, X_array))
            }
            
            logger.info(f"LIME explanation generated for instance {instance_id}")
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            return {}
    
    def explain_instance(self, X_instance: Union[pd.Series, np.ndarray],
                        instance_id: Union[str, int] = None,
                        methods: List[str] = ['shap', 'lime']) -> LocalExplanation:
        """
        Generate comprehensive explanation for a single instance
        
        Args:
            X_instance: Single instance to explain
            instance_id: Identifier for the instance
            methods: List of explanation methods to use
            
        Returns:
            LocalExplanation object with all explanations
        """
        logger.info(f"Generating comprehensive explanation for instance {instance_id}")
        
        # Get prediction
        if isinstance(X_instance, pd.Series):
            X_array = X_instance.values.reshape(1, -1)
        else:
            X_array = X_instance.reshape(1, -1)
        
        prediction = self.model.predict_proba(X_array)[0, 1]
        
        # Initialize explanation components
        shap_explanation = None
        lime_explanation = None
        feature_contributions = {}
        
        # Generate SHAP explanation
        if 'shap' in methods:
            shap_explanation = self.explain_prediction_shap(X_instance, instance_id)
            if shap_explanation:
                feature_contributions.update(shap_explanation.get('shap_values', {}))
        
        # Generate LIME explanation
        if 'lime' in methods:
            lime_explanation = self.explain_prediction_lime(X_instance, instance_id)
        
        # Combine explanations and get top features
        if not feature_contributions and lime_explanation:
            feature_contributions = lime_explanation.get('lime_contributions', {})
        
        top_features = sorted(feature_contributions.items(),
                            key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # Create local explanation object
        local_explanation = LocalExplanation(
            instance_id=instance_id or 'unknown',
            prediction=prediction,
            shap_values=shap_explanation.get('shap_values') if shap_explanation else None,
            lime_explanation=lime_explanation,
            feature_contributions=feature_contributions,
            top_features=top_features,
            explanation_date=datetime.now()
        )
        
        logger.info(f"Comprehensive explanation generated for instance {instance_id}")
        
        return local_explanation
    
    def validate_explanation_consistency(self, X_instances: pd.DataFrame,
                                       sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate consistency between SHAP and LIME explanations
        
        Args:
            X_instances: Dataset of instances to validate
            sample_size: Number of instances to sample for validation
            
        Returns:
            Dictionary with consistency validation results
        """
        logger.info("Validating explanation consistency")
        
        if len(X_instances) > sample_size:
            X_sample = X_instances.sample(n=sample_size, random_state=config.RANDOM_SEED)
        else:
            X_sample = X_instances.copy()
        
        consistency_results = {
            'n_instances': len(X_sample),
            'correlations': {},
            'agreement_rates': {},
            'consistency_score': 0.0
        }
        
        shap_rankings = []
        lime_rankings = []
        
        for idx, (_, instance) in enumerate(X_sample.iterrows()):
            try:
                # Get both explanations
                shap_exp = self.explain_prediction_shap(instance, idx)
                lime_exp = self.explain_prediction_lime(instance, idx)
                
                if shap_exp and lime_exp:
                    # Get feature rankings
                    shap_features = [feat for feat, _ in shap_exp.get('top_features', [])]
                    lime_features = [feat for feat, _ in lime_exp.get('top_features', [])]
                    
                    shap_rankings.append(shap_features)
                    lime_rankings.append(lime_features)
                
            except Exception as e:
                logger.warning(f"Error validating instance {idx}: {str(e)}")
                continue
        
        if shap_rankings and lime_rankings:
            # Calculate agreement rates for top-k features
            for k in [3, 5, 10]:
                agreements = []
                for shap_top, lime_top in zip(shap_rankings, lime_rankings):
                    shap_k = set(shap_top[:k])
                    lime_k = set(lime_top[:k])
                    if len(shap_k) > 0 and len(lime_k) > 0:
                        agreement = len(shap_k.intersection(lime_k)) / len(shap_k.union(lime_k))
                        agreements.append(agreement)
                
                if agreements:
                    consistency_results['agreement_rates'][f'top_{k}'] = np.mean(agreements)
            
            # Overall consistency score
            if consistency_results['agreement_rates']:
                consistency_results['consistency_score'] = np.mean(list(consistency_results['agreement_rates'].values()))
        
        logger.info(f"Explanation consistency validation completed. Score: {consistency_results['consistency_score']:.3f}")
        
        return consistency_results
    
    def batch_explain(self, X_instances: pd.DataFrame,
                     methods: List[str] = ['shap'],
                     max_instances: int = 100) -> List[LocalExplanation]:
        """
        Generate explanations for multiple instances
        
        Args:
            X_instances: Dataset of instances to explain
            methods: List of explanation methods to use
            max_instances: Maximum number of instances to explain
            
        Returns:
            List of LocalExplanation objects
        """
        logger.info(f"Generating batch explanations for {len(X_instances)} instances")
        
        if len(X_instances) > max_instances:
            X_sample = X_instances.sample(n=max_instances, random_state=config.RANDOM_SEED)
        else:
            X_sample = X_instances.copy()
        
        explanations = []
        
        for idx, (instance_id, instance) in enumerate(X_sample.iterrows()):
            try:
                explanation = self.explain_instance(instance, instance_id, methods)
                explanations.append(explanation)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(X_sample)} instances")
                    
            except Exception as e:
                logger.warning(f"Error explaining instance {instance_id}: {str(e)}")
                continue
        
        logger.info(f"Batch explanation completed. Generated {len(explanations)} explanations")
        
        return explanations
    
    def _create_shap_force_plot_data(self, instance: pd.Series, shap_values: np.ndarray,
                                   prediction: float, instance_id: Union[str, int]) -> Dict[str, Any]:
        """Create data for SHAP force plot visualization"""
        try:
            # Prepare data for force plot
            force_plot_data = {
                'instance_id': instance_id,
                'prediction': prediction,
                'features': [],
                'base_value': 0.5  # Default base value
            }
            
            # Add feature contributions
            for i, (feature_name, shap_value) in enumerate(zip(self.feature_names, shap_values)):
                feature_data = {
                    'name': feature_name,
                    'value': instance[feature_name] if feature_name in instance.index else 0,
                    'shap_value': float(shap_value),
                    'contribution': 'positive' if shap_value > 0 else 'negative'
                }
                force_plot_data['features'].append(feature_data)
            
            # Sort by absolute SHAP value
            force_plot_data['features'].sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return force_plot_data
            
        except Exception as e:
            logger.error(f"Error creating force plot data: {str(e)}")
            return {}
    
    def create_local_explanation_plot(self, explanation: LocalExplanation,
                                    save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create visualization for local explanation
        
        Args:
            explanation: LocalExplanation object to visualize
            save_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Feature contributions (SHAP or LIME)
            if explanation.shap_values:
                contributions = explanation.shap_values
                title = "SHAP Feature Contributions"
            elif explanation.lime_explanation:
                contributions = explanation.lime_explanation.get('lime_contributions', {})
                title = "LIME Feature Contributions"
            else:
                contributions = explanation.feature_contributions
                title = "Feature Contributions"
            
            # Get top 10 features
            top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            
            features, values = zip(*top_features)
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            axes[0].barh(range(len(features)), values, color=colors, alpha=0.7)
            axes[0].set_yticks(range(len(features)))
            axes[0].set_yticklabels(features)
            axes[0].set_xlabel('Contribution to Prediction')
            axes[0].set_title(title)
            axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Prediction summary
            axes[1].text(0.1, 0.8, f"Instance ID: {explanation.instance_id}", fontsize=12, transform=axes[1].transAxes)
            axes[1].text(0.1, 0.7, f"Churn Probability: {explanation.prediction:.3f}", fontsize=12, transform=axes[1].transAxes)
            axes[1].text(0.1, 0.6, f"Risk Level: {'High' if explanation.prediction > 0.7 else 'Medium' if explanation.prediction > 0.3 else 'Low'}", 
                        fontsize=12, transform=axes[1].transAxes)
            
            # Add top contributing factors
            axes[1].text(0.1, 0.4, "Top Contributing Factors:", fontsize=12, fontweight='bold', transform=axes[1].transAxes)
            for i, (feature, contribution) in enumerate(explanation.top_features[:5]):
                direction = "increases" if contribution > 0 else "decreases"
                axes[1].text(0.1, 0.3 - i*0.05, f"• {feature} {direction} churn risk", 
                           fontsize=10, transform=axes[1].transAxes)
            
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].axis('off')
            axes[1].set_title('Prediction Summary')
            
            plt.suptitle(f'Local Explanation - Instance {explanation.instance_id}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Local explanation plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating local explanation plot: {str(e)}")
    
    def save_explanation(self, explanation: LocalExplanation,
                        output_path: Union[str, Path]) -> None:
        """
        Save local explanation to file
        
        Args:
            explanation: LocalExplanation object to save
            output_path: Path to save the explanation
        """
        import joblib
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(explanation, output_path)
        
        logger.info(f"Local explanation saved to {output_path}")
    
    @classmethod
    def load_explanation(cls, explanation_path: Union[str, Path]) -> LocalExplanation:
        """
        Load local explanation from file
        
        Args:
            explanation_path: Path to the saved explanation
            
        Returns:
            LocalExplanation object
        """
        import joblib
        
        explanation = joblib.load(explanation_path)
        
        logger.info(f"Local explanation loaded from {explanation_path}")
        
        return explanation


@dataclass
class BusinessRecommendation:
    """Container for business recommendations"""
    customer_id: Union[str, int]
    churn_probability: float
    risk_level: str
    primary_actions: List[str]
    secondary_actions: List[str]
    priority_score: float
    explanation_summary: str
    confidence: float
    estimated_impact: Dict[str, float]


class BusinessRecommendationEngine:
    """
    Translates technical explanations into business actions and recommendations
    Creates action templates based on explanation patterns and implements prioritization logic
    """
    
    def __init__(self, business_config: Optional[Dict[str, Any]] = None):
        """
        Initialize BusinessRecommendationEngine
        
        Args:
            business_config: Configuration for business parameters and action templates
        """
        self.business_config = business_config or self._get_default_business_config()
        self.action_templates = self._initialize_action_templates()
        self.feature_action_mapping = self._initialize_feature_action_mapping()
        
        logger.info("BusinessRecommendationEngine initialized")
    
    def generate_recommendations(self, explanation: LocalExplanation,
                               customer_data: Optional[Dict[str, Any]] = None) -> BusinessRecommendation:
        """
        Generate business recommendations based on model explanation
        
        Args:
            explanation: LocalExplanation object with model insights
            customer_data: Additional customer information for context
            
        Returns:
            BusinessRecommendation object with actionable insights
        """
        logger.info(f"Generating business recommendations for customer {explanation.instance_id}")
        
        # Determine risk level
        risk_level = self._determine_risk_level(explanation.prediction)
        
        # Generate primary actions based on top contributing features
        primary_actions = self._generate_primary_actions(explanation.top_features, customer_data)
        
        # Generate secondary actions based on risk level and business rules
        secondary_actions = self._generate_secondary_actions(risk_level, explanation.top_features)
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(explanation, customer_data)
        
        # Create explanation summary
        explanation_summary = self._create_explanation_summary(explanation)
        
        # Calculate confidence in recommendations
        confidence = self._calculate_recommendation_confidence(explanation)
        
        # Estimate business impact
        estimated_impact = self._estimate_business_impact(explanation, primary_actions)
        
        recommendation = BusinessRecommendation(
            customer_id=explanation.instance_id,
            churn_probability=explanation.prediction,
            risk_level=risk_level,
            primary_actions=primary_actions,
            secondary_actions=secondary_actions,
            priority_score=priority_score,
            explanation_summary=explanation_summary,
            confidence=confidence,
            estimated_impact=estimated_impact
        )
        
        logger.info(f"Business recommendations generated for customer {explanation.instance_id}")
        
        return recommendation
    
    def batch_generate_recommendations(self, explanations: List[LocalExplanation],
                                     customer_data_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> List[BusinessRecommendation]:
        """
        Generate recommendations for multiple customers
        
        Args:
            explanations: List of LocalExplanation objects
            customer_data_dict: Dictionary mapping customer IDs to additional data
            
        Returns:
            List of BusinessRecommendation objects
        """
        logger.info(f"Generating batch recommendations for {len(explanations)} customers")
        
        recommendations = []
        
        for explanation in explanations:
            try:
                customer_data = None
                if customer_data_dict and explanation.instance_id in customer_data_dict:
                    customer_data = customer_data_dict[explanation.instance_id]
                
                recommendation = self.generate_recommendations(explanation, customer_data)
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning(f"Error generating recommendation for customer {explanation.instance_id}: {str(e)}")
                continue
        
        logger.info(f"Batch recommendations completed. Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def prioritize_customers(self, recommendations: List[BusinessRecommendation],
                           max_contacts: Optional[int] = None) -> List[BusinessRecommendation]:
        """
        Prioritize customers for retention campaigns based on multiple factors
        
        Args:
            recommendations: List of BusinessRecommendation objects
            max_contacts: Maximum number of customers to contact
            
        Returns:
            Sorted list of prioritized recommendations
        """
        logger.info(f"Prioritizing {len(recommendations)} customers for retention campaigns")
        
        # Sort by priority score (descending)
        prioritized = sorted(recommendations, key=lambda x: x.priority_score, reverse=True)
        
        # Apply contact limit if specified
        if max_contacts and max_contacts < len(prioritized):
            prioritized = prioritized[:max_contacts]
            logger.info(f"Limited to top {max_contacts} customers based on priority score")
        
        return prioritized
    
    def create_campaign_segments(self, recommendations: List[BusinessRecommendation]) -> Dict[str, List[BusinessRecommendation]]:
        """
        Segment customers into campaign groups based on risk level and recommended actions
        
        Args:
            recommendations: List of BusinessRecommendation objects
            
        Returns:
            Dictionary mapping segment names to customer lists
        """
        logger.info("Creating campaign segments")
        
        segments = {
            'high_risk_immediate': [],
            'high_risk_engagement': [],
            'medium_risk_proactive': [],
            'medium_risk_value': [],
            'low_risk_monitoring': []
        }
        
        for rec in recommendations:
            if rec.risk_level == 'High':
                # Segment high-risk customers by primary action type
                if any('immediate' in action.lower() or 'urgent' in action.lower() 
                      for action in rec.primary_actions):
                    segments['high_risk_immediate'].append(rec)
                else:
                    segments['high_risk_engagement'].append(rec)
            
            elif rec.risk_level == 'Medium':
                # Segment medium-risk customers by action focus
                if any('value' in action.lower() or 'discount' in action.lower() 
                      for action in rec.primary_actions):
                    segments['medium_risk_value'].append(rec)
                else:
                    segments['medium_risk_proactive'].append(rec)
            
            else:  # Low risk
                segments['low_risk_monitoring'].append(rec)
        
        # Log segment sizes
        for segment_name, customers in segments.items():
            logger.info(f"Segment '{segment_name}': {len(customers)} customers")
        
        return segments
    
    def generate_campaign_summary(self, segments: Dict[str, List[BusinessRecommendation]]) -> Dict[str, Any]:
        """
        Generate summary statistics for campaign segments
        
        Args:
            segments: Dictionary of campaign segments
            
        Returns:
            Dictionary with campaign summary statistics
        """
        logger.info("Generating campaign summary")
        
        total_customers = sum(len(customers) for customers in segments.values())
        
        summary = {
            'total_customers': total_customers,
            'segments': {},
            'overall_metrics': {
                'avg_churn_probability': 0.0,
                'avg_priority_score': 0.0,
                'estimated_total_impact': 0.0
            }
        }
        
        all_recommendations = []
        
        for segment_name, customers in segments.items():
            if customers:
                segment_summary = {
                    'count': len(customers),
                    'percentage': len(customers) / total_customers * 100 if total_customers > 0 else 0,
                    'avg_churn_probability': np.mean([c.churn_probability for c in customers]),
                    'avg_priority_score': np.mean([c.priority_score for c in customers]),
                    'avg_confidence': np.mean([c.confidence for c in customers]),
                    'estimated_impact': sum(c.estimated_impact.get('expected_value', 0) for c in customers),
                    'top_actions': self._get_top_actions_for_segment(customers)
                }
                
                summary['segments'][segment_name] = segment_summary
                all_recommendations.extend(customers)
        
        # Calculate overall metrics
        if all_recommendations:
            summary['overall_metrics']['avg_churn_probability'] = np.mean([r.churn_probability for r in all_recommendations])
            summary['overall_metrics']['avg_priority_score'] = np.mean([r.priority_score for r in all_recommendations])
            summary['overall_metrics']['estimated_total_impact'] = sum(r.estimated_impact.get('expected_value', 0) for r in all_recommendations)
        
        return summary
    
    def _determine_risk_level(self, churn_probability: float) -> str:
        """Determine risk level based on churn probability"""
        thresholds = self.business_config.get('risk_thresholds', {
            'high': 0.7,
            'medium': 0.3
        })
        
        if churn_probability >= thresholds['high']:
            return 'High'
        elif churn_probability >= thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_primary_actions(self, top_features: List[Tuple[str, float]],
                                customer_data: Optional[Dict[str, Any]]) -> List[str]:
        """Generate primary actions based on top contributing features"""
        actions = []
        
        for feature, contribution in top_features[:5]:  # Focus on top 5 features
            # Map feature to business actions
            if feature in self.feature_action_mapping:
                feature_actions = self.feature_action_mapping[feature]
                
                # Choose action based on contribution direction
                if contribution > 0:  # Increases churn risk
                    action_type = 'reduce_risk'
                else:  # Decreases churn risk (protective factor)
                    action_type = 'maintain_strength'
                
                if action_type in feature_actions:
                    actions.extend(feature_actions[action_type])
        
        # Remove duplicates and limit to top actions
        unique_actions = list(dict.fromkeys(actions))  # Preserves order
        return unique_actions[:3]  # Return top 3 primary actions
    
    def _generate_secondary_actions(self, risk_level: str, 
                                  top_features: List[Tuple[str, float]]) -> List[str]:
        """Generate secondary actions based on risk level and business rules"""
        secondary_actions = []
        
        # Risk-level specific actions
        if risk_level == 'High':
            secondary_actions.extend([
                "Schedule immediate customer success call",
                "Offer premium support upgrade",
                "Provide executive escalation contact"
            ])
        elif risk_level == 'Medium':
            secondary_actions.extend([
                "Send proactive engagement email",
                "Offer product training session",
                "Schedule quarterly business review"
            ])
        else:  # Low risk
            secondary_actions.extend([
                "Include in customer satisfaction survey",
                "Send product update newsletter",
                "Monitor usage patterns"
            ])
        
        # Feature-specific secondary actions
        feature_categories = self._categorize_features(top_features)
        
        if 'usage' in feature_categories:
            secondary_actions.append("Provide usage optimization recommendations")
        
        if 'billing' in feature_categories:
            secondary_actions.append("Review billing and payment options")
        
        if 'support' in feature_categories:
            secondary_actions.append("Improve support experience")
        
        return secondary_actions[:4]  # Limit to 4 secondary actions
    
    def _calculate_priority_score(self, explanation: LocalExplanation,
                                customer_data: Optional[Dict[str, Any]]) -> float:
        """Calculate priority score for customer ranking"""
        # Base score from churn probability
        base_score = explanation.prediction
        
        # Adjust for confidence in explanation
        confidence_weight = 0.2
        confidence_adjustment = (explanation.prediction * confidence_weight * 
                               len(explanation.top_features) / 10)  # Normalize by max features
        
        # Adjust for customer value (if available)
        value_adjustment = 0.0
        if customer_data:
            customer_value = customer_data.get('lifetime_value', 0)
            if customer_value > 0:
                # Normalize customer value (assuming max value of 10000)
                normalized_value = min(customer_value / 10000, 1.0)
                value_adjustment = normalized_value * 0.3
        
        # Combine components
        priority_score = base_score + confidence_adjustment + value_adjustment
        
        return min(priority_score, 1.0)  # Cap at 1.0
    
    def _create_explanation_summary(self, explanation: LocalExplanation) -> str:
        """Create human-readable explanation summary"""
        risk_level = self._determine_risk_level(explanation.prediction)
        
        summary_parts = [
            f"Customer has {risk_level.lower()} churn risk ({explanation.prediction:.1%} probability)."
        ]
        
        if explanation.top_features:
            top_feature, top_contribution = explanation.top_features[0]
            direction = "increases" if top_contribution > 0 else "decreases"
            summary_parts.append(f"Primary factor: {top_feature} {direction} churn risk.")
            
            if len(explanation.top_features) > 1:
                other_factors = [feat for feat, _ in explanation.top_features[1:3]]
                summary_parts.append(f"Other key factors: {', '.join(other_factors)}.")
        
        return " ".join(summary_parts)
    
    def _calculate_recommendation_confidence(self, explanation: LocalExplanation) -> float:
        """Calculate confidence in the recommendations"""
        # Base confidence from number of strong features
        strong_features = sum(1 for _, contrib in explanation.top_features 
                            if abs(contrib) > 0.1)  # Threshold for "strong" contribution
        
        feature_confidence = min(strong_features / 5, 1.0)  # Normalize by 5 features
        
        # Adjust for prediction certainty
        prediction_certainty = abs(explanation.prediction - 0.5) * 2  # 0 to 1 scale
        
        # Combine factors
        confidence = (feature_confidence * 0.6 + prediction_certainty * 0.4)
        
        return confidence
    
    def _estimate_business_impact(self, explanation: LocalExplanation,
                                primary_actions: List[str]) -> Dict[str, float]:
        """Estimate business impact of recommendations"""
        # Base retention probability improvement from actions
        action_effectiveness = self.business_config.get('action_effectiveness', {})
        
        total_effectiveness = 0.0
        for action in primary_actions:
            # Map action to effectiveness (simplified)
            if 'discount' in action.lower():
                total_effectiveness += action_effectiveness.get('discount', 0.15)
            elif 'support' in action.lower():
                total_effectiveness += action_effectiveness.get('support', 0.10)
            elif 'engagement' in action.lower():
                total_effectiveness += action_effectiveness.get('engagement', 0.08)
            else:
                total_effectiveness += action_effectiveness.get('default', 0.05)
        
        # Cap effectiveness
        total_effectiveness = min(total_effectiveness, 0.5)  # Max 50% improvement
        
        # Calculate expected value
        retention_probability = total_effectiveness
        customer_value = self.business_config.get('avg_customer_value', 1000)
        contact_cost = self.business_config.get('contact_cost', 50)
        
        expected_value = (retention_probability * customer_value) - contact_cost
        
        return {
            'retention_probability_improvement': total_effectiveness,
            'expected_value': expected_value,
            'contact_cost': contact_cost,
            'potential_revenue': retention_probability * customer_value
        }
    
    def _get_top_actions_for_segment(self, customers: List[BusinessRecommendation]) -> List[str]:
        """Get most common actions for a segment"""
        action_counts = {}
        
        for customer in customers:
            for action in customer.primary_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Sort by frequency and return top 3
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [action for action, _ in sorted_actions[:3]]
    
    def _categorize_features(self, top_features: List[Tuple[str, float]]) -> List[str]:
        """Categorize features into business domains"""
        categories = []
        
        for feature, _ in top_features:
            feature_lower = feature.lower()
            
            if any(usage_term in feature_lower for usage_term in ['usage', 'activity', 'login', 'session']):
                categories.append('usage')
            elif any(billing_term in feature_lower for billing_term in ['charge', 'payment', 'bill', 'cost']):
                categories.append('billing')
            elif any(support_term in feature_lower for support_term in ['support', 'ticket', 'complaint']):
                categories.append('support')
            elif any(service_term in feature_lower for service_term in ['service', 'feature', 'product']):
                categories.append('service')
        
        return list(set(categories))  # Remove duplicates
    
    def _get_default_business_config(self) -> Dict[str, Any]:
        """Get default business configuration"""
        return {
            'risk_thresholds': {
                'high': 0.7,
                'medium': 0.3
            },
            'action_effectiveness': {
                'discount': 0.15,
                'support': 0.10,
                'engagement': 0.08,
                'default': 0.05
            },
            'avg_customer_value': 1000.0,
            'contact_cost': 50.0
        }
    
    def _initialize_action_templates(self) -> Dict[str, List[str]]:
        """Initialize action templates for different scenarios"""
        return {
            'high_risk': [
                "Offer immediate retention discount (10-20%)",
                "Schedule urgent customer success call",
                "Provide dedicated account manager",
                "Offer service upgrade at no cost"
            ],
            'medium_risk': [
                "Send personalized engagement email",
                "Offer product training session",
                "Provide usage optimization tips",
                "Schedule quarterly business review"
            ],
            'low_risk': [
                "Include in customer satisfaction survey",
                "Send product update newsletter",
                "Monitor usage patterns",
                "Invite to user community events"
            ]
        }
    
    def _initialize_feature_action_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize mapping from features to business actions"""
        return {
            'monthly_charges': {
                'reduce_risk': [
                    "Offer billing discount or payment plan",
                    "Review service plan for cost optimization",
                    "Provide value demonstration session"
                ],
                'maintain_strength': [
                    "Highlight value received for current spend",
                    "Offer loyalty rewards program"
                ]
            },
            'total_charges': {
                'reduce_risk': [
                    "Offer retention credit",
                    "Review historical usage for optimization"
                ],
                'maintain_strength': [
                    "Acknowledge customer loyalty",
                    "Offer premium service benefits"
                ]
            },
            'tenure': {
                'reduce_risk': [
                    "Provide new customer onboarding support",
                    "Offer extended trial period"
                ],
                'maintain_strength': [
                    "Celebrate customer milestone",
                    "Offer long-term customer benefits"
                ]
            },
            'contract': {
                'reduce_risk': [
                    "Offer contract flexibility options",
                    "Provide contract renewal incentives"
                ],
                'maintain_strength': [
                    "Highlight contract benefits",
                    "Offer contract upgrade options"
                ]
            },
            'support_tickets': {
                'reduce_risk': [
                    "Improve support response time",
                    "Assign dedicated support representative",
                    "Provide proactive support check-ins"
                ],
                'maintain_strength': [
                    "Maintain current support quality",
                    "Offer premium support upgrade"
                ]
            }
        }
    
    def export_recommendations(self, recommendations: List[BusinessRecommendation],
                             output_path: Union[str, Path],
                             format: str = 'csv') -> None:
        """
        Export recommendations to file
        
        Args:
            recommendations: List of BusinessRecommendation objects
            output_path: Path to save the recommendations
            format: Export format ('csv', 'json', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        data = []
        for rec in recommendations:
            data.append({
                'customer_id': rec.customer_id,
                'churn_probability': rec.churn_probability,
                'risk_level': rec.risk_level,
                'priority_score': rec.priority_score,
                'primary_actions': '; '.join(rec.primary_actions),
                'secondary_actions': '; '.join(rec.secondary_actions),
                'explanation_summary': rec.explanation_summary,
                'confidence': rec.confidence,
                'expected_value': rec.estimated_impact.get('expected_value', 0),
                'retention_improvement': rec.estimated_impact.get('retention_probability_improvement', 0)
            })
        
        df = pd.DataFrame(data)
        
        # Export in specified format
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Recommendations exported to {output_path} in {format} format")


class ExplanationValidator:
    """
    Validates explanation consistency with domain knowledge and business logic
    Implements quality checks for SHAP and LIME explanations
    """
    
    def __init__(self, domain_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize ExplanationValidator
        
        Args:
            domain_rules: Dictionary with domain-specific validation rules
        """
        self.domain_rules = domain_rules or self._get_default_domain_rules()
        
        logger.info("ExplanationValidator initialized")
    
    def validate_explanation_quality(self, explanation: LocalExplanation) -> Dict[str, Any]:
        """
        Validate the quality of a local explanation
        
        Args:
            explanation: LocalExplanation object to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating explanation quality for instance {explanation.instance_id}")
        
        validation_results = {
            'instance_id': explanation.instance_id,
            'overall_quality': 'good',
            'issues': [],
            'warnings': [],
            'quality_score': 0.0,
            'checks_performed': []
        }
        
        # Check 1: Feature contribution consistency
        consistency_check = self._check_feature_consistency(explanation)
        validation_results['checks_performed'].append('feature_consistency')
        if not consistency_check['passed']:
            validation_results['issues'].extend(consistency_check['issues'])
        
        # Check 2: Domain knowledge alignment
        domain_check = self._check_domain_alignment(explanation)
        validation_results['checks_performed'].append('domain_alignment')
        if not domain_check['passed']:
            validation_results['warnings'].extend(domain_check['warnings'])
        
        # Check 3: Explanation completeness
        completeness_check = self._check_explanation_completeness(explanation)
        validation_results['checks_performed'].append('completeness')
        if not completeness_check['passed']:
            validation_results['warnings'].extend(completeness_check['warnings'])
        
        # Check 4: SHAP additivity (if SHAP values available)
        if explanation.shap_values:
            additivity_check = self._check_shap_additivity(explanation)
            validation_results['checks_performed'].append('shap_additivity')
            if not additivity_check['passed']:
                validation_results['issues'].extend(additivity_check['issues'])
        
        # Calculate overall quality score
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        # Determine overall quality
        if len(validation_results['issues']) > 0:
            validation_results['overall_quality'] = 'poor'
        elif len(validation_results['warnings']) > 2:
            validation_results['overall_quality'] = 'fair'
        else:
            validation_results['overall_quality'] = 'good'
        
        logger.info(f"Explanation validation completed. Quality: {validation_results['overall_quality']}")
        
        return validation_results
    
    def validate_business_alignment(self, recommendation: BusinessRecommendation) -> Dict[str, Any]:
        """
        Validate business recommendation alignment with business rules
        
        Args:
            recommendation: BusinessRecommendation object to validate
            
        Returns:
            Dictionary with business validation results
        """
        logger.info(f"Validating business alignment for customer {recommendation.customer_id}")
        
        validation_results = {
            'customer_id': recommendation.customer_id,
            'business_alignment': 'good',
            'issues': [],
            'suggestions': [],
            'alignment_score': 0.0
        }
        
        # Check action appropriateness for risk level
        risk_action_check = self._check_risk_action_alignment(recommendation)
        if not risk_action_check['passed']:
            validation_results['issues'].extend(risk_action_check['issues'])
        
        # Check cost-effectiveness
        cost_check = self._check_cost_effectiveness(recommendation)
        if not cost_check['passed']:
            validation_results['suggestions'].extend(cost_check['suggestions'])
        
        # Check action feasibility
        feasibility_check = self._check_action_feasibility(recommendation)
        if not feasibility_check['passed']:
            validation_results['suggestions'].extend(feasibility_check['suggestions'])
        
        # Calculate alignment score
        validation_results['alignment_score'] = self._calculate_alignment_score(validation_results)
        
        # Determine overall alignment
        if len(validation_results['issues']) > 0:
            validation_results['business_alignment'] = 'poor'
        elif len(validation_results['suggestions']) > 2:
            validation_results['business_alignment'] = 'fair'
        else:
            validation_results['business_alignment'] = 'good'
        
        return validation_results
    
    def _check_feature_consistency(self, explanation: LocalExplanation) -> Dict[str, Any]:
        """Check consistency of feature contributions"""
        issues = []
        
        # Check for contradictory contributions
        if explanation.shap_values and explanation.lime_explanation:
            shap_values = explanation.shap_values
            lime_values = explanation.lime_explanation.get('lime_contributions', {})
            
            # Compare signs of contributions for common features
            for feature in shap_values:
                if feature in lime_values:
                    shap_sign = np.sign(shap_values[feature])
                    lime_sign = np.sign(lime_values[feature])
                    
                    if shap_sign != lime_sign and abs(shap_values[feature]) > 0.01:
                        issues.append(f"Contradictory contributions for {feature}: SHAP={shap_sign}, LIME={lime_sign}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _check_domain_alignment(self, explanation: LocalExplanation) -> Dict[str, Any]:
        """Check alignment with domain knowledge"""
        warnings = []
        
        # Check for unexpected feature relationships
        for feature, contribution in explanation.top_features[:5]:
            if feature in self.domain_rules.get('expected_relationships', {}):
                expected_sign = self.domain_rules['expected_relationships'][feature]
                actual_sign = np.sign(contribution)
                
                if expected_sign != actual_sign:
                    warnings.append(f"Unexpected relationship for {feature}: expected {expected_sign}, got {actual_sign}")
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _check_explanation_completeness(self, explanation: LocalExplanation) -> Dict[str, Any]:
        """Check completeness of explanation"""
        warnings = []
        
        # Check if we have sufficient features
        if len(explanation.top_features) < 3:
            warnings.append("Explanation has fewer than 3 contributing features")
        
        # Check if contributions are meaningful
        total_contribution = sum(abs(contrib) for _, contrib in explanation.top_features)
        if total_contribution < 0.1:
            warnings.append("Feature contributions are very small, explanation may be unreliable")
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _check_shap_additivity(self, explanation: LocalExplanation) -> Dict[str, Any]:
        """Check SHAP additivity property"""
        issues = []
        
        if explanation.shap_values:
            # Check if SHAP values sum approximately to prediction difference from base
            shap_sum = sum(explanation.shap_values.values())
            base_value = 0.5  # Assuming base value of 0.5 for binary classification
            expected_prediction = base_value + shap_sum
            
            prediction_diff = abs(expected_prediction - explanation.prediction)
            
            if prediction_diff > 0.05:  # 5% tolerance
                issues.append(f"SHAP additivity violated: expected {expected_prediction:.3f}, got {explanation.prediction:.3f}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _check_risk_action_alignment(self, recommendation: BusinessRecommendation) -> Dict[str, Any]:
        """Check if actions are appropriate for risk level"""
        issues = []
        
        risk_level = recommendation.risk_level
        actions = recommendation.primary_actions
        
        # Define action intensity levels
        high_intensity_keywords = ['immediate', 'urgent', 'discount', 'credit']
        medium_intensity_keywords = ['proactive', 'engagement', 'training']
        low_intensity_keywords = ['monitor', 'survey', 'newsletter']
        
        for action in actions:
            action_lower = action.lower()
            
            if risk_level == 'Low':
                if any(keyword in action_lower for keyword in high_intensity_keywords):
                    issues.append(f"High-intensity action '{action}' not appropriate for low-risk customer")
            
            elif risk_level == 'High':
                if any(keyword in action_lower for keyword in low_intensity_keywords):
                    issues.append(f"Low-intensity action '{action}' insufficient for high-risk customer")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def _check_cost_effectiveness(self, recommendation: BusinessRecommendation) -> Dict[str, Any]:
        """Check cost-effectiveness of recommendations"""
        suggestions = []
        
        expected_value = recommendation.estimated_impact.get('expected_value', 0)
        
        if expected_value < 0:
            suggestions.append("Negative expected value - consider less costly actions")
        
        if expected_value < 100 and recommendation.risk_level == 'Low':
            suggestions.append("Low expected value for low-risk customer - consider monitoring only")
        
        return {
            'passed': len(suggestions) == 0,
            'suggestions': suggestions
        }
    
    def _check_action_feasibility(self, recommendation: BusinessRecommendation) -> Dict[str, Any]:
        """Check feasibility of recommended actions"""
        suggestions = []
        
        # Check for too many actions
        total_actions = len(recommendation.primary_actions) + len(recommendation.secondary_actions)
        if total_actions > 6:
            suggestions.append("Too many recommended actions - prioritize top 3-4 actions")
        
        # Check for conflicting actions
        all_actions = recommendation.primary_actions + recommendation.secondary_actions
        action_text = ' '.join(all_actions).lower()
        
        if 'discount' in action_text and 'upgrade' in action_text:
            suggestions.append("Conflicting actions: offering both discount and upgrade")
        
        return {
            'passed': len(suggestions) == 0,
            'suggestions': suggestions
        }
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        base_score = 1.0
        
        # Deduct for issues and warnings
        issue_penalty = len(validation_results['issues']) * 0.2
        warning_penalty = len(validation_results['warnings']) * 0.1
        
        quality_score = max(0.0, base_score - issue_penalty - warning_penalty)
        
        return quality_score
    
    def _calculate_alignment_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate business alignment score"""
        base_score = 1.0
        
        # Deduct for issues and suggestions
        issue_penalty = len(validation_results['issues']) * 0.3
        suggestion_penalty = len(validation_results['suggestions']) * 0.1
        
        alignment_score = max(0.0, base_score - issue_penalty - suggestion_penalty)
        
        return alignment_score
    
    def _get_default_domain_rules(self) -> Dict[str, Any]:
        """Get default domain validation rules"""
        return {
            'expected_relationships': {
                'monthly_charges': 1,  # Higher charges should increase churn risk
                'tenure': -1,  # Longer tenure should decrease churn risk
                'total_charges': 1,  # Higher total charges should increase churn risk
                'contract_month_to_month': 1,  # Month-to-month should increase churn risk
                'payment_method_electronic_check': 1,  # Electronic check should increase churn risk
                'internet_service_fiber_optic': -1,  # Fiber optic should decrease churn risk
            },
            'feature_importance_thresholds': {
                'minimum_contribution': 0.01,
                'maximum_features': 10
            }
        }