"""
Customer Segmentation and CLV Analysis Module

This module implements customer segmentation using clustering algorithms,
Customer Lifetime Value (CLV) calculation using statistical models,
and customer prioritization for retention campaigns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Clustering and statistical libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# CLV modeling libraries
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False
    print("Warning: lifetimes library not available. CLV functionality will be limited.")


@dataclass
class SegmentProfile:
    """Data class for storing segment characteristics"""
    segment_id: int
    size: int
    churn_rate: float
    avg_clv: float
    characteristics: Dict[str, Any]
    top_features: List[Tuple[str, float]]


@dataclass
class CLVPrediction:
    """Data class for CLV prediction results"""
    customer_id: str
    predicted_clv: float
    confidence_interval: Tuple[float, float]
    frequency: float
    recency: float
    monetary_value: float


class CustomerSegmenter:
    """
    Customer segmentation using clustering algorithms with optimal cluster selection
    and comprehensive segment profiling.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize CustomerSegmenter
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_n_clusters = None
        self.cluster_labels = None
        self.segment_profiles = None
        
    def find_optimal_clusters(self, X: pd.DataFrame, max_clusters: int = 10, 
                            method: str = 'kmeans') -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple evaluation metrics
        
        Args:
            X: Feature matrix for clustering
            max_clusters: Maximum number of clusters to test
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary with optimal cluster results and evaluation metrics
        """
        if len(X) < 10:
            raise ValueError("Need at least 10 samples for clustering")
            
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Test different numbers of clusters
        cluster_range = range(2, min(max_clusters + 1, len(X) // 2))
        metrics = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'inertia': []
        }
        
        models = {}
        
        for n_clusters in cluster_range:
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise ValueError("Method must be 'kmeans' or 'hierarchical'")
                
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            sil_score = silhouette_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            
            metrics['n_clusters'].append(n_clusters)
            metrics['silhouette'].append(sil_score)
            metrics['calinski_harabasz'].append(ch_score)
            metrics['davies_bouldin'].append(db_score)
            
            if method == 'kmeans':
                metrics['inertia'].append(model.inertia_)
            else:
                metrics['inertia'].append(0)  # Not applicable for hierarchical
                
            models[n_clusters] = model
        
        # Find optimal number of clusters (highest silhouette score)
        metrics_df = pd.DataFrame(metrics)
        best_idx = metrics_df['silhouette'].idxmax()
        self.best_n_clusters = metrics_df.loc[best_idx, 'n_clusters']
        self.best_model = models[self.best_n_clusters]
        
        return {
            'optimal_clusters': self.best_n_clusters,
            'metrics': metrics_df,
            'best_silhouette': metrics_df.loc[best_idx, 'silhouette'],
            'models': models
        }
    
    def perform_kmeans_clustering(self, X: pd.DataFrame, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform K-means clustering with optional automatic cluster selection
        
        Args:
            X: Feature matrix for clustering
            n_clusters: Number of clusters (if None, will find optimal)
            
        Returns:
            Tuple of (cluster_labels, clustering_results)
        """
        if n_clusters is None:
            # Find optimal number of clusters
            results = self.find_optimal_clusters(X, method='kmeans')
            n_clusters = results['optimal_clusters']
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.best_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = self.best_model.fit_predict(X_scaled)
        self.best_n_clusters = n_clusters
        
        # Calculate final metrics
        sil_score = silhouette_score(X_scaled, self.cluster_labels)
        ch_score = calinski_harabasz_score(X_scaled, self.cluster_labels)
        db_score = davies_bouldin_score(X_scaled, self.cluster_labels)
        
        results = {
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'inertia': self.best_model.inertia_,
            'cluster_centers': self.best_model.cluster_centers_,
            'labels': self.cluster_labels
        }
        
        return self.cluster_labels, results
    
    def perform_hierarchical_clustering(self, X: pd.DataFrame, n_clusters: Optional[int] = None,
                                      linkage_method: str = 'ward') -> Tuple[np.ndarray, Dict]:
        """
        Perform hierarchical clustering with dendrogram analysis
        
        Args:
            X: Feature matrix for clustering
            n_clusters: Number of clusters (if None, will find optimal)
            linkage_method: Linkage method for hierarchical clustering
            
        Returns:
            Tuple of (cluster_labels, clustering_results)
        """
        if n_clusters is None:
            # Find optimal number of clusters
            results = self.find_optimal_clusters(X, method='hierarchical')
            n_clusters = results['optimal_clusters']
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.best_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        self.cluster_labels = self.best_model.fit_predict(X_scaled)
        self.best_n_clusters = n_clusters
        
        # Calculate linkage matrix for dendrogram
        linkage_matrix = linkage(X_scaled, method=linkage_method)
        
        # Calculate metrics
        sil_score = silhouette_score(X_scaled, self.cluster_labels)
        ch_score = calinski_harabasz_score(X_scaled, self.cluster_labels)
        db_score = davies_bouldin_score(X_scaled, self.cluster_labels)
        
        results = {
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'linkage_matrix': linkage_matrix,
            'linkage_method': linkage_method,
            'labels': self.cluster_labels
        }
        
        return self.cluster_labels, results
    
    def profile_segments(self, df: pd.DataFrame, cluster_labels: Optional[np.ndarray] = None,
                        target_col: str = 'Churn') -> pd.DataFrame:
        """
        Create comprehensive segment profiles with descriptive statistics
        
        Args:
            df: Original dataframe with features and target
            cluster_labels: Cluster assignments (uses self.cluster_labels if None)
            target_col: Name of target column for churn analysis
            
        Returns:
            DataFrame with segment profiles
        """
        if cluster_labels is None:
            cluster_labels = self.cluster_labels
            
        if cluster_labels is None:
            raise ValueError("No cluster labels available. Run clustering first.")
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['Segment'] = cluster_labels
        
        # Initialize segment profiles
        profiles = []
        
        for segment_id in sorted(df_with_clusters['Segment'].unique()):
            segment_data = df_with_clusters[df_with_clusters['Segment'] == segment_id]
            
            # Basic segment statistics
            size = len(segment_data)
            churn_rate = segment_data[target_col].mean() if target_col in df.columns else 0.0
            
            # Feature statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            characteristics = {}
            
            # Numeric feature summaries
            for col in numeric_cols:
                if col != target_col and col in segment_data.columns:
                    characteristics[f'{col}_mean'] = segment_data[col].mean()
                    characteristics[f'{col}_median'] = segment_data[col].median()
                    characteristics[f'{col}_std'] = segment_data[col].std()
            
            # Categorical feature summaries
            for col in categorical_cols:
                if col != target_col and col in segment_data.columns:
                    mode_value = segment_data[col].mode()
                    if len(mode_value) > 0:
                        characteristics[f'{col}_mode'] = mode_value.iloc[0]
                        characteristics[f'{col}_mode_freq'] = (segment_data[col] == mode_value.iloc[0]).mean()
            
            # Calculate feature importance (difference from overall mean)
            top_features = []
            overall_means = df[numeric_cols].mean()
            
            for col in numeric_cols:
                if col != target_col and col in segment_data.columns:
                    segment_mean = segment_data[col].mean()
                    overall_mean = overall_means[col]
                    if overall_mean != 0:
                        relative_diff = abs(segment_mean - overall_mean) / abs(overall_mean)
                        top_features.append((col, relative_diff))
            
            # Sort by importance and take top 5
            top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:5]
            
            # Create segment profile
            profile = SegmentProfile(
                segment_id=segment_id,
                size=size,
                churn_rate=churn_rate,
                avg_clv=0.0,  # Will be updated when CLV is calculated
                characteristics=characteristics,
                top_features=top_features
            )
            
            profiles.append(profile)
        
        # Convert to DataFrame for easy analysis
        profile_data = []
        for profile in profiles:
            row = {
                'Segment': profile.segment_id,
                'Size': profile.size,
                'Size_Pct': profile.size / len(df) * 100,
                'Churn_Rate': profile.churn_rate,
                'Avg_CLV': profile.avg_clv
            }
            
            # Add top characteristics
            for i, (feature, importance) in enumerate(profile.top_features):
                row[f'Top_Feature_{i+1}'] = feature
                row[f'Top_Feature_{i+1}_Importance'] = importance
            
            profile_data.append(row)
        
        self.segment_profiles = profiles
        return pd.DataFrame(profile_data)
    
    def evaluate_clustering_quality(self, X: pd.DataFrame, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics
        
        Args:
            X: Feature matrix used for clustering
            labels: Cluster labels (uses self.cluster_labels if None)
            
        Returns:
            Dictionary with clustering quality metrics
        """
        if labels is None:
            labels = self.cluster_labels
            
        if labels is None:
            raise ValueError("No cluster labels available. Run clustering first.")
        
        # Standardize features (use fitted scaler)
        X_scaled = self.scaler.transform(X)
        
        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette_score(X_scaled, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin_score': davies_bouldin_score(X_scaled, labels),
            'n_clusters': len(np.unique(labels)),
            'n_samples': len(X)
        }
        
        # Add inertia if KMeans was used
        if hasattr(self.best_model, 'inertia_'):
            metrics['inertia'] = self.best_model.inertia_
        
        return metrics
    
    def plot_clustering_results(self, X: pd.DataFrame, labels: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Create visualization plots for clustering results
        
        Args:
            X: Feature matrix used for clustering
            labels: Cluster labels (uses self.cluster_labels if None)
            save_path: Path to save the plot
        """
        if labels is None:
            labels = self.cluster_labels
            
        if labels is None:
            raise ValueError("No cluster labels available. Run clustering first.")
        
        # Use PCA for 2D visualization if more than 2 features
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            X_pca = pca.fit_transform(self.scaler.transform(X))
            x_col, y_col = 'PC1', 'PC2'
            plot_data = pd.DataFrame(X_pca, columns=[x_col, y_col])
        else:
            plot_data = X.copy()
            x_col, y_col = X.columns[0], X.columns[1] if len(X.columns) > 1 else X.columns[0]
        
        plot_data['Cluster'] = labels
        
        # Create subplot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        scatter = axes[0].scatter(plot_data[x_col], plot_data[y_col], 
                                c=plot_data['Cluster'], cmap='viridis', alpha=0.7)
        axes[0].set_xlabel(x_col)
        axes[0].set_ylabel(y_col)
        axes[0].set_title('Customer Segments')
        plt.colorbar(scatter, ax=axes[0])
        
        # Cluster size distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        axes[1].bar(cluster_counts.index, cluster_counts.values)
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Number of Customers')
        axes[1].set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_dendrogram(self, X: pd.DataFrame, max_display: int = 30, 
                       save_path: Optional[str] = None) -> None:
        """
        Plot dendrogram for hierarchical clustering
        
        Args:
            X: Feature matrix
            max_display: Maximum number of clusters to display
            save_path: Path to save the plot
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate linkage matrix
        linkage_matrix = linkage(X_scaled, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, truncate_mode='lastp', p=max_display)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class CLVCalculator:
    """
    Customer Lifetime Value calculation using BG/NBD and Gamma-Gamma models
    """
    
    def __init__(self):
        """Initialize CLVCalculator"""
        self.bgnbd_model = None
        self.ggf_model = None
        self.is_fitted = False
        
    def prepare_rfm_data(self, transaction_data: pd.DataFrame, 
                        customer_col: str = 'customer_id',
                        date_col: str = 'order_date', 
                        value_col: str = 'order_value') -> pd.DataFrame:
        """
        Prepare RFM (Recency, Frequency, Monetary) data from transaction history
        
        Args:
            transaction_data: DataFrame with transaction history
            customer_col: Column name for customer identifier
            date_col: Column name for transaction date
            value_col: Column name for transaction value
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        if not LIFETIMES_AVAILABLE:
            # Fallback implementation without lifetimes library
            return self._prepare_rfm_fallback(transaction_data, customer_col, date_col, value_col)
        
        # Convert date column to datetime
        transaction_data = transaction_data.copy()
        transaction_data[date_col] = pd.to_datetime(transaction_data[date_col])
        
        # Use lifetimes utility function
        rfm_data = summary_data_from_transaction_data(
            transaction_data,
            customer_id_col=customer_col,
            datetime_col=date_col,
            monetary_value_col=value_col,
            observation_period_end=transaction_data[date_col].max()
        )
        
        return rfm_data
    
    def _prepare_rfm_fallback(self, transaction_data: pd.DataFrame,
                            customer_col: str, date_col: str, value_col: str) -> pd.DataFrame:
        """
        Fallback RFM calculation without lifetimes library
        """
        transaction_data = transaction_data.copy()
        transaction_data[date_col] = pd.to_datetime(transaction_data[date_col])
        
        # Calculate RFM metrics manually
        current_date = transaction_data[date_col].max()
        
        rfm_data = transaction_data.groupby(customer_col).agg({
            date_col: ['count', 'max'],
            value_col: ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        rfm_data.columns = [customer_col, 'frequency', 'last_purchase_date', 'monetary_value', 'avg_order_value']
        
        # Calculate recency (days since last purchase)
        rfm_data['recency'] = (current_date - rfm_data['last_purchase_date']).dt.days
        
        # Adjust frequency (number of repeat purchases)
        rfm_data['frequency'] = rfm_data['frequency'] - 1
        rfm_data['frequency'] = rfm_data['frequency'].clip(lower=0)
        
        # Calculate T (customer age in days from first purchase)
        first_purchase = transaction_data.groupby(customer_col)[date_col].min()
        rfm_data['T'] = (current_date - first_purchase).dt.days
        
        return rfm_data[[customer_col, 'frequency', 'recency', 'T', 'monetary_value']]
    
    def fit_bgnbd_model(self, rfm_data: pd.DataFrame) -> Optional[Any]:
        """
        Fit BG/NBD model for purchase prediction
        
        Args:
            rfm_data: DataFrame with RFM metrics (frequency, recency, T)
            
        Returns:
            Fitted BG/NBD model or None if lifetimes not available
        """
        if not LIFETIMES_AVAILABLE:
            print("Warning: BG/NBD model requires lifetimes library. Using fallback approach.")
            return None
        
        # Initialize and fit BG/NBD model
        self.bgnbd_model = BetaGeoFitter(penalizer_coef=0.01)
        
        # Fit model
        self.bgnbd_model.fit(
            frequency=rfm_data['frequency'],
            recency=rfm_data['recency'], 
            T=rfm_data['T']
        )
        
        return self.bgnbd_model
    
    def fit_gamma_gamma_model(self, rfm_data: pd.DataFrame) -> Optional[Any]:
        """
        Fit Gamma-Gamma model for monetary value estimation
        
        Args:
            rfm_data: DataFrame with RFM metrics including monetary_value
            
        Returns:
            Fitted Gamma-Gamma model or None if lifetimes not available
        """
        if not LIFETIMES_AVAILABLE:
            print("Warning: Gamma-Gamma model requires lifetimes library. Using fallback approach.")
            return None
        
        # Filter customers with at least one repeat purchase
        repeat_customers = rfm_data[rfm_data['frequency'] > 0]
        
        if len(repeat_customers) == 0:
            print("Warning: No repeat customers found for Gamma-Gamma model")
            return None
        
        # Initialize and fit Gamma-Gamma model
        self.ggf_model = GammaGammaFitter(penalizer_coef=0.01)
        
        # Fit model
        self.ggf_model.fit(
            frequency=repeat_customers['frequency'],
            monetary_value=repeat_customers['monetary_value']
        )
        
        return self.ggf_model
    
    def predict_clv(self, customer_data: pd.DataFrame, time_horizon: int = 365,
                   discount_rate: float = 0.01) -> pd.DataFrame:
        """
        Predict Customer Lifetime Value with confidence intervals
        
        Args:
            customer_data: DataFrame with customer RFM data
            time_horizon: Prediction horizon in days
            discount_rate: Monthly discount rate for NPV calculation
            
        Returns:
            DataFrame with CLV predictions and confidence intervals
        """
        if not LIFETIMES_AVAILABLE or self.bgnbd_model is None:
            return self._predict_clv_fallback(customer_data, time_horizon)
        
        results = []
        
        for _, customer in customer_data.iterrows():
            customer_id = customer.get('customer_id', customer.name)
            frequency = customer['frequency']
            recency = customer['recency']
            T = customer['T']
            monetary_value = customer['monetary_value']
            
            # Predict expected purchases
            expected_purchases = self.bgnbd_model.conditional_expected_number_of_purchases_up_to_time(
                time_horizon, frequency, recency, T
            )
            
            # Predict average transaction value (if Gamma-Gamma model is available)
            if self.ggf_model is not None and frequency > 0:
                expected_avg_value = self.ggf_model.conditional_expected_average_profit(
                    frequency, monetary_value
                )
            else:
                expected_avg_value = monetary_value
            
            # Calculate CLV
            predicted_clv = expected_purchases * expected_avg_value
            
            # Apply discount rate (simple approximation)
            if discount_rate > 0:
                periods = time_horizon / 30  # Convert to months
                discount_factor = (1 - (1 + discount_rate) ** -periods) / discount_rate
                predicted_clv *= discount_factor
            
            # Calculate confidence intervals (simplified approach)
            # In practice, you would use bootstrap or analytical methods
            std_error = predicted_clv * 0.2  # Assume 20% standard error
            confidence_interval = (
                max(0, predicted_clv - 1.96 * std_error),
                predicted_clv + 1.96 * std_error
            )
            
            result = CLVPrediction(
                customer_id=str(customer_id),
                predicted_clv=predicted_clv,
                confidence_interval=confidence_interval,
                frequency=frequency,
                recency=recency,
                monetary_value=monetary_value
            )
            
            results.append(result)
        
        # Convert to DataFrame
        clv_df = pd.DataFrame([
            {
                'customer_id': r.customer_id,
                'predicted_clv': r.predicted_clv,
                'clv_lower_ci': r.confidence_interval[0],
                'clv_upper_ci': r.confidence_interval[1],
                'frequency': r.frequency,
                'recency': r.recency,
                'monetary_value': r.monetary_value
            }
            for r in results
        ])
        
        self.is_fitted = True
        return clv_df
    
    def _predict_clv_fallback(self, customer_data: pd.DataFrame, time_horizon: int) -> pd.DataFrame:
        """
        Fallback CLV calculation without lifetimes library
        """
        results = []
        
        for _, customer in customer_data.iterrows():
            customer_id = customer.get('customer_id', customer.name)
            frequency = customer['frequency']
            recency = customer['recency']
            T = customer['T']
            monetary_value = customer['monetary_value']
            
            # Simple heuristic-based CLV calculation
            # Purchase rate = frequency / T (purchases per day)
            if T > 0:
                purchase_rate = frequency / T
            else:
                purchase_rate = 0
            
            # Adjust for recency (more recent customers more likely to continue)
            recency_factor = np.exp(-recency / 365)  # Decay over a year
            
            # Expected purchases in time horizon
            expected_purchases = purchase_rate * time_horizon * recency_factor
            
            # CLV calculation
            predicted_clv = expected_purchases * monetary_value
            
            # Simple confidence interval
            std_error = predicted_clv * 0.3  # Higher uncertainty without proper model
            confidence_interval = (
                max(0, predicted_clv - 1.96 * std_error),
                predicted_clv + 1.96 * std_error
            )
            
            result = CLVPrediction(
                customer_id=str(customer_id),
                predicted_clv=predicted_clv,
                confidence_interval=confidence_interval,
                frequency=frequency,
                recency=recency,
                monetary_value=monetary_value
            )
            
            results.append(result)
        
        # Convert to DataFrame
        clv_df = pd.DataFrame([
            {
                'customer_id': r.customer_id,
                'predicted_clv': r.predicted_clv,
                'clv_lower_ci': r.confidence_interval[0],
                'clv_upper_ci': r.confidence_interval[1],
                'frequency': r.frequency,
                'recency': r.recency,
                'monetary_value': r.monetary_value
            }
            for r in results
        ])
        
        return clv_df
    
    def evaluate_model_fit(self, rfm_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the fit quality of CLV models
        
        Args:
            rfm_data: DataFrame with RFM data used for fitting
            
        Returns:
            Dictionary with model evaluation metrics
        """
        if not LIFETIMES_AVAILABLE or self.bgnbd_model is None:
            return {"error": "Models not fitted or lifetimes library not available"}
        
        evaluation = {}
        
        # BG/NBD model evaluation
        if self.bgnbd_model is not None:
            evaluation['bgnbd_log_likelihood'] = self.bgnbd_model.log_likelihood_
            evaluation['bgnbd_aic'] = self.bgnbd_model.AIC_
            evaluation['bgnbd_params'] = dict(self.bgnbd_model.params_)
        
        # Gamma-Gamma model evaluation
        if self.ggf_model is not None:
            evaluation['ggf_log_likelihood'] = self.ggf_model.log_likelihood_
            evaluation['ggf_aic'] = self.ggf_model.AIC_
            evaluation['ggf_params'] = dict(self.ggf_model.params_)
        
        return evaluation


class PriorityRanker:
    """
    Customer prioritization for retention campaigns combining churn risk and CLV
    """
    
    def __init__(self):
        """Initialize PriorityRanker"""
        self.priority_scores = None
        self.ranking_results = None
    
    def calculate_expected_value(self, churn_probabilities: np.ndarray, 
                               clv_estimates: np.ndarray,
                               retention_cost: float = 50.0,
                               retention_effectiveness: float = 0.3) -> np.ndarray:
        """
        Calculate expected value of retention interventions
        
        Args:
            churn_probabilities: Array of churn probabilities
            clv_estimates: Array of CLV estimates
            retention_cost: Cost of retention intervention per customer
            retention_effectiveness: Probability that intervention prevents churn
            
        Returns:
            Array of expected values
        """
        # Expected value = (churn_prob * retention_effectiveness * CLV) - retention_cost
        expected_savings = churn_probabilities * retention_effectiveness * clv_estimates
        expected_value = expected_savings - retention_cost
        
        return expected_value
    
    def rank_customers_for_retention(self, customer_data: pd.DataFrame,
                                   churn_col: str = 'churn_probability',
                                   clv_col: str = 'predicted_clv',
                                   customer_id_col: str = 'customer_id',
                                   business_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Rank customers for retention campaigns based on expected value
        
        Args:
            customer_data: DataFrame with customer data, churn probabilities, and CLV
            churn_col: Column name for churn probabilities
            clv_col: Column name for CLV estimates
            customer_id_col: Column name for customer IDs
            business_params: Dictionary with business parameters
            
        Returns:
            DataFrame with customer rankings and priority scores
        """
        if business_params is None:
            business_params = {
                'retention_cost': 50.0,
                'retention_effectiveness': 0.3,
                'min_clv_threshold': 100.0,
                'min_churn_threshold': 0.1
            }
        
        # Calculate expected values
        expected_values = self.calculate_expected_value(
            customer_data[churn_col].values,
            customer_data[clv_col].values,
            business_params['retention_cost'],
            business_params['retention_effectiveness']
        )
        
        # Create ranking dataframe
        ranking_df = customer_data[[customer_id_col, churn_col, clv_col]].copy()
        ranking_df['expected_value'] = expected_values
        
        # Calculate priority score (normalized expected value)
        max_ev = ranking_df['expected_value'].max()
        min_ev = ranking_df['expected_value'].min()
        
        if max_ev > min_ev:
            ranking_df['priority_score'] = (ranking_df['expected_value'] - min_ev) / (max_ev - min_ev)
        else:
            ranking_df['priority_score'] = 0.5
        
        # Apply business rules
        # Filter by minimum CLV threshold
        ranking_df['meets_clv_threshold'] = ranking_df[clv_col] >= business_params['min_clv_threshold']
        
        # Filter by minimum churn threshold
        ranking_df['meets_churn_threshold'] = ranking_df[churn_col] >= business_params['min_churn_threshold']
        
        # Overall eligibility
        ranking_df['eligible_for_intervention'] = (
            ranking_df['meets_clv_threshold'] & 
            ranking_df['meets_churn_threshold'] &
            (ranking_df['expected_value'] > 0)
        )
        
        # Assign priority tiers
        ranking_df['priority_tier'] = 'Low'
        
        # High priority: top 20% of eligible customers
        eligible_customers = ranking_df[ranking_df['eligible_for_intervention']]
        if len(eligible_customers) > 0:
            high_priority_threshold = eligible_customers['priority_score'].quantile(0.8)
            medium_priority_threshold = eligible_customers['priority_score'].quantile(0.5)
            
            ranking_df.loc[
                (ranking_df['eligible_for_intervention']) & 
                (ranking_df['priority_score'] >= high_priority_threshold), 
                'priority_tier'
            ] = 'High'
            
            ranking_df.loc[
                (ranking_df['eligible_for_intervention']) & 
                (ranking_df['priority_score'] >= medium_priority_threshold) &
                (ranking_df['priority_score'] < high_priority_threshold), 
                'priority_tier'
            ] = 'Medium'
        
        # Sort by priority score (descending)
        ranking_df = ranking_df.sort_values('priority_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        self.ranking_results = ranking_df
        return ranking_df
    
    def perform_cost_benefit_analysis(self, ranking_results: Optional[pd.DataFrame] = None,
                                    campaign_budget: float = 10000.0,
                                    business_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform cost-benefit analysis for different intervention strategies
        
        Args:
            ranking_results: DataFrame with customer rankings (uses self.ranking_results if None)
            campaign_budget: Total budget available for retention campaigns
            business_params: Dictionary with business parameters
            
        Returns:
            Dictionary with cost-benefit analysis results
        """
        if ranking_results is None:
            ranking_results = self.ranking_results
            
        if ranking_results is None:
            raise ValueError("No ranking results available. Run rank_customers_for_retention first.")
        
        if business_params is None:
            business_params = {
                'retention_cost': 50.0,
                'retention_effectiveness': 0.3
            }
        
        # Filter eligible customers
        eligible_customers = ranking_results[ranking_results['eligible_for_intervention']].copy()
        
        if len(eligible_customers) == 0:
            return {
                'total_customers_targeted': 0,
                'total_cost': 0,
                'expected_savings': 0,
                'expected_roi': 0,
                'break_even_customers': 0
            }
        
        # Calculate how many customers can be targeted with budget
        retention_cost = business_params['retention_cost']
        max_customers_in_budget = int(campaign_budget / retention_cost)
        
        # Select top customers within budget
        customers_to_target = min(len(eligible_customers), max_customers_in_budget)
        targeted_customers = eligible_customers.head(customers_to_target)
        
        # Calculate costs and benefits
        total_cost = customers_to_target * retention_cost
        expected_savings = targeted_customers['expected_value'].sum() + total_cost  # Add back cost since expected_value is net
        expected_roi = (expected_savings - total_cost) / total_cost if total_cost > 0 else 0
        
        # Find break-even point
        cumulative_expected_value = eligible_customers['expected_value'].cumsum()
        break_even_idx = (cumulative_expected_value > 0).idxmax() if (cumulative_expected_value > 0).any() else 0
        break_even_customers = break_even_idx + 1 if break_even_idx > 0 else 0
        
        # Analysis by priority tier
        tier_analysis = {}
        for tier in ['High', 'Medium', 'Low']:
            tier_customers = targeted_customers[targeted_customers['priority_tier'] == tier]
            if len(tier_customers) > 0:
                tier_analysis[tier] = {
                    'count': len(tier_customers),
                    'total_expected_value': tier_customers['expected_value'].sum(),
                    'avg_churn_probability': tier_customers['churn_probability'].mean(),
                    'avg_clv': tier_customers['predicted_clv'].mean()
                }
        
        return {
            'total_customers_targeted': customers_to_target,
            'total_cost': total_cost,
            'expected_savings': expected_savings,
            'expected_net_benefit': expected_savings - total_cost,
            'expected_roi': expected_roi,
            'break_even_customers': break_even_customers,
            'budget_utilization': total_cost / campaign_budget,
            'tier_analysis': tier_analysis,
            'avg_churn_probability': targeted_customers['churn_probability'].mean(),
            'avg_clv': targeted_customers['predicted_clv'].mean(),
            'avg_priority_score': targeted_customers['priority_score'].mean()
        }
    
    def generate_campaign_recommendations(self, ranking_results: Optional[pd.DataFrame] = None,
                                        max_recommendations: int = 100) -> List[Dict[str, Any]]:
        """
        Generate specific campaign recommendations for top customers
        
        Args:
            ranking_results: DataFrame with customer rankings
            max_recommendations: Maximum number of recommendations to generate
            
        Returns:
            List of recommendation dictionaries
        """
        if ranking_results is None:
            ranking_results = self.ranking_results
            
        if ranking_results is None:
            raise ValueError("No ranking results available. Run rank_customers_for_retention first.")
        
        # Get top eligible customers
        eligible_customers = ranking_results[ranking_results['eligible_for_intervention']]
        top_customers = eligible_customers.head(max_recommendations)
        
        recommendations = []
        
        for _, customer in top_customers.iterrows():
            # Determine recommendation type based on characteristics
            churn_prob = customer['churn_probability']
            clv = customer['predicted_clv']
            priority_tier = customer['priority_tier']
            
            # Recommendation logic
            if churn_prob >= 0.7 and clv >= 1000:
                recommendation_type = "Immediate Personal Outreach"
                urgency = "Critical"
                suggested_actions = [
                    "Schedule immediate call with account manager",
                    "Offer personalized retention package",
                    "Investigate specific pain points"
                ]
            elif churn_prob >= 0.5 and clv >= 500:
                recommendation_type = "Targeted Retention Campaign"
                urgency = "High"
                suggested_actions = [
                    "Send personalized email with special offer",
                    "Provide loyalty program benefits",
                    "Offer product usage consultation"
                ]
            elif churn_prob >= 0.3:
                recommendation_type = "Proactive Engagement"
                urgency = "Medium"
                suggested_actions = [
                    "Include in next newsletter campaign",
                    "Offer educational content",
                    "Monitor usage patterns closely"
                ]
            else:
                recommendation_type = "Standard Monitoring"
                urgency = "Low"
                suggested_actions = [
                    "Include in regular communication",
                    "Track engagement metrics"
                ]
            
            recommendation = {
                'customer_id': customer['customer_id'],
                'rank': customer['rank'],
                'priority_tier': priority_tier,
                'churn_probability': churn_prob,
                'clv_estimate': clv,
                'expected_value': customer['expected_value'],
                'recommendation_type': recommendation_type,
                'urgency': urgency,
                'suggested_actions': suggested_actions,
                'estimated_intervention_cost': 50.0 if urgency in ['Critical', 'High'] else 25.0
            }
            
            recommendations.append(recommendation)
        
        return recommendations