"""
Business evaluation and metrics module for Customer Churn ML Pipeline
Implements business-focused performance assessment, fairness analysis, and reporting
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

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class LiftTableRow:
    """Single row in a lift table"""
    decile: int
    threshold_min: float
    threshold_max: float
    n_customers: int
    n_churners: int
    churn_rate: float
    lift: float
    cumulative_lift: float
    capture_rate: float
    cumulative_capture_rate: float


class BusinessMetrics:
    """
    Calculates business-focused performance metrics including lift tables,
    gains charts, expected savings, and ROI computations
    """
    
    def __init__(self, business_config: Optional[Dict[str, float]] = None):
        """
        Initialize BusinessMetrics
        
        Args:
            business_config: Dictionary with business parameters
                - retention_value: Value of retaining a customer
                - contact_cost: Cost of contacting a customer
                - churn_cost: Cost of losing a customer
        """
        self.business_config = business_config or config.get_business_config()
        
        self.retention_value = self.business_config.get('retention_value', 1000.0)
        self.contact_cost = self.business_config.get('contact_cost', 50.0)
        self.churn_cost = self.business_config.get('churn_cost', 500.0)
        
        logger.info(f"BusinessMetrics initialized with retention_value={self.retention_value}, "
                   f"contact_cost={self.contact_cost}, churn_cost={self.churn_cost}")
    
    def calculate_lift_table(self, y_true: np.ndarray, y_prob: np.ndarray, 
                           n_deciles: int = 10) -> pd.DataFrame:
        """
        Calculate lift table showing model performance across probability deciles
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_deciles: Number of deciles to create
            
        Returns:
            DataFrame with lift table metrics
        """
        logger.info(f"Calculating lift table with {n_deciles} deciles")
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'y_true': y_true,
            'y_prob': y_prob
        })
        
        # Sort by probability (descending)
        df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
        
        # Calculate decile boundaries
        decile_size = len(df) // n_deciles
        
        # Calculate overall churn rate for lift calculation
        overall_churn_rate = y_true.mean()
        
        lift_rows = []
        cumulative_churners = 0
        total_churners = y_true.sum()
        
        for decile in range(1, n_deciles + 1):
            # Define decile boundaries
            start_idx = (decile - 1) * decile_size
            if decile == n_deciles:
                end_idx = len(df)  # Include remaining customers in last decile
            else:
                end_idx = decile * decile_size
            
            # Extract decile data
            decile_data = df.iloc[start_idx:end_idx]
            
            # Calculate metrics for this decile
            n_customers = len(decile_data)
            n_churners = decile_data['y_true'].sum()
            churn_rate = n_churners / n_customers if n_customers > 0 else 0
            
            # Calculate lift (ratio of decile churn rate to overall churn rate)
            lift = churn_rate / overall_churn_rate if overall_churn_rate > 0 else 0
            
            # Calculate capture rate (percentage of total churners in this decile)
            capture_rate = n_churners / total_churners if total_churners > 0 else 0
            
            # Calculate cumulative metrics
            cumulative_churners += n_churners
            cumulative_customers = end_idx
            cumulative_churn_rate = cumulative_churners / cumulative_customers if cumulative_customers > 0 else 0
            cumulative_lift = cumulative_churn_rate / overall_churn_rate if overall_churn_rate > 0 else 0
            cumulative_capture_rate = cumulative_churners / total_churners if total_churners > 0 else 0
            
            # Get probability thresholds for this decile
            threshold_max = decile_data['y_prob'].max()
            threshold_min = decile_data['y_prob'].min()
            
            lift_row = LiftTableRow(
                decile=decile,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                n_customers=n_customers,
                n_churners=n_churners,
                churn_rate=churn_rate,
                lift=lift,
                cumulative_lift=cumulative_lift,
                capture_rate=capture_rate,
                cumulative_capture_rate=cumulative_capture_rate
            )
            
            lift_rows.append(lift_row)
        
        # Convert to DataFrame
        lift_df = pd.DataFrame([
            {
                'decile': row.decile,
                'threshold_min': row.threshold_min,
                'threshold_max': row.threshold_max,
                'n_customers': row.n_customers,
                'n_churners': row.n_churners,
                'churn_rate': row.churn_rate,
                'lift': row.lift,
                'cumulative_lift': row.cumulative_lift,
                'capture_rate': row.capture_rate,
                'cumulative_capture_rate': row.cumulative_capture_rate
            }
            for row in lift_rows
        ])
        
        logger.info(f"Lift table calculated with {len(lift_df)} deciles")
        
        return lift_df
    
    def calculate_expected_savings(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 threshold: float, 
                                 business_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate expected savings and ROI based on business parameters
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            threshold: Probability threshold for predictions
            business_params: Override business parameters
            
        Returns:
            Dictionary with savings and ROI metrics
        """
        logger.info(f"Calculating expected savings with threshold={threshold}")
        
        # Use provided business parameters or defaults
        if business_params:
            retention_value = business_params.get('retention_value', self.retention_value)
            contact_cost = business_params.get('contact_cost', self.contact_cost)
            churn_cost = business_params.get('churn_cost', self.churn_cost)
        else:
            retention_value = self.retention_value
            contact_cost = self.contact_cost
            churn_cost = self.churn_cost
        
        # Make predictions based on threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        
        total_customers = len(y_true)
        total_churners = y_true.sum()
        
        # Calculate business metrics
        
        # Revenue from successfully retaining customers who would churn
        retention_revenue = tp * retention_value
        
        # Cost of contacting customers (both TP and FP)
        total_contact_costs = (tp + fp) * contact_cost
        
        # Cost of customers who churn despite not being contacted (FN)
        missed_churn_costs = fn * churn_cost
        
        # Total profit with model
        total_profit_with_model = retention_revenue - total_contact_costs - missed_churn_costs
        
        # Baseline scenario (no model - all customers churn)
        baseline_churn_cost = total_churners * churn_cost
        
        # Net savings compared to baseline
        net_savings = total_profit_with_model + baseline_churn_cost
        
        # ROI calculation
        total_investment = total_contact_costs
        roi = (net_savings / total_investment * 100) if total_investment > 0 else 0
        
        # Calculate rates and percentages
        contact_rate = (tp + fp) / total_customers
        retention_success_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        churn_prevention_rate = tp / total_churners if total_churners > 0 else 0
        
        # Calculate cost per retained customer
        cost_per_retained = total_contact_costs / tp if tp > 0 else float('inf')
        
        # Calculate value metrics
        value_per_customer = net_savings / total_customers
        
        savings_metrics = {
            # Core financial metrics
            'total_profit': total_profit_with_model,
            'net_savings': net_savings,
            'roi_percentage': roi,
            'retention_revenue': retention_revenue,
            'total_contact_costs': total_contact_costs,
            'missed_churn_costs': missed_churn_costs,
            'baseline_churn_cost': baseline_churn_cost,
            
            # Efficiency metrics
            'cost_per_retained_customer': cost_per_retained,
            'value_per_customer': value_per_customer,
            
            # Rate metrics
            'contact_rate': contact_rate,
            'retention_success_rate': retention_success_rate,
            'churn_prevention_rate': churn_prevention_rate,
            
            # Volume metrics
            'customers_contacted': tp + fp,
            'customers_retained': tp,
            'customers_lost': fn,
            'total_customers': total_customers,
            'total_churners': total_churners,
            
            # Confusion matrix
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            
            # Business parameters used
            'retention_value_used': retention_value,
            'contact_cost_used': contact_cost,
            'churn_cost_used': churn_cost,
            'threshold_used': threshold
        }
        
        logger.info(f"Expected savings calculated: net_savings=${net_savings:,.2f}, ROI={roi:.1f}%")
        
        return savings_metrics
    
    def generate_gains_chart(self, y_true: np.ndarray, y_prob: np.ndarray,
                           save_path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
        """
        Generate gains chart data and visualization
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            save_path: Path to save the chart
            
        Returns:
            Dictionary with gains chart data
        """
        logger.info("Generating gains chart")
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate cumulative gains
        total_positives = y_true.sum()
        cumulative_positives = np.cumsum(y_true_sorted)
        
        # Calculate percentage of population and percentage of positives captured
        population_percentages = np.arange(1, len(y_true) + 1) / len(y_true) * 100
        gains_percentages = cumulative_positives / total_positives * 100
        
        # Create gains chart
        plt.figure(figsize=(10, 8))
        
        # Plot gains curve
        plt.plot(population_percentages, gains_percentages, 'b-', linewidth=2, label='Model')
        
        # Plot random baseline (diagonal line)
        plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Random')
        
        # Add perfect model line (if we could perfectly rank all positives first)
        perfect_x = [0, total_positives / len(y_true) * 100, 100]
        perfect_y = [0, 100, 100]
        plt.plot(perfect_x, perfect_y, 'g--', linewidth=1, label='Perfect Model')
        
        plt.xlabel('Percentage of Population Contacted')
        plt.ylabel('Percentage of Churners Captured')
        plt.title('Gains Chart - Model Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add annotations for key points
        for pct in [10, 20, 30, 50]:
            idx = int(len(y_true) * pct / 100) - 1
            if idx < len(gains_percentages):
                plt.annotate(f'{gains_percentages[idx]:.1f}%', 
                           xy=(pct, gains_percentages[idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gains chart saved to {save_path}")
        
        plt.show()
        
        # Return gains data
        gains_data = {
            'population_percentages': population_percentages,
            'gains_percentages': gains_percentages,
            'total_positives': total_positives,
            'cumulative_positives': cumulative_positives
        }
        
        return gains_data
    
    def calculate_decile_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                                n_deciles: int = 10) -> Dict[str, Any]:
        """
        Perform comprehensive decile analysis with business metrics
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_deciles: Number of deciles to analyze
            
        Returns:
            Dictionary with decile analysis results
        """
        logger.info(f"Performing decile analysis with {n_deciles} deciles")
        
        # Get lift table
        lift_table = self.calculate_lift_table(y_true, y_prob, n_deciles)
        
        # Calculate business metrics for each decile
        business_metrics = []
        
        for _, row in lift_table.iterrows():
            decile = row['decile']
            threshold = row['threshold_min']
            
            # Calculate expected savings if we contact only this decile and above
            decile_mask = y_prob >= threshold
            
            if decile_mask.sum() > 0:
                savings = self.calculate_expected_savings(
                    y_true[decile_mask], 
                    y_prob[decile_mask], 
                    threshold=threshold
                )
                
                business_metrics.append({
                    'decile': decile,
                    'threshold': threshold,
                    'customers_contacted': savings['customers_contacted'],
                    'expected_profit': savings['total_profit'],
                    'roi_percentage': savings['roi_percentage'],
                    'cost_per_retained': savings['cost_per_retained_customer']
                })
        
        business_df = pd.DataFrame(business_metrics)
        
        # Calculate summary statistics
        summary_stats = {
            'best_decile_lift': lift_table['lift'].max(),
            'worst_decile_lift': lift_table['lift'].min(),
            'top_3_deciles_capture': lift_table.head(3)['capture_rate'].sum(),
            'top_5_deciles_capture': lift_table.head(5)['capture_rate'].sum(),
            'concentration_ratio': lift_table.head(2)['capture_rate'].sum(),  # Top 20% capture rate
        }
        
        # Find optimal decile for business metrics
        if len(business_df) > 0:
            optimal_profit_decile = business_df.loc[business_df['expected_profit'].idxmax(), 'decile']
            optimal_roi_decile = business_df.loc[business_df['roi_percentage'].idxmax(), 'decile']
            
            summary_stats.update({
                'optimal_profit_decile': optimal_profit_decile,
                'optimal_roi_decile': optimal_roi_decile,
                'max_expected_profit': business_df['expected_profit'].max(),
                'max_roi_percentage': business_df['roi_percentage'].max()
            })
        
        decile_analysis = {
            'lift_table': lift_table,
            'business_metrics': business_df,
            'summary_stats': summary_stats,
            'n_deciles': n_deciles,
            'total_customers': len(y_true),
            'total_churners': y_true.sum(),
            'overall_churn_rate': y_true.mean()
        }
        
        logger.info("Decile analysis completed")
        
        return decile_analysis
    
    def calculate_capture_rate_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                                     contact_percentages: List[float] = None) -> pd.DataFrame:
        """
        Calculate capture rates for different contact percentages
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            contact_percentages: List of population percentages to analyze
            
        Returns:
            DataFrame with capture rate metrics
        """
        if contact_percentages is None:
            contact_percentages = [5, 10, 15, 20, 25, 30, 40, 50]
        
        logger.info(f"Calculating capture rates for {len(contact_percentages)} contact percentages")
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        total_churners = y_true.sum()
        total_customers = len(y_true)
        
        capture_metrics = []
        
        for contact_pct in contact_percentages:
            # Calculate number of customers to contact
            n_contact = int(total_customers * contact_pct / 100)
            
            if n_contact > 0:
                # Get churners captured in top n_contact customers
                churners_captured = y_true_sorted[:n_contact].sum()
                capture_rate = churners_captured / total_churners if total_churners > 0 else 0
                
                # Calculate threshold for this contact percentage
                threshold = y_prob[sorted_indices[n_contact-1]] if n_contact <= len(y_prob) else y_prob.min()
                
                # Calculate business metrics
                savings = self.calculate_expected_savings(y_true, y_prob, threshold)
                
                capture_metrics.append({
                    'contact_percentage': contact_pct,
                    'customers_contacted': n_contact,
                    'churners_captured': churners_captured,
                    'capture_rate': capture_rate,
                    'threshold': threshold,
                    'expected_profit': savings['total_profit'],
                    'roi_percentage': savings['roi_percentage'],
                    'contact_cost': savings['total_contact_costs'],
                    'retention_revenue': savings['retention_revenue']
                })
        
        capture_df = pd.DataFrame(capture_metrics)
        
        logger.info("Capture rate metrics calculated")
        
        return capture_df
    
    def plot_lift_chart(self, lift_table: pd.DataFrame, 
                       save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot lift chart from lift table
        
        Args:
            lift_table: DataFrame from calculate_lift_table
            save_path: Path to save the chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Lift by decile
        ax1.bar(lift_table['decile'], lift_table['lift'], alpha=0.7, color='skyblue')
        ax1.axhline(y=1, color='red', linestyle='--', label='Baseline (Lift = 1)')
        ax1.set_xlabel('Decile')
        ax1.set_ylabel('Lift')
        ax1.set_title('Lift by Decile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(lift_table['lift']):
            ax1.text(i + 1, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
        
        # Plot 2: Cumulative capture rate
        ax2.plot(lift_table['decile'], lift_table['cumulative_capture_rate'] * 100, 
                marker='o', linewidth=2, markersize=6, color='green')
        ax2.set_xlabel('Decile')
        ax2.set_ylabel('Cumulative Capture Rate (%)')
        ax2.set_title('Cumulative Capture Rate by Decile')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(lift_table['cumulative_capture_rate'] * 100):
            ax2.text(i + 1, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Lift chart saved to {save_path}")
        
        plt.show()
    
    def calculate_model_value_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    threshold: float) -> Dict[str, float]:
        """
        Calculate comprehensive model value metrics
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            threshold: Probability threshold
            
        Returns:
            Dictionary with value metrics
        """
        logger.info("Calculating comprehensive model value metrics")
        
        # Get expected savings
        savings = self.calculate_expected_savings(y_true, y_prob, threshold)
        
        # Calculate additional value metrics
        total_customers = len(y_true)
        
        # Customer Lifetime Value impact
        clv_impact = savings['customers_retained'] * self.retention_value
        
        # Efficiency metrics
        precision = savings['true_positives'] / (savings['true_positives'] + savings['false_positives']) if (savings['true_positives'] + savings['false_positives']) > 0 else 0
        recall = savings['true_positives'] / (savings['true_positives'] + savings['false_negatives']) if (savings['true_positives'] + savings['false_negatives']) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Business efficiency
        revenue_per_contact = savings['retention_revenue'] / savings['customers_contacted'] if savings['customers_contacted'] > 0 else 0
        profit_margin = savings['total_profit'] / savings['retention_revenue'] if savings['retention_revenue'] > 0 else 0
        
        value_metrics = {
            **savings,  # Include all savings metrics
            'clv_impact': clv_impact,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'revenue_per_contact': revenue_per_contact,
            'profit_margin': profit_margin,
            'break_even_contact_rate': self.contact_cost / self.retention_value if self.retention_value > 0 else 0
        }
        
        return value_metrics

class SegmentAnalyzer:
    """
    Performs fairness and subgroup analysis across demographics
    Implements calibration fairness checks and bias detection
    """
    
    def __init__(self, fairness_threshold: float = 0.1):
        """
        Initialize SegmentAnalyzer
        
        Args:
            fairness_threshold: Threshold for detecting significant fairness violations
        """
        self.fairness_threshold = fairness_threshold
        self.segment_results = {}
        
        logger.info(f"SegmentAnalyzer initialized with fairness_threshold={fairness_threshold}")
    
    def analyze_subgroup_performance(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   segments: pd.Series, 
                                   segment_name: str = 'segment') -> pd.DataFrame:
        """
        Analyze model performance across different demographic subgroups
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            segments: Series with segment labels for each sample
            segment_name: Name of the segmentation variable
            
        Returns:
            DataFrame with subgroup performance metrics
        """
        logger.info(f"Analyzing subgroup performance for {segment_name}")
        
        # Ensure segments is a pandas Series
        if not isinstance(segments, pd.Series):
            segments = pd.Series(segments)
        
        unique_segments = segments.unique()
        subgroup_results = []
        
        for segment in unique_segments:
            # Get data for this segment
            segment_mask = segments == segment
            segment_y_true = y_true[segment_mask]
            segment_y_prob = y_prob[segment_mask]
            
            if len(segment_y_true) == 0:
                logger.warning(f"No samples found for segment {segment}")
                continue
            
            # Calculate performance metrics
            try:
                # Basic metrics
                n_samples = len(segment_y_true)
                n_positives = segment_y_true.sum()
                base_rate = n_positives / n_samples if n_samples > 0 else 0
                
                # ROC-AUC (only if both classes present)
                if len(np.unique(segment_y_true)) > 1:
                    roc_auc = roc_auc_score(segment_y_true, segment_y_prob)
                    pr_auc = average_precision_score(segment_y_true, segment_y_prob)
                else:
                    roc_auc = np.nan
                    pr_auc = np.nan
                
                # Brier score and log loss
                brier_score = brier_score_loss(segment_y_true, segment_y_prob)
                logloss = log_loss(segment_y_true, segment_y_prob)
                
                # Calibration metrics
                if len(np.unique(segment_y_true)) > 1:
                    cal_slope, cal_intercept = self._calculate_calibration_slope(
                        segment_y_true, segment_y_prob
                    )
                else:
                    cal_slope, cal_intercept = np.nan, np.nan
                
                # Prediction statistics
                mean_pred_prob = segment_y_prob.mean()
                std_pred_prob = segment_y_prob.std()
                
                subgroup_results.append({
                    'segment': segment,
                    'segment_name': segment_name,
                    'n_samples': n_samples,
                    'n_positives': n_positives,
                    'base_rate': base_rate,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'brier_score': brier_score,
                    'log_loss': logloss,
                    'calibration_slope': cal_slope,
                    'calibration_intercept': cal_intercept,
                    'mean_predicted_prob': mean_pred_prob,
                    'std_predicted_prob': std_pred_prob
                })
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for segment {segment}: {str(e)}")
                continue
        
        subgroup_df = pd.DataFrame(subgroup_results)
        
        # Store results
        self.segment_results[segment_name] = subgroup_df
        
        logger.info(f"Subgroup analysis completed for {len(subgroup_df)} segments")
        
        return subgroup_df
    
    def check_calibration_fairness(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 segments: pd.Series, 
                                 segment_name: str = 'segment') -> Dict[str, Any]:
        """
        Check calibration fairness across demographic groups
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            segments: Series with segment labels
            segment_name: Name of the segmentation variable
            
        Returns:
            Dictionary with calibration fairness results
        """
        logger.info(f"Checking calibration fairness for {segment_name}")
        
        # Ensure segments is a pandas Series
        if not isinstance(segments, pd.Series):
            segments = pd.Series(segments)
        
        unique_segments = segments.unique()
        calibration_results = []
        
        for segment in unique_segments:
            segment_mask = segments == segment
            segment_y_true = y_true[segment_mask]
            segment_y_prob = y_prob[segment_mask]
            
            if len(segment_y_true) == 0 or len(np.unique(segment_y_true)) <= 1:
                continue
            
            # Calculate calibration curve
            try:
                fraction_pos, mean_pred = calibration_curve(
                    segment_y_true, segment_y_prob, n_bins=10
                )
                
                # Calculate Expected Calibration Error (ECE)
                ece = np.mean(np.abs(fraction_pos - mean_pred))
                
                # Calculate Maximum Calibration Error (MCE)
                mce = np.max(np.abs(fraction_pos - mean_pred))
                
                # Calculate calibration slope and intercept
                cal_slope, cal_intercept = self._calculate_calibration_slope(
                    segment_y_true, segment_y_prob
                )
                
                calibration_results.append({
                    'segment': segment,
                    'ece': ece,
                    'mce': mce,
                    'calibration_slope': cal_slope,
                    'calibration_intercept': cal_intercept,
                    'n_samples': len(segment_y_true),
                    'base_rate': segment_y_true.mean()
                })
                
            except Exception as e:
                logger.warning(f"Error calculating calibration for segment {segment}: {str(e)}")
                continue
        
        calibration_df = pd.DataFrame(calibration_results)
        
        # Detect calibration fairness violations
        fairness_violations = []
        
        if len(calibration_df) > 1:
            # Check ECE differences
            max_ece = calibration_df['ece'].max()
            min_ece = calibration_df['ece'].min()
            ece_gap = max_ece - min_ece
            
            if ece_gap > self.fairness_threshold:
                fairness_violations.append({
                    'type': 'ECE_gap',
                    'value': ece_gap,
                    'threshold': self.fairness_threshold,
                    'description': f'ECE gap of {ece_gap:.3f} exceeds threshold {self.fairness_threshold}'
                })
            
            # Check calibration slope differences
            slopes = calibration_df['calibration_slope'].dropna()
            if len(slopes) > 1:
                slope_std = slopes.std()
                if slope_std > self.fairness_threshold:
                    fairness_violations.append({
                        'type': 'calibration_slope_variance',
                        'value': slope_std,
                        'threshold': self.fairness_threshold,
                        'description': f'Calibration slope variance of {slope_std:.3f} exceeds threshold'
                    })
        
        fairness_results = {
            'segment_name': segment_name,
            'calibration_metrics': calibration_df,
            'fairness_violations': fairness_violations,
            'is_fair': len(fairness_violations) == 0,
            'n_segments': len(calibration_df)
        }
        
        logger.info(f"Calibration fairness check completed. Fair: {fairness_results['is_fair']}")
        
        return fairness_results
    
    def detect_bias(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
                   segments: pd.Series, segment_name: str = 'segment') -> Dict[str, Any]:
        """
        Detect various types of bias across demographic groups
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            y_pred: Binary predictions
            segments: Series with segment labels
            segment_name: Name of the segmentation variable
            
        Returns:
            Dictionary with bias detection results
        """
        logger.info(f"Detecting bias for {segment_name}")
        
        # Ensure segments is a pandas Series
        if not isinstance(segments, pd.Series):
            segments = pd.Series(segments)
        
        unique_segments = segments.unique()
        bias_metrics = []
        
        for segment in unique_segments:
            segment_mask = segments == segment
            segment_y_true = y_true[segment_mask]
            segment_y_prob = y_prob[segment_mask]
            segment_y_pred = y_pred[segment_mask]
            
            if len(segment_y_true) == 0:
                continue
            
            # Calculate bias metrics
            n_samples = len(segment_y_true)
            
            # Demographic parity (equal positive prediction rates)
            positive_pred_rate = segment_y_pred.mean()
            
            # Equal opportunity (equal TPR across groups)
            if segment_y_true.sum() > 0:
                tpr = np.sum((segment_y_true == 1) & (segment_y_pred == 1)) / segment_y_true.sum()
            else:
                tpr = np.nan
            
            # Equalized odds (equal TPR and FPR across groups)
            if (segment_y_true == 0).sum() > 0:
                fpr = np.sum((segment_y_true == 0) & (segment_y_pred == 1)) / (segment_y_true == 0).sum()
            else:
                fpr = np.nan
            
            # Predictive parity (equal PPV across groups)
            if segment_y_pred.sum() > 0:
                ppv = np.sum((segment_y_true == 1) & (segment_y_pred == 1)) / segment_y_pred.sum()
            else:
                ppv = np.nan
            
            # Statistical parity difference
            base_rate = segment_y_true.mean()
            statistical_parity_diff = positive_pred_rate - base_rate
            
            bias_metrics.append({
                'segment': segment,
                'n_samples': n_samples,
                'base_rate': base_rate,
                'positive_prediction_rate': positive_pred_rate,
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'positive_predictive_value': ppv,
                'statistical_parity_difference': statistical_parity_diff
            })
        
        bias_df = pd.DataFrame(bias_metrics)
        
        # Detect bias violations
        bias_violations = []
        
        if len(bias_df) > 1:
            # Check demographic parity
            ppr_values = bias_df['positive_prediction_rate'].dropna()
            if len(ppr_values) > 1:
                ppr_range = ppr_values.max() - ppr_values.min()
                if ppr_range > self.fairness_threshold:
                    bias_violations.append({
                        'type': 'demographic_parity',
                        'value': ppr_range,
                        'threshold': self.fairness_threshold,
                        'description': f'Positive prediction rate range of {ppr_range:.3f} exceeds threshold'
                    })
            
            # Check equal opportunity
            tpr_values = bias_df['true_positive_rate'].dropna()
            if len(tpr_values) > 1:
                tpr_range = tpr_values.max() - tpr_values.min()
                if tpr_range > self.fairness_threshold:
                    bias_violations.append({
                        'type': 'equal_opportunity',
                        'value': tpr_range,
                        'threshold': self.fairness_threshold,
                        'description': f'True positive rate range of {tpr_range:.3f} exceeds threshold'
                    })
            
            # Check equalized odds (TPR and FPR)
            fpr_values = bias_df['false_positive_rate'].dropna()
            if len(fpr_values) > 1:
                fpr_range = fpr_values.max() - fpr_values.min()
                if fpr_range > self.fairness_threshold:
                    bias_violations.append({
                        'type': 'equalized_odds_fpr',
                        'value': fpr_range,
                        'threshold': self.fairness_threshold,
                        'description': f'False positive rate range of {fpr_range:.3f} exceeds threshold'
                    })
        
        bias_results = {
            'segment_name': segment_name,
            'bias_metrics': bias_df,
            'bias_violations': bias_violations,
            'is_fair': len(bias_violations) == 0,
            'n_segments': len(bias_df)
        }
        
        logger.info(f"Bias detection completed. Fair: {bias_results['is_fair']}")
        
        return bias_results
    
    def create_segment_performance_report(self, y_true: np.ndarray, y_prob: np.ndarray,
                                        segments_dict: Dict[str, pd.Series],
                                        threshold: float = 0.5) -> Dict[str, Any]:
        """
        Create comprehensive segment-specific performance report
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            segments_dict: Dictionary mapping segment names to segment series
            threshold: Probability threshold for binary predictions
            
        Returns:
            Dictionary with comprehensive segment analysis
        """
        logger.info("Creating comprehensive segment performance report")
        
        y_pred = (y_prob >= threshold).astype(int)
        
        segment_report = {
            'threshold_used': threshold,
            'overall_metrics': self._calculate_overall_metrics(y_true, y_prob, y_pred),
            'segment_analyses': {},
            'fairness_summary': {},
            'recommendations': []
        }
        
        for segment_name, segments in segments_dict.items():
            logger.info(f"Analyzing segment: {segment_name}")
            
            # Subgroup performance analysis
            subgroup_perf = self.analyze_subgroup_performance(
                y_true, y_prob, segments, segment_name
            )
            
            # Calibration fairness check
            calibration_fairness = self.check_calibration_fairness(
                y_true, y_prob, segments, segment_name
            )
            
            # Bias detection
            bias_results = self.detect_bias(
                y_true, y_prob, y_pred, segments, segment_name
            )
            
            segment_report['segment_analyses'][segment_name] = {
                'subgroup_performance': subgroup_perf,
                'calibration_fairness': calibration_fairness,
                'bias_detection': bias_results
            }
            
            # Summarize fairness for this segment
            segment_report['fairness_summary'][segment_name] = {
                'is_calibration_fair': calibration_fairness['is_fair'],
                'is_bias_free': bias_results['is_fair'],
                'n_calibration_violations': len(calibration_fairness['fairness_violations']),
                'n_bias_violations': len(bias_results['bias_violations'])
            }
            
            # Generate recommendations
            recommendations = self._generate_fairness_recommendations(
                segment_name, calibration_fairness, bias_results, subgroup_perf
            )
            segment_report['recommendations'].extend(recommendations)
        
        # Overall fairness assessment
        all_fair = all(
            summary['is_calibration_fair'] and summary['is_bias_free']
            for summary in segment_report['fairness_summary'].values()
        )
        
        segment_report['overall_fairness'] = {
            'is_fair': all_fair,
            'segments_analyzed': list(segments_dict.keys()),
            'total_violations': sum(
                summary['n_calibration_violations'] + summary['n_bias_violations']
                for summary in segment_report['fairness_summary'].values()
            )
        }
        
        logger.info("Segment performance report completed")
        
        return segment_report
    
    def _calculate_calibration_slope(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """
        Calculate calibration slope and intercept using logistic regression
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Tuple of (slope, intercept)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Convert probabilities to logits
            epsilon = 1e-15
            y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
            logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
            
            # Fit logistic regression
            cal_model = LogisticRegression()
            cal_model.fit(logits.reshape(-1, 1), y_true)
            
            slope = cal_model.coef_[0][0]
            intercept = cal_model.intercept_[0]
            
            return slope, intercept
            
        except Exception as e:
            logger.warning(f"Error calculating calibration slope: {str(e)}")
            return np.nan, np.nan
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate overall model performance metrics"""
        try:
            metrics = {
                'n_samples': len(y_true),
                'base_rate': y_true.mean(),
                'roc_auc': roc_auc_score(y_true, y_prob),
                'pr_auc': average_precision_score(y_true, y_prob),
                'brier_score': brier_score_loss(y_true, y_prob),
                'log_loss': log_loss(y_true, y_prob),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            logger.warning(f"Error calculating overall metrics: {str(e)}")
            return {}
    
    def _generate_fairness_recommendations(self, segment_name: str, 
                                         calibration_results: Dict[str, Any],
                                         bias_results: Dict[str, Any],
                                         subgroup_perf: pd.DataFrame) -> List[str]:
        """Generate recommendations based on fairness analysis"""
        recommendations = []
        
        # Calibration recommendations
        if not calibration_results['is_fair']:
            recommendations.append(
                f"Consider recalibrating the model for {segment_name} segments due to "
                f"{len(calibration_results['fairness_violations'])} calibration violations"
            )
        
        # Bias recommendations
        if not bias_results['is_fair']:
            violation_types = [v['type'] for v in bias_results['bias_violations']]
            recommendations.append(
                f"Address bias in {segment_name} segments. Detected violations: {', '.join(violation_types)}"
            )
        
        # Performance recommendations
        if len(subgroup_perf) > 1:
            auc_range = subgroup_perf['roc_auc'].max() - subgroup_perf['roc_auc'].min()
            if auc_range > 0.1:  # Significant performance difference
                worst_segment = subgroup_perf.loc[subgroup_perf['roc_auc'].idxmin(), 'segment']
                recommendations.append(
                    f"Consider collecting more data or feature engineering for {segment_name} "
                    f"segment '{worst_segment}' which shows lower performance"
                )
        
        return recommendations
    
    def plot_segment_comparison(self, segment_name: str, 
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot segment comparison charts
        
        Args:
            segment_name: Name of the segment to plot
            save_path: Path to save the plot
        """
        if segment_name not in self.segment_results:
            logger.error(f"No results found for segment {segment_name}")
            return
        
        subgroup_df = self.segment_results[segment_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: ROC-AUC by segment
        axes[0, 0].bar(range(len(subgroup_df)), subgroup_df['roc_auc'], alpha=0.7)
        axes[0, 0].set_xlabel('Segment')
        axes[0, 0].set_ylabel('ROC-AUC')
        axes[0, 0].set_title(f'ROC-AUC by {segment_name}')
        axes[0, 0].set_xticks(range(len(subgroup_df)))
        axes[0, 0].set_xticklabels(subgroup_df['segment'], rotation=45)
        
        # Plot 2: Base rate by segment
        axes[0, 1].bar(range(len(subgroup_df)), subgroup_df['base_rate'], alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Base Rate')
        axes[0, 1].set_title(f'Base Rate by {segment_name}')
        axes[0, 1].set_xticks(range(len(subgroup_df)))
        axes[0, 1].set_xticklabels(subgroup_df['segment'], rotation=45)
        
        # Plot 3: Calibration slope by segment
        axes[1, 0].bar(range(len(subgroup_df)), subgroup_df['calibration_slope'], alpha=0.7, color='green')
        axes[1, 0].axhline(y=1, color='red', linestyle='--', label='Perfect Calibration')
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Calibration Slope')
        axes[1, 0].set_title(f'Calibration Slope by {segment_name}')
        axes[1, 0].set_xticks(range(len(subgroup_df)))
        axes[1, 0].set_xticklabels(subgroup_df['segment'], rotation=45)
        axes[1, 0].legend()
        
        # Plot 4: Sample size by segment
        axes[1, 1].bar(range(len(subgroup_df)), subgroup_df['n_samples'], alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_title(f'Sample Size by {segment_name}')
        axes[1, 1].set_xticks(range(len(subgroup_df)))
        axes[1, 1].set_xticklabels(subgroup_df['segment'], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segment comparison plot saved to {save_path}")
        
        plt.show()
    
    def export_segment_analysis(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Export segment analysis results to files
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping result types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for segment_name, results_df in self.segment_results.items():
            # Save segment performance results
            results_path = output_dir / f"segment_analysis_{segment_name}.csv"
            results_df.to_csv(results_path, index=False)
            saved_paths[f"segment_analysis_{segment_name}"] = str(results_path)
        
        logger.info(f"Exported segment analysis results to {output_dir}")
        
        return saved_paths


class ReportGenerator:
    """
    Creates automated evaluation reports with figures and tables
    Implements performance comparison across models and business impact summaries
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ReportGenerator
        
        Args:
            output_dir: Directory to save reports (defaults to config.REPORTS_PATH)
        """
        self.output_dir = Path(output_dir) if output_dir else config.REPORTS_PATH
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        logger.info(f"ReportGenerator initialized with output_dir={self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    model_results: Dict[str, Dict[str, Any]],
                                    business_metrics: Optional[BusinessMetrics] = None,
                                    segment_analyzer: Optional[SegmentAnalyzer] = None,
                                    report_title: str = "Customer Churn Model Evaluation Report") -> str:
        """
        Generate comprehensive evaluation report with all analyses
        
        Args:
            model_results: Dictionary with model evaluation results
                Format: {model_name: {y_true, y_prob, y_pred, metadata}}
            business_metrics: BusinessMetrics instance for business analysis
            segment_analyzer: SegmentAnalyzer instance for fairness analysis
            report_title: Title for the report
            
        Returns:
            Path to generated HTML report
        """
        logger.info("Generating comprehensive evaluation report")
        
        # Initialize business metrics if not provided
        if business_metrics is None:
            business_metrics = BusinessMetrics()
        
        # Generate all report sections
        report_sections = []
        
        # 1. Executive Summary
        exec_summary = self._generate_executive_summary(model_results, business_metrics)
        report_sections.append(exec_summary)
        
        # 2. Model Performance Comparison
        perf_comparison = self._generate_performance_comparison(model_results)
        report_sections.append(perf_comparison)
        
        # 3. Business Impact Analysis
        business_impact = self._generate_business_impact_analysis(model_results, business_metrics)
        report_sections.append(business_impact)
        
        # 4. Detailed Model Analysis
        detailed_analysis = self._generate_detailed_model_analysis(model_results, business_metrics)
        report_sections.append(detailed_analysis)
        
        # 5. Fairness and Bias Analysis (if segment analyzer provided)
        if segment_analyzer is not None:
            fairness_analysis = self._generate_fairness_analysis(model_results, segment_analyzer)
            report_sections.append(fairness_analysis)
        
        # 6. Recommendations
        recommendations = self._generate_recommendations(model_results, business_metrics)
        report_sections.append(recommendations)
        
        # Combine all sections into HTML report
        html_report = self._create_html_report(report_title, report_sections)
        
        # Save report
        report_path = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Comprehensive report generated: {report_path}")
        
        return str(report_path)
    
    def generate_model_comparison_table(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate model comparison table with key metrics
        
        Args:
            model_results: Dictionary with model evaluation results
            
        Returns:
            DataFrame with model comparison
        """
        logger.info("Generating model comparison table")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_prob = results['y_prob']
            y_pred = results.get('y_pred', (y_prob >= 0.5).astype(int))
            metadata = results.get('metadata', {})
            
            # Calculate technical metrics
            try:
                metrics = {
                    'Model': model_name,
                    'ROC-AUC': roc_auc_score(y_true, y_prob),
                    'PR-AUC': average_precision_score(y_true, y_prob),
                    'Brier Score': brier_score_loss(y_true, y_prob),
                    'Log Loss': log_loss(y_true, y_prob),
                    'Accuracy': accuracy_score(y_true, y_pred),
                    'Precision': precision_score(y_true, y_pred),
                    'Recall': recall_score(y_true, y_pred),
                    'F1 Score': f1_score(y_true, y_pred)
                }
                
                # Add metadata if available
                if 'training_time' in metadata:
                    metrics['Training Time (s)'] = metadata['training_time']
                if 'n_features' in metadata:
                    metrics['Features Used'] = metadata['n_features']
                
                comparison_data.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for {model_name}: {str(e)}")
                continue
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by ROC-AUC (descending)
        if 'ROC-AUC' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        # Save table
        table_path = self.tables_dir / "model_comparison.csv"
        comparison_df.to_csv(table_path, index=False)
        
        logger.info(f"Model comparison table saved to {table_path}")
        
        return comparison_df
    
    def generate_business_impact_summary(self, model_results: Dict[str, Dict[str, Any]],
                                       business_metrics: BusinessMetrics,
                                       threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate business impact summary for all models
        
        Args:
            model_results: Dictionary with model evaluation results
            business_metrics: BusinessMetrics instance
            threshold: Probability threshold for business calculations
            
        Returns:
            DataFrame with business impact metrics
        """
        logger.info("Generating business impact summary")
        
        business_data = []
        
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            try:
                # Calculate business metrics
                savings = business_metrics.calculate_expected_savings(y_true, y_prob, threshold)
                
                business_summary = {
                    'Model': model_name,
                    'Expected Profit ($)': savings['total_profit'],
                    'Net Savings ($)': savings['net_savings'],
                    'ROI (%)': savings['roi_percentage'],
                    'Customers Contacted': savings['customers_contacted'],
                    'Customers Retained': savings['customers_retained'],
                    'Contact Rate (%)': savings['contact_rate'] * 100,
                    'Retention Success Rate (%)': savings['retention_success_rate'] * 100,
                    'Cost per Retained Customer ($)': savings['cost_per_retained_customer']
                }
                
                business_data.append(business_summary)
                
            except Exception as e:
                logger.warning(f"Error calculating business metrics for {model_name}: {str(e)}")
                continue
        
        business_df = pd.DataFrame(business_data)
        
        # Sort by expected profit (descending)
        if 'Expected Profit ($)' in business_df.columns:
            business_df = business_df.sort_values('Expected Profit ($)', ascending=False)
        
        # Save table
        table_path = self.tables_dir / "business_impact_summary.csv"
        business_df.to_csv(table_path, index=False)
        
        logger.info(f"Business impact summary saved to {table_path}")
        
        return business_df
    
    def create_performance_visualizations(self, model_results: Dict[str, Dict[str, Any]],
                                        business_metrics: BusinessMetrics) -> Dict[str, str]:
        """
        Create comprehensive performance visualizations
        
        Args:
            model_results: Dictionary with model evaluation results
            business_metrics: BusinessMetrics instance
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Creating performance visualizations")
        
        saved_plots = {}
        
        # 1. ROC Curves Comparison
        roc_path = self._plot_roc_curves_comparison(model_results)
        saved_plots['roc_curves'] = roc_path
        
        # 2. Precision-Recall Curves Comparison
        pr_path = self._plot_pr_curves_comparison(model_results)
        saved_plots['pr_curves'] = pr_path
        
        # 3. Calibration Plots
        cal_path = self._plot_calibration_comparison(model_results)
        saved_plots['calibration_plots'] = cal_path
        
        # 4. Lift Charts for best model
        best_model_name = self._get_best_model_name(model_results)
        if best_model_name:
            lift_path = self._plot_lift_chart_for_model(
                model_results[best_model_name], business_metrics, best_model_name
            )
            saved_plots['lift_chart'] = lift_path
        
        # 5. Gains Chart for best model
        if best_model_name:
            gains_path = self._plot_gains_chart_for_model(
                model_results[best_model_name], business_metrics, best_model_name
            )
            saved_plots['gains_chart'] = gains_path
        
        # 6. Business Metrics Comparison
        business_path = self._plot_business_metrics_comparison(model_results, business_metrics)
        saved_plots['business_comparison'] = business_path
        
        logger.info(f"Created {len(saved_plots)} performance visualizations")
        
        return saved_plots
    
    def _generate_executive_summary(self, model_results: Dict[str, Dict[str, Any]],
                                  business_metrics: BusinessMetrics) -> str:
        """Generate executive summary section"""
        
        # Find best model
        best_model_name = self._get_best_model_name(model_results)
        
        if best_model_name:
            best_results = model_results[best_model_name]
            y_true = best_results['y_true']
            y_prob = best_results['y_prob']
            
            # Calculate key metrics
            roc_auc = roc_auc_score(y_true, y_prob)
            savings = business_metrics.calculate_expected_savings(y_true, y_prob, 0.5)
            
            summary = f"""
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Best Performing Model:</strong> {best_model_name}</li>
                    <li><strong>Model Performance:</strong> ROC-AUC of {roc_auc:.3f}</li>
                    <li><strong>Expected Annual Profit:</strong> ${savings['total_profit']:,.0f}</li>
                    <li><strong>ROI:</strong> {savings['roi_percentage']:.1f}%</li>
                    <li><strong>Customers to Contact:</strong> {savings['customers_contacted']:,} ({savings['contact_rate']*100:.1f}% of total)</li>
                    <li><strong>Expected Customers Retained:</strong> {savings['customers_retained']:,}</li>
                </ul>
                
                <h3>Business Impact</h3>
                <p>The recommended model can generate an expected profit of <strong>${savings['total_profit']:,.0f}</strong> 
                by contacting {savings['contact_rate']*100:.1f}% of customers with the highest churn risk. 
                This represents a {savings['roi_percentage']:.1f}% return on investment.</p>
                
                <h3>Recommendations</h3>
                <ul>
                    <li>Deploy the {best_model_name} model for production use</li>
                    <li>Implement retention campaigns targeting the top {savings['contact_rate']*100:.1f}% of customers by churn risk</li>
                    <li>Monitor model performance and retrain quarterly</li>
                </ul>
            </div>
            """
        else:
            summary = """
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <p>No valid model results found for analysis.</p>
            </div>
            """
        
        return summary
    
    def _generate_performance_comparison(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate performance comparison section"""
        
        comparison_df = self.generate_model_comparison_table(model_results)
        
        if len(comparison_df) > 0:
            # Convert DataFrame to HTML table
            table_html = comparison_df.round(4).to_html(index=False, classes='comparison-table')
            
            section = f"""
            <h2>Model Performance Comparison</h2>
            <div class="table-container">
                {table_html}
            </div>
            <p><strong>Note:</strong> Models are ranked by ROC-AUC score. Higher values indicate better performance.</p>
            """
        else:
            section = """
            <h2>Model Performance Comparison</h2>
            <p>No model comparison data available.</p>
            """
        
        return section
    
    def _generate_business_impact_analysis(self, model_results: Dict[str, Dict[str, Any]],
                                         business_metrics: BusinessMetrics) -> str:
        """Generate business impact analysis section"""
        
        business_df = self.generate_business_impact_summary(model_results, business_metrics)
        
        if len(business_df) > 0:
            # Convert DataFrame to HTML table
            table_html = business_df.round(2).to_html(index=False, classes='business-table')
            
            # Calculate summary statistics
            total_profit_range = business_df['Expected Profit ($)'].max() - business_df['Expected Profit ($)'].min()
            best_roi = business_df['ROI (%)'].max()
            
            section = f"""
            <h2>Business Impact Analysis</h2>
            <div class="table-container">
                {table_html}
            </div>
            
            <div class="analysis-box">
                <h3>Business Insights</h3>
                <ul>
                    <li><strong>Profit Range:</strong> ${total_profit_range:,.0f} difference between best and worst models</li>
                    <li><strong>Best ROI:</strong> {best_roi:.1f}% return on investment</li>
                    <li><strong>Business Parameters Used:</strong> 
                        Retention Value: ${business_metrics.retention_value:,.0f}, 
                        Contact Cost: ${business_metrics.contact_cost:,.0f}</li>
                </ul>
            </div>
            """
        else:
            section = """
            <h2>Business Impact Analysis</h2>
            <p>No business impact data available.</p>
            """
        
        return section
    
    def _generate_detailed_model_analysis(self, model_results: Dict[str, Dict[str, Any]],
                                        business_metrics: BusinessMetrics) -> str:
        """Generate detailed model analysis section"""
        
        best_model_name = self._get_best_model_name(model_results)
        
        if best_model_name:
            results = model_results[best_model_name]
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            # Calculate lift table
            lift_table = business_metrics.calculate_lift_table(y_true, y_prob)
            lift_html = lift_table.round(3).to_html(index=False, classes='lift-table')
            
            # Calculate decile analysis
            decile_analysis = business_metrics.calculate_decile_analysis(y_true, y_prob)
            
            section = f"""
            <h2>Detailed Analysis - {best_model_name}</h2>
            
            <h3>Lift Table Analysis</h3>
            <div class="table-container">
                {lift_html}
            </div>
            
            <div class="analysis-box">
                <h3>Key Insights</h3>
                <ul>
                    <li><strong>Best Decile Lift:</strong> {decile_analysis['summary_stats']['best_decile_lift']:.2f}x</li>
                    <li><strong>Top 30% Capture Rate:</strong> {decile_analysis['summary_stats']['top_3_deciles_capture']*100:.1f}%</li>
                    <li><strong>Concentration Ratio:</strong> {decile_analysis['summary_stats']['concentration_ratio']*100:.1f}% of churners in top 20%</li>
                </ul>
            </div>
            """
        else:
            section = """
            <h2>Detailed Model Analysis</h2>
            <p>No model selected for detailed analysis.</p>
            """
        
        return section
    
    def _generate_fairness_analysis(self, model_results: Dict[str, Dict[str, Any]],
                                  segment_analyzer: SegmentAnalyzer) -> str:
        """Generate fairness and bias analysis section"""
        
        # This would require segment data to be provided
        # For now, return a placeholder
        section = """
        <h2>Fairness and Bias Analysis</h2>
        <div class="analysis-box">
            <p>Fairness analysis requires demographic segment data. 
            Please provide segment information to enable bias detection and calibration fairness checks.</p>
            
            <h3>Recommended Fairness Checks</h3>
            <ul>
                <li>Demographic Parity: Equal positive prediction rates across groups</li>
                <li>Equal Opportunity: Equal true positive rates across groups</li>
                <li>Calibration Fairness: Consistent probability calibration across groups</li>
                <li>Equalized Odds: Equal true positive and false positive rates across groups</li>
            </ul>
        </div>
        """
        
        return section
    
    def _generate_recommendations(self, model_results: Dict[str, Dict[str, Any]],
                                business_metrics: BusinessMetrics) -> str:
        """Generate recommendations section"""
        
        recommendations = []
        
        # Model selection recommendation
        best_model_name = self._get_best_model_name(model_results)
        if best_model_name:
            recommendations.append(f"Deploy the <strong>{best_model_name}</strong> model for production use based on superior ROC-AUC performance.")
        
        # Business recommendations
        if best_model_name:
            results = model_results[best_model_name]
            savings = business_metrics.calculate_expected_savings(results['y_true'], results['y_prob'], 0.5)
            
            recommendations.extend([
                f"Target the top {savings['contact_rate']*100:.1f}% of customers by churn probability for retention campaigns.",
                f"Expected to retain {savings['customers_retained']} customers with an ROI of {savings['roi_percentage']:.1f}%.",
                "Monitor model performance monthly and retrain quarterly or when performance degrades.",
                "Consider A/B testing different retention strategies for high-risk customers."
            ])
        
        # Technical recommendations
        recommendations.extend([
            "Implement model monitoring to detect data drift and performance degradation.",
            "Set up automated retraining pipelines with performance validation.",
            "Consider ensemble methods if individual model performance is insufficient.",
            "Validate model fairness across different customer segments before deployment."
        ])
        
        section = f"""
        <h2>Recommendations</h2>
        <div class="recommendations-box">
            <h3>Implementation Recommendations</h3>
            <ol>
                {''.join(f'<li>{rec}</li>' for rec in recommendations)}
            </ol>
            
            <h3>Next Steps</h3>
            <ol>
                <li>Validate model performance on holdout test set</li>
                <li>Conduct pilot retention campaign with high-risk customers</li>
                <li>Measure actual retention rates and business impact</li>
                <li>Refine model and business parameters based on results</li>
                <li>Scale to full customer base</li>
            </ol>
        </div>
        """
        
        return section
    
    def _create_html_report(self, title: str, sections: List[str]) -> str:
        """Create complete HTML report"""
        
        css_styles = """
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; margin-top: 30px; }
            h3 { color: #7f8c8d; }
            .summary-box, .analysis-box, .recommendations-box { 
                background-color: #f8f9fa; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0; 
            }
            .table-container { overflow-x: auto; margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .comparison-table, .business-table, .lift-table { font-size: 0.9em; }
            .metric-highlight { background-color: #e8f5e8; font-weight: bold; }
            ul, ol { padding-left: 20px; }
            li { margin: 5px 0; }
        </style>
        """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            {css_styles}
        </head>
        <body>
            <h1>{title}</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            {''.join(sections)}
            
            <hr>
            <p><em>This report was automatically generated by the Customer Churn ML Pipeline evaluation system.</em></p>
        </body>
        </html>
        """
        
        return html_content
    
    def _get_best_model_name(self, model_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Get the name of the best performing model by ROC-AUC"""
        best_score = -1
        best_model = None
        
        for model_name, results in model_results.items():
            try:
                score = roc_auc_score(results['y_true'], results['y_prob'])
                if score > best_score:
                    best_score = score
                    best_model = model_name
            except Exception:
                continue
        
        return best_model
    
    def _plot_roc_curves_comparison(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """Plot ROC curves comparison"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            try:
                y_true = results['y_true']
                y_prob = results['y_prob']
                
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = roc_auc_score(y_true, y_prob)
                
                plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                logger.warning(f"Error plotting ROC curve for {model_name}: {str(e)}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.figures_dir / "roc_curves_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_pr_curves_comparison(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """Plot Precision-Recall curves comparison"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            try:
                y_true = results['y_true']
                y_prob = results['y_prob']
                
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
                
                plt.plot(recall, precision, linewidth=2, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
                
            except Exception as e:
                logger.warning(f"Error plotting PR curve for {model_name}: {str(e)}")
                continue
        
        # Add baseline
        baseline = np.mean([results['y_true'].mean() for results in model_results.values()])
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.figures_dir / "pr_curves_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_calibration_comparison(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """Plot calibration comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Calibration curves
        for model_name, results in model_results.items():
            try:
                y_true = results['y_true']
                y_prob = results['y_prob']
                
                fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
                
                axes[0].plot(mean_pred, fraction_pos, marker='o', linewidth=2, label=model_name)
                
            except Exception as e:
                logger.warning(f"Error plotting calibration for {model_name}: {str(e)}")
                continue
        
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Calibration Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Brier scores
        model_names = []
        brier_scores = []
        
        for model_name, results in model_results.items():
            try:
                y_true = results['y_true']
                y_prob = results['y_prob']
                brier = brier_score_loss(y_true, y_prob)
                
                model_names.append(model_name)
                brier_scores.append(brier)
                
            except Exception:
                continue
        
        if model_names:
            axes[1].bar(range(len(model_names)), brier_scores, alpha=0.7)
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Brier Score (Lower is Better)')
            axes[1].set_title('Brier Score Comparison')
            axes[1].set_xticks(range(len(model_names)))
            axes[1].set_xticklabels(model_names, rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.figures_dir / "calibration_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_lift_chart_for_model(self, results: Dict[str, Any], 
                                  business_metrics: BusinessMetrics, 
                                  model_name: str) -> str:
        """Plot lift chart for specific model"""
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        lift_table = business_metrics.calculate_lift_table(y_true, y_prob)
        business_metrics.plot_lift_chart(lift_table)
        
        plot_path = self.figures_dir / f"lift_chart_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_gains_chart_for_model(self, results: Dict[str, Any],
                                   business_metrics: BusinessMetrics,
                                   model_name: str) -> str:
        """Plot gains chart for specific model"""
        y_true = results['y_true']
        y_prob = results['y_prob']
        
        gains_data = business_metrics.generate_gains_chart(y_true, y_prob)
        
        plot_path = self.figures_dir / f"gains_chart_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_business_metrics_comparison(self, model_results: Dict[str, Dict[str, Any]],
                                        business_metrics: BusinessMetrics) -> str:
        """Plot business metrics comparison"""
        business_df = self.generate_business_impact_summary(model_results, business_metrics)
        
        if len(business_df) == 0:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Expected Profit
        axes[0, 0].bar(range(len(business_df)), business_df['Expected Profit ($)'], alpha=0.7)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Expected Profit ($)')
        axes[0, 0].set_title('Expected Profit Comparison')
        axes[0, 0].set_xticks(range(len(business_df)))
        axes[0, 0].set_xticklabels(business_df['Model'], rotation=45)
        
        # Plot 2: ROI
        axes[0, 1].bar(range(len(business_df)), business_df['ROI (%)'], alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('ROI (%)')
        axes[0, 1].set_title('Return on Investment Comparison')
        axes[0, 1].set_xticks(range(len(business_df)))
        axes[0, 1].set_xticklabels(business_df['Model'], rotation=45)
        
        # Plot 3: Contact Rate vs Retention Rate
        axes[1, 0].scatter(business_df['Contact Rate (%)'], business_df['Retention Success Rate (%)'], 
                          s=100, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Contact Rate (%)')
        axes[1, 0].set_ylabel('Retention Success Rate (%)')
        axes[1, 0].set_title('Contact Rate vs Retention Success Rate')
        
        # Add model labels
        for i, model in enumerate(business_df['Model']):
            axes[1, 0].annotate(model, 
                              (business_df.iloc[i]['Contact Rate (%)'], 
                               business_df.iloc[i]['Retention Success Rate (%)']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Plot 4: Cost per Retained Customer
        axes[1, 1].bar(range(len(business_df)), business_df['Cost per Retained Customer ($)'], 
                      alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Cost per Retained Customer ($)')
        axes[1, 1].set_title('Cost Efficiency Comparison')
        axes[1, 1].set_xticks(range(len(business_df)))
        axes[1, 1].set_xticklabels(business_df['Model'], rotation=45)
        
        plt.tight_layout()
        
        plot_path = self.figures_dir / "business_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)