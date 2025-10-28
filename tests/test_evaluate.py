"""
Tests for business evaluation and metrics module
Tests business metrics calculations, fairness analysis, and report generation
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

import sys
sys.path.append('src')

from evaluate import BusinessMetrics, SegmentAnalyzer, ReportGenerator
from config import config


class TestBusinessMetrics:
    """Test cases for BusinessMetrics class"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing"""
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
    def business_metrics(self):
        """Create BusinessMetrics instance"""
        business_config = {
            'retention_value': 1000.0,
            'contact_cost': 50.0,
            'churn_cost': 500.0
        }
        return BusinessMetrics(business_config)
    
    def test_business_metrics_initialization(self, business_metrics):
        """Test BusinessMetrics initialization"""
        assert business_metrics.retention_value == 1000.0
        assert business_metrics.contact_cost == 50.0
        assert business_metrics.churn_cost == 500.0
    
    def test_lift_table_calculation(self, business_metrics, sample_predictions):
        """Test lift table calculation"""
        y_true, y_prob = sample_predictions
        
        lift_table = business_metrics.calculate_lift_table(y_true, y_prob, n_deciles=10)
        
        # Check lift table structure
        assert isinstance(lift_table, pd.DataFrame)
        assert len(lift_table) == 10  # 10 deciles
        
        expected_columns = [
            'decile', 'threshold_min', 'threshold_max', 'n_customers', 'n_churners',
            'churn_rate', 'lift', 'cumulative_lift', 'capture_rate', 'cumulative_capture_rate'
        ]
        assert all(col in lift_table.columns for col in expected_columns)
        
        # Check that deciles are ordered correctly
        assert lift_table['decile'].tolist() == list(range(1, 11))
        
        # Check that thresholds are decreasing (highest prob first)
        assert lift_table['threshold_max'].iloc[0] >= lift_table['threshold_max'].iloc[-1]
        
        # Check that cumulative capture rate is increasing
        cumulative_rates = lift_table['cumulative_capture_rate'].values
        assert all(cumulative_rates[i] <= cumulative_rates[i+1] for i in range(len(cumulative_rates)-1))
    
    def test_expected_savings_calculation(self, business_metrics, sample_predictions):
        """Test expected savings calculation"""
        y_true, y_prob = sample_predictions
        threshold = 0.5
        
        savings = business_metrics.calculate_expected_savings(y_true, y_prob, threshold)
        
        # Check that all expected keys are present
        expected_keys = [
            'total_profit', 'net_savings', 'roi_percentage', 'retention_revenue',
            'total_contact_costs', 'missed_churn_costs', 'customers_contacted',
            'customers_retained', 'contact_rate', 'retention_success_rate'
        ]
        assert all(key in savings for key in expected_keys)
        
        # Check that values are reasonable
        assert savings['total_customers'] == len(y_true)
        assert savings['total_churners'] == y_true.sum()
        assert 0 <= savings['contact_rate'] <= 1
        assert 0 <= savings['retention_success_rate'] <= 1
        
        # Check business logic
        assert savings['customers_contacted'] >= savings['customers_retained']
        assert savings['retention_revenue'] == savings['customers_retained'] * business_metrics.retention_value
        assert savings['total_contact_costs'] == savings['customers_contacted'] * business_metrics.contact_cost
    
    def test_gains_chart_generation(self, business_metrics, sample_predictions):
        """Test gains chart generation"""
        y_true, y_prob = sample_predictions
        
        with patch('matplotlib.pyplot.show'):  # Prevent plot display during testing
            gains_data = business_metrics.generate_gains_chart(y_true, y_prob)
        
        # Check gains data structure
        assert isinstance(gains_data, dict)
        expected_keys = ['population_percentages', 'gains_percentages', 'total_positives', 'cumulative_positives']
        assert all(key in gains_data for key in expected_keys)
        
        # Check data properties
        assert len(gains_data['population_percentages']) == len(y_true)
        assert len(gains_data['gains_percentages']) == len(y_true)
        assert gains_data['total_positives'] == y_true.sum()
        
        # Check that gains are monotonically increasing
        gains = gains_data['gains_percentages']
        assert all(gains[i] <= gains[i+1] for i in range(len(gains)-1))
    
    def test_decile_analysis(self, business_metrics, sample_predictions):
        """Test comprehensive decile analysis"""
        y_true, y_prob = sample_predictions
        
        decile_analysis = business_metrics.calculate_decile_analysis(y_true, y_prob, n_deciles=5)
        
        # Check analysis structure
        assert isinstance(decile_analysis, dict)
        expected_keys = ['lift_table', 'business_metrics', 'summary_stats', 'n_deciles']
        assert all(key in decile_analysis for key in expected_keys)
        
        # Check lift table
        assert len(decile_analysis['lift_table']) == 5
        
        # Check business metrics
        business_df = decile_analysis['business_metrics']
        assert isinstance(business_df, pd.DataFrame)
        assert 'expected_profit' in business_df.columns
        assert 'roi_percentage' in business_df.columns
        
        # Check summary stats
        summary = decile_analysis['summary_stats']
        assert 'best_decile_lift' in summary
        assert 'concentration_ratio' in summary
    
    def test_capture_rate_metrics(self, business_metrics, sample_predictions):
        """Test capture rate metrics calculation"""
        y_true, y_prob = sample_predictions
        
        contact_percentages = [10, 20, 30]
        capture_df = business_metrics.calculate_capture_rate_metrics(y_true, y_prob, contact_percentages)
        
        # Check capture metrics structure
        assert isinstance(capture_df, pd.DataFrame)
        assert len(capture_df) == len(contact_percentages)
        
        expected_columns = [
            'contact_percentage', 'customers_contacted', 'churners_captured',
            'capture_rate', 'threshold', 'expected_profit'
        ]
        assert all(col in capture_df.columns for col in expected_columns)
        
        # Check that capture rates are reasonable
        assert all(0 <= rate <= 1 for rate in capture_df['capture_rate'])
        
        # Check that more contact leads to more churners captured (generally)
        assert capture_df['churners_captured'].iloc[0] <= capture_df['churners_captured'].iloc[-1]


class TestSegmentAnalyzer:
    """Test cases for SegmentAnalyzer class"""
    
    @pytest.fixture
    def sample_data_with_segments(self):
        """Create sample data with demographic segments"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create predictions
        y_true = np.random.binomial(1, 0.3, n_samples)
        y_prob = np.random.beta(2, 5, n_samples)
        y_prob[y_true == 1] += 0.3
        y_prob = np.clip(y_prob, 0.01, 0.99)
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Create demographic segments
        segments = pd.Series(np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_samples))
        
        return y_true, y_prob, y_pred, segments
    
    @pytest.fixture
    def segment_analyzer(self):
        """Create SegmentAnalyzer instance"""
        return SegmentAnalyzer(fairness_threshold=0.1)
    
    def test_segment_analyzer_initialization(self, segment_analyzer):
        """Test SegmentAnalyzer initialization"""
        assert segment_analyzer.fairness_threshold == 0.1
        assert isinstance(segment_analyzer.segment_results, dict)
    
    def test_subgroup_performance_analysis(self, segment_analyzer, sample_data_with_segments):
        """Test subgroup performance analysis"""
        y_true, y_prob, y_pred, segments = sample_data_with_segments
        
        subgroup_df = segment_analyzer.analyze_subgroup_performance(
            y_true, y_prob, segments, 'demographic'
        )
        
        # Check subgroup analysis structure
        assert isinstance(subgroup_df, pd.DataFrame)
        assert len(subgroup_df) == len(segments.unique())
        
        expected_columns = [
            'segment', 'segment_name', 'n_samples', 'n_positives', 'base_rate',
            'roc_auc', 'pr_auc', 'brier_score', 'mean_predicted_prob'
        ]
        assert all(col in subgroup_df.columns for col in expected_columns)
        
        # Check that all segments are represented
        assert set(subgroup_df['segment']) == set(segments.unique())
        
        # Check that metrics are reasonable
        assert all(0 <= rate <= 1 for rate in subgroup_df['base_rate'])
        assert all(0 <= auc <= 1 for auc in subgroup_df['roc_auc'].dropna())
    
    def test_calibration_fairness_check(self, segment_analyzer, sample_data_with_segments):
        """Test calibration fairness check"""
        y_true, y_prob, y_pred, segments = sample_data_with_segments
        
        fairness_results = segment_analyzer.check_calibration_fairness(
            y_true, y_prob, segments, 'demographic'
        )
        
        # Check fairness results structure
        assert isinstance(fairness_results, dict)
        expected_keys = ['segment_name', 'calibration_metrics', 'fairness_violations', 'is_fair']
        assert all(key in fairness_results for key in expected_keys)
        
        # Check calibration metrics
        cal_df = fairness_results['calibration_metrics']
        assert isinstance(cal_df, pd.DataFrame)
        assert 'ece' in cal_df.columns
        assert 'mce' in cal_df.columns
        
        # Check fairness assessment
        assert isinstance(fairness_results['is_fair'], bool)
        assert isinstance(fairness_results['fairness_violations'], list)
    
    def test_bias_detection(self, segment_analyzer, sample_data_with_segments):
        """Test bias detection"""
        y_true, y_prob, y_pred, segments = sample_data_with_segments
        
        bias_results = segment_analyzer.detect_bias(
            y_true, y_prob, y_pred, segments, 'demographic'
        )
        
        # Check bias results structure
        assert isinstance(bias_results, dict)
        expected_keys = ['segment_name', 'bias_metrics', 'bias_violations', 'is_fair']
        assert all(key in bias_results for key in expected_keys)
        
        # Check bias metrics
        bias_df = bias_results['bias_metrics']
        assert isinstance(bias_df, pd.DataFrame)
        
        expected_columns = [
            'segment', 'base_rate', 'positive_prediction_rate',
            'true_positive_rate', 'false_positive_rate'
        ]
        assert all(col in bias_df.columns for col in expected_columns)
        
        # Check that rates are valid probabilities
        assert all(0 <= rate <= 1 for rate in bias_df['base_rate'])
        assert all(0 <= rate <= 1 for rate in bias_df['positive_prediction_rate'])
    
    def test_segment_performance_report(self, segment_analyzer, sample_data_with_segments):
        """Test comprehensive segment performance report"""
        y_true, y_prob, y_pred, segments = sample_data_with_segments
        
        segments_dict = {'demographic': segments}
        
        report = segment_analyzer.create_segment_performance_report(
            y_true, y_prob, segments_dict, threshold=0.5
        )
        
        # Check report structure
        assert isinstance(report, dict)
        expected_keys = [
            'threshold_used', 'overall_metrics', 'segment_analyses',
            'fairness_summary', 'overall_fairness', 'recommendations'
        ]
        assert all(key in report for key in expected_keys)
        
        # Check segment analyses
        assert 'demographic' in report['segment_analyses']
        segment_analysis = report['segment_analyses']['demographic']
        assert 'subgroup_performance' in segment_analysis
        assert 'calibration_fairness' in segment_analysis
        assert 'bias_detection' in segment_analysis
        
        # Check fairness summary
        assert 'demographic' in report['fairness_summary']
        fairness_summary = report['fairness_summary']['demographic']
        assert 'is_calibration_fair' in fairness_summary
        assert 'is_bias_free' in fairness_summary


class TestReportGenerator:
    """Test cases for ReportGenerator class"""
    
    @pytest.fixture
    def sample_model_results(self):
        """Create sample model results for testing"""
        np.random.seed(42)
        n_samples = 500
        
        # Create two models with different performance
        model_results = {}
        
        for i, model_name in enumerate(['Model_A', 'Model_B']):
            y_true = np.random.binomial(1, 0.3, n_samples)
            
            # Make Model_A slightly better
            if model_name == 'Model_A':
                y_prob = np.random.beta(2, 4, n_samples)
                y_prob[y_true == 1] += 0.4
            else:
                y_prob = np.random.beta(2, 5, n_samples)
                y_prob[y_true == 1] += 0.3
            
            y_prob = np.clip(y_prob, 0.01, 0.99)
            y_pred = (y_prob >= 0.5).astype(int)
            
            model_results[model_name] = {
                'y_true': y_true,
                'y_prob': y_prob,
                'y_pred': y_pred,
                'metadata': {'training_time': 10 + i, 'n_features': 20}
            }
        
        return model_results
    
    @pytest.fixture
    def report_generator(self):
        """Create ReportGenerator instance with temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ReportGenerator(output_dir=temp_dir)
    
    def test_report_generator_initialization(self, report_generator):
        """Test ReportGenerator initialization"""
        assert report_generator.output_dir.exists()
        assert report_generator.figures_dir.exists()
        assert report_generator.tables_dir.exists()
    
    def test_model_comparison_table(self, report_generator, sample_model_results):
        """Test model comparison table generation"""
        comparison_df = report_generator.generate_model_comparison_table(sample_model_results)
        
        # Check comparison table structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(sample_model_results)
        
        expected_columns = [
            'Model', 'ROC-AUC', 'PR-AUC', 'Brier Score', 'Accuracy', 'Precision', 'Recall'
        ]
        assert all(col in comparison_df.columns for col in expected_columns)
        
        # Check that models are present
        assert set(comparison_df['Model']) == set(sample_model_results.keys())
        
        # Check that metrics are reasonable
        assert all(0 <= auc <= 1 for auc in comparison_df['ROC-AUC'])
        assert all(0 <= acc <= 1 for acc in comparison_df['Accuracy'])
    
    def test_business_impact_summary(self, report_generator, sample_model_results):
        """Test business impact summary generation"""
        from evaluate import BusinessMetrics
        business_metrics = BusinessMetrics()
        
        business_df = report_generator.generate_business_impact_summary(
            sample_model_results, business_metrics, threshold=0.5
        )
        
        # Check business summary structure
        assert isinstance(business_df, pd.DataFrame)
        assert len(business_df) == len(sample_model_results)
        
        expected_columns = [
            'Model', 'Expected Profit ($)', 'Net Savings ($)', 'ROI (%)',
            'Customers Contacted', 'Customers Retained'
        ]
        assert all(col in business_df.columns for col in expected_columns)
        
        # Check that all models are present
        assert set(business_df['Model']) == set(sample_model_results.keys())
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_performance_visualizations(self, mock_savefig, mock_show, 
                                      report_generator, sample_model_results):
        """Test performance visualizations creation"""
        from evaluate import BusinessMetrics
        business_metrics = BusinessMetrics()
        
        saved_plots = report_generator.create_performance_visualizations(
            sample_model_results, business_metrics
        )
        
        # Check that visualizations are created
        assert isinstance(saved_plots, dict)
        expected_plots = ['roc_curves', 'pr_curves', 'calibration_plots']
        assert all(plot in saved_plots for plot in expected_plots)
        
        # Check that files would be saved (mocked)
        assert mock_savefig.call_count >= len(expected_plots)
    
    def test_comprehensive_report_generation(self, report_generator, sample_model_results):
        """Test comprehensive report generation"""
        from evaluate import BusinessMetrics
        business_metrics = BusinessMetrics()
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            report_path = report_generator.generate_comprehensive_report(
                sample_model_results, business_metrics, report_title="Test Report"
            )
        
        # Check that report is generated
        assert Path(report_path).exists()
        assert report_path.endswith('.html')
        
        # Check report content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'Test Report' in content
        assert 'Executive Summary' in content
        assert 'Model Performance Comparison' in content
        assert 'Business Impact Analysis' in content


if __name__ == '__main__':
    pytest.main([__file__])