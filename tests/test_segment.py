"""
Unit tests for customer segmentation and CLV components
Tests clustering stability, CLV model fitting, and priority ranking logic
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.segment import (
    CustomerSegmenter, CLVCalculator, PriorityRanker,
    SegmentProfile, CLVPrediction
)


class TestCustomerSegmenter:
    """Test cases for CustomerSegmenter class"""
    
    @pytest.fixture
    def sample_customer_data(self):
        """Create sample customer data for testing"""
        np.random.seed(42)
        n_customers = 100
        
        data = {
            'customer_id': [f'cust_{i:04d}' for i in range(n_customers)],
            'age': np.random.normal(40, 15, n_customers).clip(18, 80),
            'tenure': np.random.exponential(2, n_customers).clip(0, 10),
            'monthly_charges': np.random.normal(70, 20, n_customers).clip(10, 150),
            'total_charges': np.random.normal(1500, 800, n_customers).clip(0, 5000),
            'num_services': np.random.poisson(3, n_customers).clip(1, 8),
            'support_tickets': np.random.poisson(1, n_customers),
            'Churn': np.random.binomial(1, 0.3, n_customers)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def segmenter(self):
        """Create CustomerSegmenter instance"""
        return CustomerSegmenter(random_state=42)
    
    def test_initialization(self, segmenter):
        """Test CustomerSegmenter initialization"""
        assert segmenter.random_state == 42
        assert segmenter.best_model is None
        assert segmenter.best_n_clusters is None
        assert segmenter.cluster_labels is None
    
    def test_find_optimal_clusters_kmeans(self, segmenter, sample_customer_data):
        """Test optimal cluster selection for K-means"""
        # Use numeric features only
        features = sample_customer_data[['age', 'tenure', 'monthly_charges', 'total_charges', 'num_services']]
        
        results = segmenter.find_optimal_clusters(features, max_clusters=5, method='kmeans')
        
        assert 'optimal_clusters' in results
        assert 'metrics' in results
        assert 'best_silhouette' in results
        assert results['optimal_clusters'] >= 2
        assert results['optimal_clusters'] <= 5
        assert isinstance(results['metrics'], pd.DataFrame)
        assert len(results['metrics']) > 0
    
    def test_find_optimal_clusters_hierarchical(self, segmenter, sample_customer_data):
        """Test optimal cluster selection for hierarchical clustering"""
        features = sample_customer_data[['age', 'tenure', 'monthly_charges', 'total_charges']]
        
        results = segmenter.find_optimal_clusters(features, max_clusters=4, method='hierarchical')
        
        assert 'optimal_clusters' in results
        assert results['optimal_clusters'] >= 2
        assert results['optimal_clusters'] <= 4
    
    def test_perform_kmeans_clustering(self, segmenter, sample_customer_data):
        """Test K-means clustering execution"""
        features = sample_customer_data[['age', 'tenure', 'monthly_charges', 'total_charges']]
        
        labels, results = segmenter.perform_kmeans_clustering(features, n_clusters=3)
        
        assert len(labels) == len(features)
        assert len(np.unique(labels)) == 3
        assert 'silhouette_score' in results
        assert 'inertia' in results
        assert results['n_clusters'] == 3
        assert segmenter.cluster_labels is not None
        assert segmenter.best_model is not None
    
    def test_perform_hierarchical_clustering(self, segmenter, sample_customer_data):
        """Test hierarchical clustering execution"""
        features = sample_customer_data[['age', 'tenure', 'monthly_charges']]
        
        labels, results = segmenter.perform_hierarchical_clustering(features, n_clusters=3)
        
        assert len(labels) == len(features)
        assert len(np.unique(labels)) == 3
        assert 'silhouette_score' in results
        assert 'linkage_matrix' in results
        assert results['n_clusters'] == 3
    
    def test_profile_segments(self, segmenter, sample_customer_data):
        """Test segment profiling functionality"""
        features = sample_customer_data[['age', 'tenure', 'monthly_charges', 'total_charges']]
        
        # First perform clustering
        labels, _ = segmenter.perform_kmeans_clustering(features, n_clusters=3)
        
        # Then profile segments
        profiles = segmenter.profile_segments(sample_customer_data, labels)
        
        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) == 3  # Number of clusters
        assert 'Segment' in profiles.columns
        assert 'Size' in profiles.columns
        assert 'Churn_Rate' in profiles.columns
        assert profiles['Size'].sum() == len(sample_customer_data)
    
    def test_evaluate_clustering_quality(self, segmenter, sample_customer_data):
        """Test clustering quality evaluation"""
        features = sample_customer_data[['age', 'tenure', 'monthly_charges']]
        
        # Perform clustering first
        labels, _ = segmenter.perform_kmeans_clustering(features, n_clusters=3)
        
        # Evaluate quality
        metrics = segmenter.evaluate_clustering_quality(features, labels)
        
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
        assert 'n_clusters' in metrics
        assert metrics['n_clusters'] == 3
    
    def test_clustering_reproducibility(self, sample_customer_data):
        """Test that clustering results are reproducible with same random seed"""
        features = sample_customer_data[['age', 'tenure', 'monthly_charges']]
        
        # First run
        segmenter1 = CustomerSegmenter(random_state=42)
        labels1, _ = segmenter1.perform_kmeans_clustering(features, n_clusters=3)
        
        # Second run with same seed
        segmenter2 = CustomerSegmenter(random_state=42)
        labels2, _ = segmenter2.perform_kmeans_clustering(features, n_clusters=3)
        
        # Results should be identical
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_small_dataset_handling(self, segmenter):
        """Test handling of very small datasets"""
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="Need at least 10 samples"):
            segmenter.find_optimal_clusters(small_data)


class TestCLVCalculator:
    """Test cases for CLVCalculator class"""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for CLV testing"""
        np.random.seed(42)
        n_transactions = 500
        n_customers = 50
        
        # Generate transaction data
        customer_ids = [f'cust_{i:03d}' for i in range(n_customers)]
        
        transactions = []
        base_date = datetime(2023, 1, 1)
        
        for customer_id in customer_ids:
            # Random number of transactions per customer
            n_trans = np.random.poisson(10) + 1
            
            for i in range(n_trans):
                transaction_date = base_date + timedelta(days=np.random.randint(0, 365))
                order_value = np.random.exponential(100) + 20
                
                transactions.append({
                    'customer_id': customer_id,
                    'order_date': transaction_date,
                    'order_value': order_value
                })
        
        return pd.DataFrame(transactions)
    
    @pytest.fixture
    def sample_rfm_data(self):
        """Create sample RFM data for testing"""
        np.random.seed(42)
        n_customers = 50
        
        data = {
            'customer_id': [f'cust_{i:03d}' for i in range(n_customers)],
            'frequency': np.random.poisson(5, n_customers),
            'recency': np.random.exponential(30, n_customers).clip(0, 365),
            'T': np.random.exponential(180, n_customers).clip(30, 730),
            'monetary_value': np.random.exponential(150, n_customers) + 50
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def clv_calculator(self):
        """Create CLVCalculator instance"""
        return CLVCalculator()
    
    def test_initialization(self, clv_calculator):
        """Test CLVCalculator initialization"""
        assert clv_calculator.bgnbd_model is None
        assert clv_calculator.ggf_model is None
        assert clv_calculator.is_fitted is False
    
    def test_prepare_rfm_data_fallback(self, clv_calculator, sample_transaction_data):
        """Test RFM data preparation using fallback method"""
        rfm_data = clv_calculator.prepare_rfm_data(sample_transaction_data)
        
        assert isinstance(rfm_data, pd.DataFrame)
        assert 'customer_id' in rfm_data.columns
        assert 'frequency' in rfm_data.columns
        assert 'recency' in rfm_data.columns
        assert 'T' in rfm_data.columns
        assert 'monetary_value' in rfm_data.columns
        assert len(rfm_data) > 0
        assert rfm_data['frequency'].min() >= 0
        assert rfm_data['recency'].min() >= 0
    
    def test_predict_clv_fallback(self, clv_calculator, sample_rfm_data):
        """Test CLV prediction using fallback method"""
        clv_predictions = clv_calculator.predict_clv(sample_rfm_data, time_horizon=365)
        
        assert isinstance(clv_predictions, pd.DataFrame)
        assert 'customer_id' in clv_predictions.columns
        assert 'predicted_clv' in clv_predictions.columns
        assert 'clv_lower_ci' in clv_predictions.columns
        assert 'clv_upper_ci' in clv_predictions.columns
        assert len(clv_predictions) == len(sample_rfm_data)
        assert clv_predictions['predicted_clv'].min() >= 0
        assert (clv_predictions['clv_upper_ci'] >= clv_predictions['clv_lower_ci']).all()
    
    @patch('src.segment.LIFETIMES_AVAILABLE', True)
    @patch('src.segment.BetaGeoFitter')
    def test_fit_bgnbd_model_with_lifetimes(self, mock_bgnbd, clv_calculator, sample_rfm_data):
        """Test BG/NBD model fitting when lifetimes is available"""
        mock_model = MagicMock()
        mock_bgnbd.return_value = mock_model
        
        result = clv_calculator.fit_bgnbd_model(sample_rfm_data)
        
        assert result == mock_model
        mock_bgnbd.assert_called_once_with(penalizer_coef=0.01)
        mock_model.fit.assert_called_once()
    
    @patch('src.segment.LIFETIMES_AVAILABLE', False)
    def test_fit_bgnbd_model_without_lifetimes(self, clv_calculator, sample_rfm_data):
        """Test BG/NBD model fitting when lifetimes is not available"""
        result = clv_calculator.fit_bgnbd_model(sample_rfm_data)
        assert result is None
    
    def test_clv_prediction_consistency(self, clv_calculator, sample_rfm_data):
        """Test that CLV predictions are consistent across runs"""
        # First prediction
        clv1 = clv_calculator.predict_clv(sample_rfm_data, time_horizon=365)
        
        # Second prediction
        clv2 = clv_calculator.predict_clv(sample_rfm_data, time_horizon=365)
        
        # Should be identical (deterministic calculation)
        pd.testing.assert_frame_equal(clv1, clv2)
    
    def test_clv_time_horizon_scaling(self, clv_calculator, sample_rfm_data):
        """Test that CLV scales appropriately with time horizon"""
        clv_short = clv_calculator.predict_clv(sample_rfm_data, time_horizon=180)
        clv_long = clv_calculator.predict_clv(sample_rfm_data, time_horizon=360)
        
        # Longer horizon should generally produce higher CLV
        assert clv_long['predicted_clv'].mean() > clv_short['predicted_clv'].mean()


class TestPriorityRanker:
    """Test cases for PriorityRanker class"""
    
    @pytest.fixture
    def sample_customer_scores(self):
        """Create sample customer data with churn probabilities and CLV"""
        np.random.seed(42)
        n_customers = 100
        
        data = {
            'customer_id': [f'cust_{i:04d}' for i in range(n_customers)],
            'churn_probability': np.random.beta(2, 5, n_customers),  # Skewed toward lower probabilities
            'predicted_clv': np.random.exponential(500, n_customers) + 100,
            'segment': np.random.choice(['A', 'B', 'C'], n_customers)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def priority_ranker(self):
        """Create PriorityRanker instance"""
        return PriorityRanker()
    
    def test_initialization(self, priority_ranker):
        """Test PriorityRanker initialization"""
        assert priority_ranker.priority_scores is None
        assert priority_ranker.ranking_results is None
    
    def test_calculate_expected_value(self, priority_ranker):
        """Test expected value calculation"""
        churn_probs = np.array([0.1, 0.5, 0.8])
        clv_estimates = np.array([1000, 500, 2000])
        
        expected_values = priority_ranker.calculate_expected_value(
            churn_probs, clv_estimates, retention_cost=50, retention_effectiveness=0.3
        )
        
        assert len(expected_values) == 3
        assert isinstance(expected_values, np.ndarray)
        # High churn, high CLV should have highest expected value
        assert expected_values[2] > expected_values[1]
    
    def test_rank_customers_for_retention(self, priority_ranker, sample_customer_scores):
        """Test customer ranking functionality"""
        ranking_results = priority_ranker.rank_customers_for_retention(sample_customer_scores)
        
        assert isinstance(ranking_results, pd.DataFrame)
        assert 'customer_id' in ranking_results.columns
        assert 'expected_value' in ranking_results.columns
        assert 'priority_score' in ranking_results.columns
        assert 'priority_tier' in ranking_results.columns
        assert 'eligible_for_intervention' in ranking_results.columns
        assert 'rank' in ranking_results.columns
        
        # Check that ranking is in descending order of priority score
        assert (ranking_results['priority_score'].diff().dropna() <= 0).all()
        
        # Check that ranks are sequential
        assert list(ranking_results['rank']) == list(range(1, len(ranking_results) + 1))
    
    def test_business_rules_application(self, priority_ranker, sample_customer_scores):
        """Test that business rules are properly applied"""
        business_params = {
            'retention_cost': 50.0,
            'retention_effectiveness': 0.3,
            'min_clv_threshold': 300.0,
            'min_churn_threshold': 0.2
        }
        
        ranking_results = priority_ranker.rank_customers_for_retention(
            sample_customer_scores, business_params=business_params
        )
        
        # Check CLV threshold
        eligible_customers = ranking_results[ranking_results['eligible_for_intervention']]
        assert (eligible_customers['predicted_clv'] >= 300.0).all()
        
        # Check churn threshold
        assert (eligible_customers['churn_probability'] >= 0.2).all()
        
        # Check positive expected value
        assert (eligible_customers['expected_value'] > 0).all()
    
    def test_priority_tier_assignment(self, priority_ranker, sample_customer_scores):
        """Test priority tier assignment logic"""
        ranking_results = priority_ranker.rank_customers_for_retention(sample_customer_scores)
        
        # Check that tiers are assigned
        tiers = ranking_results['priority_tier'].unique()
        assert 'High' in tiers or 'Medium' in tiers or 'Low' in tiers
        
        # High priority customers should have higher priority scores than medium
        if 'High' in tiers and 'Medium' in tiers:
            high_scores = ranking_results[ranking_results['priority_tier'] == 'High']['priority_score']
            medium_scores = ranking_results[ranking_results['priority_tier'] == 'Medium']['priority_score']
            assert high_scores.min() >= medium_scores.max()
    
    def test_cost_benefit_analysis(self, priority_ranker, sample_customer_scores):
        """Test cost-benefit analysis functionality"""
        # First rank customers
        ranking_results = priority_ranker.rank_customers_for_retention(sample_customer_scores)
        
        # Then perform cost-benefit analysis
        analysis = priority_ranker.perform_cost_benefit_analysis(
            ranking_results, campaign_budget=5000.0
        )
        
        assert 'total_customers_targeted' in analysis
        assert 'total_cost' in analysis
        assert 'expected_savings' in analysis
        assert 'expected_roi' in analysis
        assert 'budget_utilization' in analysis
        
        # Budget utilization should be <= 1.0
        assert analysis['budget_utilization'] <= 1.0
        
        # Total cost should not exceed budget
        assert analysis['total_cost'] <= 5000.0
    
    def test_campaign_recommendations(self, priority_ranker, sample_customer_scores):
        """Test campaign recommendation generation"""
        # First rank customers
        ranking_results = priority_ranker.rank_customers_for_retention(sample_customer_scores)
        
        # Generate recommendations
        recommendations = priority_ranker.generate_campaign_recommendations(
            ranking_results, max_recommendations=10
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert 'customer_id' in rec
            assert 'recommendation_type' in rec
            assert 'urgency' in rec
            assert 'suggested_actions' in rec
            assert isinstance(rec['suggested_actions'], list)
    
    def test_empty_eligible_customers(self, priority_ranker):
        """Test handling when no customers are eligible for intervention"""
        # Create data where no customers meet thresholds
        data = pd.DataFrame({
            'customer_id': ['cust_001', 'cust_002'],
            'churn_probability': [0.05, 0.08],  # Below threshold
            'predicted_clv': [50, 80]  # Below threshold
        })
        
        business_params = {
            'retention_cost': 50.0,
            'retention_effectiveness': 0.3,
            'min_clv_threshold': 100.0,
            'min_churn_threshold': 0.1
        }
        
        ranking_results = priority_ranker.rank_customers_for_retention(
            data, business_params=business_params
        )
        
        # Should have no eligible customers
        eligible_count = ranking_results['eligible_for_intervention'].sum()
        assert eligible_count == 0
        
        # Cost-benefit analysis should handle this gracefully
        analysis = priority_ranker.perform_cost_benefit_analysis(ranking_results)
        assert analysis['total_customers_targeted'] == 0
        assert analysis['total_cost'] == 0


class TestIntegration:
    """Integration tests for segmentation and CLV components"""
    
    @pytest.fixture
    def integrated_customer_data(self):
        """Create comprehensive customer data for integration testing"""
        np.random.seed(42)
        n_customers = 200
        
        # Customer features
        data = {
            'customer_id': [f'cust_{i:04d}' for i in range(n_customers)],
            'age': np.random.normal(40, 15, n_customers).clip(18, 80),
            'tenure': np.random.exponential(2, n_customers).clip(0, 10),
            'monthly_charges': np.random.normal(70, 20, n_customers).clip(10, 150),
            'total_charges': np.random.normal(1500, 800, n_customers).clip(0, 5000),
            'frequency': np.random.poisson(5, n_customers),
            'recency': np.random.exponential(30, n_customers).clip(0, 365),
            'T': np.random.exponential(180, n_customers).clip(30, 730),
            'monetary_value': np.random.exponential(150, n_customers) + 50,
            'churn_probability': np.random.beta(2, 5, n_customers),
            'Churn': np.random.binomial(1, 0.3, n_customers)
        }
        
        return pd.DataFrame(data)
    
    def test_end_to_end_workflow(self, integrated_customer_data):
        """Test complete segmentation and prioritization workflow"""
        # Step 1: Customer Segmentation
        segmenter = CustomerSegmenter(random_state=42)
        features = integrated_customer_data[['age', 'tenure', 'monthly_charges', 'total_charges']]
        
        labels, clustering_results = segmenter.perform_kmeans_clustering(features, n_clusters=4)
        segment_profiles = segmenter.profile_segments(integrated_customer_data, labels)
        
        # Step 2: CLV Calculation
        clv_calculator = CLVCalculator()
        rfm_data = integrated_customer_data[['customer_id', 'frequency', 'recency', 'T', 'monetary_value']]
        clv_predictions = clv_calculator.predict_clv(rfm_data)
        
        # Step 3: Merge data for prioritization
        customer_scores = integrated_customer_data[['customer_id', 'churn_probability']].merge(
            clv_predictions[['customer_id', 'predicted_clv']], on='customer_id'
        )
        
        # Step 4: Priority Ranking
        priority_ranker = PriorityRanker()
        ranking_results = priority_ranker.rank_customers_for_retention(customer_scores)
        
        # Verify integration
        assert len(segment_profiles) == 4  # Number of clusters
        assert len(clv_predictions) == len(integrated_customer_data)
        assert len(ranking_results) == len(customer_scores)
        assert 'priority_score' in ranking_results.columns
        assert 'priority_tier' in ranking_results.columns
    
    def test_segment_clv_correlation(self, integrated_customer_data):
        """Test that segments show different CLV patterns"""
        # Perform segmentation
        segmenter = CustomerSegmenter(random_state=42)
        features = integrated_customer_data[['age', 'tenure', 'monthly_charges']]
        labels, _ = segmenter.perform_kmeans_clustering(features, n_clusters=3)
        
        # Calculate CLV
        clv_calculator = CLVCalculator()
        rfm_data = integrated_customer_data[['customer_id', 'frequency', 'recency', 'T', 'monetary_value']]
        clv_predictions = clv_calculator.predict_clv(rfm_data)
        
        # Merge segment and CLV data
        analysis_data = integrated_customer_data[['customer_id']].copy()
        analysis_data['segment'] = labels
        analysis_data = analysis_data.merge(clv_predictions[['customer_id', 'predicted_clv']], on='customer_id')
        
        # Check that segments have different CLV distributions
        segment_clv_means = analysis_data.groupby('segment')['predicted_clv'].mean()
        
        # Should have variation across segments
        assert segment_clv_means.std() > 0
        assert len(segment_clv_means) == 3