"""
Tests for explainability and interpretability module
Tests SHAP value consistency, explanation quality, and business recommendation logic
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.explain import (
    GlobalExplainer, LocalExplainer, BusinessRecommendationEngine, ExplanationValidator,
    GlobalExplanation, LocalExplanation, BusinessRecommendation
)


class TestGlobalExplainer:
    """Test GlobalExplainer functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5,
            n_redundant=2, random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(10)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y, feature_names
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model for testing"""
        X_df, y, _ = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        return model
    
    @pytest.fixture
    def global_explainer(self, trained_model, sample_data):
        """Create GlobalExplainer instance"""
        _, _, feature_names = sample_data
        
        explainer = GlobalExplainer(
            model=trained_model,
            feature_names=feature_names,
            model_name='test_model'
        )
        
        return explainer
    
    def test_global_explainer_initialization(self, global_explainer, sample_data):
        """Test GlobalExplainer initialization"""
        _, _, feature_names = sample_data
        
        assert global_explainer.model is not None
        assert global_explainer.feature_names == feature_names
        assert global_explainer.model_name == 'test_model'
        assert global_explainer.explainer is None
        assert global_explainer.shap_values is None
    
    @patch('src.explain.HAS_SHAP', True)
    @patch('src.explain.shap')
    def test_fit_explainer_tree(self, mock_shap, global_explainer, sample_data):
        """Test fitting SHAP explainer for tree model"""
        X_df, _, _ = sample_data
        
        # Mock SHAP TreeExplainer
        mock_tree_explainer = Mock()
        mock_shap.TreeExplainer.return_value = mock_tree_explainer
        
        global_explainer.fit_explainer(X_df, explainer_type='tree')
        
        assert global_explainer.explainer == mock_tree_explainer
        assert global_explainer.background_data is not None
        mock_shap.TreeExplainer.assert_called_once_with(global_explainer.model)
    
    @patch('src.explain.HAS_SHAP', True)
    @patch('src.explain.shap')
    def test_fit_explainer_auto_detection(self, mock_shap, global_explainer, sample_data):
        """Test automatic explainer type detection"""
        X_df, _, _ = sample_data
        
        mock_tree_explainer = Mock()
        mock_shap.TreeExplainer.return_value = mock_tree_explainer
        
        global_explainer.fit_explainer(X_df, explainer_type='auto')
        
        # Should detect RandomForest as tree model
        mock_shap.TreeExplainer.assert_called_once()
    
    @patch('src.explain.HAS_SHAP', True)
    @patch('src.explain.shap')
    def test_generate_shap_summary(self, mock_shap, global_explainer, sample_data):
        """Test SHAP summary generation"""
        X_df, _, feature_names = sample_data
        
        # Mock explainer and SHAP values
        mock_explainer = Mock()
        mock_shap_values = np.random.randn(50, 10)  # 50 samples, 10 features
        mock_explainer.shap_values.return_value = [mock_shap_values, mock_shap_values]  # Binary classification
        
        global_explainer.explainer = mock_explainer
        
        summary = global_explainer.generate_shap_summary(X_df.head(50))
        
        assert 'mean_abs_shap' in summary
        assert 'feature_ranking' in summary
        assert 'top_10_features' in summary
        assert len(summary['feature_ranking']) == len(feature_names)
        assert summary['samples_explained'] == 50
    
    @patch('src.explain.HAS_SHAP', True)
    @patch('src.explain.shap')
    def test_calculate_feature_interactions(self, mock_shap, global_explainer, sample_data):
        """Test feature interaction calculation"""
        X_df, _, _ = sample_data
        
        # Mock explainer with interaction values
        mock_explainer = Mock()
        mock_interaction_values = np.random.randn(20, 10, 10)  # 20 samples, 10x10 interaction matrix
        mock_explainer.shap_interaction_values.return_value = [mock_interaction_values, mock_interaction_values]
        
        global_explainer.explainer = mock_explainer
        
        interactions = global_explainer.calculate_feature_interactions(X_df.head(20))
        
        assert isinstance(interactions, pd.DataFrame)
        assert 'feature_1' in interactions.columns
        assert 'feature_2' in interactions.columns
        assert 'interaction_strength' in interactions.columns
    
    def test_create_partial_dependence_plots(self, global_explainer, sample_data):
        """Test partial dependence plot creation"""
        X_df, _, feature_names = sample_data
        
        # Test with specific features
        test_features = feature_names[:3]
        pd_data = global_explainer.create_partial_dependence_plots(X_df, features=test_features)
        
        assert len(pd_data) <= len(test_features)
        
        for feature in pd_data:
            assert 'values' in pd_data[feature]
            assert 'average' in pd_data[feature]
            assert 'feature_type' in pd_data[feature]
    
    @patch('src.explain.HAS_SHAP', True)
    @patch('src.explain.shap')
    def test_generate_global_explanation(self, mock_shap, global_explainer, sample_data):
        """Test comprehensive global explanation generation"""
        X_df, _, _ = sample_data
        
        # Mock SHAP components
        mock_explainer = Mock()
        mock_shap_values = np.random.randn(50, 10)
        mock_explainer.shap_values.return_value = [mock_shap_values, mock_shap_values]
        mock_explainer.shap_interaction_values.return_value = [
            np.random.randn(20, 10, 10), np.random.randn(20, 10, 10)
        ]
        
        global_explainer.explainer = mock_explainer
        
        explanation = global_explainer.generate_global_explanation(X_df)
        
        assert isinstance(explanation, GlobalExplanation)
        assert explanation.model_name == 'test_model'
        assert explanation.feature_importance is not None
        assert explanation.explanation_date is not None


class TestLocalExplainer:
    """Test LocalExplainer functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(
            n_samples=100, n_features=8, n_informative=4,
            n_redundant=2, random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(8)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y, feature_names
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model for testing"""
        X_df, y, _ = sample_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_df, y)
        
        return model
    
    @pytest.fixture
    def local_explainer(self, trained_model, sample_data):
        """Create LocalExplainer instance"""
        _, _, feature_names = sample_data
        
        explainer = LocalExplainer(
            model=trained_model,
            feature_names=feature_names
        )
        
        return explainer
    
    def test_local_explainer_initialization(self, local_explainer, sample_data):
        """Test LocalExplainer initialization"""
        _, _, feature_names = sample_data
        
        assert local_explainer.model is not None
        assert local_explainer.feature_names == feature_names
        assert local_explainer.lime_explainer is None
    
    @patch('src.explain.HAS_LIME', True)
    @patch('src.explain.lime.lime_tabular.LimeTabularExplainer')
    def test_fit_lime_explainer(self, mock_lime_explainer, local_explainer, sample_data):
        """Test fitting LIME explainer"""
        X_df, _, _ = sample_data
        
        mock_explainer_instance = Mock()
        mock_lime_explainer.return_value = mock_explainer_instance
        
        local_explainer.fit_lime_explainer(X_df)
        
        assert local_explainer.lime_explainer == mock_explainer_instance
        assert local_explainer.background_data is not None
        mock_lime_explainer.assert_called_once()
    
    def test_explain_prediction_shap_without_global_explainer(self, local_explainer, sample_data):
        """Test SHAP explanation without global explainer"""
        X_df, _, _ = sample_data
        instance = X_df.iloc[0]
        
        explanation = local_explainer.explain_prediction_shap(instance, 'test_instance')
        
        # Should return empty dict when no global explainer
        assert explanation == {}
    
    @patch('src.explain.HAS_LIME', True)
    def test_explain_prediction_lime(self, local_explainer, sample_data):
        """Test LIME explanation generation"""
        X_df, _, _ = sample_data
        
        # Mock LIME explainer
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_0 <= 1.0', 0.5), ('feature_1 > 0.5', -0.3)]
        mock_explanation.score = 0.8
        mock_explanation.local_pred = [0.3, 0.7]
        
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        local_explainer.lime_explainer = mock_lime_explainer
        
        instance = X_df.iloc[0]
        explanation = local_explainer.explain_prediction_lime(instance, 'test_instance')
        
        assert explanation['instance_id'] == 'test_instance'
        assert 'prediction' in explanation
        assert 'lime_contributions' in explanation
        assert 'top_features' in explanation
    
    def test_explain_instance_with_lime_only(self, local_explainer, sample_data):
        """Test comprehensive instance explanation with LIME only"""
        X_df, _, _ = sample_data
        
        # Mock LIME explainer
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_0 <= 1.0', 0.5)]
        mock_explanation.score = 0.8
        mock_explanation.local_pred = [0.3, 0.7]
        
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        local_explainer.lime_explainer = mock_lime_explainer
        
        instance = X_df.iloc[0]
        explanation = local_explainer.explain_instance(instance, 'test_instance', methods=['lime'])
        
        assert isinstance(explanation, LocalExplanation)
        assert explanation.instance_id == 'test_instance'
        assert explanation.prediction is not None
        assert explanation.lime_explanation is not None
    
    def test_validate_explanation_consistency(self, local_explainer, sample_data):
        """Test explanation consistency validation"""
        X_df, _, _ = sample_data
        
        # Mock both explainers to return consistent results
        mock_global_explainer = Mock()
        mock_shap_explainer = Mock()
        mock_shap_explainer.shap_values.return_value = [np.random.randn(5, 8), np.random.randn(5, 8)]
        mock_global_explainer.explainer = mock_shap_explainer
        
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_0 <= 1.0', 0.5), ('feature_1 > 0.5', 0.3)]
        mock_explanation.score = 0.8
        mock_explanation.local_pred = [0.3, 0.7]
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        
        local_explainer.global_explainer = mock_global_explainer
        local_explainer.lime_explainer = mock_lime_explainer
        
        consistency = local_explainer.validate_explanation_consistency(X_df.head(5))
        
        assert 'n_instances' in consistency
        assert 'agreement_rates' in consistency
        assert 'consistency_score' in consistency
        assert consistency['n_instances'] == 5
    
    def test_batch_explain(self, local_explainer, sample_data):
        """Test batch explanation generation"""
        X_df, _, _ = sample_data
        
        # Mock LIME explainer for batch processing
        mock_lime_explainer = Mock()
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [('feature_0 <= 1.0', 0.5)]
        mock_explanation.score = 0.8
        mock_explanation.local_pred = [0.3, 0.7]
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        
        local_explainer.lime_explainer = mock_lime_explainer
        
        explanations = local_explainer.batch_explain(X_df.head(5), methods=['lime'])
        
        assert len(explanations) == 5
        assert all(isinstance(exp, LocalExplanation) for exp in explanations)


class TestBusinessRecommendationEngine:
    """Test BusinessRecommendationEngine functionality"""
    
    @pytest.fixture
    def recommendation_engine(self):
        """Create BusinessRecommendationEngine instance"""
        return BusinessRecommendationEngine()
    
    @pytest.fixture
    def sample_explanation(self):
        """Create sample LocalExplanation for testing"""
        return LocalExplanation(
            instance_id='customer_123',
            prediction=0.75,
            shap_values={'monthly_charges': 0.3, 'tenure': -0.2, 'support_tickets': 0.4},
            lime_explanation=None,
            feature_contributions={'monthly_charges': 0.3, 'tenure': -0.2, 'support_tickets': 0.4},
            top_features=[('support_tickets', 0.4), ('monthly_charges', 0.3), ('tenure', -0.2)],
            explanation_date=datetime.now()
        )
    
    def test_recommendation_engine_initialization(self, recommendation_engine):
        """Test BusinessRecommendationEngine initialization"""
        assert recommendation_engine.business_config is not None
        assert recommendation_engine.action_templates is not None
        assert recommendation_engine.feature_action_mapping is not None
    
    def test_generate_recommendations(self, recommendation_engine, sample_explanation):
        """Test business recommendation generation"""
        recommendation = recommendation_engine.generate_recommendations(sample_explanation)
        
        assert isinstance(recommendation, BusinessRecommendation)
        assert recommendation.customer_id == 'customer_123'
        assert recommendation.churn_probability == 0.75
        assert recommendation.risk_level == 'High'
        assert len(recommendation.primary_actions) > 0
        assert len(recommendation.secondary_actions) > 0
        assert recommendation.priority_score > 0
        assert recommendation.confidence > 0
    
    def test_determine_risk_level(self, recommendation_engine):
        """Test risk level determination"""
        assert recommendation_engine._determine_risk_level(0.8) == 'High'
        assert recommendation_engine._determine_risk_level(0.5) == 'Medium'
        assert recommendation_engine._determine_risk_level(0.2) == 'Low'
    
    def test_generate_primary_actions(self, recommendation_engine):
        """Test primary action generation"""
        top_features = [('monthly_charges', 0.3), ('support_tickets', 0.4)]
        
        actions = recommendation_engine._generate_primary_actions(top_features, None)
        
        assert isinstance(actions, list)
        assert len(actions) <= 3  # Should limit to top 3 actions
    
    def test_calculate_priority_score(self, recommendation_engine, sample_explanation):
        """Test priority score calculation"""
        customer_data = {'lifetime_value': 5000}
        
        score = recommendation_engine._calculate_priority_score(sample_explanation, customer_data)
        
        assert 0 <= score <= 1.0
        assert score > sample_explanation.prediction  # Should be higher due to customer value
    
    def test_batch_generate_recommendations(self, recommendation_engine, sample_explanation):
        """Test batch recommendation generation"""
        explanations = [sample_explanation] * 3
        
        recommendations = recommendation_engine.batch_generate_recommendations(explanations)
        
        assert len(recommendations) == 3
        assert all(isinstance(rec, BusinessRecommendation) for rec in recommendations)
    
    def test_prioritize_customers(self, recommendation_engine, sample_explanation):
        """Test customer prioritization"""
        # Create multiple recommendations with different priority scores
        recommendations = []
        for i, score in enumerate([0.9, 0.7, 0.8]):
            rec = BusinessRecommendation(
                customer_id=f'customer_{i}',
                churn_probability=score,
                risk_level='High',
                primary_actions=['action'],
                secondary_actions=['action'],
                priority_score=score,
                explanation_summary='test',
                confidence=0.8,
                estimated_impact={'expected_value': 100}
            )
            recommendations.append(rec)
        
        prioritized = recommendation_engine.prioritize_customers(recommendations, max_contacts=2)
        
        assert len(prioritized) == 2
        assert prioritized[0].priority_score >= prioritized[1].priority_score
    
    def test_create_campaign_segments(self, recommendation_engine):
        """Test campaign segmentation"""
        # Create recommendations with different risk levels
        recommendations = []
        for risk_level in ['High', 'Medium', 'Low']:
            rec = BusinessRecommendation(
                customer_id=f'customer_{risk_level}',
                churn_probability=0.5,
                risk_level=risk_level,
                primary_actions=['test action'],
                secondary_actions=['test action'],
                priority_score=0.5,
                explanation_summary='test',
                confidence=0.8,
                estimated_impact={'expected_value': 100}
            )
            recommendations.append(rec)
        
        segments = recommendation_engine.create_campaign_segments(recommendations)
        
        assert isinstance(segments, dict)
        assert 'high_risk_immediate' in segments or 'high_risk_engagement' in segments
        assert 'low_risk_monitoring' in segments
    
    def test_generate_campaign_summary(self, recommendation_engine):
        """Test campaign summary generation"""
        # Create sample segments
        segments = {
            'high_risk': [
                BusinessRecommendation(
                    customer_id='customer_1',
                    churn_probability=0.8,
                    risk_level='High',
                    primary_actions=['action'],
                    secondary_actions=['action'],
                    priority_score=0.8,
                    explanation_summary='test',
                    confidence=0.8,
                    estimated_impact={'expected_value': 200}
                )
            ],
            'low_risk': []
        }
        
        summary = recommendation_engine.generate_campaign_summary(segments)
        
        assert 'total_customers' in summary
        assert 'segments' in summary
        assert 'overall_metrics' in summary
        assert summary['total_customers'] == 1


class TestExplanationValidator:
    """Test ExplanationValidator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create ExplanationValidator instance"""
        return ExplanationValidator()
    
    @pytest.fixture
    def sample_explanation(self):
        """Create sample LocalExplanation for testing"""
        return LocalExplanation(
            instance_id='customer_123',
            prediction=0.75,
            shap_values={'monthly_charges': 0.3, 'tenure': -0.2},
            lime_explanation={'lime_contributions': {'monthly_charges': 0.25, 'tenure': -0.15}},
            feature_contributions={'monthly_charges': 0.3, 'tenure': -0.2},
            top_features=[('monthly_charges', 0.3), ('tenure', -0.2)],
            explanation_date=datetime.now()
        )
    
    @pytest.fixture
    def sample_recommendation(self):
        """Create sample BusinessRecommendation for testing"""
        return BusinessRecommendation(
            customer_id='customer_123',
            churn_probability=0.75,
            risk_level='High',
            primary_actions=['Offer immediate retention discount'],
            secondary_actions=['Schedule customer success call'],
            priority_score=0.8,
            explanation_summary='High risk customer',
            confidence=0.8,
            estimated_impact={'expected_value': 200}
        )
    
    def test_validator_initialization(self, validator):
        """Test ExplanationValidator initialization"""
        assert validator.domain_rules is not None
        assert 'expected_relationships' in validator.domain_rules
    
    def test_validate_explanation_quality(self, validator, sample_explanation):
        """Test explanation quality validation"""
        validation = validator.validate_explanation_quality(sample_explanation)
        
        assert 'instance_id' in validation
        assert 'overall_quality' in validation
        assert 'issues' in validation
        assert 'warnings' in validation
        assert 'quality_score' in validation
        assert 'checks_performed' in validation
        
        assert validation['instance_id'] == 'customer_123'
        assert validation['overall_quality'] in ['good', 'fair', 'poor']
        assert 0 <= validation['quality_score'] <= 1.0
    
    def test_validate_business_alignment(self, validator, sample_recommendation):
        """Test business recommendation alignment validation"""
        validation = validator.validate_business_alignment(sample_recommendation)
        
        assert 'customer_id' in validation
        assert 'business_alignment' in validation
        assert 'issues' in validation
        assert 'suggestions' in validation
        assert 'alignment_score' in validation
        
        assert validation['customer_id'] == 'customer_123'
        assert validation['business_alignment'] in ['good', 'fair', 'poor']
        assert 0 <= validation['alignment_score'] <= 1.0
    
    def test_check_feature_consistency(self, validator, sample_explanation):
        """Test feature consistency checking"""
        result = validator._check_feature_consistency(sample_explanation)
        
        assert 'passed' in result
        assert 'issues' in result
        assert isinstance(result['passed'], bool)
        assert isinstance(result['issues'], list)
    
    def test_check_domain_alignment(self, validator, sample_explanation):
        """Test domain knowledge alignment checking"""
        result = validator._check_domain_alignment(sample_explanation)
        
        assert 'passed' in result
        assert 'warnings' in result
        assert isinstance(result['passed'], bool)
        assert isinstance(result['warnings'], list)
    
    def test_check_shap_additivity(self, validator, sample_explanation):
        """Test SHAP additivity property checking"""
        result = validator._check_shap_additivity(sample_explanation)
        
        assert 'passed' in result
        assert 'issues' in result
        assert isinstance(result['passed'], bool)
        assert isinstance(result['issues'], list)
    
    def test_check_risk_action_alignment(self, validator, sample_recommendation):
        """Test risk-action alignment checking"""
        result = validator._check_risk_action_alignment(sample_recommendation)
        
        assert 'passed' in result
        assert 'issues' in result
        assert isinstance(result['passed'], bool)
        assert isinstance(result['issues'], list)
    
    def test_check_cost_effectiveness(self, validator, sample_recommendation):
        """Test cost-effectiveness checking"""
        result = validator._check_cost_effectiveness(sample_recommendation)
        
        assert 'passed' in result
        assert 'suggestions' in result
        assert isinstance(result['passed'], bool)
        assert isinstance(result['suggestions'], list)


class TestIntegration:
    """Integration tests for explainability components"""
    
    @pytest.fixture
    def complete_setup(self):
        """Create complete explainability setup"""
        # Create sample data
        X, y = make_classification(
            n_samples=100, n_features=6, n_informative=4,
            n_redundant=1, random_state=42
        )
        
        feature_names = ['monthly_charges', 'tenure', 'support_tickets', 
                        'contract', 'payment_method', 'internet_service']
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        # Create explainers
        global_explainer = GlobalExplainer(model, feature_names, 'test_model')
        local_explainer = LocalExplainer(model, feature_names, global_explainer)
        
        # Create business components
        recommendation_engine = BusinessRecommendationEngine()
        validator = ExplanationValidator()
        
        return {
            'data': (X_df, y),
            'model': model,
            'global_explainer': global_explainer,
            'local_explainer': local_explainer,
            'recommendation_engine': recommendation_engine,
            'validator': validator
        }
    
    def test_end_to_end_explanation_workflow(self, complete_setup):
        """Test complete explanation workflow"""
        setup = complete_setup
        X_df, y = setup['data']
        
        # Test instance
        test_instance = X_df.iloc[0]
        
        # Generate local explanation (without SHAP/LIME due to mocking complexity)
        explanation = LocalExplanation(
            instance_id='test_customer',
            prediction=0.75,
            shap_values=None,
            lime_explanation=None,
            feature_contributions={'monthly_charges': 0.3, 'tenure': -0.2},
            top_features=[('monthly_charges', 0.3), ('tenure', -0.2)],
            explanation_date=datetime.now()
        )
        
        # Generate business recommendation
        recommendation = setup['recommendation_engine'].generate_recommendations(explanation)
        
        # Validate explanation and recommendation
        explanation_validation = setup['validator'].validate_explanation_quality(explanation)
        business_validation = setup['validator'].validate_business_alignment(recommendation)
        
        # Assertions
        assert isinstance(recommendation, BusinessRecommendation)
        assert recommendation.customer_id == 'test_customer'
        assert recommendation.risk_level in ['High', 'Medium', 'Low']
        
        assert explanation_validation['overall_quality'] in ['good', 'fair', 'poor']
        assert business_validation['business_alignment'] in ['good', 'fair', 'poor']
    
    def test_batch_processing_workflow(self, complete_setup):
        """Test batch processing of explanations and recommendations"""
        setup = complete_setup
        X_df, y = setup['data']
        
        # Create multiple explanations
        explanations = []
        for i in range(5):
            explanation = LocalExplanation(
                instance_id=f'customer_{i}',
                prediction=0.5 + i * 0.1,
                shap_values=None,
                lime_explanation=None,
                feature_contributions={'monthly_charges': 0.2 + i * 0.1},
                top_features=[('monthly_charges', 0.2 + i * 0.1)],
                explanation_date=datetime.now()
            )
            explanations.append(explanation)
        
        # Generate batch recommendations
        recommendations = setup['recommendation_engine'].batch_generate_recommendations(explanations)
        
        # Create campaign segments
        segments = setup['recommendation_engine'].create_campaign_segments(recommendations)
        
        # Generate campaign summary
        summary = setup['recommendation_engine'].generate_campaign_summary(segments)
        
        # Assertions
        assert len(recommendations) == 5
        assert isinstance(segments, dict)
        assert summary['total_customers'] == 5
        assert 'overall_metrics' in summary


if __name__ == '__main__':
    pytest.main([__file__])