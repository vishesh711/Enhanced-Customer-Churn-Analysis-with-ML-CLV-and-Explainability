"""
Customer Churn Analytics Dashboard
Interactive Streamlit application for churn prediction insights and customer analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src and root to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

from data_prep import DataLoader, DataCleaner
from train import ModelTrainer
from evaluate import BusinessMetrics, SegmentAnalyzer
from explain import GlobalExplainer, LocalExplainer
from segment import CustomerSegmenter, CLVCalculator, PriorityRanker
from config import config

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

class ChurnDashboard:
    """Main dashboard class for customer churn analytics"""
    
    def __init__(self):
        """Initialize dashboard with session state management"""
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'model_metadata' not in st.session_state:
            st.session_state.model_metadata = {}
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'business_metrics' not in st.session_state:
            st.session_state.business_metrics = None
        if 'segments' not in st.session_state:
            st.session_state.segments = None
        if 'clv_data' not in st.session_state:
            st.session_state.clv_data = None
    
    def render_sidebar(self):
        """Render sidebar with navigation and data upload"""
        st.sidebar.markdown("## 🎛️ Navigation")
        
        # Page selection
        pages = [
            "📊 Overview Dashboard",
            "⚠️ Customer Risk Analysis", 
            "👤 Customer Details",
            "🎯 Segment Analysis & CLV",
            "📁 Data Management"
        ]
        
        selected_page = st.sidebar.selectbox("Select Page", pages)
        
        st.sidebar.markdown("---")
        
        # Data upload section
        st.sidebar.markdown("## 📁 Data Upload")
        self.render_data_upload_section()
        
        st.sidebar.markdown("---")
        
        # Model management section
        st.sidebar.markdown("## 🤖 Model Management")
        self.render_model_management_section()
        
        return selected_page
    
    def render_data_upload_section(self):
        """Render data upload and processing section"""
        # Dataset selection
        dataset_option = st.sidebar.selectbox(
            "Choose Data Source",
            ["Upload CSV File", "Load Telco Dataset", "Load Olist Dataset"]
        )
        
        if dataset_option == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload a CSV file with customer data"
            )
            
            if uploaded_file is not None:
                if st.sidebar.button("Process Uploaded Data"):
                    self.process_uploaded_data(uploaded_file)
        
        elif dataset_option == "Load Telco Dataset":
            if st.sidebar.button("Download & Load Telco Data"):
                self.load_telco_dataset()
        
        elif dataset_option == "Load Olist Dataset":
            if st.sidebar.button("Download & Load Olist Data"):
                self.load_olist_dataset()
        
        # Data status
        if st.session_state.data_loaded:
            st.sidebar.success(f"✅ Data loaded: {len(st.session_state.current_data)} rows")
        else:
            st.sidebar.info("ℹ️ No data loaded")
    
    def render_model_management_section(self):
        """Render model management section"""
        # Model loading
        model_dir = st.sidebar.text_input(
            "Model Directory Path",
            value="models/",
            help="Path to directory containing trained models"
        )
        
        if st.sidebar.button("Load Models"):
            self.load_trained_models(model_dir)
        
        # Model status
        if st.session_state.models_loaded:
            st.sidebar.success(f"✅ Models loaded: {len(st.session_state.trained_models)}")
            
            # Model selection
            if st.session_state.trained_models:
                selected_model = st.sidebar.selectbox(
                    "Select Active Model",
                    list(st.session_state.trained_models.keys())
                )
                st.session_state.selected_model = selected_model
        else:
            st.sidebar.info("ℹ️ No models loaded")
    
    def process_uploaded_data(self, uploaded_file):
        """Process uploaded CSV data"""
        try:
            with st.spinner("Processing uploaded data..."):
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Basic data cleaning
                cleaner = DataCleaner()
                
                # Detect dataset type and apply appropriate cleaning
                if 'customerID' in df.columns and 'Churn' in df.columns:
                    # Telco-like dataset
                    df_clean = cleaner.clean_telco_data(df)
                else:
                    # Generic dataset - apply basic cleaning
                    df_clean = df.copy()
                    # Handle missing values
                    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                    categorical_cols = df_clean.select_dtypes(include=['object']).columns
                    
                    for col in numeric_cols:
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    
                    for col in categorical_cols:
                        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
                
                st.session_state.current_data = df_clean
                st.session_state.data_loaded = True
                
                st.sidebar.success("Data processed successfully!")
                
        except Exception as e:
            st.sidebar.error(f"Error processing data: {str(e)}")
    
    def load_telco_dataset(self):
        """Load and process Telco dataset"""
        try:
            with st.spinner("Downloading and processing Telco dataset..."):
                loader = DataLoader()
                df = loader.load_telco_data()
                
                cleaner = DataCleaner()
                df_clean = cleaner.clean_telco_data(df)
                
                st.session_state.current_data = df_clean
                st.session_state.data_loaded = True
                
                st.sidebar.success("Telco dataset loaded successfully!")
                
        except Exception as e:
            st.sidebar.error(f"Error loading Telco dataset: {str(e)}")
    
    def load_olist_dataset(self):
        """Load and process Olist dataset"""
        try:
            with st.spinner("Downloading and processing Olist dataset..."):
                loader = DataLoader()
                df = loader.load_olist_data()
                
                cleaner = DataCleaner()
                df_clean = cleaner.clean_olist_data(df)
                
                st.session_state.current_data = df_clean
                st.session_state.data_loaded = True
                
                st.sidebar.success("Olist dataset loaded successfully!")
                
        except Exception as e:
            st.sidebar.error(f"Error loading Olist dataset: {str(e)}")
    
    def load_trained_models(self, model_dir):
        """Load trained models from directory"""
        try:
            model_path = Path(model_dir)
            if not model_path.exists():
                st.sidebar.error("Model directory does not exist")
                return
            
            with st.spinner("Loading trained models..."):
                trainer = ModelTrainer.load_models(model_path)
                
                st.session_state.trained_models = trainer.trained_models
                st.session_state.model_metadata = trainer.model_metadata
                st.session_state.models_loaded = True
                
                st.sidebar.success("Models loaded successfully!")
                
        except Exception as e:
            st.sidebar.error(f"Error loading models: {str(e)}")
    
    def run(self):
        """Main dashboard execution"""
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render main content based on selected page
        if selected_page == "📊 Overview Dashboard":
            self.render_overview_page()
        elif selected_page == "⚠️ Customer Risk Analysis":
            self.render_risk_analysis_page()
        elif selected_page == "👤 Customer Details":
            self.render_customer_details_page()
        elif selected_page == "🎯 Segment Analysis & CLV":
            self.render_segment_analysis_page()
        elif selected_page == "📁 Data Management":
            self.render_data_management_page()
    
    def render_overview_page(self):
        """Render overview dashboard page"""
        st.markdown('<h1 class="main-header">Customer Churn Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first using the sidebar")
            return
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ Please load trained models first using the sidebar")
            return
        
        # Generate predictions if not already done
        if st.session_state.predictions is None:
            self.generate_predictions()
        
        # Display KPIs (will show N/A if no predictions available)
        self.display_kpi_metrics()
        
        # Show additional guidance if no predictions are available
        if st.session_state.predictions is None:
            st.info("💡 **Getting Started:**\n"
                   "1. Load data using the sidebar (Telco dataset recommended for demo)\n"
                   "2. Load trained models from the models/ directory\n"
                   "3. Predictions will be generated automatically")
            return
        
        st.markdown("---")
        
        # Display model performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Performance Comparison")
            self.display_model_comparison()
        
        with col2:
            st.subheader("📊 Lift Chart")
            self.display_lift_chart()
        
        st.markdown("---")
        
        # Display gains chart and business metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Gains Chart")
            self.display_gains_chart()
        
        with col2:
            st.subheader("💰 Business Impact")
            self.display_business_impact()
    
    def render_risk_analysis_page(self):
        """Render customer risk analysis page"""
        st.markdown('<h1 class="main-header">Customer Risk Analysis</h1>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded or not st.session_state.models_loaded:
            st.warning("⚠️ Please load data and models first")
            return
        
        # Generate predictions if not already done
        if st.session_state.predictions is None:
            self.generate_predictions()
        
        # Check if predictions are available
        if st.session_state.predictions is None:
            st.info("💡 **Getting Started with Risk Analysis:**\n"
                   "1. Load data using the sidebar (Telco dataset recommended for demo)\n"
                   "2. Load trained models from the models/ directory\n"
                   "3. Risk analysis will be available once predictions are generated")
            return
        
        # Risk threshold selection
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            risk_threshold = st.slider(
                "Risk Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Customers above this threshold are considered high-risk"
            )
        
        with col2:
            top_n = st.number_input(
                "Show Top N Customers",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
        
        # Filter high-risk customers
        high_risk_customers = self.get_high_risk_customers(risk_threshold, top_n)
        
        # Display high-risk customer table
        st.subheader(f"⚠️ High-Risk Customers (Threshold: {risk_threshold:.2f})")
        self.display_high_risk_table(high_risk_customers)
        
        st.markdown("---")
        
        # Batch scoring section
        st.subheader("📊 Batch Scoring")
        self.render_batch_scoring_section()
    
    def render_customer_details_page(self):
        """Render individual customer detail pages"""
        st.markdown('<h1 class="main-header">Customer Details</h1>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded or not st.session_state.models_loaded:
            st.warning("⚠️ Please load data and models first")
            return
        
        # Check if data is loaded
        if st.session_state.current_data is None:
            st.info("💡 **Getting Started with Customer Details:**\n"
                   "1. Load data using the sidebar (Telco dataset recommended for demo)\n"
                   "2. Load trained models from the models/ directory\n"
                   "3. Customer details will be available once data and predictions are loaded")
            return
        
        # Customer selection
        if 'customerID' in st.session_state.current_data.columns:
            customer_ids = st.session_state.current_data['customerID'].unique()
            selected_customer = st.selectbox(
                "Select Customer ID",
                customer_ids,
                help="Choose a customer to view detailed analysis"
            )
        else:
            # Use index if no customer ID column
            selected_customer = st.number_input(
                "Select Customer Index",
                min_value=0,
                max_value=len(st.session_state.current_data) - 1,
                value=0
            )
        
        if st.button("Analyze Customer"):
            self.display_customer_profile(selected_customer)
    
    def render_segment_analysis_page(self):
        """Render segment analysis and CLV visualization page"""
        st.markdown('<h1 class="main-header">Segment Analysis & Customer Lifetime Value</h1>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
            return
        
        # Perform segmentation if not already done
        if st.session_state.segments is None:
            self.perform_customer_segmentation()
        
        # Display segment analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Customer Segments")
            self.display_segment_profiles()
        
        with col2:
            st.subheader("💰 CLV Analysis")
            self.display_clv_analysis()
        
        st.markdown("---")
        
        # Campaign targeting section
        st.subheader("📢 Campaign Targeting")
        self.render_campaign_targeting_section()
    
    def render_data_management_page(self):
        """Render data management and exploration page"""
        st.markdown('<h1 class="main-header">Data Management</h1>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first using the sidebar")
            return
        
        df = st.session_state.current_data
        
        # Data overview
        st.subheader("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(df))
        
        with col2:
            if 'Churn' in df.columns:
                churn_rate = df['Churn'].mean()
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            else:
                st.metric("Columns", len(df.columns))
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Data quality report
        st.subheader("📋 Data Quality Report")
        self.display_data_quality_report(df)


    def generate_predictions(self):
        """Generate predictions using the selected model"""
        try:
            if 'selected_model' not in st.session_state or st.session_state.selected_model is None:
                st.warning("⚠️ No model selected. Please load models first using the sidebar.")
                return
            
            model_name = st.session_state.selected_model
            if model_name not in st.session_state.trained_models:
                st.error(f"Model '{model_name}' not found in loaded models")
                return
                
            model = st.session_state.trained_models[model_name]
            
            # Prepare data for prediction
            df = st.session_state.current_data.copy()
            
            # Remove target column if present
            feature_cols = [col for col in df.columns if col not in ['Churn', 'customerID', 'customer_id']]
            X = df[feature_cols]
            
            # Handle categorical variables (simple encoding for demo)
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.Categorical(X[col]).codes
            
            # Generate predictions
            predictions = model.predict_proba(X)[:, 1]
            
            # Create predictions dataframe
            pred_df = df.copy()
            pred_df['churn_probability'] = predictions
            pred_df['risk_level'] = pd.cut(
                predictions, 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low', 'Medium', 'High']
            )
            
            st.session_state.predictions = pred_df
            
            # Calculate business metrics
            if 'Churn' in df.columns:
                y_true = df['Churn'].values
                business_metrics = BusinessMetrics()
                
                # Calculate lift table
                lift_table = business_metrics.calculate_lift_table(y_true, predictions)
                
                # Calculate expected savings
                savings = business_metrics.calculate_expected_savings(
                    y_true, predictions, threshold=0.5
                )
                
                st.session_state.business_metrics = {
                    'lift_table': lift_table,
                    'savings': savings
                }
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
    
    def display_kpi_metrics(self):
        """Display key performance indicators"""
        df = st.session_state.predictions
        
        # Handle case where predictions haven't been generated yet
        if df is None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", "N/A")
            with col2:
                st.metric("High Risk Customers", "N/A")
            with col3:
                st.metric("Average Risk Score", "N/A")
            with col4:
                st.metric("Expected Savings", "N/A")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            high_risk = len(df[df['churn_probability'] > 0.7])
            high_risk_pct = (high_risk / total_customers) * 100
            st.metric("High Risk Customers", f"{high_risk:,}", f"{high_risk_pct:.1f}%")
        
        with col3:
            avg_risk = df['churn_probability'].mean()
            st.metric("Average Risk Score", f"{avg_risk:.3f}")
        
        with col4:
            if st.session_state.business_metrics:
                expected_savings = st.session_state.business_metrics['savings']['net_savings']
                st.metric("Expected Savings", f"${expected_savings:,.0f}")
            else:
                st.metric("Model Accuracy", "N/A")
    
    def display_model_comparison(self):
        """Display model performance comparison"""
        if not st.session_state.model_metadata:
            st.info("📊 Model performance comparison will appear here once models are loaded")
            return
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metadata in st.session_state.model_metadata.items():
            metrics = metadata.performance_metrics
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': metrics.get('roc_auc', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) > 0:
            # Create bar chart
            fig = px.bar(
                comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                x='Model',
                y='Score',
                color='Metric',
                title="Model Performance Comparison",
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model performance data available")
    
    def display_lift_chart(self):
        """Display lift chart"""
        if not st.session_state.business_metrics:
            st.info("📈 Lift chart will appear here once predictions are generated")
            return
        
        lift_table = st.session_state.business_metrics['lift_table']
        
        # Create lift chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=lift_table['decile'],
            y=lift_table['lift'],
            name='Lift',
            marker_color='lightblue'
        ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Baseline (Lift = 1)")
        
        fig.update_layout(
            title="Lift Chart by Decile",
            xaxis_title="Decile",
            yaxis_title="Lift",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_gains_chart(self):
        """Display gains chart"""
        if not st.session_state.business_metrics:
            st.info("📈 Gains chart will appear here once predictions are generated")
            return
        
        df = st.session_state.predictions
        if 'Churn' not in df.columns:
            st.info("Gains chart requires actual churn labels")
            return
        
        # Calculate gains
        y_true = df['Churn'].values
        y_prob = df['churn_probability'].values
        
        # Sort by probability
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate cumulative gains
        total_positives = y_true.sum()
        cumulative_positives = np.cumsum(y_true_sorted)
        
        population_pct = np.arange(1, len(y_true) + 1) / len(y_true) * 100
        gains_pct = cumulative_positives / total_positives * 100
        
        # Create gains chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=population_pct,
            y=gains_pct,
            mode='lines',
            name='Model',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Random',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Gains Chart",
            xaxis_title="% of Population Contacted",
            yaxis_title="% of Churners Captured",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_business_impact(self):
        """Display business impact metrics"""
        if not st.session_state.business_metrics:
            st.info("💰 Business impact analysis will appear here once predictions are generated")
            return
        
        savings = st.session_state.business_metrics['savings']
        
        # Display key business metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Net Savings", f"${savings['net_savings']:,.0f}")
            st.metric("ROI", f"{savings['roi_percentage']:.1f}%")
            st.metric("Customers Contacted", f"{savings['customers_contacted']:,}")
        
        with col2:
            st.metric("Customers Retained", f"{savings['customers_retained']:,}")
            st.metric("Contact Cost", f"${savings['total_contact_costs']:,.0f}")
            st.metric("Retention Revenue", f"${savings['retention_revenue']:,.0f}")
        
        # Business impact visualization
        categories = ['Contact Costs', 'Retention Revenue', 'Net Savings']
        values = [
            -savings['total_contact_costs'],
            savings['retention_revenue'],
            savings['net_savings']
        ]
        colors = ['red', 'green', 'blue']
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Business Impact Analysis",
            yaxis_title="Amount ($)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_high_risk_customers(self, threshold, top_n):
        """Get high-risk customers above threshold"""
        df = st.session_state.predictions
        
        # Handle case where predictions haven't been generated yet
        if df is None:
            return pd.DataFrame()  # Return empty dataframe
        
        # Filter high-risk customers
        high_risk = df[df['churn_probability'] >= threshold].copy()
        
        # Sort by probability (descending) and take top N
        high_risk = high_risk.sort_values('churn_probability', ascending=False).head(top_n)
        
        return high_risk
    
    def display_high_risk_table(self, high_risk_df):
        """Display high-risk customers table"""
        if len(high_risk_df) == 0:
            st.info("No high-risk customers found with the current threshold")
            return
        
        # Select relevant columns for display
        display_cols = ['churn_probability', 'risk_level']
        
        # Add customer ID if available
        if 'customerID' in high_risk_df.columns:
            display_cols = ['customerID'] + display_cols
        
        # Add a few key features
        numeric_cols = high_risk_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Churn', 'churn_probability']][:5]
        display_cols.extend(feature_cols)
        
        # Filter columns that exist
        display_cols = [col for col in display_cols if col in high_risk_df.columns]
        
        # Format the dataframe for display
        display_df = high_risk_df[display_cols].copy()
        
        if 'churn_probability' in display_df.columns:
            display_df['churn_probability'] = display_df['churn_probability'].round(3)
        
        # Add search functionality
        search_term = st.text_input("🔍 Search customers", placeholder="Enter customer ID or value...")
        
        if search_term:
            # Simple search across all columns
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
        
        # Display table with styling
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download High-Risk Customers",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )
    
    def render_batch_scoring_section(self):
        """Render batch scoring interface"""
        st.markdown("### Upload New Data for Scoring")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file for batch scoring",
            type=['csv'],
            help="Upload customer data to get churn predictions"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                new_data = pd.read_csv(uploaded_file)
                
                st.write("**Data Preview:**")
                st.dataframe(new_data.head(), use_container_width=True)
                
                if st.button("Generate Predictions"):
                    with st.spinner("Generating predictions..."):
                        # Process new data similar to training data
                        processed_data = self.process_new_data_for_scoring(new_data)
                        
                        # Generate predictions
                        model_name = st.session_state.selected_model
                        model = st.session_state.trained_models[model_name]
                        
                        predictions = model.predict_proba(processed_data)[:, 1]
                        
                        # Add predictions to dataframe
                        result_df = new_data.copy()
                        result_df['churn_probability'] = predictions
                        result_df['risk_level'] = pd.cut(
                            predictions,
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        st.success("Predictions generated successfully!")
                        
                        # Display results
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing batch data: {str(e)}")
    
    def process_new_data_for_scoring(self, new_data):
        """Process new data to match training data format"""
        # This is a simplified version - in practice, you'd want to use
        # the same preprocessing pipeline used during training
        
        processed_data = new_data.copy()
        
        # Remove non-feature columns
        feature_cols = [col for col in processed_data.columns 
                       if col not in ['customerID', 'customer_id', 'Churn']]
        processed_data = processed_data[feature_cols]
        
        # Handle categorical variables
        for col in processed_data.select_dtypes(include=['object']).columns:
            processed_data[col] = pd.Categorical(processed_data[col]).codes
        
        # Handle missing values
        processed_data.fillna(processed_data.median(), inplace=True)
        
        return processed_data
    
    def display_customer_profile(self, customer_id):
        """Display detailed customer profile and explanations"""
        df = st.session_state.predictions
        
        # Handle case where predictions haven't been generated yet
        if df is None:
            st.warning("⚠️ No predictions available. Please generate predictions first by loading data and models.")
            return
        
        # Get customer data
        if 'customerID' in df.columns:
            customer_data = df[df['customerID'] == customer_id]
        else:
            customer_data = df.iloc[[customer_id]]
        
        if len(customer_data) == 0:
            st.error("Customer not found")
            return
        
        customer = customer_data.iloc[0]
        
        # Display customer overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Customer ID", customer_id)
            st.metric("Churn Probability", f"{customer['churn_probability']:.3f}")
        
        with col2:
            st.metric("Risk Level", customer['risk_level'])
            if 'Churn' in customer:
                actual_churn = "Yes" if customer['Churn'] == 1 else "No"
                st.metric("Actual Churn", actual_churn)
        
        with col3:
            # Add CLV if available
            if st.session_state.clv_data is not None:
                clv_info = st.session_state.clv_data[
                    st.session_state.clv_data['customer_id'] == str(customer_id)
                ]
                if len(clv_info) > 0:
                    clv_value = clv_info.iloc[0]['predicted_clv']
                    st.metric("Predicted CLV", f"${clv_value:.0f}")
        
        st.markdown("---")
        
        # Feature contributions (simplified SHAP-like explanation)
        st.subheader("🔍 Feature Contributions")
        self.display_feature_contributions(customer)
        
        st.markdown("---")
        
        # Recommended actions
        st.subheader("💡 Recommended Actions")
        self.display_recommended_actions(customer)
    
    def display_feature_contributions(self, customer):
        """Display feature contributions for the customer"""
        # This is a simplified version - in practice, you'd use SHAP values
        
        # Check if predictions are available
        if st.session_state.predictions is None:
            st.info("Feature contributions will be available once predictions are generated")
            return
        
        # Get numeric features
        numeric_features = []
        for col in customer.index:
            if col not in ['customerID', 'customer_id', 'Churn', 'churn_probability', 'risk_level']:
                if pd.api.types.is_numeric_dtype(type(customer[col])):
                    numeric_features.append(col)
        
        if len(numeric_features) == 0:
            st.info("No numeric features available for contribution analysis")
            return
        
        # Calculate simple feature importance based on correlation with prediction
        # (This is a placeholder - real implementation would use SHAP)
        df = st.session_state.predictions
        correlations = df[numeric_features].corrwith(df['churn_probability']).abs()
        
        # Get top contributing features
        top_features = correlations.nlargest(10)
        
        # Create contribution chart
        fig = go.Figure(go.Bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Top Feature Contributions (Correlation with Churn Probability)",
            xaxis_title="Absolute Correlation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature values
        st.write("**Customer Feature Values:**")
        feature_df = pd.DataFrame({
            'Feature': numeric_features[:10],
            'Value': [customer[col] for col in numeric_features[:10]]
        })
        st.dataframe(feature_df, use_container_width=True)
    
    def display_recommended_actions(self, customer):
        """Display recommended retention actions"""
        risk_level = customer['risk_level']
        churn_prob = customer['churn_probability']
        
        # Generate recommendations based on risk level
        if risk_level == 'High':
            recommendations = [
                "🚨 **Immediate Action Required**",
                "📞 Contact customer within 24 hours",
                "💰 Offer retention discount or upgrade",
                "🎯 Assign to senior customer success manager",
                "📊 Conduct satisfaction survey"
            ]
        elif risk_level == 'Medium':
            recommendations = [
                "⚠️ **Monitor Closely**",
                "📧 Send personalized engagement email",
                "🎁 Offer loyalty rewards or benefits",
                "📱 Increase touchpoint frequency",
                "🔍 Analyze usage patterns for intervention opportunities"
            ]
        else:
            recommendations = [
                "✅ **Low Risk - Maintain Engagement**",
                "📬 Include in regular newsletter campaigns",
                "⭐ Encourage referrals and reviews",
                "🆙 Suggest product upgrades when appropriate",
                "📈 Monitor for any changes in behavior"
            ]
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Add specific recommendations based on features (simplified)
        st.markdown("**Specific Recommendations:**")
        
        # This would be more sophisticated in practice
        if churn_prob > 0.8:
            st.markdown("- Consider offering a significant discount or free trial extension")
        elif churn_prob > 0.6:
            st.markdown("- Schedule a check-in call to understand pain points")
        else:
            st.markdown("- Focus on upselling and cross-selling opportunities")
    
    def perform_customer_segmentation(self):
        """Perform customer segmentation analysis"""
        try:
            df = st.session_state.current_data
            
            # Select numeric features for segmentation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['Churn']]
            
            if len(feature_cols) < 2:
                st.error("Need at least 2 numeric features for segmentation")
                return
            
            X = df[feature_cols].fillna(df[feature_cols].median())
            
            # Perform segmentation
            segmenter = CustomerSegmenter()
            labels, results = segmenter.perform_kmeans_clustering(X, n_clusters=4)
            
            # Create segment profiles
            segment_profiles = segmenter.profile_segments(df, labels)
            
            st.session_state.segments = {
                'labels': labels,
                'profiles': segment_profiles,
                'results': results
            }
            
        except Exception as e:
            st.error(f"Error performing segmentation: {str(e)}")
    
    def display_segment_profiles(self):
        """Display customer segment profiles"""
        if st.session_state.segments is None:
            st.info("Performing customer segmentation...")
            self.perform_customer_segmentation()
            return
        
        profiles = st.session_state.segments['profiles']
        
        # Display segment summary
        st.dataframe(profiles, use_container_width=True)
        
        # Segment distribution chart
        fig = px.pie(
            profiles,
            values='Size',
            names='Segment',
            title="Customer Segment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_clv_analysis(self):
        """Display CLV analysis"""
        # Simplified CLV calculation for demo
        if st.session_state.clv_data is None:
            st.info("Calculating CLV estimates...")
            self.calculate_clv_estimates()
        
        if st.session_state.clv_data is not None:
            clv_data = st.session_state.clv_data
            
            # CLV distribution
            fig = px.histogram(
                clv_data,
                x='predicted_clv',
                title="Customer Lifetime Value Distribution",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # CLV summary stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_clv = clv_data['predicted_clv'].mean()
                st.metric("Average CLV", f"${avg_clv:.0f}")
            
            with col2:
                median_clv = clv_data['predicted_clv'].median()
                st.metric("Median CLV", f"${median_clv:.0f}")
            
            with col3:
                total_clv = clv_data['predicted_clv'].sum()
                st.metric("Total CLV", f"${total_clv:.0f}")
    
    def calculate_clv_estimates(self):
        """Calculate simplified CLV estimates"""
        try:
            df = st.session_state.current_data
            
            # Simple CLV calculation based on available features
            # This is a placeholder - real implementation would use proper CLV models
            
            clv_data = []
            
            for idx, row in df.iterrows():
                # Simple heuristic CLV calculation
                if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
                    monthly_charges = row.get('MonthlyCharges', 50)
                    tenure = row.get('tenure', 12)
                    
                    # Simple CLV = Monthly charges * expected remaining tenure
                    expected_remaining_tenure = max(24 - tenure, 6)  # At least 6 months
                    predicted_clv = monthly_charges * expected_remaining_tenure
                else:
                    # Fallback random CLV for demo
                    predicted_clv = np.random.normal(1000, 300)
                
                customer_id = row.get('customerID', f'customer_{idx}')
                
                clv_data.append({
                    'customer_id': str(customer_id),
                    'predicted_clv': max(predicted_clv, 0),  # Ensure positive
                    'confidence_interval_lower': predicted_clv * 0.8,
                    'confidence_interval_upper': predicted_clv * 1.2
                })
            
            st.session_state.clv_data = pd.DataFrame(clv_data)
            
        except Exception as e:
            st.error(f"Error calculating CLV: {str(e)}")
    
    def render_campaign_targeting_section(self):
        """Render campaign targeting and ROI analysis"""
        st.markdown("### 📢 Campaign Targeting Tool")
        
        if st.session_state.predictions is None:
            st.info("📢 Campaign targeting tools will appear here once predictions are generated")
            return
        
        # Campaign parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            campaign_budget = st.number_input(
                "Campaign Budget ($)",
                min_value=1000,
                max_value=1000000,
                value=50000,
                step=1000
            )
        
        with col2:
            cost_per_contact = st.number_input(
                "Cost per Contact ($)",
                min_value=1,
                max_value=500,
                value=50,
                step=5
            )
        
        with col3:
            retention_value = st.number_input(
                "Retention Value ($)",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        # Calculate optimal targeting
        max_contacts = int(campaign_budget / cost_per_contact)
        
        df = st.session_state.predictions
        
        # Sort by churn probability and select top customers
        top_customers = df.nlargest(max_contacts, 'churn_probability')
        
        # Calculate campaign metrics
        total_contacts = len(top_customers)
        total_cost = total_contacts * cost_per_contact
        
        # Estimate retention (simplified)
        avg_churn_prob = top_customers['churn_probability'].mean()
        expected_retentions = total_contacts * avg_churn_prob * 0.3  # 30% retention rate
        expected_revenue = expected_retentions * retention_value
        expected_roi = ((expected_revenue - total_cost) / total_cost) * 100
        
        # Display campaign results
        st.markdown("### 📊 Campaign Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Customers to Contact", f"{total_contacts:,}")
        
        with col2:
            st.metric("Total Campaign Cost", f"${total_cost:,.0f}")
        
        with col3:
            st.metric("Expected Retentions", f"{expected_retentions:.0f}")
        
        with col4:
            st.metric("Expected ROI", f"{expected_roi:.1f}%")
        
        # ROI analysis chart
        contact_percentages = [10, 20, 30, 40, 50]
        roi_values = []
        
        for pct in contact_percentages:
            n_contacts = int(len(df) * pct / 100)
            top_n = df.nlargest(n_contacts, 'churn_probability')
            cost = n_contacts * cost_per_contact
            avg_prob = top_n['churn_probability'].mean()
            retentions = n_contacts * avg_prob * 0.3
            revenue = retentions * retention_value
            roi = ((revenue - cost) / cost) * 100 if cost > 0 else 0
            roi_values.append(roi)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=contact_percentages,
            y=roi_values,
            mode='lines+markers',
            name='ROI',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="ROI by Contact Percentage",
            xaxis_title="% of Customers Contacted",
            yaxis_title="ROI (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_data_quality_report(self, df):
        """Display comprehensive data quality report"""
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            st.write("**Missing Values:**")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found")
        
        # Data types
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Summary:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)


def main():
    """Main application entry point"""
    dashboard = ChurnDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
