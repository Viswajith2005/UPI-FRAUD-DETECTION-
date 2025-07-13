import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from datetime import datetime as dt
import time
import base64
import pickle 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBClassifier

# Page configuration with professional settings
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #34495e;
        --accent-color: #3498db;
        --success-color: #27ae60;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --light-gray: #ecf0f1;
        --dark-gray: #7f8c8d;
        --white: #ffffff;
    }
    
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin: 1rem 0 0 0;
        font-weight: 300;
    }
    
    /* Card styling */
    .stCard {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #ecf0f1;
        transition: box-shadow 0.3s ease;
    }
    
    .stCard:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9, #3498db);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        transform: translateY(-1px);
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 3px solid #3498db;
    }
    
    .metric-card h4 {
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Status boxes */
    .success-box {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(39, 174, 96, 0.2);
        border-left: 4px solid #2ecc71;
    }
    
    .error-box {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(231, 76, 60, 0.2);
        border-left: 4px solid #c0392b;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(243, 156, 18, 0.2);
        border-left: 4px solid #e67e22;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #bdc3c7;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        background: #f8f9fa;
    }
    
    .stFileUploader > div:hover {
        border-color: #3498db;
        background: #ecf0f1;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3498db, #2980b9);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #ecf0f1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #bdc3c7;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #95a5a6;
    }
    
    /* Professional form styling */
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 1px solid #bdc3c7;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #bdc3c7;
    }
    
    .stDateInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #bdc3c7;
    }
</style>
""", unsafe_allow_html=True)

# Professional header
st.markdown("""
<div class="main-header">
    <h1>UPI Fraud Detection System</h1>
    <p>Advanced Machine Learning-powered Transaction Security Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# Load the model with professional error handling
@st.cache_resource
def load_model():
    try:
        with open("UPI Fraud Detection Final.pkl", 'rb') as file:
            model = pickle.load(file)
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Show loading animation
with st.spinner("Loading AI Model..."):
    loaded_model = load_model()

if loaded_model:
    st.success("AI Model loaded successfully")

# Define the expected feature lists
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 
      'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 
      'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 
      'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 
      'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

def preprocess_for_prediction(df):
    """
    Preprocess new data to match the training data format
    """
    # Initialize the output DataFrame with zeros for all features
    all_columns = ['amount', 'Year', 'Month']
    
    # Add Transaction_Type columns
    all_columns.extend([f'Transaction_Type_{type}' for type in tt])
    
    # Add Payment_Gateway columns
    all_columns.extend([f'Payment_Gateway_{gateway}' for gateway in pg])
    
    # Add Transaction_State columns
    all_columns.extend([f'Transaction_State_{state}' for state in ts])
    
    # Add Merchant_Category columns
    all_columns.extend([f'Merchant_Category_{cat}' for cat in mc])
    
    # Create DataFrame with zeros
    output = pd.DataFrame(0, index=df.index, columns=all_columns)
    
    # Copy numeric values
    output['amount'] = df['amount']
    output['Year'] = df['Year']
    output['Month'] = df['Month']
    
    # Set categorical variables
    for idx, row in df.iterrows():
        # Transaction Type
        col_name = f'Transaction_Type_{row["Transaction_Type"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
            
        # Payment Gateway
        col_name = f'Payment_Gateway_{row["Payment_Gateway"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
            
        # Transaction State
        col_name = f'Transaction_State_{row["Transaction_State"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
            
        # Merchant Category
        col_name = f'Merchant_Category_{row["Merchant_Category"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
    
    return output

def validate_input(amount, date):
    if amount <= 0:
        st.error("Amount must be greater than 0")
        return False
    if date > datetime.datetime.now().date():
        st.error("Transaction date cannot be in the future")
        return False
    return True

def display_prediction(prediction, probability):
    fraud_prob = probability[0][1]
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.markdown(
            "<span style='color:#b00020; font-weight:600; font-size:1.1rem;'>Fraudulent Transaction Detected</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='color:#006400; font-weight:600; font-size:1.1rem;'>Legitimate Transaction</span>",
            unsafe_allow_html=True
        )
    st.markdown(f"**Fraud Probability:** {fraud_prob:.2%}")

def display_statistics(amount, merchant_cat, state):
    st.markdown("### Transaction Analytics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Amount**")
        st.markdown(f"<span style='font-size:1.3rem; font-weight:600;'>{amount:,.2f} ₹</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Category**")
        st.markdown(f"<span style='font-size:1.1rem; font-weight:500;'>{merchant_cat}</span>", unsafe_allow_html=True)
    with col3:
        st.markdown("**Location**")
        st.markdown(f"<span style='font-size:1.1rem; font-weight:500;'>{state}</span>", unsafe_allow_html=True)
    with col4:
        st.markdown("**Risk Level**")
        risk_level = "Low" if amount < 10000 else "Medium" if amount < 50000 else "High"
        risk_color = "#27ae60" if risk_level == "Low" else "#f39c12" if risk_level == "Medium" else "#e74c3c"
        st.markdown(
            f"<span style='font-size:1.1rem; font-weight:600; color:{risk_color};'>{risk_level}</span>",
            unsafe_allow_html=True
        )

def process_batch_file(df):
    try:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Processing data...")
        progress_bar.progress(25)
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        progress_bar.progress(50)
        
        # Extract Year and Month
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        progress_bar.progress(75)
        
        # Preprocess the data
        processed_df = preprocess_for_prediction(df)
        progress_bar.progress(90)
        
        if loaded_model is None:
            st.error("Model not loaded properly")
            return None
            
        # Make predictions
        predictions = loaded_model.predict(processed_df)
        probabilities = loaded_model.predict_proba(processed_df)
        progress_bar.progress(100)
        
        # Add predictions to original dataframe
        df['Predicted_Fraud'] = predictions
        df['Fraud_Probability'] = probabilities[:, 1]
        
        status_text.text("Processing complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return df
        
    except Exception as e:
        st.error(f"Error in process_batch_file: {str(e)}")
        return None

def create_analytics_dashboard(results):
    """Create comprehensive analytics dashboard"""
    st.markdown("### Analytics Dashboard")
    
    # Summary metrics
    total_transactions = len(results)
    fraud_count = sum(results['Predicted_Fraud'])
    fraud_percentage = (fraud_count / total_transactions) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    
    with col3:
        st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
    
    with col4:
        avg_amount = results['amount'].mean()
        st.metric("Average Amount", f"₹{avg_amount:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud distribution pie chart
        fraud_data = results['Predicted_Fraud'].value_counts()
        fig_pie = px.pie(
            values=fraud_data.values,
            names=['Legitimate', 'Fraudulent'],
            title="Transaction Distribution",
            color_discrete_map={0: '#27ae60', 1: '#e74c3c'}
        )
        fig_pie.update_layout(
            font={'color': "#2c3e50"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Amount distribution histogram
        fig_hist = px.histogram(
            results,
            x='amount',
            color='Predicted_Fraud',
            title="Amount Distribution by Fraud Status",
            color_discrete_map={0: '#27ae60', 1: '#e74c3c'}
        )
        fig_hist.update_layout(
            font={'color': "#2c3e50"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Risk analysis by category
    st.markdown("#### Risk Analysis by Category")
    
    if 'Merchant_Category' in results.columns:
        category_risk = results.groupby('Merchant_Category')['Predicted_Fraud'].agg(['count', 'sum']).reset_index()
        category_risk['fraud_rate'] = (category_risk['sum'] / category_risk['count']) * 100
        
        fig_category = px.bar(
            category_risk,
            x='Merchant_Category',
            y='fraud_rate',
            title="Fraud Rate by Merchant Category",
            color='fraud_rate',
            color_continuous_scale='RdYlGn_r'
        )
        fig_category.update_layout(
            xaxis_tickangle=-45,
            font={'color': "#2c3e50"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_category, use_container_width=True)

# Professional sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h3>Navigation</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Analysis Mode",
    ["Single Transaction", "Batch Processing"]
)

# Add spacing
st.sidebar.markdown("---")

# Add model info
if loaded_model:
    st.sidebar.markdown("""
    <div style="background: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4>AI Model Status</h4>
        <p style="color: #27ae60;">Model Loaded</p>
        <p style="font-size: 0.8rem; color: #7f8c8d;">XGBoost Classifier</p>
    </div>
    """, unsafe_allow_html=True)

if page == "Single Transaction":
    st.markdown("## Single Transaction Analysis")
    st.markdown("Enter transaction details below to analyze for potential fraud.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Transaction Details")
        amount = st.number_input(
            "Transaction Amount (₹)", 
            min_value=0.0, 
            value=1000.0, 
            step=100.0,
            format="%.2f",
            help="Enter the transaction amount in Indian Rupees"
        )
        
        transaction_type = st.selectbox(
            "Transaction Type",
            tt,
            help="Select the type of transaction"
        )
        
        payment_gateway = st.selectbox(
            "Payment Gateway",
            pg,
            help="Select the payment gateway used"
        )
    
    with col2:
        st.markdown("#### Location & Category")
        transaction_state = st.selectbox(
            "Transaction State",
            ts,
            help="Select the state where the transaction occurred"
        )
        
        merchant_category = st.selectbox(
            "Merchant Category",
            mc,
            help="Select the merchant category"
        )
        
        transaction_date = st.date_input(
            "Transaction Date", 
            datetime.datetime.now(),
            help="Select the transaction date"
        )
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Professional button
    if st.button("Analyze Transaction", use_container_width=True):
        if validate_input(amount, transaction_date):
            with st.spinner("AI is analyzing the transaction..."):
                # Create input dataframe
                input_df = pd.DataFrame({
                    'amount': [amount],
                    'Transaction_Type': [transaction_type],
                    'Payment_Gateway': [payment_gateway],
                    'Transaction_State': [transaction_state],
                    'Merchant_Category': [merchant_category],
                    'Year': [transaction_date.year],
                    'Month': [transaction_date.month]
                })
                
                # Preprocess data
                processed_data = preprocess_for_prediction(input_df)
                
                try:
                    # Make prediction
                    prediction = loaded_model.predict(processed_data)
                    probability = loaded_model.predict_proba(processed_data)
                    
                    # Display results
                    st.markdown("<br>", unsafe_allow_html=True)
                    display_statistics(amount, merchant_category, transaction_state)
                    st.markdown("<br>", unsafe_allow_html=True)
                    display_prediction(prediction[0], probability)
                    
                    # Add professional insights
                    st.markdown("### AI Insights")
                    if prediction[0] == 1:
                        st.warning("""
                        **High-risk indicators detected:**
                        - Unusual transaction pattern
                        - Suspicious amount or location
                        - Anomalous merchant category
                        
                        **Recommendation:** Review this transaction manually and consider blocking if necessary.
                        """)
                    else:
                        st.success("""
                        **Transaction appears legitimate:**
                        - Normal transaction pattern
                        - Expected amount and location
                        - Standard merchant category
                        
                        **Recommendation:** Transaction can proceed normally.
                        """)
                    
                    # --- About & Instructions Section ---
                    # about_and_team_section() # Moved to the very end
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

else:
    st.markdown("## Batch Processing")
    st.markdown("Upload a CSV file with multiple transactions for bulk analysis.")
    
    # Professional file uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file containing transaction data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show file info
            st.success(f"File uploaded successfully! ({len(df)} transactions)")
            
            # Show sample of uploaded data
            with st.expander("Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Process button
            if st.button("Process All Transactions", use_container_width=True):
                # Process the file
                results = process_batch_file(df)
                
                if results is not None:
                    # Create comprehensive dashboard
                    create_analytics_dashboard(results)
                    
                    # Download results
                    st.markdown("### Download Results")
                    csv = results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="fraud_detection_results.csv" class="stButton">Download Results CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Help section
    with st.expander("Expected CSV Format", expanded=False):
        st.markdown("""
        Your CSV file should contain the following columns:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | Date | Transaction date (DD-MM-YYYY) | 15-01-2024 |
        | amount | Transaction amount | 1500.00 |
        | Transaction_Type | Type of transaction | Purchase |
        | Payment_Gateway | Payment method | Google Pay |
        | Transaction_State | State location | Maharashtra |
        | Merchant_Category | Merchant category | Purchases |
        
        **Note:** Column names are case-sensitive.
        """)

# --- Developed by Section (with LinkedIn links) ---
def about_and_team_section():
    st.markdown("---")
    st.markdown("### About & Instructions")
    st.markdown("""
    <div style='font-size:1rem;'>
        <b>About this Application:</b><br>
        This UPI Fraud Detection system uses machine learning to analyze transaction patterns and identify potential fraud. <br><br>
        <b>Required Inputs:</b>
        <ul>
            <li><b>Transaction Amount</b> (₹): Enter the value of the transaction.</li>
            <li><b>Transaction Type</b>: Select the type of transaction (e.g., Bill Payment, Purchase).</li>
            <li><b>Payment Gateway</b>: Choose the payment method used.</li>
            <li><b>Transaction State</b>: Select the state where the transaction occurred.</li>
            <li><b>Merchant Category</b>: Choose the merchant category.</li>
            <li><b>Transaction Date</b>: Enter the date of the transaction.</li>
        </ul>
        <b>How to Use:</b>
        <ol>
            <li>Fill in all required transaction details in the form above.</li>
            <li>Click <b>Analyze Transaction</b> to get a fraud prediction and insights.</li>
            <li>Review the results and AI insights for further action.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align:center; font-size:1.1rem; font-weight:600;'>Developed by</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='text-align:center; padding:0.5rem 0;'>
            <span style='font-weight:500;'>Talluri Ranga Sai Varun</span><br>
            <a href='https://www.linkedin.com/in/talluri-ranga-sai-varun/' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='24' style='margin-top:4px;'/>
            </a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:0.5rem 0;'>
            <span style='font-weight:500;'>Telagamsetty Viswajith Gupta</span><br>
            <a href='https://www.linkedin.com/in/viswajith-gupta/' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='24' style='margin-top:4px;'/>
            </a>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='text-align:center; padding:0.5rem 0;'>
            <span style='font-weight:500;'>Dokala Manoj Kumar</span><br>
            <a href='https://www.linkedin.com/in/dokala-manoj-kumar/' target='_blank'>
                <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='24' style='margin-top:4px;'/>
            </a>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; color: #7f8c8d; font-size: 0.9rem;'>
        UPI Fraud Detection System - Powered by Machine Learning
    </div>
    """, unsafe_allow_html=True)

# Clean, minimal footer
# about_and_team_section() # Moved to the very end

about_and_team_section()

st.markdown("""
<style>
.github-float {
    position: fixed;
    bottom: 32px;
    right: 32px;
    z-index: 9999;
}
.github-float a {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fff;
    border: 2px solid #0366d6;
    border-radius: 50%;
    width: 56px;
    height: 56px;
    box-shadow: 0 2px 12px rgba(3,102,214,0.10);
    transition: background 0.2s, border 0.2s, box-shadow 0.2s;
    text-decoration: none;
}
.github-float a:hover {
    background: #0366d6;
    border: 2px solid #0366d6;
    box-shadow: 0 4px 16px rgba(3,102,214,0.18);
}
.github-float a img {
    filter: none;
    width: 32px;
    height: 32px;
    transition: filter 0.2s;
}
.github-float a:hover img {
    filter: invert(1) brightness(2);
}
@media (max-width: 600px) {
    .github-float {
        bottom: 16px;
        right: 16px;
    }
    .github-float a {
        width: 40px;
        height: 40px;
    }
    .github-float a img {
        width: 22px;
        height: 22px;
    }
}
</style>
<div class="github-float">
    <a href="https://github.com/viswajith2005/UPI-Fraud-Detection" target="_blank" title="View on GitHub">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" />
    </a>
</div>
""", unsafe_allow_html=True)
