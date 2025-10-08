import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# ----------------- THEME CONFIG -----------------
st.markdown("""
<style>
/* Background and Text */
.stApp {
    background: linear-gradient(160deg, #0d1b0d, #1a2e1a, #253524);
    color: #e6f0e6;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111c11 !important;
    padding: 10px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] * {
    color: #d9ead3 !important;
}

/* Hover Animations */
div[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(90,143,41,0.3) !important;
    transform: scale(1.05);
    transition: all 0.3s ease-in-out;
    border-radius: 8px;
}

/* Headers */
h1, h2, h3 {
    color: #cce5cc;
    font-family: 'Trebuchet MS', sans-serif;
    font-weight: bold;
    text-shadow: 0px 0px 6px rgba(100,255,100,0.4);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(30, 60, 30, 0.65);
    color: #f0f0f0;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.7);
    backdrop-filter: blur(10px);
    text-align: center;
    transition: transform 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px) scale(1.02);
}

/* Animated float effect */
@keyframes float {
  0% {transform: translateY(0);}
  50% {transform: translateY(-6px);}
  100% {transform: translateY(0);}
}
[data-testid="stMetric"] { animation: float 4s ease-in-out infinite; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #5a8f29, #9acd32);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0px 4px 12px rgba(154,205,50,0.6);
}

/* Footer */
.footer {
    text-align: center;
    color: #bcd9b2;
    font-size: 15px;
    padding: 10px;
    margin-top: 20px;
}
.footer span {
    font-weight: bold;
    color: #9acd32;
}

/* Tabs styling */
.stTabs [role="tab"] {
    border: 1px solid #5a8f29;
    border-radius: 10px;
    background: rgba(40, 60, 40, 0.3);
    margin: 3px;
}
.stTabs [role="tab"]:hover {
    background: rgba(90,143,41,0.3);
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR MENU -----------------
with st.sidebar:
    selected = option_menu(
        "🌱 Soil Health App", 
        ["📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"], 
        icons=["cloud-upload", "bar-chart", "robot", "graph-up", "lightbulb"],
        menu_icon="list", 
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#111c11"},
            "icon": {"color": "#9acd32", "font-size": "20px"},
            "nav-link": {"color":"#d9ead3","font-size": "16px"},
            "nav-link-selected": {"background-color": "#5a8f29"},
        }
    )

# ----------------- COLUMN MAPPING -----------------
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc']
}
required_columns = list(column_mapping.keys())

# ----------------- UPLOAD DATA (with Preprocessing) -----------------
if selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    uploaded_files = st.file_uploader(
        "Upload multiple datasets (.csv or .xlsx)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True
    )
    cleaned_dfs = []

    if uploaded_files:
        with st.spinner("🧹 Cleaning and merging datasets..."):
            time.sleep(1.2)
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                    renamed = {}
                    for std_col, alt_names in column_mapping.items():
                        for alt in alt_names:
                            if alt in df.columns:
                                renamed[alt] = std_col
                                break
                    df.rename(columns=renamed, inplace=True)
                    df = df[[col for col in required_columns if col in df.columns]]
                    df.drop_duplicates(inplace=True)
                    cleaned_dfs.append(df)
                    st.success(f"✅ Cleaned: {file.name} ({df.shape[0]} rows)")
                except Exception as e:
                    st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True)
            st.subheader("🔗 Merged & Cleaned Dataset")
            st.dataframe(df.head())
            st.session_state["df"] = df

            # ✅ Download cleaned dataset
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Cleaned & Merged Dataset (CSV)",
                data=csv,
                file_name="cleaned_merged_soil_dataset.csv",
                mime="text/csv"
            )

            # ✅ NEW SECTION: Missing Value Preprocessing
            st.markdown("---")
            st.subheader("🧹 Preprocess Missing Values")

            before_missing = df.isnull().sum().sum()
            if st.button("🔧 Fill Missing Values (Mean/Mode)"):
                df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].mean(), inplace=True)
                cat_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in cat_cols:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                after_missing = df.isnull().sum().sum()
                st.success(f"✨ Filled {int(before_missing - after_missing)} missing values successfully!")
                st.session_state["df"] = df

                st.write("📋 Missing Values Summary After Preprocessing:")
                st.dataframe(df.isnull().sum().to_frame("Missing Count"))

                # Download preprocessed dataset
                pre_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Preprocessed Dataset (CSV)",
                    data=pre_csv,
                    file_name="preprocessed_soil_dataset.csv",
                    mime="text/csv"
                )

            st.balloons()

# ----------------- (Rest of your code remains unchanged below) -----------------
