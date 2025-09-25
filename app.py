import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    page_icon="🌿",
    layout="wide"
)

# ----------------- CUSTOM CSS (Dark Forest Theme) -----------------
st.markdown("""
    <style>
    /* Main background: smooth dark forest */
    .stApp {
        background: linear-gradient(160deg, #0d1b0d, #1a2e1a, #253524);
        background-attachment: fixed;
        color: #e6f0e6;
    }

    /* Sidebar background and text */
    section[data-testid="stSidebar"] {
        background: #111c11;
        color: #d9ead3;
        border-right: 2px solid #2f4f2f;
    }

    section[data-testid="stSidebar"] * {
        color: #d9ead3 !important;
        font-weight: 500;
    }

    /* Headers */
    h1, h2, h3 {
        color: #cce5cc;
        font-family: 'Trebuchet MS', sans-serif;
        font-weight: bold;
    }

    /* Metric cards */
    .stMetric {
        background: #1a2e1a;
        color: #f0f0f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
        text-align: center;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.4);
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
    </style>
""", unsafe_allow_html=True)

# ----------------- APP TITLE -----------------
st.title("🌱 Soil Health ML App")
st.markdown("Upload soil datasets, merge them, clean the data, and run predictions using multiple ML models.")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ App Settings")

# Column mapping
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc']
}
required_columns = list(column_mapping.keys())

# Upload datasets
uploaded_files = st.file_uploader("📁 Upload multiple datasets (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True)
cleaned_dfs = []

if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

            # Rename columns
            renamed = {}
            for std_col, alt_names in column_mapping.items():
                for alt in alt_names:
                    if alt in df.columns:
                        renamed[alt] = std_col
                        break
            df.rename(columns=renamed, inplace=True)

            # Clean data
            df = df[[col for col in required_columns if col in df.columns]]
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            cleaned_dfs.append(df)
            st.success(f"✅ Cleaned: {file.name} ({df.shape[0]} rows)")

        except Exception as e:
            st.warning(f"⚠️ Skipped {file.name}: {e}")

# ----------------- MERGE DATA -----------------
if cleaned_dfs:
    df = pd.concat(cleaned_dfs, ignore_index=True)
    st.subheader("🔗 Merged & Cleaned Dataset")
    st.dataframe(df.head())

    # Filter by pH
    if 'pH' in df.columns:
        ph_range = st.sidebar.slider("Filter pH", float(df['pH'].min()), float(df['pH'].max()), (5.5, 7.5))
        df = df[(df['pH'] >= ph_range[0]) & (df['pH'] <= ph_range[1])]

    # ----------------- FEATURE DISTRIBUTION -----------------
    st.subheader("📊 Feature Distribution")
    selected_feature = st.selectbox("Choose a feature to plot:", df.columns)
    fig1 = px.histogram(df, x=selected_feature, nbins=20, marginal="box", template="plotly_dark",
                        color_discrete_sequence=["#9acd32"])
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------- PREDICTION TASK -----------------
    task = st.sidebar.radio("🧠 Prediction Task", ["Classification", "Regression"])
    if task == "Classification":
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
    else:
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])

    # Features and labels
    if task == "Classification":
        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for classification.")
        else:
            df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
            X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
            y = df['Fertility_Level']
    else:
        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for regression.")
        else:
            X = df.drop(columns=['Nitrogen'])
            y = df['Nitrogen']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model function
    def get_model(name, task):
        if task == "Classification":
            return {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC()
            }[name]
        else:
            return {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVM": SVR(),
                "Linear Regression": LinearRegression()
            }[name]

    # Train and predict
    model = get_model(model_name, task)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ----------------- RESULTS -----------------
    st.subheader("📈 Results")
    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2f}", delta="✅ Good" if acc > 0.75 else "⚠️ Needs Improvement")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.metric("RMSE", f"{rmse:.2f}", delta="✅ Good" if rmse < 5 else "⚠️ High Error")
        st.metric("R² Score", f"{r2:.2f}", delta="✅ Good" if r2 > 0.7 else "⚠️ Low Fit")

        # Scatter plot
        fig2 = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                          title="Actual vs Predicted", template="plotly_dark",
                          color_discrete_sequence=["#9acd32"])
        fig2.add_shape(type="line",
                       x0=min(y_test), y0=min(y_test),
                       x1=max(y_test), y1=max(y_test),
                       line=dict(color="red", dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown(
    """
    <div class="footer">
        🌿 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span>
    </div>
    """,
    unsafe_allow_html=True
)
