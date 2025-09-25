import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --------------------------
# 🌱 PAGE CONFIG & STYLING
# --------------------------
st.set_page_config(page_title="🌱 Soil Health ML Dashboard", layout="wide")
st.markdown("""
    <style>
    body { background-color: #f5f5f0; }
    .stApp { background-color: #f9f9f4; }
    h1, h2, h3 { color: #2e4600; }
    .good { color: green; font-weight: bold; }
    .moderate { color: orange; font-weight: bold; }
    .poor { color: red; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Soil Health Machine Learning Dashboard")
st.markdown("Upload soil datasets, clean & merge them, and run predictions using ML models with an agriculture-inspired design.")

# --------------------------
# 📂 SIDEBAR SETTINGS
# --------------------------
st.sidebar.header("⚙️ App Settings")
task = st.sidebar.radio("🧠 Prediction Task", ["Classification", "Regression"])

# --------------------------
# 🔑 COLUMN MAPPING
# --------------------------
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc']
}
required_columns = list(column_mapping.keys())

# --------------------------
# 📥 UPLOAD DATA
# --------------------------
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
            df = df[[col for col in required_columns if col in df.columns]]
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            cleaned_dfs.append(df)
            st.success(f"✅ Cleaned: {file.name} ({df.shape[0]} rows)")
        except Exception as e:
            st.warning(f"⚠️ Skipped {file.name}: {e}")

# --------------------------
# 🔗 MERGE DATA
# --------------------------
if cleaned_dfs:
    df = pd.concat(cleaned_dfs, ignore_index=True)
    with st.expander("🔎 View Merged & Cleaned Dataset"):
        st.dataframe(df.head())

    # Filter by pH
    if 'pH' in df.columns:
        ph_range = st.sidebar.slider("Filter by pH", float(df['pH'].min()), float(df['pH'].max()), (5.5, 7.5))
        df = df[(df['pH'] >= ph_range[0]) & (df['pH'] <= ph_range[1])]

    # --------------------------
    # 📊 FEATURE DISTRIBUTION
    # --------------------------
    with st.expander("📊 Feature Distribution"):
        selected_feature = st.selectbox("Choose a feature:", df.columns)
        fig = px.histogram(df, x=selected_feature, nbins=20, marginal="box",
                           title=f"Distribution of {selected_feature}",
                           color_discrete_sequence=["#6b8e23"])
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # 🧠 MODEL SELECTION
    # --------------------------
    if task == "Classification":
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for classification.")
        else:
            df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
            X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
            y = df['Fertility_Level']
    else:
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])
        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for regression.")
        else:
            X = df.drop(columns=['Nitrogen'])
            y = df['Nitrogen']

    # --------------------------
    # 🔄 TRAIN / TEST SPLIT
    # --------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --------------------------
    # 🔧 MODEL FUNCTION
    # --------------------------
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

    # --------------------------
    # 🚀 TRAIN & PREDICT
    # --------------------------
    model = get_model(model_name, task)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --------------------------
    # 📈 RESULTS
    # --------------------------
    st.subheader("📈 Model Performance")

    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        status = "good" if acc > 0.8 else "moderate" if acc > 0.6 else "poor"
        st.markdown(f"**Accuracy:** <span class='{status}'>{acc:.2f}</span>", unsafe_allow_html=True)

        report = classification_report(y_test, y_pred, output_dict=True)
        fig = px.bar(pd.DataFrame(report).transpose(), title="Classification Report", color_discrete_sequence=["#228b22"])
        st.plotly_chart(fig, use_container_width=True)

    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        status_rmse = "good" if rmse < 5 else "moderate" if rmse < 10 else "poor"
        status_r2 = "good" if r2 > 0.7 else "moderate" if r2 > 0.4 else "poor"

        st.markdown(f"**RMSE:** <span class='{status_rmse}'>{rmse:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"**R² Score:** <span class='{status_r2}'>{r2:.2f}</span>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                 marker=dict(color='green', size=8, opacity=0.6),
                                 name="Predictions"))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                 y=[y_test.min(), y_test.max()],
                                 mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
        fig.update_layout(title="Actual vs Predicted Nitrogen", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)
