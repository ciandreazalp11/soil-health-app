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

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# Custom CSS for agriculture feel
st.markdown("""
    <style>
    body {
        background-color: #f5f5dc;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #2e4600;
    }
    .stMetric {
        background-color: #eaf4e1;
        padding: 10px;
        border-radius: 10px;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Soil Health Machine Learning App")
st.caption("📊 Upload soil datasets, clean them, visualize distributions, and predict soil fertility using ML models.")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ App Settings")
st.sidebar.info("Developed by **Andre Plaza** & **[Partner’s Name]** 🌿")

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

# Tabs
tabs = st.tabs(["📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"])

# ----------------- UPLOAD TAB -----------------
with tabs[0]:
    uploaded_files = st.file_uploader("📁 Upload multiple datasets (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True)
    cleaned_dfs = []

    if uploaded_files:
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
                df.dropna(inplace=True)
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

# ----------------- VISUALIZATION TAB -----------------
with tabs[1]:
    if "df" in st.session_state:
        df = st.session_state["df"]

        st.subheader("📊 Feature Distribution")
        feature = st.selectbox("Select a feature", df.columns)
        fig = px.histogram(df, x=feature, nbins=20, marginal="box", color_discrete_sequence=["#5a8f29"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌐 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="YlGn")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data first.")

# ----------------- MODELING TAB -----------------
with tabs[2]:
    if "df" in st.session_state:
        df = st.session_state["df"]

        task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])
        if task == "Classification":
            model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
        else:
            model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])

        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for modeling.")
        else:
            if task == "Classification":
                df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
                X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
                y = df['Fertility_Level']
            else:
                X = df.drop(columns=['Nitrogen'])
                y = df['Nitrogen']

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

            model = get_model(model_name, task)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state["results"] = {"task": task, "y_test": y_test, "y_pred": y_pred}
    else:
        st.info("Please upload data first.")

# ----------------- RESULTS TAB -----------------
with tabs[3]:
    if "results" in st.session_state:
        results = st.session_state["results"]
        y_test, y_pred, task = results["y_test"], results["y_pred"], results["task"]

        st.subheader("📈 Model Performance")

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            color = "green" if acc > 0.8 else "orange" if acc > 0.6 else "red"
            st.metric("Accuracy", f"{acc:.2f}", delta=None, label_visibility="visible")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=acc,
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color}},
                title={'text': "Accuracy Gauge"}
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            rmse_color = "green" if rmse < 10 else "orange" if rmse < 20 else "red"
            r2_color = "green" if r2 > 0.7 else "orange" if r2 > 0.4 else "red"

            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("R² Score", f"{r2:.2f}")

            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#5a8f29"])
            fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please run a model first.")

# ----------------- INSIGHTS TAB -----------------
with tabs[4]:
    st.subheader("🌿 Insights & Recommendations")
    st.markdown("""
    - If **Nitrogen is low**, soil may need **nitrogen-based fertilizers**.
    - If **pH < 5.5**, consider **lime treatment** to reduce acidity.
    - High **Organic Matter** usually indicates healthier soil.
    - Always validate ML predictions with **field tests**.
    """)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("👨‍💻 Developed by **Andre Plaza** & **[Partner’s Name]** | 🌱 Soil Health Capstone Project")
