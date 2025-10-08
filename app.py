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

st.set_page_config(page_title="🌱 Soil Health ML App", layout="wide", page_icon="🌿")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "🌱 Soil Health App",
        ["📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"],
        icons=["cloud-upload", "bar-chart", "robot", "graph-up", "lightbulb"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5!important"},
            "icon": {"color": "#9acd32", "font-size": "20px"},
            "nav-link": {"color": "#d9ead3", "font-size": "16px"},
            "nav-link-selected": {"background-color": "#5a8f29"},
        },
    )

# Column mapping for standard names
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc']
}
required_columns = list(column_mapping.keys())

# ========== 📂 UPLOAD & PREPROCESS ==========
if selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True)
    cleaned_dfs = []

    if uploaded_files:
        with st.spinner("Cleaning and merging datasets..."):
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
                    st.success(f"✅ Loaded: {file.name} ({df.shape[0]} rows)")
                except Exception as e:
                    st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True)

            # ---------- AUTO PREPROCESS ----------
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(exclude=[np.number]).columns

            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)

            df.dropna(how='all', inplace=True)
            st.session_state["df"] = df  # ✅ Persist data across pages

            st.subheader("🔗 Final Preprocessed Dataset")
            st.dataframe(df.head())

            csv_final = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Final Preprocessed Dataset",
                data=csv_final,
                file_name="final_preprocessed_soil_dataset.csv",
                mime="text/csv"
            )

            st.success("✨ Dataset cleaned, merged, and preprocessed successfully!")
            st.balloons()

# ========== 📊 VISUALIZATION ==========
elif selected == "📊 Visualization":
    st.title("📊 Data Visualization")
    if "df" in st.session_state:
        df = st.session_state["df"]
        feature = st.selectbox("Select a feature", df.columns)
        fig = px.histogram(df, x=feature, nbins=20, color_discrete_sequence=["#9acd32"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌐 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload and preprocess data first.")

# ========== 🤖 MODELING ==========
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling & Prediction")
    if "df" not in st.session_state:
        st.info("Please upload data first.")
        st.stop()

    df = st.session_state["df"]
    task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])

    if task == "Classification":
        model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
    else:
        model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])

    # --- Parameters per model
    params = {}
    if model_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 300, 100)
        max_depth = st.slider("Max Depth", 2, 20, 10)
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
    elif model_name == "Decision Tree":
        params = {"max_depth": st.slider("Max Depth", 2, 20, 5)}
    elif model_name == "KNN":
        params = {"n_neighbors": st.slider("K Neighbors", 1, 15, 5)}
    elif model_name == "SVM":
        params = {"kernel": st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])}

    # --- Prepare Data
    if task == "Classification":
        df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Medium', 'High'])
        X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
        y = df['Fertility_Level']
    else:
        X = df.drop(columns=['Nitrogen'])
        y = df['Nitrogen']

    X = X.select_dtypes(include=[np.number])
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Get model safely
    def get_model(name, task, params):
        if task == "Classification":
            if name == "Random Forest": return RandomForestClassifier(random_state=42, **params)
            elif name == "Decision Tree": return DecisionTreeClassifier(random_state=42, **params)
            elif name == "KNN": return KNeighborsClassifier(**params)
            elif name == "SVM": return SVC(**params)
        else:
            if name == "Random Forest": return RandomForestRegressor(random_state=42, **params)
            elif name == "Decision Tree": return DecisionTreeRegressor(random_state=42, **params)
            elif name == "KNN": return KNeighborsRegressor(**params)
            elif name == "SVM": return SVR(**params)
            elif name == "Linear Regression": return LinearRegression()

    # --- Train
    with st.spinner("Training model..."):
        model = get_model(model_name, task, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.session_state["results"] = {"task": task, "y_test": y_test, "y_pred": y_pred, "model": model, "model_name": model_name}

    st.success("✅ Training complete! Go to 📈 Results.")
    joblib.dump(model, "soil_model.pkl")
    st.download_button("⬇️ Download Trained Model", data=open("soil_model.pkl", "rb"), file_name="soil_model.pkl")

# ========== 📈 RESULTS ==========
elif selected == "📈 Results":
    st.title("📈 Model Results")
    if "results" not in st.session_state:
        st.info("Please run the model first.")
        st.stop()

    results = st.session_state["results"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    model = results["model"]
    task = results["task"]

    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2f}")
        st.text(classification_report(y_test, y_pred))
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

# ========== 🌿 INSIGHTS ==========
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights")
    if "df" in st.session_state:
        df = st.session_state["df"]
        avg_ph = df["pH"].mean()
        st.markdown(f"**Average Soil pH:** {avg_ph:.2f}")
        if avg_ph < 5.5:
            st.warning("⚠️ Soil is acidic — consider lime application.")
        elif avg_ph > 7.5:
            st.info("ℹ️ Soil is alkaline — add organic matter or sulfur.")
        else:
            st.success("✅ Soil pH is within optimal range (5.5–7.5).")
    else:
        st.info("Please upload a dataset first.")
