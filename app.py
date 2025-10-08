import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from io import BytesIO

st.set_page_config(page_title="Soil Health Prediction System", layout="wide")

# ---------------- SESSION STATE ----------------
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'task' not in st.session_state:
    st.session_state.task = None

# ---------------- DATA HANDLING ----------------
def clean_merge_preprocess(df):
    # Clean: drop duplicates
    df = df.drop_duplicates()
    # Merge placeholder (single file upload case)
    df = df.copy()

    # Preprocess: fill missing numeric with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    # Fill missing categorical with mode
    for col in df.select_dtypes(exclude=np.number).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# ---------------- DOWNLOAD HELPERS ----------------
def download_dataframe(df, filename="final_dataset.csv"):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="⬇️ Download Cleaned + Preprocessed Dataset",
        data=buffer,
        file_name=filename,
        mime="text/csv"
    )

# ---------------- MODEL GETTER ----------------
def get_model(model_name, task, params):
    if task == "Classification":
        if model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(random_state=42, **params)
    else:
        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=42, **params)
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42, **params)

# ---------------- APP LAYOUT ----------------
st.title("🌾 Soil Health Prediction System")

menu = st.sidebar.radio("Navigation", ["📂 Data", "📊 Visualization", "🤖 Modeling", "🧠 Prediction Insights"])

# ---------------- PAGE: DATA ----------------
if menu == "📂 Data":
    st.header("Upload, Clean, Merge & Preprocess Data")

    uploaded = st.file_uploader("Upload your soil dataset (CSV)", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader("Raw Uploaded Data")
        st.dataframe(df.head())

        df_clean = clean_merge_preprocess(df)
        st.session_state.cleaned_data = df_clean

        st.success("✅ Data cleaned, merged, and preprocessed automatically.")
        st.subheader("Processed Data Preview")
        st.dataframe(df_clean.head())

        download_dataframe(df_clean)

# ---------------- PAGE: VISUALIZATION ----------------
elif menu == "📊 Visualization":
    st.header("Dataset Visualization")

    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        columns = df.columns.tolist()
        x_col = st.selectbox("Select X-axis", columns, key="xcol")
        y_col = st.selectbox("Select Y-axis", columns, key="ycol")

        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", color_discrete_sequence=["#4CAF50"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload and preprocess data first.")

# ---------------- PAGE: MODELING ----------------
elif menu == "🤖 Modeling":
    st.header("Train and Evaluate Model")

    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        target_col = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        task = st.radio("Select Task Type", ["Classification", "Regression"])
        model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree"])

        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        params = {}

        model = get_model(model_name, task, params)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.task = task

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Accuracy: {acc:.2f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.info(f"📉 Model Mean Squared Error: {mse:.2f}")
    else:
        st.warning("Please complete preprocessing first.")

# ---------------- PAGE: INSIGHTS ----------------
elif menu == "🧠 Prediction Insights":
    st.header("Soil Health Prediction Insights")

    if st.session_state.model is not None and st.session_state.cleaned_data is not None:
        model = st.session_state.model
        df = st.session_state.cleaned_data

        st.subheader("Legend / Indicator")
        st.markdown("""
        🟢 **Good Soil** — Healthy nutrient levels, ideal for planting  
        🟡 **Moderate Soil** — Needs minor adjustment or fertilizer  
        🔴 **Poor Soil** — Nutrient imbalance or contamination risk
        """)

        st.subheader("Try New Data for Prediction")
        sample = {}
        for col in df.columns[:-1]:
            sample[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        sample_df = pd.DataFrame([sample])

        if st.button("Predict Soil Health"):
            prediction = model.predict(sample_df)[0]
            st.write(f"🔍 **Predicted Soil Health:** {prediction}")

            # Interpret the result
            if str(prediction).lower() in ["good", "healthy", "1"]:
                st.success("🟢 Soil Health: GOOD\n✅ Nutrients are balanced. Ideal for most crops.")
            elif str(prediction).lower() in ["moderate", "medium", "2"]:
                st.warning("🟡 Soil Health: MODERATE\n⚠️ Some nutrient imbalance. Consider minor adjustments.")
            else:
                st.error("🔴 Soil Health: POOR\n🚫 Deficient or toxic levels detected. Improve before planting.")
    else:
        st.warning("Please train a model first in the Modeling section.")
