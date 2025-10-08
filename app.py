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
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import joblib
from io import BytesIO

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Soil Health Prediction System", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #d4fc79, #96e6a1);
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #2f4f4f;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #006400;
            font-size: 20px;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #228B22;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #006400;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌱 Soil Health Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Random Forest and Streamlit</div>', unsafe_allow_html=True)

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.title("🔍 Navigation")
tabs = ["Upload & Clean Data", "Visualization", "Modeling", "Results", "Insights"]
choice = st.sidebar.radio("Go to:", tabs)

# ----------------------------- UPLOAD TAB -----------------------------
if choice == "Upload & Clean Data":
    st.header("📂 Upload and Clean Dataset")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")

        st.subheader("📊 Preview of Dataset")
        st.dataframe(df.head())

        st.subheader("🧹 Data Cleaning")
        df = df.drop_duplicates()
        df = df.dropna()

        st.write("✅ Duplicates removed and missing values handled.")
        st.write(f"📏 Cleaned Dataset Shape: {df.shape}")

        # Save to session state
        st.session_state["df"] = df

        # ----------------------------- DOWNLOAD CLEANED DATASET -----------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv,
            file_name="cleaned_soil_dataset.csv",
            mime="text/csv"
        )

# ----------------------------- VISUALIZATION TAB -----------------------------
elif choice == "Visualization":
    st.header("📈 Data Visualization")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and clean a dataset first.")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.subheader("Feature Distribution")
        col = st.selectbox("Select a column:", numeric_cols)
        st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}"))

        st.subheader("Correlation Heatmap")
        corr = df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap"))

# ----------------------------- MODELING TAB -----------------------------
elif choice == "Modeling":
    st.header("🤖 Modeling and Training")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload and clean a dataset first.")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        target = st.selectbox("Select Target Column", numeric_cols)
        features = st.multiselect("Select Feature Columns", [col for col in numeric_cols if col != target])

        if features:
            X = df[features]
            y = df[target]

            task = st.radio("Select Task Type:", ["Regression", "Classification"])
            model_name = st.selectbox("Choose Algorithm:", 
                ["Random Forest", "Decision Tree", "K-Nearest Neighbors", "Support Vector Machine", "Linear Regression"])

            # Hyperparameter tuning
            st.subheader("⚙️ Hyperparameter Settings")
            n_estimators = st.slider("Number of Trees (for Random Forest)", 10, 500, 100)
            max_depth = st.slider("Max Depth (for Trees)", 2, 20, 10)
            k_value = st.slider("K Value (for KNN)", 1, 15, 5)

            test_size = st.slider("Test Size (Ratio)", 0.1, 0.5, 0.2)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = None
            if task == "Classification":
                if model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                elif model_name == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(n_neighbors=k_value)
                elif model_name == "Support Vector Machine":
                    model = SVC()
            else:
                if model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif model_name == "Decision Tree":
                    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                elif model_name == "K-Nearest Neighbors":
                    model = KNeighborsRegressor(n_neighbors=k_value)
                elif model_name == "Support Vector Machine":
                    model = SVR()
                elif model_name == "Linear Regression":
                    model = LinearRegression()

            if st.button("Train Model"):
                model.fit(X_train, y_train)
                st.session_state["model"] = model
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.success("✅ Model trained successfully!")

# ----------------------------- RESULTS TAB -----------------------------
elif choice == "Results":
    st.header("📊 Model Evaluation Results")

    if "model" not in st.session_state:
        st.warning("⚠️ Please train a model first.")
    else:
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)

        if isinstance(model, (RandomForestRegressor, DecisionTreeRegressor, KNeighborsRegressor, SVR, LinearRegression)):
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MSE", f"{mse:.3f}")
            st.metric("MAE", f"{mae:.3f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc*100:.2f}%")
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, title="Confusion Matrix"))

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, name="Importance")
            importance.index = st.session_state["df"].select_dtypes(include=np.number).columns.drop(st.session_state["df"].select_dtypes(include=np.number).columns[-1])
            fig = px.bar(importance.sort_values(ascending=True), orientation="h", title="Feature Importance", color=importance, color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)

        # Download trained model
        buffer = BytesIO()
        joblib.dump(model, buffer)
        st.download_button("⬇️ Download Trained Model", data=buffer.getvalue(), file_name="trained_soil_model.pkl")

# ----------------------------- INSIGHTS TAB -----------------------------
elif choice == "Insights":
    st.header("💡 Data Insights and Recommendations")

    if "df" in st.session_state:
        df = st.session_state["df"]
        avg_ph = df["pH"].mean() if "pH" in df.columns else None

        if avg_ph:
            if avg_ph < 5.5:
                st.warning("⚠️ Soil tends to be acidic. Consider applying lime.")
            elif avg_ph > 7.5:
                st.info("ℹ️ Soil tends to be alkaline. Add organic matter or sulfur.")
            else:
                st.success("✅ Soil pH is optimal for most crops.")

        st.write("💬 General Recommendation: Regular soil testing improves long-term fertility and crop yield.")

    else:
        st.warning("⚠️ Please upload and clean a dataset first.")
