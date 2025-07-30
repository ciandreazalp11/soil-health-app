import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Column mapping
column_mapping = {
    'pH': ['pH', 'ph', 'soil_pH', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level', 'Soil_TN', 'Soil_AN'],
    'Phosphorus': ['Phosphorus', 'P', 'Phosphorus_Level', 'Soil_TP', 'Soil_AP'],
    'Potassium': ['Potassium', 'K', 'Potassium_Level'],
    'Moisture': ['Moisture', 'soil_moisture_%', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc', 'Organic_Carbon']
}
required_columns = list(column_mapping.keys())

st.set_page_config(page_title="Soil Health ML App", layout="wide")
st.title("🌱 Soil Health ML Predictor with Visualization")
st.markdown("Upload a soil dataset (.csv or .xlsx) to analyze and predict soil nutrient values.")

uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

mode = st.selectbox("Choose prediction mode:", ["Classification (Fertility Level)", "Regression (Nitrogen Value)"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Rename columns based on mapping
        renamed_columns = {}
        for standard_col, alternatives in column_mapping.items():
            for alt in alternatives:
                if alt in df.columns:
                    renamed_columns[alt] = standard_col
                    break
        df.rename(columns=renamed_columns, inplace=True)

        # Filter columns
        present_cols = [col for col in required_columns if col in df.columns]
        if len(present_cols) < 4:
            st.error("Not enough required columns found. Please upload a valid soil dataset.")
            st.stop()

        df = df[present_cols].dropna().drop_duplicates()

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df)

        # Sidebar filter
        st.sidebar.header("🔍 Filter Data")
        if 'pH' in df.columns:
            ph_range = st.sidebar.slider("pH Range", float(df['pH'].min()), float(df['pH'].max()), (5.5, 7.5))
            df = df[(df['pH'] >= ph_range[0]) & (df['pH'] <= ph_range[1])]

        # Feature distribution
        st.subheader("📉 Feature Distribution")
        feature_to_plot = st.selectbox("Select feature for histogram:", present_cols)
        fig1, ax1 = plt.subplots()
        sns.histplot(df[feature_to_plot], kde=True, ax=ax1)
        st.pyplot(fig1)

        # Normalize
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=present_cols)

        if mode.startswith("Classification"):
            if 'Nitrogen' not in df.columns:
                st.error("Nitrogen column is required for classification.")
                st.stop()

            df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
            y = df['Fertility_Level']
            X = df.drop(columns=['Fertility_Level', 'Nitrogen'])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("✅ Classification Complete")
            st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Feature importance chart
            st.subheader("🔬 Feature Importance")
            importance = model.feature_importances_
            fig2, ax2 = plt.subplots()
            ax2.barh(X.columns, importance)
            ax2.set_title("Random Forest Feature Importance")
            st.pyplot(fig2)

        else:
            if 'Nitrogen' not in df.columns:
                st.error("Nitrogen column is required for regression.")
                st.stop()

            y = df['Nitrogen']
            X = df.drop(columns=['Nitrogen'])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.success("✅ Regression Complete")
            st.write("**RMSE:**", rmse)
            st.write("**R² Score:**", r2)

            st.subheader("📈 Actual vs Predicted Nitrogen")
            fig3, ax3 = plt.subplots()
            ax3.scatter(y_test, y_pred, alpha=0.6)
            ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax3.set_xlabel("Actual Nitrogen")
            ax3.set_ylabel("Predicted Nitrogen")
            ax3.set_title("Actual vs Predicted")
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error: {e}")
