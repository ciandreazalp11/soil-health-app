import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Streamlit config
st.set_page_config(page_title="Soil Health ML App", layout="wide")
st.title("🌱 Soil Health ML App with Merging, Cleaning & Visualization")

# Column mapping (for flexible naming)
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level', 'Soil_TN', 'Soil_AN'],
    'Phosphorus': ['Phosphorus', 'P', 'Phosphorus_Level', 'Soil_TP', 'Soil_AP'],
    'Potassium': ['Potassium', 'K', 'Potassium_Level'],
    'Moisture': ['Moisture', 'Soil_Moisture', 'soil_moisture_%'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc', 'Organic_Carbon']
}
required_columns = list(column_mapping.keys())

# Upload multiple files
uploaded_files = st.file_uploader("📁 Upload multiple soil datasets (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True)

cleaned_dfs = []

# Clean and merge logic
if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

            # Rename columns
            renamed = {}
            for std, alts in column_mapping.items():
                for alt in alts:
                    if alt in df.columns:
                        renamed[alt] = std
                        break
            df.rename(columns=renamed, inplace=True)

            # Keep only matched columns
            present_cols = [col for col in required_columns if col in df.columns]
            df = df[present_cols].dropna().drop_duplicates()
            cleaned_dfs.append(df)

            st.success(f"✅ Processed: {file.name} ({df.shape[0]} rows)")
        except Exception as e:
            st.warning(f"⚠️ Skipped {file.name}: {e}")

# Merge and display
if cleaned_dfs:
    df = pd.concat(cleaned_dfs, ignore_index=True)
    st.subheader("🔗 Merged & Cleaned Dataset")
    st.dataframe(df.head())

    # Sidebar filter
    st.sidebar.header("🔍 Filter")
    if 'pH' in df.columns:
        ph_range = st.sidebar.slider("pH Range", float(df['pH'].min()), float(df['pH'].max()), (5.5, 7.5))
        df = df[(df['pH'] >= ph_range[0]) & (df['pH'] <= ph_range[1])]

    present_cols = df.columns.tolist()

    # Feature Distribution Chart
    st.subheader("📉 Feature Distribution")
    feature = st.selectbox("Choose a feature to visualize", present_cols)
    fig1, ax1 = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax1)
    st.pyplot(fig1)

    # Select Prediction Mode
    mode = st.selectbox("🎯 Prediction Mode", ["Classification (Fertility Level)", "Regression (Nitrogen Value)"])

    # Normalize
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=present_cols)

    # ML Model Logic
    if mode.startswith("Classification"):
        if 'Nitrogen' not in df.columns:
            st.error("❌ Classification requires the 'Nitrogen' column.")
            st.stop()

        df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
        y = df['Fertility_Level']
        X = df.drop(columns=['Fertility_Level', 'Nitrogen'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Classification Complete")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📊 Feature Importance")
        fig2, ax2 = plt.subplots()
        ax2.barh(X.columns, model.feature_importances_)
        ax2.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig2)

    else:
        if 'Nitrogen' not in df.columns:
            st.error("❌ Regression requires the 'Nitrogen' column.")
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
        st.write("RMSE:", rmse)
        st.write("R² Score:", r2)

        st.subheader("📈 Actual vs Predicted Nitrogen")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, y_pred, alpha=0.6)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        st.pyplot(fig3)
