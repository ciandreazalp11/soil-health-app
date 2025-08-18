import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="🌿 Plaza and Baliling Capstone Soil Health ML App", layout="wide")
st.title("🌿 Soil Health ML App with Merging, Cleaning & Multiple Models")
st.markdown("Upload your soil datasets, merge them, clean the data, and run predictions using various ML models.")

# Sidebar Setup
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

# Upload multiple files
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

            # Filter + clean
            df = df[[col for col in required_columns if col in df.columns]]
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            cleaned_dfs.append(df)
            st.success(f"✅ Cleaned: {file.name} ({df.shape[0]} rows)")

        except Exception as e:
            st.warning(f"⚠️ Skipped {file.name}: {e}")

# Merge all cleaned data
if cleaned_dfs:
    df = pd.concat(cleaned_dfs, ignore_index=True)
    st.subheader("🔗 Merged & Cleaned Dataset")
    st.dataframe(df.head())

    # Filter by pH
    if 'pH' in df.columns:
        ph_range = st.sidebar.slider("Filter pH", float(df['pH'].min()), float(df['pH'].max()), (5.5, 7.5))
        df = df[(df['pH'] >= ph_range[0]) & (df['pH'] <= ph_range[1])]

    # Feature distribution
    st.subheader("📊 Feature Distribution")
    selected_feature = st.selectbox("Choose a feature to plot:", df.columns)
    fig1, ax1 = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax1)
    st.pyplot(fig1)

    # Choose task and model
    task = st.sidebar.radio("🧠 Prediction Task", ["Classification", "Regression"])
    if task == "Classification":
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
    else:
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])

    # Select features and labels
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

    # Results
    st.subheader("📈 Results")
    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")
        # Scatter plot
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred, alpha=0.6)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        st.pyplot(fig2)

