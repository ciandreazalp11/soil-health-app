import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from io import BytesIO
import joblib
import time

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="🌱 Soil Health ML App", layout="wide", page_icon="🌿")

# ----------------- STYLES -----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #0d1b0d, #1a2e1a, #253524);
    color: #e6f0e6;
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #111c11 !important;
    padding: 10px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] * {
    color: #d9ead3 !important;
}
h1, h2, h3 {
    color: #cce5cc;
    font-weight: bold;
    text-shadow: 0px 0px 6px rgba(100,255,100,0.4);
}
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
            "nav-link": {"color": "#d9ead3", "font-size": "16px"},
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

# ----------------- PREPROCESS FUNCTION -----------------
def preprocess_data(df):
    df = df.replace(["", " ", "NA", "NaN", None], np.nan)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# ----------------- UPLOAD DATA -----------------
if selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    uploaded_files = st.file_uploader("Upload multiple datasets (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True)
    cleaned_dfs = []

    if uploaded_files:
        with st.spinner("🧹 Cleaning, merging, and preprocessing datasets..."):
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
                    df = preprocess_data(df)
                    cleaned_dfs.append(df)
                    st.success(f"✅ Processed: {file.name} ({df.shape[0]} rows)")
                except Exception as e:
                    st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True)
            st.session_state["df"] = df
            st.subheader("🔗 Final Preprocessed Dataset")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Preprocessed Dataset (CSV)",
                data=csv,
                file_name="final_preprocessed_soil_dataset.csv",
                mime="text/csv"
            )
            st.balloons()

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Soil Data Visualization")
    if "df" in st.session_state:
        df = st.session_state["df"]
        feature = st.selectbox("Select a feature", df.columns)
        fig = px.histogram(df, x=feature, nbins=20, marginal="box", color_discrete_sequence=["#9acd32"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌐 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Greens")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data first.")

# ----------------- MODELING -----------------
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling & Prediction (Random Forest Only)")
    if "df" not in st.session_state:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state["df"]
        task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])

        if task == "Classification":
            target_col = st.selectbox("Select Target Column", df.columns)
            df[target_col] = pd.qcut(df[target_col], q=3, labels=['Low', 'Moderate', 'High'])
            X = df.drop(columns=[target_col])
            y = df[target_col]
            model = RandomForestClassifier(random_state=42)
        else:
            target_col = st.selectbox("Select Target Column", df.columns)
            X = df.drop(columns=[target_col])
            y = df[target_col]
            model = RandomForestRegressor(random_state=42)

        X = X.select_dtypes(include=[np.number])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        with st.spinner("🧠 Training Random Forest model..."):
            time.sleep(1.5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        st.session_state["results"] = {
            "task": task,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "model": model,
            "target": target_col
        }

        joblib.dump(model, 'soil_model.pkl')
        st.download_button("⬇️ Download Trained Model", data=open('soil_model.pkl', 'rb'), file_name='soil_model.pkl')
        st.success("✅ Model training completed! Go to 📈 Results to view performance.")
        st.snow()

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results & Soil Health Insights")
    if "results" not in st.session_state:
        st.info("Please train a model first.")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2f}")
            color = "green" if acc > 0.8 else "orange" if acc > 0.6 else "red"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=acc,
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color}},
                title={'text': "Accuracy Gauge"}
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Soil Health Interpretation
            pred_labels, counts = np.unique(y_pred, return_counts=True)
            dominant = pred_labels[np.argmax(counts)]
            if dominant == 'High':
                st.success("🌿 Soil Health: GOOD — High fertility and nutrients present.")
            elif dominant == 'Moderate':
                st.warning("⚠️ Soil Health: MODERATE — Some nutrient imbalance detected.")
            else:
                st.error("🚫 Soil Health: POOR — Low fertility, corrective measures needed.")

        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("R² Score", f"{r2:.2f}")
            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#9acd32"])
            fig.add_trace(go.Scatter(x=[np.min(y_test), np.max(y_test)], y=[np.min(y_test), np.max(y_test)],
                                     mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            avg_pred = np.mean(y_pred)
            st.markdown(f"**Average Predicted Nutrient Value:** {avg_pred:.2f}")
            if avg_pred >= np.percentile(y_test, 66):
                st.success("🌿 Soil Health: GOOD — High nutrient level.")
            elif avg_pred >= np.percentile(y_test, 33):
                st.warning("⚠️ Soil Health: MODERATE — Medium nutrient level.")
            else:
                st.error("🚫 Soil Health: POOR — Low nutrient level detected.")

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights")
    if "df" not in st.session_state:
        st.info("Upload and process data first.")
    else:
        df = st.session_state["df"]
        avg_ph = df["pH"].mean()
        st.markdown(f"**Average Soil pH:** {avg_ph:.2f}")
        if avg_ph < 5.5:
            st.warning("⚠️ Soil is acidic — consider lime application.")
        elif avg_ph > 7.5:
            st.info("ℹ️ Soil is alkaline — add organic matter or sulfur.")
        else:
            st.success("✅ Soil pH is within optimal range (5.5–7.5).")

# ----------------- FOOTER -----------------
st.markdown("<div class='footer'>👨‍💻 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> | 🌱 Capstone Project</div>", unsafe_allow_html=True)
