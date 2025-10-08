import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# ----------------- THEME CONFIG -----------------
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
div[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(90,143,41,0.3) !important;
    transform: scale(1.05);
    transition: all 0.3s ease-in-out;
    border-radius: 8px;
}
h1, h2, h3 {
    color: #cce5cc;
    font-family: 'Trebuchet MS', sans-serif;
    font-weight: bold;
    text-shadow: 0px 0px 6px rgba(100,255,100,0.4);
}
[data-testid="stMetric"] {
    background: rgba(30, 60, 30, 0.65);
    color: #f0f0f0;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.7);
    backdrop-filter: blur(10px);
    text-align: center;
    transition: transform 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px) scale(1.02);
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
            "nav-link": {"color":"#d9ead3","font-size": "16px"},
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
                    cleaned_dfs.append(df)
                    st.success(f"✅ Loaded: {file.name} ({df.shape[0]} rows)")
                except Exception as e:
                    st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True)
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            # Fill missing numeric and categorical
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            df.dropna(how='all', inplace=True)

            st.session_state["df"] = df
            st.subheader("🔗 Final Preprocessed Dataset")
            st.dataframe(df.head())

            csv_preprocessed = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Final Preprocessed Dataset (CSV)",
                data=csv_preprocessed,
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
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌐 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Greens")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data first.")

# ----------------- MODELING -----------------
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling & Prediction")
    if "df" in st.session_state:
        df = st.session_state["df"]
        task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])

        if task == "Classification":
            df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
            X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
            y = df['Fertility_Level']
            model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10)
        else:
            X = df.drop(columns=['Nitrogen'])
            y = df['Nitrogen']
            model = RandomForestRegressor(random_state=42, n_estimators=150, max_depth=10)

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
            "model": model
        }
        joblib.dump(model, 'soil_model.pkl')
        st.download_button("⬇️ Download Trained Model", data=open('soil_model.pkl','rb'), file_name='soil_model.pkl')
        st.success("✅ Model training completed! Go to 📈 Results to view performance.")
        st.snow()
    else:
        st.info("Please upload data first.")

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results")
    if "results" in st.session_state:
        results = st.session_state["results"]
        task = results["task"]
        model = results["model"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        if task == "Classification":
            try:
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc:.2f}")
                if acc >= 0.8:
                    st.success("🌿 Excellent Prediction: Soil condition predictions are highly reliable.")
                elif acc >= 0.6:
                    st.warning("🌾 Moderate Prediction: Model performance is acceptable but could improve.")
                else:
                    st.error("🌱 Weak Prediction: Predictions may not be reliable, check data quality.")

                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
            except Exception as e:
                st.error(f"⚠️ Error evaluating classification: {e}")

        else:
            try:
                # Safe RMSE calculation for all sklearn versions
                try:
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                except TypeError:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("MAE", f"{mae:.2f}")
                col3.metric("R²", f"{r2:.2f}")

                if r2 >= 0.8:
                    st.success("🌿 Excellent Fit: Model explains most of the soil health variation.")
                elif r2 >= 0.6:
                    st.warning("🌾 Moderate Fit: Model is okay but can be tuned.")
                else:
                    st.error("🌱 Poor Fit: Consider improving dataset quality or parameters.")
            except Exception as e:
                st.error(f"⚠️ Error evaluating regression: {e}")

    else:
        st.info("Please run a model first.")

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights & Recommendations")
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

        avg_n = df["Nitrogen"].mean()
        if avg_n < df["Nitrogen"].quantile(0.33):
            st.warning("Low Nitrogen — apply nitrogen-rich fertilizers.")
        elif avg_n > df["Nitrogen"].quantile(0.67):
            st.success("Good Nitrogen Level — healthy soil nutrient balance.")
        else:
            st.info("Moderate Nitrogen — acceptable but monitor regularly.")
    else:
        st.info("Upload a dataset to generate soil insights.")

# ----------------- FOOTER -----------------
st.markdown("<div class='footer'>👨‍💻 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> | 🌱 Capstone Project</div>", unsafe_allow_html=True)
