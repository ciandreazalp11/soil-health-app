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

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# ----------------- THEME CONFIG -----------------
st.markdown("""
<style>
/* Background and Text */
.stApp {
    background: linear-gradient(160deg, #0d1b0d, #1a2e1a, #253524);
    color: #e6f0e6;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111c11 !important;
    padding: 10px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] * {
    color: #d9ead3 !important;
}

/* Hover Animations */
div[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(90,143,41,0.3) !important;
    transform: scale(1.05);
    transition: all 0.3s ease-in-out;
    border-radius: 8px;
}

/* Headers */
h1, h2, h3 {
    color: #cce5cc;
    font-family: 'Trebuchet MS', sans-serif;
    font-weight: bold;
    text-shadow: 0px 0px 6px rgba(100,255,100,0.4);
}

/* Metric cards */
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

/* Animated float effect */
@keyframes float {
  0% {transform: translateY(0);}
  50% {transform: translateY(-6px);}
  100% {transform: translateY(0);}
}
[data-testid="stMetric"] { animation: float 4s ease-in-out infinite; }

/* Buttons */
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

/* Footer */
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

/* Tabs styling */
.stTabs [role="tab"] {
    border: 1px solid #5a8f29;
    border-radius: 10px;
    background: rgba(40, 60, 40, 0.3);
    margin: 3px;
}
.stTabs [role="tab"]:hover {
    background: rgba(90,143,41,0.3);
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
        with st.spinner("🧹 Cleaning and merging datasets..."):
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

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Cleaned & Merged Dataset (CSV)",
                data=csv,
                file_name="cleaned_merged_soil_dataset.csv",
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

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])

        if task == "Classification":
            model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
        else:
            model_name = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])

        st.subheader("⚙️ Model Hyperparameters")
        params = {}
        if model_name == "Random Forest":
            params["n_estimators"] = st.slider("Number of Trees", 50, 500, 100)
            params["max_depth"] = st.slider("Max Depth", 2, 20, 10)
        elif model_name == "Decision Tree":
            params["max_depth"] = st.slider("Max Depth", 2, 20, 5)
        elif model_name == "KNN":
            params["n_neighbors"] = st.slider("K Neighbors", 1, 20, 5)
        elif model_name == "SVM":
            params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

        if task == "Classification":
            df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
            X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
            y = df['Fertility_Level']
        else:
            X = df.drop(columns=['Nitrogen'])
            y = df['Nitrogen']

        X = X.select_dtypes(include=[np.number])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def get_model(name, task, params):
            if task == "Classification":
                return {
                    "Random Forest": RandomForestClassifier(random_state=42, **params),
                    "Decision Tree": DecisionTreeClassifier(random_state=42, **params),
                    "KNN": KNeighborsClassifier(**params),
                    "SVM": SVC(**params)
                }[name]
            else:
                return {
                    "Random Forest": RandomForestRegressor(random_state=42, **params),
                    "Decision Tree": DecisionTreeRegressor(random_state=42, **params),
                    "KNN": KNeighborsRegressor(**params),
                    "SVM": SVR(**params),
                    "Linear Regression": LinearRegression()
                }[name]

        with st.spinner("🧠 Training model..."):
            time.sleep(1.5)
            model = get_model(model_name, task, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        st.session_state["results"] = {
            "task": task,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "model_name": model_name,
            "X_columns": X.columns.tolist(),
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
            acc = accuracy_score(y_test, y_pred)
            color = "green" if acc > 0.8 else "orange" if acc > 0.6 else "red"
            st.metric("Accuracy", f"{acc:.2f}")
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

        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("R² Score", f"{r2:.2f}")

            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#9acd32"])
            fig.add_trace(go.Scatter(x=[np.min(y_test), np.max(y_test)], y=[np.min(y_test), np.max(y_test)], mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        if hasattr(model, "feature_importances_"):
            st.subheader("🌾 Feature Importance")
            X_columns = results["X_columns"]
            importance = pd.Series(model.feature_importances_, index=X_columns).sort_values(ascending=True)
            fig = px.bar(importance, orientation='h', title="Feature Importance", color=importance, color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
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

        st.markdown("""
        **General Recommendations:**
        - Low Nitrogen → Apply nitrogen-rich fertilizers.
        - Low Phosphorus → Use phosphate-based fertilizers.
        - High Organic Matter → Indicates good soil health.
        - Validate predictions with on-site soil testing.
        """)
    else:
        st.info("Upload a dataset to generate soil insights.")

# ----------------- FOOTER -----------------
st.markdown("<div class='footer'>👨‍💻 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> | 🌱 Capstone Project</div>", unsafe_allow_html=True)
