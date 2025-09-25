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
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(160deg, #0d1b0d, #1a2e1a, #253524);
        color: #e6f0e6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111c11 !important;
        padding: 10px;
        border-radius: 12px;
        color: #d9ead3;
    }
    section[data-testid="stSidebar"] * {
        color: #d9ead3 !important;
        font-weight: 500;
    }

    /* Hover animation for menu */
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
    }

    /* Metric cards - holographic style */
    .stMetric {
        background: rgba(30, 60, 30, 0.65);
        color: #f0f0f0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.7);
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px) scale(1.02);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.4);
    }

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

# ----------------- COMMON SETTINGS -----------------
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

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results")
    if "results" in st.session_state:
        results = st.session_state["results"]
        y_test, y_pred, task = results["y_test"], results["y_pred"], results["task"]

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
            r2 = r2_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("R² Score", f"{r2:.2f}")

            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#9acd32"])
            fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please run a model first.")

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights & Recommendations")
    st.markdown("""
    - If **Nitrogen is low**, soil may need **nitrogen-based fertilizers**.
    - If **pH < 5.5**, consider **lime treatment** to reduce acidity.
    - High **Organic Matter** usually indicates healthier soil.
    - Always validate ML predictions with **field tests**.
    """)

# ----------------- FOOTER -----------------
st.markdown("<div class='footer'>👨‍💻 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> | 🌱 Capstone Project</div>", unsafe_allow_html=True)
