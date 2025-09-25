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
        background: linear-gradient(135deg, #eaf4e1, #f5f5dc);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eaf4e1 !important;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0px 0px 12px rgba(0,0,0,0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: #2e4600;
        font-family: 'Trebuchet MS', sans-serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Metric cards */
    .stMetric {
        background: rgba(255,255,255,0.8);
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stMetric:hover {
        transform: scale(1.03);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Holographic hover effect for buttons */
    button[kind="primary"] {
        background: linear-gradient(45deg, #5a8f29, #b4ec51, #429321);
        background-size: 300% 300%;
        animation: gradientShift 5s ease infinite;
        border: none;
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 8px 16px;
    }
    button[kind="primary"]:hover {
        box-shadow: 0px 0px 20px rgba(90,143,41,0.6);
        transform: scale(1.05);
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
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
            "container": {"padding": "5!important", "background-color": "#eaf4e1"},
            "icon": {"color": "#2e4600", "font-size": "20px"},
            "nav-link": {"color":"#2e4600","font-size": "16px"},
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
        fig = px.histogram(df, x=feature, nbins=20, marginal="box", color_discrete_sequence=["#5a8f29"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌐 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="YlGn")
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

            # Define interpretation ranges
            if acc < 0.6:
                interpretation = "❌ Poor Accuracy – Model needs significant improvement."
                color = "red"
            elif 0.6 <= acc < 0.75:
                interpretation = "⚠️ Fair Accuracy – Model may be usable but not reliable."
                color = "orange"
            elif 0.75 <= acc < 0.9:
                interpretation = "✅ Good Accuracy – Model performs well."
                color = "yellowgreen"
            else:
                interpretation = "🌟 Excellent Accuracy – Model is highly reliable."
                color = "green"

            st.metric("Accuracy", f"{acc:.2f}")
            st.markdown(f"**Legend:** {interpretation}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=acc,
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 0.6], 'color': "lightcoral"},
                        {'range': [0.6, 0.75], 'color': "orange"},
                        {'range': [0.75, 0.9], 'color': "yellowgreen"},
                        {'range': [0.9, 1], 'color': "green"}
                    ]
                },
                title={'text': "Accuracy Gauge"}
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("R² Score", f"{r2:.2f}")

            # Add interpretation legend for R²
            if r2 < 0.3:
                interp = "❌ Poor fit – Model explains very little variance."
            elif 0.3 <= r2 < 0.6:
                interp = "⚠️ Fair fit – Model explains some variance."
            elif 0.6 <= r2 < 0.8:
                interp = "✅ Good fit – Model explains most variance."
            else:
                interp = "🌟 Excellent fit – Model explains nearly all variance."

            st.markdown(f"**Legend:** {interp}")

            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={"x": "Actual", "y": "Predicted"},
                color_discrete_sequence=["#5a8f29"]
            )
            fig.add_trace(go.Scatter(
                x=[min(y_test), max(y_test)], 
                y=[min(y_test), max(y_test)], 
                mode="lines", 
                name="Ideal", 
                line=dict(color="red", dash="dash")
            ))
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
st.markdown("---")
st.markdown("👨‍💻 Developed by **Andre Plaza** | 🌱 Capstone Project")
