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

# ----------------- STYLES: animated bg, holographic buttons, hover, floating logo -----------------
st.markdown("""
    <style>
    /* Animated dark-forest background (soft, non-glaring) */
    .stApp {
        background: linear-gradient(120deg, #07140a 0%, #122612 35%, #203826 70%, #2c5130 100%);
        background-size: 300% 300%;
        animation: gradientShift 18s ease infinite;
        color: #e7f3e7;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar base & readable text */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1a0b 0%, #102212 100%) !important;
        border-right: 2px solid rgba(60,100,60,0.6);
        color: #e7f3e7;
        padding: 18px;
    }
    section[data-testid="stSidebar"] * {
        color: #e6f6e6 !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #cfe9cf !important;
    }

    /* Header styles */
    .css-1d391kg h1, h1 {
        color: #dff3d8 !important;
    }
    h2, h3 {
        color: #d6f0c6 !important;
    }

    /* Holographic button & hover */
    .stButton>button {
        background: linear-gradient(135deg, rgba(40,80,40,0.95) 0%, rgba(80,140,80,0.95) 100%);
        color: #fff;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease, border-image .4s linear;
        border: 2px solid transparent;
        border-image: linear-gradient(45deg, rgba(0,255,200,0.9), rgba(160,255,120,0.9), rgba(120,200,255,0.9)) 1;
        box-shadow: 0 6px 18px rgba(0,0,0,0.55);
    }
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 12px 30px rgba(20,120,60,0.35), 0 0 18px rgba(120,255,200,0.08);
        border-image: linear-gradient(90deg, rgba(255,0,200,0.9), rgba(120,255,120,0.9), rgba(0,200,255,0.9)) 1;
    }

    /* Selectboxes, radios, sliders visibility */
    div[data-baseweb="select"] > div, .stSelectbox > div, div[role="radiogroup"] {
        background: rgba(18,40,18,0.65) !important;
        color: #e9f6e9 !important;
        border-radius: 8px;
    }
    .stRadio > label, .stRadio > div, .stSelectbox > label {
        color: #e9f6e9 !important;
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: rgba(40,80,40,0.7) !important;
    }

    /* Metric cards with hover zoom & soft glow */
    .stMetric {
        background: rgba(18,38,18,0.75);
        color: #eefaf0;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        transition: transform .22s ease, box-shadow .22s ease;
    }
    .stMetric:hover {
        transform: scale(1.04);
        box-shadow: 0 10px 30px rgba(0,140,60,0.18);
    }

    /* Holographic card border accent */
    .holo-card {
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: inset 0 0 0 1px rgba(120,255,200,0.03), 0 6px 20px rgba(0,0,0,0.5);
        background: linear-gradient(180deg, rgba(6,20,6,0.4), rgba(8,28,8,0.45));
        position: relative;
        overflow: hidden;
    }
    .holo-card:before {
        content: "";
        position: absolute;
        left: -50%;
        top: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, rgba(255,0,200,0.06), rgba(120,255,120,0.06), rgba(0,200,255,0.06));
        transform: rotate(20deg);
        animation: holoShift 6s linear infinite;
        pointer-events: none;
    }
    @keyframes holoShift {
        0% { transform: translateX(-10%) translateY(0) rotate(20deg); opacity: 0.12; }
        50% { transform: translateX(10%) translateY(6%) rotate(25deg); opacity: 0.2; }
        100% { transform: translateX(-10%) translateY(0) rotate(20deg); opacity: 0.12; }
    }

    /* Floating logo (style, not required to exist) */
    .floating-logo {
        width: 84px; height: 84px; border-radius: 50%;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        border: 2px solid rgba(255,255,255,0.03);
        animation: floatLogo 5s ease-in-out infinite;
    }
    @keyframes floatLogo {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }

    /* Dataframe rounded */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #cfe9cf;
        margin-top: 18px;
        font-size: 14px;
    }
    .footer span { color: #b8f28a; font-weight:700; }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR MENU (option_menu) -----------------
with st.sidebar:
    Selected = option_menu(
        title="🌱 Soil Health App",
        options=["📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"],
        icons=["cloud-upload", "bar-chart", "robot", "graph-up", "lightbulb"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "8px", "background-color": "transparent"},
            "icon": {"color": "#9bd98a", "font-size": "18px"},
            "nav-link": {"color": "#e7f3e7", "font-size": "15px", "text-align": "left"},
            "nav-link-selected": {"background-color": "rgba(100,170,100,0.12)", "color": "#bdf4b0"}
        }
    )

# preserve old variable name used in original code
selected = Selected

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
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload multiple datasets (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True)
    cleaned_dfs = []

    if uploaded_files:
        for file in uploaded_files:
            try:
                df = pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file)
                # Rename columns (case-insensitive partial match)
                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    for alt in alt_names:
                        for c in df.columns:
                            if c.lower() == alt.lower() or alt.lower() in c.lower():
                                renamed[c] = std_col
                                break
                        if any(v == std_col for v in renamed.values()):
                            break
                df.rename(columns=renamed, inplace=True)

                # Keep only required mapped columns
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
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Soil Data Visualization")
    if "df" in st.session_state:
        df = st.session_state["df"]

        feature = st.selectbox("Select a feature", df.columns)
        fig = px.histogram(df, x=feature, nbins=20, marginal="box",
                           color_discrete_sequence=["#9bd98a"], template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌐 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="Greens", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
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
                # robust qcut fallback
                try:
                    df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
                except Exception:
                    df['Fertility_Level'] = pd.cut(df['Nitrogen'], bins=3, labels=['Low', 'Moderate', 'High'])
                X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
                y = df['Fertility_Level']
            else:
                X = df.drop(columns=['Nitrogen'])
                y = df['Nitrogen']

            # numeric columns only & scaling
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric features available after preprocessing.")
            else:
                X = X[numeric_cols]
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                def get_model(name, task):
                    if task == "Classification":
                        return {
                            "Random Forest": RandomForestClassifier(random_state=42),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "KNN": KNeighborsClassifier(),
                            "SVM": SVC()
                        }[name]
                    else:
                        return {
                            "Random Forest": RandomForestRegressor(random_state=42),
                            "Decision Tree": DecisionTreeRegressor(random_state=42),
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
            color = "lime" if acc > 0.8 else "gold" if acc > 0.6 else "tomato"
            st.metric("Accuracy", f"{acc:.2f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=acc,
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color}},
                title={'text': "Accuracy Gauge"}
            ))
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("R² Score", f"{r2:.2f}")

            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                             color_discrete_sequence=["#9bd98a"], template="plotly_dark")
            fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                                     mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
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
st.markdown('👨‍💻 Developed by **Andre Plaza** & **Rica Baliling** | 🌱 Capstone Project')
