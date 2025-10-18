# app_b.py
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

st.set_page_config(
    page_title="🌱 Soil Health ML App — B (Neon)",
    layout="wide",
    page_icon="🌿"
)

# ----------------- NEON THEME (UI ONLY) -----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
.stApp { font-family: 'Montserrat', sans-serif; color:#e6f7ff; min-height:100vh; background: linear-gradient(120deg, #071e1f 0%, #0b2a2b 40%, #07112a 100%); background-attachment: fixed; }
section[data-testid="stSidebar"] { background: rgba(5,10,15,0.9) !important; border-radius:12px; padding:18px; box-shadow: 0 12px 40px rgba(0,0,0,0.6); }
div[data-testid="stSidebarNav"] a { color:#bfefff !important; border-radius:8px; padding:10px; }
div[data-testid="stSidebarNav"] a:hover { background: rgba(67,97,238,0.06) !important; transform: translateX(6px); transition: all .18s; }
h1,h2,h3 { color:#a8ffea; text-shadow: 0 0 10px rgba(0,255,200,0.12); }
.stButton button { background: linear-gradient(90deg,#00f5d4,#00b4ff) !important; color:#031017 !important; border-radius:12px; padding:0.55rem 1rem; font-weight:700; box-shadow: 0 12px 34px rgba(0,180,255,0.12); }
.stButton button:hover { transform: scale(1.04); box-shadow: 0 18px 46px rgba(0,180,255,0.2); }
.legend { background: linear-gradient(90deg, rgba(10,20,40,0.6), rgba(10,20,40,0.4)); border-radius:12px; padding:10px; color:#e6fbff; }
.header-glow { position:absolute; right:20px; top:12px; width:78px; height:78px; opacity:0.95; animation: shimmer 8s linear infinite; }
@keyframes shimmer { 0% { transform: rotate(0deg) translateY(0) } 50% { transform: rotate(6deg) translateY(-10px) } 100% { transform: rotate(0deg) translateY(0) } }
.footer { text-align:center; color:#bdeeff; font-size:13px; padding:10px; margin-top:18px; }
</style>

<div style="position:relative;">
  <svg class="header-glow" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <circle cx="32" cy="32" r="28" fill="#00f5d4" opacity="0.08"/>
    <path fill="#00b4ff" d="M20 44c6-12 24-24 40-28-4 18-18 34-34 34-1 0-3-1-6-6z" opacity="0.28"/>
  </svg>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR MENU -----------------
with st.sidebar:
    selected = option_menu(
        "🌱 Soil Health App",
        ["📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"],
        icons=["cloud-upload", "bar-chart", "robot", "graph-up", "lightbulb"],
        menu_icon="list",
        default_index=0,
        styles={"container": {"padding": "5!important", "background-color": "#111c11"},
                "icon": {"color": "#9acd32", "font-size": "20px"},
                "nav-link": {"color": "#d9ead3", "font-size": "16px"},
                "nav-link-selected": {"background-color": "#5a8f29"},}
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

if "df" not in st.session_state:
    st.session_state["df"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "y_train_quantiles" not in st.session_state:
    st.session_state["y_train_quantiles"] = None

def safe_to_numeric_columns(df, cols):
    numeric_found = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_found.append(c)
    return numeric_found

def download_df_button(df, filename="final_preprocessed_soil_dataset.csv", label="⬇️ Download Cleaned & Preprocessed Data"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=filename, mime="text/csv")

def create_fertility_label(df, col="Nitrogen", q=3):
    try:
        labels = ['Low', 'Moderate', 'High']
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates='drop')
        if fert.nunique() < 3:
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        labels = ['Low', 'Moderate', 'High']
        fert = pd.cut(df[col], bins=3, labels=labels)
    return fert.astype(str)

def interpret_label(label):
    l = str(label).lower()
    if l in ["high", "good", "healthy", "3", "2"]:
        return ("Good", "green", "✅ Nutrients are balanced. Ideal for most crops.")
    if l in ["moderate", "medium", "2"]:
        return ("Moderate", "orange", "⚠️ Some nutrient imbalance. Consider minor adjustments.")
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")

# ----------------- UPLOAD DATA -----------------
if selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    uploaded_files = st.file_uploader("Upload multiple datasets (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True, key="uploader")
    if st.session_state["df"] is not None and not uploaded_files:
        st.info(f"✅ A preprocessed dataset is already loaded ({st.session_state['df'].shape[0]} rows, {st.session_state['df'].shape[1]} cols).")
        st.dataframe(st.session_state["df"].head())
        if st.button("🔁 Clear current dataset and upload new ones"):
            st.session_state["df"] = None
            st.experimental_rerun()
    cleaned_dfs = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                df_file = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    for alt in alt_names:
                        if alt in df_file.columns:
                            renamed[alt] = std_col
                            break
                df_file.rename(columns=renamed, inplace=True)
                cols_to_keep = [col for col in required_columns if col in df_file.columns]
                df_file = df_file[cols_to_keep]
                safe_to_numeric_columns(df_file, cols_to_keep)
                df_file.drop_duplicates(inplace=True)
                cleaned_dfs.append(df_file)
                st.success(f"✅ Cleaned: {file.name} ({df_file.shape[0]} rows, kept cols: {len(cols_to_keep)})")
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")
        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            safe_to_numeric_columns(df, required_columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                medians = df[numeric_cols].median()
                df[numeric_cols] = df[numeric_cols].fillna(medians)
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                try:
                    if df[c].isnull().sum() > 0:
                        df[c].fillna(df[c].mode().iloc[0], inplace=True)
                except Exception:
                    df[c].fillna(method='ffill', inplace=True)
            df.dropna(how='all', inplace=True)
            st.session_state["df"] = df
            st.subheader("🔗 Final Merged, Cleaned & Preprocessed Dataset")
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df, filename="final_preprocessed_soil_dataset.csv", label="⬇️ Download Cleaned & Preprocessed Data")
            st.success("✨ Auto preprocessing applied and dataset saved to session (used for Visualization/Modeling/Results).")
            

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Soil Data Visualization")
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for plotting.")
        else:
            feature = st.selectbox("Select a numeric feature", numeric_cols)
            fig = px.histogram(df, x=feature, nbins=30, marginal="box", color_discrete_sequence=["#00f5d4"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("🌐 Correlation Heatmap")
            corr = df.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload and preprocess data first (Upload Data tab).")

# ----------------- MODELING (Random Forest only) -----------------
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling & Prediction Using Random Forest")
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Please upload data first.")
    else:
        df = st.session_state["df"]
        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for modeling (used as target or to build fertility label).")
        else:
            task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])
            st.subheader("⚙️ Random Forest Hyperparameters")
            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100)
            max_depth = st.slider("Max Depth (max_depth)", 2, 50, 10)
            if task == "Classification":
                try:
                    df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
                except Exception:
                    df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
                X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
                y = df['Fertility_Level']
            else:
                X = df.drop(columns=['Nitrogen'])
                y = df['Nitrogen']
            X = X.select_dtypes(include=[np.number])
            if X.shape[1] == 0:
                st.error("No numeric features available for modeling after preprocessing.")
            else:
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if task == "Classification":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                with st.spinner("🧠 Training Random Forest..."):
                    time.sleep(0.5)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                st.session_state["results"] = {"task": task, "y_test": y_test.tolist(), "y_pred": y_pred.tolist(), "model_name": "Random Forest", "X_columns": X.columns.tolist()}
                st.session_state["model"] = model
                if task == "Regression":
                    st.session_state["y_train_quantiles"] = df['Nitrogen'].quantile([0.33, 0.66]).tolist()
                st.success("✅ Random Forest training completed! Go to 📈 Results to view performance.")

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results & Soil Health Interpretation")
    if not st.session_state.get("results"):
        st.info("Please run a model first (Modeling tab).")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])
        if len(y_test) != len(y_pred):
            st.error("⚠️ Mismatch between test and prediction lengths. Please retrain your model.")
        else:
            if task == "Classification":
                try:
                    acc = accuracy_score(y_test, y_pred)
                except Exception:
                    acc = 0
                color = "green" if acc > 0.8 else "orange" if acc > 0.6 else "red"
                st.metric("Accuracy", f"{acc:.2f}")
                fig = go.Figure(go.Indicator(mode="gauge+number", value=acc, gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color}}, title={'text': "Accuracy Gauge"}))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Classification Report")
                try:
                    st.text(classification_report(y_test, y_pred))
                except Exception as e:
                    st.warning(f"⚠️ Could not generate classification report: {e}")
                unique, counts = np.unique(y_pred, return_counts=True)
                pred_counts = dict(zip(unique, counts))
                st.subheader("Prediction Distribution (test set)")
                st.write(pred_counts)
                majority_label = unique[np.argmax(counts)]
                label_text, label_color, label_explanation = interpret_label(majority_label)
                st.markdown(f"### Overall Soil Health: <span style='color:{label_color}'>{label_text}</span>", unsafe_allow_html=True)
                st.write(label_explanation)
                st.markdown("""<div class='legend'><span style='background:green'></span> High / Good → Soil is healthy  <span style='background:orange'></span> Moderate → Needs attention  <span style='background:red'></span> Low / Poor → Improvement needed  </div>""", unsafe_allow_html=True)
            else:
                try:
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                except TypeError:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                except Exception as e:
                    rmse = float('nan')
                    st.warning(f"⚠️ Error computing RMSE: {e}")
                try:
                    r2 = r2_score(y_test, y_pred)
                except Exception as e:
                    r2 = float('nan')
                    st.warning(f"⚠️ Error computing R² score: {e}")
                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("R² Score", f"{r2:.2f}")
                fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#00f5d4"])
                fig.add_trace(go.Scatter(x=[np.min(y_test), np.max(y_test)], y=[np.min(y_test), np.max(y_test)], mode="lines", name="Ideal", line=dict(color="magenta", dash="dash")))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                q = st.session_state.get("y_train_quantiles")
                if q:
                    low_th, high_th = q[0], q[1]
                else:
                    df = st.session_state.get("df")
                    low_th, high_th = df['Nitrogen'].quantile([0.33, 0.66]).tolist()
                avg_pred = float(np.mean(y_pred))
                if avg_pred >= high_th:
                    st.markdown("### Overall Soil Health: <span style='color:green'>Good</span>", unsafe_allow_html=True)
                    st.write("✅ Predicted nutrient level is high — likely healthy.")
                elif avg_pred >= low_th:
                    st.markdown("### Overall Soil Health: <span style='color:orange'>Moderate</span>", unsafe_allow_html=True)
                    st.write("⚠️ Predicted nutrient level is moderate — consider adjustments.")
                else:
                    st.markdown("### Overall Soil Health: <span style='color:red'>Poor</span>", unsafe_allow_html=True)
                    st.write("🚫 Predicted nutrient level is low — corrective action recommended.")
                st.markdown("""<div class='legend'><span style='background:green'></span> Ranges above 66th percentile → Good  <span style='background:orange'></span> 33rd–66th percentile → Moderate  <span style='background:red'></span> Below 33rd percentile → Poor  </div>""", unsafe_allow_html=True)

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights & Recommendations")
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.subheader("Dataset summary (selected metrics)")
        if 'pH' in df.columns:
            st.write(f"- Average pH: {df['pH'].mean():.2f}")
        if 'Nitrogen' in df.columns:
            st.write(f"- Average Nitrogen: {df['Nitrogen'].mean():.2f}")
        if 'Moisture' in df.columns:
            st.write(f"- Average Moisture: {df['Moisture'].mean():.2f}")
        if 'Organic Matter' in df.columns:
            st.write(f"- Average Organic Matter: {df['Organic Matter'].mean():.2f}")
        st.markdown("""**Practical Recommendations**  - If **Nitrogen** is low → apply nitrogen-rich fertilizer (e.g., urea, ammonium nitrate).  - If **pH < 5.5** → consider lime application to reduce acidity.  - If **Moisture** is low (<20%) → improve irrigation / water retention.  - If **Organic Matter** > 3% → generally good; maintain with compost.  """)
    else:
        st.info("Please upload a dataset to generate insights.")

# ----------------- FOOTER -----------------
st.markdown("<div class='footer'>👨‍💻 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> | 🌱 Capstone Project</div>", unsafe_allow_html=True)

