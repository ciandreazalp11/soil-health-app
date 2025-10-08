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
st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# ----------------- ENHANCED VISUAL STYLES (only UI changes) -----------------
st.markdown("""
<!-- Google Font -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
/* Page base */
.stApp {
    font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    color: #e6f0e6;
    min-height: 100vh;
    /* animated subtle radial gradient background */
    background:
      radial-gradient(circle at 10% 20%, rgba(35,85,35,0.18), transparent 15%),
      radial-gradient(circle at 90% 80%, rgba(50,120,50,0.12), transparent 12%),
      linear-gradient(160deg, #09120a 0%, #122612 40%, #1e3a1e 100%);
    background-attachment: fixed;
}

/* Floating leaf decorative element in header (non-invasive) */
.header-deco {
    position: absolute;
    right: 30px;
    top: 18px;
    width: 56px;
    height: 56px;
    pointer-events: none;
    opacity: 0.9;
    transform-origin: center;
    animation: floaty 6s ease-in-out infinite;
}
@keyframes floaty {
  0% { transform: translateY(0) rotate(0deg); }
  50% { transform: translateY(-8px) rotate(6deg); }
  100% { transform: translateY(0) rotate(0deg); }
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,20,10,0.9), rgba(8,18,8,0.95)) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    border-radius: 12px;
    padding: 18px;
    margin: 10px 8px 10px 10px;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #d9ead3 !important;
}
[data-testid="stSidebar"] .css-1d391kg { padding-top: 6px; }

/* Sidebar nav links */
div[data-testid="stSidebarNav"] a {
    color: #d9ead3 !important;
    border-radius: 8px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
}
div[data-testid="stSidebarNav"] a:hover {
    background: rgba(90,143,41,0.12) !important;
    transform: translateX(4px);
    transition: all 0.18s ease;
}

/* Headings */
h1, h2, h3 {
    color: #cce5cc;
    font-weight: 700;
    margin: 6px 0;
    text-shadow: 0 2px 14px rgba(0,0,0,0.45);
}

/* Card/metric look */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,60,30,0.55), rgba(20,40,20,0.35));
    padding: 12px 14px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #5a8f29, #9acd32) !important;
    color: white !important;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}

/* DataFrame styling container */
.stDataFrame, .stTable {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}

/* Legend boxes */
.legend {
    background: linear-gradient(90deg, rgba(20,40,20,0.6), rgba(20,40,20,0.3));
    border-radius: 10px;
    padding: 10px;
    font-size: 14px;
    color: #eaf6e8;
}

/* Footer */
.footer {
    text-align: center;
    color: #bcd9b2;
    font-size: 13px;
    padding: 10px;
    margin-top: 22px;
}

/* Make charts pop a little */
.plotly-graph-div {
    border-radius: 10px;
    padding: 6px;
}

/* Responsive tweaks */
@media (max-width: 768px) {
    .header-deco { display: none; }
}
</style>
""", unsafe_allow_html=True)

# decorative floating leaf (pure CSS svg) inserted at top-right of page
st.markdown("""
<div style="position:relative;">
  <svg class="header-deco" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <path fill="#9acd32" d="M12 44c6-12 24-24 40-28-4 18-18 34-34 34-1 0-3-1-6-6z" opacity="0.95"/>
    <path fill="#5a8f29" d="M50 14c-10 6-24 18-34 30 12-6 28-18 34-30z" opacity="0.12"/>
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
        styles={
            "container": {"padding": "5!important", "background-color": "#111c11"},
            "icon": {"color": "#9acd32", "font-size": "20px"},
            "nav-link": {"color": "#d9ead3", "font-size": "16px"},
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

# Initialize session_state keys we will use (safe)
if "df" not in st.session_state:
    st.session_state["df"] = None           # final preprocessed dataframe
if "results" not in st.session_state:
    st.session_state["results"] = None      # training results
if "model" not in st.session_state:
    st.session_state["model"] = None        # trained RF model
if "y_train_quantiles" not in st.session_state:
    st.session_state["y_train_quantiles"] = None  # for regression interpretation

# ----------------- HELPERS -----------------
def safe_to_numeric_columns(df, cols):
    """Coerce listed cols to numeric (inplace), return list of numeric cols found."""
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
    """Create Fertility_Level label robustly. Return series of labels."""
    try:
        # attempt qcut; catch exceptions for duplicates
        labels = ['Low', 'Moderate', 'High']
        # if qcut fails due to duplicate edges, try cut with equal-width bins
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates='drop')
        # if qcut produced fewer categories because of duplicates, fall back to cut
        if fert.nunique() < 3:
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        labels = ['Low', 'Moderate', 'High']
        fert = pd.cut(df[col], bins=3, labels=labels)
    return fert.astype(str)

def interpret_label(label):
    """Map label to interpretation text & color"""
    l = str(label).lower()
    if l in ["high", "good", "healthy", "3", "2"]:  # some label possibilities
        return ("Good", "green", "✅ Nutrients are balanced. Ideal for most crops.")
    if l in ["moderate", "medium", "2"]:
        return ("Moderate", "orange", "⚠️ Some nutrient imbalance. Consider minor adjustments.")
    # default low/others
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")

# ----------------- UPLOAD DATA -----------------
if selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    uploaded_files = st.file_uploader(
        "Upload multiple datasets (.csv or .xlsx)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True,
        key="uploader"  # stable widget key
    )

    # If there's already a preprocessed df in session, show it and allow clearing
    if st.session_state["df"] is not None and not uploaded_files:
        st.info(f"✅ A preprocessed dataset is already loaded ({st.session_state['df'].shape[0]} rows, {st.session_state['df'].shape[1]} cols).")
        st.dataframe(st.session_state["df"].head())
        if st.button("🔁 Clear current dataset and upload new ones"):
            st.session_state["df"] = None
            st.experimental_rerun()

    cleaned_dfs = []
    if uploaded_files:
        # Process uploaded files
        for file in uploaded_files:
            try:
                df_file = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                # rename alternative column names to standard
                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    for alt in alt_names:
                        if alt in df_file.columns:
                            renamed[alt] = std_col
                            break
                df_file.rename(columns=renamed, inplace=True)

                # keep only the required columns that exist in this file
                cols_to_keep = [col for col in required_columns if col in df_file.columns]
                df_file = df_file[cols_to_keep]

                # coerce required numeric columns to numeric (inplace)
                safe_to_numeric_columns(df_file, cols_to_keep)

                # drop duplicates (row-wise)
                df_file.drop_duplicates(inplace=True)

                cleaned_dfs.append(df_file)
                st.success(f"✅ Cleaned: {file.name} ({df_file.shape[0]} rows, kept cols: {len(cols_to_keep)})")
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs:
            # Merge all uploaded datasets
            df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)

            # ---------- AUTO PREPROCESSING ----------
            # Replace empty strings with NaN
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # Ensure numeric columns present are numeric
            safe_to_numeric_columns(df, required_columns)

            # Fill numeric missing with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                medians = df[numeric_cols].median()
                df[numeric_cols] = df[numeric_cols].fillna(medians)

            # Fill categorical missing with mode (if any categorical cols)
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                try:
                    if df[c].isnull().sum() > 0:
                        df[c].fillna(df[c].mode().iloc[0], inplace=True)
                except Exception:
                    df[c].fillna(method='ffill', inplace=True)

            # Drop rows that are completely empty
            df.dropna(how='all', inplace=True)

            # Persist final dataframe to session state
            st.session_state["df"] = df

            # Show results
            st.subheader("🔗 Final Merged, Cleaned & Preprocessed Dataset")
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())

            # Single download button for the final preprocessed data
            download_df_button(df, filename="final_preprocessed_soil_dataset.csv",
                               label="⬇️ Download Cleaned & Preprocessed Data")

            st.success("✨ Auto preprocessing applied and dataset saved to session (used for Visualization/Modeling/Results).")
            st.balloons()

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Soil Data Visualization")
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        # choose a numeric feature to plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for plotting.")
        else:
            feature = st.selectbox("Select a numeric feature", numeric_cols)
            fig = px.histogram(df, x=feature, nbins=30, marginal="box", color_discrete_sequence=["#9acd32"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🌐 Correlation Heatmap")
            corr = df.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Greens")
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

        # Require nitrogen column for training target base
        if 'Nitrogen' not in df.columns:
            st.error("❗ 'Nitrogen' column required for modeling (used as target or to build fertility label).")
        else:
            # Task selection
            task = st.radio("🧠 Prediction Task", ["Classification", "Regression"])

            # Hyperparameters for Random Forest
            st.subheader("⚙️ Random Forest Hyperparameters")
            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100)
            max_depth = st.slider("Max Depth (max_depth)", 2, 50, 10)

            # Prepare X and y
            if task == "Classification":
                # create fertility level label robustly
                try:
                    df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
                except Exception:
                    df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
                X = df.drop(columns=['Nitrogen', 'Fertility_Level'])
                y = df['Fertility_Level']
            else:
                X = df.drop(columns=['Nitrogen'])
                y = df['Nitrogen']

            # select only numeric features for X
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

                # store results and model in session_state
                st.session_state["results"] = {
                    "task": task,
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "model_name": "Random Forest",
                    "X_columns": X.columns.tolist()
                }
                st.session_state["model"] = model

                # For regression interpretation: save training quantiles of original Nitrogen
                if task == "Regression":
                    # use original df['Nitrogen'] quantiles
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

        # --- SAFETY CHECK ---
        if len(y_test) != len(y_pred):
            st.error("⚠️ Mismatch between test and prediction lengths. Please retrain your model.")
        else:
            # Classification results
            if task == "Classification":
                try:
                    acc = accuracy_score(y_test, y_pred)
                except Exception:
                    acc = 0
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

                st.markdown("""
                <div class='legend'>
                    <span style='background:green'></span> High / Good → Soil is healthy  
                    <span style='background:orange'></span> Moderate → Needs attention  
                    <span style='background:red'></span> Low / Poor → Improvement needed  
                </div>
                """, unsafe_allow_html=True)

            # Regression results
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

                fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=["#9acd32"])
                fig.add_trace(go.Scatter(
                    x=[np.min(y_test), np.max(y_test)], 
                    y=[np.min(y_test), np.max(y_test)], 
                    mode="lines", 
                    name="Ideal", 
                    line=dict(color="red", dash="dash")
                ))
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

                st.markdown("""
                <div class='legend'>
                    <span style='background:green'></span> Ranges above 66th percentile → Good  
                    <span style='background:orange'></span> 33rd–66th percentile → Moderate  
                    <span style='background:red'></span> Below 33rd percentile → Poor  
                </div>
                """, unsafe_allow_html=True)

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights & Recommendations")
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]

        # Basic stats for indicators
        st.subheader("Dataset summary (selected metrics)")
        if 'pH' in df.columns:
            st.write(f"- Average pH: {df['pH'].mean():.2f}")
        if 'Nitrogen' in df.columns:
            st.write(f"- Average Nitrogen: {df['Nitrogen'].mean():.2f}")
        if 'Moisture' in df.columns:
            st.write(f"- Average Moisture: {df['Moisture'].mean():.2f}")
        if 'Organic Matter' in df.columns:
            st.write(f"- Average Organic Matter: {df['Organic Matter'].mean():.2f}")

        st.markdown("""
        **Practical Recommendations**  
        - If **Nitrogen** is low → apply nitrogen-rich fertilizer (e.g., urea, ammonium nitrate).  
        - If **pH < 5.5** → consider lime application to reduce acidity.  
        - If **Moisture** is low (<20%) → improve irrigation / water retention.  
        - If **Organic Matter** > 3% → generally good; maintain with compost.  
        """)
    else:
        st.info("Please upload a dataset to generate insights.")

# ----------------- FOOTER -----------------
st.markdown("<div class='footer'>👨‍💻 Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> | 🌱 Capstone Project</div>", unsafe_allow_html=True)
