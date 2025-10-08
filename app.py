import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import base64
import io

st.set_page_config(page_title="Soil Health Prediction System", layout="wide")

# ------------------- STYLES -------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg, #222, #000);}
h1, h2, h3, h4, h5 {color: #9acd32;}
legend, .legend span {display: inline-block; width: 20px; height: 10px; margin-right: 5px;}
</style>
""", unsafe_allow_html=True)

# ------------------- FUNCTIONS -------------------
def preprocess_data(df):
    df = df.copy()
    df = df.replace(["", " ", "NA", "NaN", None], np.nan)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def interpret_label(label):
    label = str(label).lower()
    if label in ["good", "high", "fertile"]:
        return "Good", "green", "✅ Soil is healthy and rich in nutrients."
    elif label in ["moderate", "medium"]:
        return "Moderate", "orange", "⚠️ Soil health is average, consider fertilizer adjustments."
    else:
        return "Poor", "red", "🚫 Soil quality is poor and needs attention."

# ------------------- SIDEBAR MENU -------------------
with st.sidebar:
    selected = st.radio("📊 Navigation", [
        "📁 Upload Data",
        "🧹 Cleaning & Preprocessing",
        "📈 Visualization",
        "🤖 Modeling",
        "📉 Results"
    ])

# ------------------- UPLOAD DATA -------------------
if selected == "📁 Upload Data":
    st.title("📁 Upload Datasets")
    uploaded_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            dfs.append(df)
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        st.session_state["raw_data"] = merged_df
        st.success(f"✅ Successfully merged {len(uploaded_files)} files.")
        st.dataframe(merged_df.head())

# ------------------- CLEANING & PREPROCESSING -------------------
elif selected == "🧹 Cleaning & Preprocessing":
    st.title("🧹 Data Cleaning & Preprocessing")
    if "raw_data" not in st.session_state:
        st.warning("Please upload dataset(s) first.")
    else:
        df = st.session_state["raw_data"].copy()
        df = df.drop_duplicates()
        df = df.dropna(how="all")
        df = preprocess_data(df)
        st.session_state["clean_data"] = df
        st.success("✅ Data cleaned and preprocessed automatically.")
        st.dataframe(df.head())

        # One download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned & Preprocessed Data", csv, "cleaned_merged_preprocessed.csv", "text/csv")

# ------------------- VISUALIZATION -------------------
elif selected == "📈 Visualization":
    st.title("📊 Data Visualization")
    if "clean_data" not in st.session_state:
        st.warning("Please clean and preprocess data first.")
    else:
        df = st.session_state["clean_data"]
        st.write("### Feature Overview")
        st.dataframe(df.describe())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=["#9acd32"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# ------------------- MODELING -------------------
elif selected == "🤖 Modeling":
    st.title("🤖 Model Training (Random Forest)")
    if "clean_data" not in st.session_state:
        st.warning("Please clean and preprocess data first.")
    else:
        df = st.session_state["clean_data"]
        target_col = st.selectbox("Select Target Column (to predict)", df.columns)
        task = st.radio("Select Task Type", ["Classification", "Regression"])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if st.button("🚀 Train Random Forest Model"):
            if task == "Classification":
                model = RandomForestClassifier(random_state=42)
            else:
                model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state["results"] = {
                "task": task,
                "y_test": y_test,
                "y_pred": y_pred
            }
            st.success("✅ Model training completed!")

# ------------------- RESULTS -------------------
elif selected == "📉 Results":
    st.title("📉 Model Results & Soil Health Insights")
    if not st.session_state.get("results"):
        st.info("Please train a model first.")
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

                avg_pred = float(np.mean(y_pred))
                q_low, q_high = np.percentile(y_test, [33, 66])

                if avg_pred >= q_high:
                    st.markdown("### Overall Soil Health: <span style='color:green'>Good</span>", unsafe_allow_html=True)
                    st.write("✅ Predicted nutrient level is high — likely healthy.")
                elif avg_pred >= q_low:
                    st.markdown("### Overall Soil Health: <span style='color:orange'>Moderate</span>", unsafe_allow_html=True)
                    st.write("⚠️ Predicted nutrient level is moderate — consider adjustments.")
                else:
                    st.markdown("### Overall Soil Health: <span style='color:red'>Poor</span>", unsafe_allow_html=True)
                    st.write("🚫 Predicted nutrient level is low — corrective action recommended.")

                st.markdown("""
                <div class='legend'>
                    <span style='background:green'></span> Above 66th percentile → Good  
                    <span style='background:orange'></span> 33rd–66th percentile → Moderate  
                    <span style='background:red'></span> Below 33rd percentile → Poor  
                </div>
                """, unsafe_allow_html=True)
