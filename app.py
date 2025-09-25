import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Soil Health Predictor", page_icon="🌱", layout="wide")

# ----------------- CUSTOM CSS -----------------
st.markdown("""
    <style>
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0d1b0d, #1a2e1a, #253524, #3d6b3d);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #e6f0e6;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0e1a0e !important;
        border-right: 2px solid #2f4f2f;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f8f0 !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #d9f2d9 !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #cce5cc !important;
        font-family: 'Trebuchet MS', sans-serif;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2e4f2e, #4a6b4a);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border: 2px solid;
        border-image: linear-gradient(45deg, #00ffcc, #99ff33, #00ccff) 1;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3d6b3d, #6b9e6b);
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 18px rgba(0,0,0,0.6);
        border-image: linear-gradient(45deg, #ff33cc, #33ff99, #3399ff) 1;
    }

    /* Metric cards */
    .stMetric {
        background: #1a2e1a;
        color: #f0f0f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stMetric:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,255,100,0.4);
    }

    /* Floating animation (apply to logo/icons/images) */
    img {
        animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.4);
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

# ----------------- APP CONTENT -----------------
st.title("🌱 Soil Health Prediction App")
st.markdown("Predict soil health indicators and visualize insights with AI-powered models.")

# Sidebar inputs
st.sidebar.header("🔧 Input Soil Parameters")
pH = st.sidebar.slider("Soil pH", 3.0, 9.0, 6.5)
nitrogen = st.sidebar.slider("Nitrogen (mg/kg)", 0, 100, 50)
phosphorus = st.sidebar.slider("Phosphorus (mg/kg)", 0, 100, 40)
potassium = st.sidebar.slider("Potassium (mg/kg)", 0, 100, 45)
moisture = st.sidebar.slider("Moisture (%)", 0, 100, 30)
organic_matter = st.sidebar.slider("Organic Matter (%)", 0.0, 10.0, 2.5)

# Generate dummy dataset for model training
np.random.seed(42)
data = pd.DataFrame({
    "pH": np.random.uniform(4, 8, 200),
    "Nitrogen": np.random.uniform(10, 90, 200),
    "Phosphorus": np.random.uniform(5, 95, 200),
    "Potassium": np.random.uniform(10, 90, 200),
    "Moisture": np.random.uniform(5, 60, 200),
    "OrganicMatter": np.random.uniform(0.5, 8, 200),
})
data["SoilHealthIndex"] = (
    0.2*data["pH"] +
    0.25*data["Nitrogen"]/100 +
    0.2*data["Phosphorus"]/100 +
    0.2*data["Potassium"]/100 +
    0.1*data["Moisture"]/100 +
    0.05*data["OrganicMatter"]/10
)

# Train model
X = data.drop("SoilHealthIndex", axis=1)
y = data["SoilHealthIndex"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Prediction
input_data = pd.DataFrame([[pH, nitrogen, phosphorus, potassium, moisture, organic_matter]],
                          columns=X.columns)
prediction = model.predict(input_data)[0]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):.3f}")
col2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
col3.metric("Predicted Soil Health Index", f"{prediction:.2f}")

# Visualization
st.subheader("📊 Soil Parameter Distribution")
fig = px.scatter_matrix(data, dimensions=["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "OrganicMatter"],
                        color="SoilHealthIndex", title="Soil Parameter Relationships",
                        color_continuous_scale="YlGn")
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
Developed by <span>Andre Plaza</span> & <span>Rica Baliling</span> 🌱
</div>
""", unsafe_allow_html=True)
