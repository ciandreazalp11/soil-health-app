import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# optional: Plotly if available (graceful fallback)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="🌿 Soil Health ML App", layout="wide")

# --- Simple soil theme CSS ---
st.markdown("""
    <style>
      .stApp { background: #fbfbf6; }
      .title { color: #234d20; font-weight:700; }
      .metric-good { color: #1b8a36; font-weight:700; }
      .metric-moderate { color: #d38a14; font-weight:700; }
      .metric-poor { color: #c92a2a; font-weight:700; }
      .card { background:#ffffff; border-radius:12px; padding:12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Soil Health ML App")
st.write("Upload soil datasets, merge & clean, and run classification/regression models.")

# Sidebar
st.sidebar.header("⚙️ App Settings")
task = st.sidebar.radio("Prediction task", ["Classification", "Regression"])

# Column mapping (common alt names)
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc']
}
required_columns = list(column_mapping.keys())

# Upload files
uploaded_files = st.file_uploader("Upload dataset files (.csv or .xlsx)", type=['csv', 'xlsx'], accept_multiple_files=True)
cleaned_dfs = []

def read_file(file):
    try:
        if file.name.lower().endswith('.csv'):
            return pd.read_csv(file)
        else:
            # for .xlsx
            return pd.read_excel(file)
    except Exception as e:
        st.warning(f"Failed to read {file.name}: {e}")
        return None

def standardize_columns(df):
    renamed = {}
    cols = df.columns.astype(str).tolist()
    for std_col, alt_names in column_mapping.items():
        for alt in alt_names:
            # match case-insensitive and partial matches
            for c in cols:
                if c.lower() == alt.lower() or alt.lower() in c.lower():
                    renamed[c] = std_col
                    break
            if any(r == std_col for r in renamed.values()):
                break
    df = df.rename(columns=renamed)
    return df

if uploaded_files:
    for f in uploaded_files:
        df_local = read_file(f)
        if df_local is None:
            continue
        df_local = standardize_columns(df_local)
        # keep only matching columns
        keep_cols = [c for c in required_columns if c in df_local.columns]
        if not keep_cols:
            st.warning(f"No mapped columns found in {f.name}, skipping.")
            continue
        df_local = df_local[keep_cols].copy()
        df_local = df_local.dropna().drop_duplicates()
        cleaned_dfs.append(df_local)
        st.success(f"Cleaned: {f.name} — {df_local.shape[0]} rows")

if not cleaned_dfs:
    st.info("Upload at least one CSV/XLSX with soil columns (pH, Nitrogen, Phosphorus, Potassium, Moisture, Organic Matter).")
    st.stop()

# Merge
df = pd.concat(cleaned_dfs, ignore_index=True)
st.subheader("Merged dataset (preview)")
st.dataframe(df.head())

# pH filter
if 'pH' in df.columns:
    ph_min, ph_max = float(df['pH'].min()), float(df['pH'].max())
    ph_range = st.sidebar.slider("Filter pH range", min_value=ph_min, max_value=ph_max, value=(max(ph_min,5.0), min(ph_max,8.0)))
    df = df[(df['pH'] >= ph_range[0]) & (df['pH'] <= ph_range[1])]

# Feature selection for plotting
st.subheader("Feature distribution")
feature = st.selectbox("Choose a feature to inspect", df.columns)
if PLOTLY_AVAILABLE:
    fig = px.histogram(df, x=feature, nbins=30, marginal="box", title=f"Distribution of {feature}")
    st.plotly_chart(fig, use_container_width=True)
else:
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)

# Prepare X and y
if task == "Classification":
    if 'Nitrogen' not in df.columns:
        st.error("Classification requires 'Nitrogen' column.")
        st.stop()
    # create categorical fertility labels
    try:
        df['Fertility_Level'] = pd.qcut(df['Nitrogen'], q=3, labels=['Low', 'Moderate', 'High'])
    except Exception:
        df['Fertility_Level'] = pd.cut(df['Nitrogen'], bins=3, labels=['Low', 'Moderate', 'High'])
    y = df['Fertility_Level']
    X = df.drop(columns=['Nitrogen', 'Fertility_Level']) if 'Nitrogen' in df.columns else df.drop(columns=['Fertility_Level'])
else:
    if 'Nitrogen' not in df.columns:
        st.error("Regression requires 'Nitrogen' column.")
        st.stop()
    y = df['Nitrogen']
    X = df.drop(columns=['Nitrogen'])

# keep only numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric features available after preprocessing. Ensure at least one numeric column is present.")
    st.stop()
X = X[numeric_cols]

# normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Select model
if task == "Classification":
    model_name = st.sidebar.selectbox("Choose model", ["Random Forest", "Decision Tree", "KNN", "SVM"])
else:
    model_name = st.sidebar.selectbox("Choose model", ["Random Forest", "Decision Tree", "KNN", "SVM", "Linear Regression"])

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

# Train/test split
test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = get_model(model_name, task)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance helper
def label_for(metric, typ):
    if typ == 'acc':
        return "metric-good" if metric >= 0.8 else "metric-moderate" if metric >= 0.6 else "metric-poor"
    if typ == 'r2':
        return "metric-good" if metric >= 0.7 else "metric-moderate" if metric >= 0.4 else "metric-poor"
    if typ == 'rmse':
        return "metric-good" if metric < 5 else "metric-moderate" if metric < 10 else "metric-poor"
    return "metric-poor"

st.subheader("Model results")
if task == "Classification":
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"**Accuracy:** <span class='{label_for(acc,'acc')}'>{acc:.3f}</span>", unsafe_allow_html=True)
    st.text("Classification report:")
    st.text(classification_report(y_test, y_pred))
    # confusion matrix (matplotlib fallback)
    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        if PLOTLY_AVAILABLE:
            cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
            fig = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted'); ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    except Exception:
        st.write("Could not render confusion matrix.")
else:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    st.markdown(f"**RMSE:** <span class='{label_for(rmse,'rmse')}'>{rmse:.3f}</span>", unsafe_allow_html=True)
    st.markdown(f"**R²:** <span class='{label_for(r2,'r2')}'>{r2:.3f}</span>", unsafe_allow_html=True)

    # Actual vs Predicted plot
    if PLOTLY_AVAILABLE:
        scatter = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title="Actual vs Predicted")
        scatter.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash='dash', color='red'))
        st.plotly_chart(scatter, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

# Feature importance for tree-based models
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    st.subheader("Feature importance")
    if PLOTLY_AVAILABLE:
        fig = px.bar(fi, x=fi.values, y=fi.index, orientation='h', labels={'x':'Importance','y':'Feature'}, title="Feature importances")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        fi.plot(kind='barh', ax=ax)
        ax.set_xlabel("Importance"); ax.set_title("Feature importances")
        st.pyplot(fig)

# Prepare predictions for download
out_df = X_test.copy()
out_df['actual'] = y_test.values
out_df['predicted'] = y_pred
csv = out_df.to_csv(index=False)
st.download_button("Download predictions (CSV)", csv, file_name="predictions.csv", mime="text/csv")

# Notify about Plotly availability & how to enable it in deployment
if not PLOTLY_AVAILABLE:
    st.info("Plotly is not available in this environment — interactive Plotly charts are disabled. "
            "To enable them, add `plotly` to your requirements.txt or install `plotly` in the environment.")
