import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model_utils import load_model

# CONFIG
st.set_page_config(
    page_title="MODMA-MDD Multimodal Dashboard",
    layout="wide"
)

# =====================================================
# PATHS
DATA_PATH = "data_processed/MODMA_multimodal_features.csv"
META_PATH = "metadata.csv"
MODEL_DIR = "../Model_Output_Joblib"
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

# DEFINE AVAILABLE MODELS
# Ensure these filenames match exactly what your training script saved
AVAILABLE_MODELS = {
    "Random Forest": "model_RandomForest.joblib",
    "Logistic Regression": "model_LogisticRegression.joblib"
}

# SIDEBAR (Moved up to select model before loading)
st.sidebar.title("MODMA-MDD Dashboard")

# 1. Model Selector
selected_model_name = st.sidebar.selectbox(
    "Select Model", 
    list(AVAILABLE_MODELS.keys())
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    meta = pd.read_csv(META_PATH)

    # --- FIX: Normalize subject_id ---
    df["subject_id"] = df["subject_id"].astype(str).str.replace(r'\.0$', '', regex=True)
    meta["subject_id"] = meta["subject_id"].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Merge labels
    df = df.merge(meta[["subject_id", "label"]], on="subject_id", how="left")
    df["diagnosis"] = df["label"].map({1: "MDD", 0: "Healthy Control"})

    return df

# LOAD MODEL (Dynamic based on selection)
@st.cache_resource
def get_model(model_name):
    model_filename = AVAILABLE_MODELS[model_name]
    model_path = os.path.join(MODEL_DIR, model_filename)
    return load_model(model_path, PREPROC_PATH)

try:
    df = load_data()
    model, preproc = get_model(selected_model_name)
except FileNotFoundError as e:
    st.error(f"❌ Could not load model or data: {e}")
    st.stop()

# NAVIGATION
page = st.sidebar.radio(
    "Navigate",
    [
        "Dataset Sanity Check",
        "Dataset Overview",
        "Label Distribution",
        "Feature Exploration",
        "Model Prediction",
        "Feature Importance"
    ]
)

# PAGE 0: DATASET SANITY CHECK
if page == "Dataset Sanity Check":
    st.title("🧪 Dataset Sanity Check")
    st.write(f"**Current Model:** {selected_model_name}")

    st.write("**Columns present:**")
    st.code(", ".join(df.columns.tolist()))

    st.write("**Diagnosis counts:**")
    st.write(df["diagnosis"].value_counts(dropna=False))

    st.write("**Missing values (top 15):**")
    st.dataframe(df.isna().mean().sort_values(ascending=False).head(15))

# PAGE 1: DATASET OVERVIEW
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("**Total subjects:**", df["subject_id"].nunique())
    st.write("**Labeled subjects:**", df["label"].notna().sum())
    st.dataframe(df.head(20), width="stretch")

# PAGE 2: LABEL DISTRIBUTION
elif page == "Label Distribution":
    st.title("🧠 MDD vs Healthy Control")
    labeled = df.dropna(subset=["label"])
    if labeled.empty:
        st.warning("No labeled subjects available.")
    else:
        counts = labeled["diagnosis"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_ylabel("Number of Subjects")
        st.pyplot(fig)

# PAGE 3: FEATURE EXPLORATION
elif page == "Feature Exploration":
    st.title("🔍 Feature Exploration")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in ["label"]:
        if c in numeric_cols: numeric_cols.remove(c)

    feature = st.selectbox("Select a feature", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(
        data=df.dropna(subset=["label"]),
        x=feature,
        hue="diagnosis",
        kde=False,
        bins=30,
        ax=ax
    )
    ax.set_title(f"{feature} distribution by diagnosis")
    st.pyplot(fig)

# PAGE 4: MODEL PREDICTION
elif page == "Model Prediction":
    st.title(f"🤖 Prediction using {selected_model_name}")

    subject_id = st.selectbox("Select Subject ID", df["subject_id"].unique())
    row = df[df["subject_id"] == subject_id]

    X_input = row.drop(columns=["subject_id", "label", "diagnosis"], errors="ignore")

    # Align Features
    if hasattr(preproc["scaler"], "feature_names_in_"):
        expected_cols = preproc["scaler"].feature_names_in_
    elif hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
    else:
        st.error("Model missing feature names metadata.")
        st.stop()

    X_aligned = pd.DataFrame(columns=expected_cols)
    for col in expected_cols:
        X_aligned.loc[0, col] = X_input[col].values[0] if col in X_input.columns else 0

    if X_aligned.empty:
        st.error("Could not align features.")
    else:
        try:
            if isinstance(preproc, dict):
                X_imp = preproc["imputer"].transform(X_aligned)
                X_scaled = preproc["scaler"].transform(X_imp)
            else:
                X_imp = preproc.named_steps["imputer"].transform(X_aligned)
                X_scaled = preproc.named_steps["scaler"].transform(X_imp)

            proba = model.predict_proba(X_scaled)[0, 1]
            pred = model.predict(X_scaled)[0]

            col1, col2 = st.columns(2)
            col1.metric("Predicted Probability (MDD)", f"{proba:.2%}")
            
            if pred == 1:
                col2.error("Prediction: MDD")
            else:
                col2.success("Prediction: Healthy Control")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("### Subject Data")
    st.dataframe(row, width="stretch")

# PAGE 5: FEATURE IMPORTANCE (Adaptive)
elif page == "Feature Importance":
    st.title(f"⭐ Feature Importance ({selected_model_name})")

    # 1. Get Feature Names
    feature_names = None
    if isinstance(preproc, dict) and hasattr(preproc["scaler"], "feature_names_in_"):
        feature_names = preproc["scaler"].feature_names_in_
    elif hasattr(preproc, "named_steps") and hasattr(preproc.named_steps["scaler"], "feature_names_in_"):
        feature_names = preproc.named_steps["scaler"].feature_names_in_
    
    if feature_names is None:
        st.warning("Could not extract feature names.")
        st.stop()

    # 2. Get Importances based on Model Type
    importances = None
    title = ""

    # Random Forest (feature_importances_)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Top 20 Features (Gini Importance)"
    
    # Logistic Regression (coef_)
    elif hasattr(model, "coef_"):
        importances = model.coef_[0] # coef_ is shape (1, n_features)
        title = "Top 20 Features (Coefficients)"
        st.info("Positive values increase MDD risk, negative values decrease it.")

    # 3. Plot
    if importances is not None:
        # Create DataFrame for easier sorting
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })
        
        # Sort by absolute value to find most influential features
        imp_df["AbsImportance"] = imp_df["Importance"].abs()
        imp_df = imp_df.sort_values(by="AbsImportance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Use color to show direction (Positive/Negative) for Logistic Regression
        colors = ['red' if x > 0 else 'blue' for x in imp_df["Importance"]]
        
        sns.barplot(
            x=imp_df["Importance"], 
            y=imp_df["Feature"], 
            ax=ax, 
            palette=colors if hasattr(model, "coef_") else "viridis"
        )
        
        ax.set_title(title)
        st.pyplot(fig)
    else:
        st.warning(f"This model ({selected_model_name}) does not provide feature importances.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Labels: MDD = 1, Healthy Control = 0"
)
