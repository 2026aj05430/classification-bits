import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="Classification Models Performance",
    page_icon="📊",
    layout="centered"
)

st.title("Classification Model Evaluation on UCI student dropout database")
st.write("Upload the test dataset in github repo, select a trained model, and observe Evaluation metrics.")

# --- 1. MODEL SELECTION DROPDOWN ---
st.sidebar.header("Configuration")
model_options = {
    "Logistic Regression": {
        "model_path": "model/logistic_regression_model.pkl",
        "scaler_path": "model/scaler.pkl",
        "needs_scaling": True
    },
    "Decision Tree": {
        "model_path": "model/decision_tree_model.pkl",
        "scaler_path": None,
        "needs_scaling": False
    },
    "K-Nearest Neighbors (KNN)": {
        "model_path": "model/knn_model.pkl",
        "scaler_path": "model/knn_scaler.pkl",
        "needs_scaling": True
    },
    "Gaussian Naive Bayes": {
        "model_path": "model/gaussian_nb_model.pkl",
        "scaler_path": None,
        "needs_scaling": False
    },
    "Support Vector Machine (SVM)": {
        "model_path": "model/svm_model.pkl",
        "scaler_path": "model/svm_scaler.pkl",
        "needs_scaling": True
    }
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model", 
    list(model_options.keys())
)

# --- 2. DATASET UPLOAD OPTION (CSV) ---
st.subheader("📁 Upload Test Dataset provided in github")
uploaded_file = st.file_uploader(
    "Upload your test CSV file (e.g., test_data.csv)", 
    type=["csv"]
)

@st.cache_resource
def load_artifacts(model_path, scaler_path, needs_scaling):
    """Helper function to load model and scaler safely."""
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if needs_scaling and os.path.exists(scaler_path) else None
    return model, scaler

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file , sep=';', encoding='utf-8')
        st.success(f"Successfully uploaded! Dataset shape: {test_df.shape}")
        
        with st.expander("Preview Uploaded Data"):
            st.dataframe(test_df.head())
            
        # Assuming the target column is named 'Target' (consistent with your training scripts)
        target_col = 'Target' if 'Target' in test_df.columns else test_df.columns[-1]
        
        if target_col not in test_df.columns:
            st.error(f"Could not find target column in dataset. Please ensure a column named 'Target' or label column exists.")
        else:
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            # Load selected model configuration and artifacts
            config = model_options[selected_model_name]
            model, scaler = load_artifacts(config["model_path"], config["scaler_path"], config["needs_scaling"])

            if model is None:
                st.error(f"Model artifact not found for {selected_model_name}. Please ensure model training scripts have been run and saved to the 'models/' directory.")
            else:
                # Preprocess features if required
                if config["needs_scaling"] and scaler is not None:
                    X_test_processed = scaler.transform(X_test)
                else:
                    X_test_processed = X_test

                # Predictions
                y_pred = model.predict(X_test_processed)
                
                # Handle Probabilities / AUC score safely
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test_processed)
                    except Exception:
                        pass

                # --- 3. DISPLAY OF EVALUATION METRICS ---
                st.markdown("---")
                st.subheader(f"📈 Evaluation Metrics: {selected_model_name}")
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Precision", f"{prec:.4f}")
                col3.metric("Recall", f"{rec:.4f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("F1 Score", f"{f1:.4f}")
                col5.metric("MCC Score", f"{mcc:.4f}")
                col6.metric("AUC Score", f"{auc_score:.4f}" if isinstance(auc_score, float) else auc_score)

                # --- 4. CONFUSION MATRIX & CLASSIFICATION REPORT ---
                st.markdown("---")
                st.subheader("🔍 Detailed Classification Report & Confusion Matrix")
                
                tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    ax.set_title(f"Confusion Matrix - {selected_model_name}")
                    st.pyplot(fig)
                    
                with tab2:
                    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format(precision=4))

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload your test CSV file (`test_data.csv`) using the file uploader above to begin evaluation.")
