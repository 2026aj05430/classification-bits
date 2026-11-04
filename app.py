import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Machine Learning Model Demonstrator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Model Classification Demonstrator")
st.write("Select a machine learning model, input your feature values from the test set, and instantly see predictions and probabilities!")

# --- LOAD DATA TO EXTRACT COLUMN NAMES & DEFAULTS ---
@st.cache_data
def load_data():
    # Load test data to get sample rows and feature names
    test_df = pd.read_csv('test_data.csv', sep=';', encoding='utf-8')
    test_df.columns = test_df.columns.str.strip()
    target_col = test_df.columns[-1]
    feature_cols = test_df.columns[:-1].tolist()
    return test_df, feature_cols, target_col

try:
    test_df, feature_cols, target_col = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Make sure 'test_data.csv' is in the directory.")
    st.stop()

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Classifier:",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors (KNN)",
        "Gaussian Naive Bayes",
        "Support Vector Machine (SVM)"
    )
)

# --- LOAD SELECTED MODEL & SCALER ---
@st.cache_resource
def load_model_and_scaler(choice):
    model, scaler = None, None
    try:
        if choice == "Logistic Regression":
            model = joblib.load('logistic_regression_model.pkl')
            scaler = joblib.load('scaler.pkl')
        elif choice == "Decision Tree":
            model = joblib.load('decision_tree_model.pkl')
        elif choice == "K-Nearest Neighbors (KNN)":
            model = joblib.load('knn_model.pkl')
            scaler = joblib.load('knn_scaler.pkl')
        elif choice == "Gaussian Naive Bayes":
            model = joblib.load('gaussian_nb_model.pkl')
        elif choice == "Support Vector Machine (SVM)":
            model = joblib.load('svm_model.pkl')
            scaler = joblib.load('svm_scaler.pkl')
    except Exception as e:
        st.sidebar.error(f"Could not load files for {choice}. Did you run the training script?")
    return model, scaler

model, scaler = load_model_and_scaler(model_choice)

# --- MAIN INTERFACE: INPUT FEATURES ---
st.subheader(f"Input Features for: `{model_choice}` (using `test_data.csv`)")
st.write("Adjust the values below to test the model.")

# Create input widgets dynamically for each feature column
input_data = {}

# Use columns for a cleaner layout (3 widgets per row)
cols = st.columns(3)
for i, col_name in enumerate(feature_cols):
    with cols[i % 3]:
        # Get default values from the first row of test data
        default_val = float(test_df[col_name].iloc[0])
        min_val = float(test_df[col_name].min())
        max_val = float(test_df[col_name].max())
        step_val = (max_val - min_val) / 100 if max_val != min_val else 0.1
        
        input_data[col_name] = st.number_input(
            label=col_name,
            value=default_val,
            min_value=min_val - abs(min_val)*0.5,
            max_value=max_val + abs(max_val)*0.5,
            step=step_val if step_val > 0 else 1.0
        )

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

# --- PREDICTION BUTTON ---
st.markdown("---")
if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
    if model is None:
        st.error(f"The model for **{model_choice}** is not loaded. Please ensure the `.pkl` file exists.")
    else:
        # Preprocess if the model requires a scaler
        if scaler is not None:
            processed_input = scaler.transform(input_df)
        else:
            processed_input = input_df

        # Make Prediction
        prediction = model.predict(processed_input)[0]
        probabilities = model.predict_proba(processed_input)[0]
        classes = model.classes_

        # Display Results
        st.success(f"### Predicted Class / Category: **{prediction}**")
        
        # Display Probabilities breakdown
        st.markdown("### Class Probability Breakdown:")
        prob_df = pd.DataFrame({
            'Class': classes,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False)
        
        st.dataframe(prob_df.set_index('Class'), use_container_width=True)
        st.bar_chart(prob_df.set_index('Class'))

