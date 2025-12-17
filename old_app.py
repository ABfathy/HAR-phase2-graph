import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_scaler():
    return joblib.load("./models/scaler.pkl")


@st.cache_resource
def load_pca():
    return joblib.load("./models/pca.pkl")


@st.cache_resource
def load_random_forest():
    return joblib.load("./models/random_forest.pkl")


@st.cache_resource
def load_sgd():
    return joblib.load("./models/sgd.pkl")


CLASS_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Load shared resources
scaler = load_scaler()
pca = load_pca()

# Sidebar - Model Selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ("Random Forest", "SGD"))

# Load selected model
if model_choice == "Random Forest":
    model = load_random_forest()
else:
    model = load_sgd()

# Main UI
st.title("Human Activity Recognition")
st.write(f"**Current Model:** {model_choice}")
st.write("Upload a CSV file for prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)

        st.write("### Preview of Data")
        st.dataframe(df.head())

        # Preprocess
        x_scaled = scaler.transform(df)
        x_pca = pca.transform(x_scaled)

        # Predict
        y_pred = model.predict(x_pca)
        y_pred_proba = model.predict_proba(x_pca)

        # Build results dataframe
        results = pd.DataFrame(
            {
                "Predicted Activity": [CLASS_NAMES[i] for i in y_pred],
                "Confidence (%)": [
                    f"{np.max(proba) * 100:.2f}%" for proba in y_pred_proba
                ],
            }
        )

        st.write("### Prediction Results")
        st.dataframe(results)

    except Exception as e:
        st.error(f"Error processing file: {e}")
