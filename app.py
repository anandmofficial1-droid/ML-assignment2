from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from Source_code import MODEL_DIR, MODEL_FILES, evaluate

BASE_DIR = Path(__file__).parent
TARGET = "diagnosis"


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_data
def load_test_data():
    return pd.read_csv(BASE_DIR / "test_data.csv")


def label_name(value):
    return "Malignant" if value == 1 else "Benign"


st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Wisconsin (Diagnostic) Classifier")

models = load_models()
test_df = load_test_data()
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

st.subheader("Metrics for all models (test set)")
all_metrics = {name: evaluate(model, X_test, y_test) for name, model in models.items()}
st.dataframe(pd.DataFrame(all_metrics).T.round(4), width="stretch")

st.subheader("Predict a test sample")
model_name = st.selectbox("Model", list(models))
index = st.number_input("Test sample index", min_value=0, max_value=len(X_test) - 1, value=0, step=1)

prediction = models[model_name].predict(X_test.iloc[[index]])[0]
actual = y_test.iloc[index]
st.write(f"Predicted: **{label_name(prediction)}** | Actual: **{label_name(actual)}**")
st.dataframe(X_test.iloc[[index]], width="stretch")
