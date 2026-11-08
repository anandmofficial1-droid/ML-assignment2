from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from Source_code import MODEL_DIR, MODEL_FILES, evaluate

BASE_DIR = Path(__file__).parent
TARGET = "diagnosis"


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_data
def load_default_test_data():
    return pd.read_csv(BASE_DIR / "test_data.csv")


def load_uploaded_test_data(uploaded_file, features):
    df = pd.read_csv(uploaded_file)
    missing = [col for col in features + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("The uploaded file has no rows.")
    non_numeric = [col for col in features if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {', '.join(non_numeric)}")
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"M": 1, "B": 0})
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"'{TARGET}' must contain 1/0 or M/B labels.")
    return df


def label_name(value):
    return "Malignant" if value == 1 else "Benign"


st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Wisconsin (Diagnostic) Classifier")

models = load_models()
default_df = load_default_test_data()
FEATURES = [col for col in default_df.columns if col != TARGET]

st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(models))
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")

if uploaded_file is None:
    test_df = default_df
    st.sidebar.caption("Using bundled test split (test_data.csv).")
else:
    try:
        test_df = load_uploaded_test_data(uploaded_file, FEATURES)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    st.sidebar.caption(f"Using uploaded file: {len(test_df)} rows.")

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)
model = models[model_name]
y_pred = model.predict(X_test)

st.subheader(f"Evaluation metrics: {model_name}")
metrics = evaluate(model, X_test, y_test)
st.table(pd.DataFrame([metrics]).round(4).T.rename(columns={0: "Score"}))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"],
        ),
        width="stretch",
    )
with col2:
    st.subheader("Classification report")
    st.code(
        classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["Benign", "Malignant"], zero_division=0
        )
    )

st.subheader("Comparison of all models")
all_metrics = {name: evaluate(m, X_test, y_test) for name, m in models.items()}
st.dataframe(pd.DataFrame(all_metrics).T.round(4), width="stretch")

st.subheader("Predict a sample")
index = st.number_input("Sample index", min_value=0, max_value=len(X_test) - 1, value=0, step=1)
prediction = model.predict(X_test.iloc[[index]])[0]
actual = y_test.iloc[index]
st.write(f"Predicted: **{label_name(prediction)}** | Actual: **{label_name(actual)}**")
st.dataframe(X_test.iloc[[index]], width="stretch")
