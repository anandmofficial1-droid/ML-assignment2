"""Classify Breast Cancer Wisconsin (Diagnostic) tumors with five ML models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).parent / "model"
TEST_DATA_PATH = Path(__file__).parent / "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_data():
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = (dataset.data.targets.iloc[:, 0] == "M").astype(int)
    return X, y


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE)
        ),
    }


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def save_artifacts(models, X_test, y_test):
    MODEL_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
    X_test.assign(diagnosis=y_test).to_csv(TEST_DATA_PATH, index=False)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results[name] = evaluate(model, X_test, y_test)

    save_artifacts(fitted_models, X_test, y_test)

    results_df = pd.DataFrame(results).T.round(4)
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Test size: {len(X_test)}")
    print(results_df.to_string())


if __name__ == "__main__":
    main()







