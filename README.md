# ML-assignment2
BITS WILP 1st sem ML assignment 2

## a. Problem Statement

Predict whether a breast mass is malignant or benign from features computed from a digitized image of a fine needle aspirate (FNA) of the mass. This is a binary classification problem. Five classifiers are trained on the same dataset and compared using Accuracy, AUC, Precision, Recall, F1 and MCC.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository (id 17), DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)
- **Task:** Binary classification. Target `Diagnosis`: M = malignant (encoded as 1), B = benign (encoded as 0)
- **Instances:** 569
- **Features:** 30 real-valued features (mean, standard error and worst value of 10 cell-nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Missing values:** None
- **Split:** 80% train / 20% test (114 samples), stratified, `random_state=42`. The test set (42 malignant, 72 benign) is saved to [test_data.csv](test_data.csv).

## c. GitHub Repository

[https://github.com/anandmofficial1-droid/ML-assignment2](https://github.com/anandmofficial1-droid/ML-assignment2)

Repository contents:

```
ML-assignment2/
├── Source_code.py        # Data loading, training, evaluation, artifact export
├── app.py                # Streamlit app
├── requirements.txt      # Pinned dependencies
├── README.md
├── test_data.csv         # Held-out test split (features + diagnosis)
└── model/                # Saved trained pipelines (.joblib)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## d. Models Used

All models are scikit-learn estimators. Logistic Regression, K-Nearest Neighbors and SVM are wrapped in a pipeline with `StandardScaler`, fitted on the training split only. Decision Tree and Gaussian Naive Bayes use the raw features.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | default, `random_state=42` |
| K-Nearest Neighbors | `n_neighbors=5` |
| Naive Bayes | `GaussianNB` |
| Support Vector Machine | RBF kernel, `random_state=42` |

AUC is computed from predicted probabilities for Logistic Regression, Decision Tree, KNN and Naive Bayes, and from the decision function for SVM.

### Comparison Table (test set, 114 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and balanced: highest AUC (0.9960) and 0.9649 accuracy. Misses 3 of 42 malignant cases and makes 1 false positive. |
| Decision Tree | Weakest model overall (accuracy 0.9298, MCC 0.8492). Errors on both sides, with 4 false negatives and 4 false positives. |
| K-Nearest Neighbors | Solid AUC (0.9823) and precision (0.9744), but recall is lower (0.9048), so 4 malignant cases are missed. |
| Naive Bayes | Very high AUC (0.9934) and perfect precision, but the most conservative at its default threshold: misses 7 of 42 malignant cases. Its ranking is strong while its positive predictions are sparse. |
| Support Vector Machine | Best on accuracy (0.9737), F1 (0.9630) and MCC (0.9442). Zero false positives and 3 false negatives. |

**Overall winner: Support Vector Machine.** It has the best accuracy, F1 and MCC, and no false positives, which matters most for a diagnostic screen. Logistic Regression is a close second and has the highest AUC, so it is a reasonable choice if a simpler, more interpretable model is needed. The gap between the top two models is only one test case, so the ranking is not statistically strong on a single 114-sample split.

## Running the Project

```
pip install -r requirements.txt
python Source_code.py      # trains models, prints metrics, writes model/ and test_data.csv
streamlit run app.py       # launches the web app
```

