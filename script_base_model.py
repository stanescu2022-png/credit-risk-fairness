# %% [markdown]
# Stabilization: ID is metadata only. Exact model-predictor groups share one partition, following the lowest-ID member of the original split. All records are retained. See `../outputs/stabilization_report.md`. Run `script_prepare_data.py` first. Hyperparameters, preprocessing and the default 0.5 decision boundary are unchanged.

# %%
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baseline import DATA_PATH, TABLES, load_split, build_model, save_pipeline, CATEGORICAL, NUMERIC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

pd.set_option("display.max_columns", None)


df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
df.head()

# %% [markdown]
# Define target and sensitive attribute

# %%
TARGET = "default"

y = df[TARGET]

# Keep sex for fairness auditing, but don't give it to the baseline model
A_sex = df["SEX"]

X = df.drop(columns=["ID", TARGET, "SEX"])

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts(normalize=True))

# %%
categorical_features = CATEGORICAL
numeric_features = NUMERIC
print("Predictors:", len(CATEGORICAL) + len(NUMERIC))

# %% [markdown]
# Train/test split

# %%
X_train, X_test, y_train, y_test, metadata_train, metadata_test = load_split(df)
A_train, A_test = metadata_train['SEX'], metadata_test['SEX']
print('Training observations:', len(X_train))
print('Test observations:', len(X_test))
print('Training default rate:', y_train.mean() * 100)
print('Test default rate:', y_test.mean() * 100)


# %% [markdown]
# Preprocessing pipeline

# %% [markdown]
# continuous financial variables
#         ↓
# StandardScaler
# 
# categorical variables
#         ↓
# One-Hot Encoding

# %%
# Preprocessing and classifier definitions are shared with the standalone scripts.
# StandardScaler for numeric columns; OneHotEncoder(handle_unknown="ignore") for categorical columns.

# %% [markdown]
# Logistic Regression

# %%
baseline_model = build_model("logistic")
baseline_model

# %% [markdown]
# Train

# %%
baseline_model.fit(X_train, y_train)

print("Baseline model trained successfully.")

# %% [markdown]
# Generate predictions

# %%
y_pred = baseline_model.predict(X_test)

# %%
y_prob = baseline_model.predict_proba(X_test)[:, 1]

print(y_pred[:10])
print(y_prob[:10])

# %%
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
    "Average Precision (AP)": average_precision_score(y_test, y_prob)
}

metrics_df = pd.DataFrame(
    metrics.items(),
    columns=["Metric", "Value"]
)

metrics_df

# %% [markdown]
# Full classification report

# %%
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Default", "Default"]
    )
)

# %% [markdown]
# Confusion matrix

# %%
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

disp.plot()
plt.title("Baseline Logistic Regression - Confusion Matrix")
plt.show()

# %% [markdown]
# ROC curve

# %%
RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Baseline Logistic Regression - ROC Curve")
plt.show()

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_prob), 4)
)

# %%
OUTPUT_PATH = ROOT / Path(
    "outputs/tables/baseline_logistic_metrics.csv"
)

metrics_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Saved:", OUTPUT_PATH.resolve())

# %%
predictions = pd.DataFrame({
    "ID": metadata_test["ID"].values,
    "actual": y_test.values,
    "predicted": y_pred,
    "probability_default": y_prob,
    "SEX": A_test.values
})

PRED_PATH = ROOT / Path(
    "outputs/tables/baseline_logistic_predictions.csv"
)

predictions.to_csv(
    PRED_PATH,
    index=False
)

predictions.head()

# %%
save_pipeline(baseline_model, "logistic", X_test, y_pred, y_prob)

