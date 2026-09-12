# %% [markdown]
# Stabilization: ID is metadata only. Exact model-predictor groups share one partition, following the lowest-ID member of the original split. All records are retained. See `../outputs/stabilization_report.md`. Run `script_prepare_data.py` first. Hyperparameters, preprocessing and the default 0.5 decision boundary are unchanged.

# %% [markdown]
# Both models were trained and evaluated using identical data partitions and preprocessing procedures; only the classification algorithm differed.

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
from sklearn.ensemble import GradientBoostingClassifier

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

# %%

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
df.head()

# %%
TARGET = "default"

y = df[TARGET]

# Protected attribute retained for fairness auditing,
# but excluded from model training
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

# %%
X_train, X_test, y_train, y_test, metadata_train, metadata_test = load_split(df)
A_train, A_test = metadata_train['SEX'], metadata_test['SEX']
print('Training observations:', len(X_train))
print('Test observations:', len(X_test))
print('Training default rate:', y_train.mean() * 100)
print('Test default rate:', y_test.mean() * 100)


# %%
# Preprocessing and classifier definitions are shared with the standalone scripts.
# StandardScaler for numeric columns; OneHotEncoder(handle_unknown="ignore") for categorical columns.

# %%
boost_model = build_model("gradient_boosting")
boost_model

# %% [markdown]
# Train

# %%
boost_model.fit(X_train, y_train)

print("Gradient Boosting model trained successfully.")

# %% [markdown]
# Generate predictions

# %%
y_pred_boost = boost_model.predict(X_test)

y_prob_boost = boost_model.predict_proba(X_test)[:, 1]

print("Predicted classes:")
print(y_pred_boost[:10])

print("\nPredicted default probabilities:")
print(y_prob_boost[:10])

# %% [markdown]
# Metrics

# %%
boost_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_boost),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred_boost),
    "Precision": precision_score(y_test, y_pred_boost),
    "Recall": recall_score(y_test, y_pred_boost),
    "F1": f1_score(y_test, y_pred_boost),
    "ROC-AUC": roc_auc_score(y_test, y_prob_boost),
    "Average Precision (AP)": average_precision_score(
        y_test,
        y_prob_boost
    )
}

boost_metrics_df = pd.DataFrame(
    boost_metrics.items(),
    columns=["Metric", "Value"]
)

boost_metrics_df

# %% [markdown]
# Classification report

# %%
print(
    classification_report(
        y_test,
        y_pred_boost,
        target_names=["No Default", "Default"]
    )
)

# %% [markdown]
# Confusion matrix

# %%
cm_boost = confusion_matrix(
    y_test,
    y_pred_boost
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_boost,
    display_labels=["No Default", "Default"]
)

disp.plot()

plt.title(
    "Gradient Boosting - Confusion Matrix"
)

plt.show()

# %% [markdown]
# ROC curve

# %%
RocCurveDisplay.from_predictions(
    y_test,
    y_prob_boost
)

plt.title(
    "Gradient Boosting - ROC Curve"
)

plt.show()

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            y_prob_boost
        ),
        4
    )
)

# %%
BOOST_METRICS_PATH = ROOT / Path(
    "outputs/tables/baseline_gradient_boosting_metrics.csv"
)

boost_metrics_df.to_csv(
    BOOST_METRICS_PATH,
    index=False
)

print(
    "Saved:",
    BOOST_METRICS_PATH.resolve()
)

# %%
boost_predictions = pd.DataFrame({
    "ID": metadata_test["ID"].values,
    "actual": y_test.values,
    "predicted": y_pred_boost,
    "probability_default": y_prob_boost,
    "SEX": A_test.values
})

BOOST_PRED_PATH = ROOT / Path(
    "outputs/tables/baseline_gradient_boosting_predictions.csv"
)

boost_predictions.to_csv(
    BOOST_PRED_PATH,
    index=False
)

print(
    "Saved:",
    BOOST_PRED_PATH.resolve()
)

boost_predictions.head()

# %%
save_pipeline(boost_model, "gradient_boosting", X_test, y_pred_boost, y_prob_boost)

