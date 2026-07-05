"""
AI Data Quality Studio — ML Module
====================================
Given a cleaned DataFrame and a target column, auto-detects whether this is
a regression or classification problem, trains a small set of standard
models, evaluates them, and returns a ranked comparison.

Kept dependency-light: only scikit-learn (no XGBoost/LightGBM) so the whole
app still installs with a single `pip install -r requirements.txt` and
nothing that needs compiled binaries.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)


@dataclass
class ModelResult:
    name: str
    metrics: dict
    primary_score: float          # the metric used to rank models (R2 for regression, F1 for classification)
    predictions: np.ndarray = None
    confusion: np.ndarray = None


@dataclass
class MLReport:
    problem_type: str             # "regression" | "classification"
    target: str
    features_used: list
    results: list = field(default_factory=list)
    best_model: str = ""
    class_labels: list = None

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"Model": r.name}
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


def detect_problem_type(y: pd.Series) -> str:
    """Heuristic: numeric + many unique values -> regression. Otherwise classification."""
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        if n_unique > 15 and n_unique / len(y) > 0.05:
            return "regression"
        return "classification"
    return "classification"


def _prepare_features(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop obviously useless columns: all-null, single-value, or ID-like high-cardinality text
    drop_cols = []
    for col in X.columns:
        is_numeric = pd.api.types.is_numeric_dtype(X[col])
        is_datetime = pd.api.types.is_datetime64_any_dtype(X[col])
        is_text = not is_numeric and not is_datetime  # covers 'object' and pandas' newer 'string'/'str' dtypes

        if X[col].isna().all() or X[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)
        elif is_text and X[col].nunique(dropna=True) > 0.9 * len(X):
            drop_cols.append(col)
        elif is_datetime:
            drop_cols.append(col)  # keep it simple; date features are a good "next step" item
    X = X.drop(columns=drop_cols)

    # Encode categoricals, impute numerics
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float)
        else:
            X[col] = X[col].fillna("Missing")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleanup — pick a different target or add more columns.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X_imputed, y, drop_cols


def train_and_evaluate(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> MLReport:
    df = df.dropna(subset=[target]).copy()
    problem_type = detect_problem_type(df[target])

    X, y, dropped = _prepare_features(df, target)

    class_labels = None
    if problem_type == "classification":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        class_labels = list(le.classes_)
    else:
        y_encoded = y.astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state,
        stratify=y_encoded if problem_type == "classification" and pd.Series(y_encoded).value_counts().min() >= 2 else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=random_state),
        }
        for name, model in models.items():
            use_scaled = name == "Linear Regression"
            model.fit(X_train_scaled if use_scaled else X_train, y_train)
            preds = model.predict(X_test_scaled if use_scaled else X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            results.append(ModelResult(
                name=name,
                metrics={"R2": round(r2, 3), "MAE": round(mae, 3), "RMSE": round(rmse, 3)},
                primary_score=r2, predictions=preds,
            ))
        results.sort(key=lambda r: r.primary_score, reverse=True)

    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=random_state),
        }
        n_classes = len(set(y_train))
        for name, model in models.items():
            use_scaled = name == "Logistic Regression"
            model.fit(X_train_scaled if use_scaled else X_train, y_train)
            preds = model.predict(X_test_scaled if use_scaled else X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            metrics = {"Accuracy": round(acc, 3), "Precision": round(prec, 3), "Recall": round(rec, 3), "F1": round(f1, 3)}

            if n_classes == 2 and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test_scaled if use_scaled else X_test)[:, 1]
                    metrics["ROC AUC"] = round(roc_auc_score(y_test, proba), 3)
                except Exception:
                    pass

            cm = confusion_matrix(y_test, preds)
            results.append(ModelResult(name=name, metrics=metrics, primary_score=f1, predictions=preds, confusion=cm))
        results.sort(key=lambda r: r.primary_score, reverse=True)

    return MLReport(
        problem_type=problem_type,
        target=target,
        features_used=list(X.columns),
        results=results,
        best_model=results[0].name if results else "",
        class_labels=class_labels,
    )
