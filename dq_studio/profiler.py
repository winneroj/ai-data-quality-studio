"""
AI Data Quality Studio — Profiler Engine
==========================================
Scans a pandas DataFrame and produces a structured report of data quality
issues, plus a single 0-100 Data Quality Score.

Design notes:
- Every check returns a small, serializable dict so the UI layer can render
  it directly without knowing anything about pandas internals.
- The score is a weighted deduction model: start at 100, subtract penalty
  points per issue category, floor at 0. Weights are tuned so that a
  genuinely messy dataset (like sample_data/messy_retail_data.csv) scores
  in the 40-60 range, and a clean dataset scores 90+.
"""

from __future__ import annotations
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[\d\s\-()]{7,15}$")


@dataclass
class Issue:
    category: str          # e.g. "Missing Values"
    severity: str           # "high" | "medium" | "low"
    column: str | None      # None for table-level issues
    description: str
    affected_count: int = 0
    affected_pct: float = 0.0
    penalty: float = 0.0


@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    score: float
    grade: str
    issues: list[Issue] = field(default_factory=list)
    column_summary: dict = field(default_factory=dict)

    def issues_by_column(self, column: str) -> list[Issue]:
        return [i for i in self.issues if i.column == column]

    def as_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["Category", "Severity", "Column", "Description", "Affected", "% of rows"])
        return pd.DataFrame([{
            "Category": i.category,
            "Severity": i.severity,
            "Column": i.column or "(table-wide)",
            "Description": i.description,
            "Affected": i.affected_count,
            "% of rows": round(i.affected_pct, 1),
        } for i in self.issues])


def _grade(score: float) -> str:
    if score >= 90: return "A — Excellent"
    if score >= 75: return "B — Good"
    if score >= 60: return "C — Needs Attention"
    if score >= 40: return "D — Poor"
    return "F — Critical"


def looks_like_email_column(col: pd.Series) -> bool:
    return "email" in str(col.name).lower()


def looks_like_phone_column(col: pd.Series) -> bool:
    name = str(col.name).lower()
    return "phone" in name or "mobile" in name or "tel" in name


def looks_like_date_column(col: pd.Series) -> bool:
    name = str(col.name).lower()
    return "date" in name or "time" in name or "dob" in name


def profile_dataframe(df: pd.DataFrame) -> QualityReport:
    issues: list[Issue] = []
    n_rows, n_cols = df.shape
    column_summary = {}

    # --- 0. Column name hygiene ---
    for col in df.columns:
        stripped = str(col).strip()
        if stripped != col:
            issues.append(Issue(
                category="Column Naming", severity="low", column=col,
                description="Column name has leading/trailing whitespace.",
                affected_count=1, affected_pct=0, penalty=0.5,
            ))

    # --- 1. Duplicate rows ---
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        pct = 100 * n_dup / max(n_rows, 1)
        issues.append(Issue(
            category="Duplicate Rows", severity="high" if pct > 5 else "medium", column=None,
            description=f"{n_dup} exact duplicate row(s) found.",
            affected_count=n_dup, affected_pct=pct,
            penalty=min(15, pct * 1.5),
        ))

    # --- 2. Duplicate columns (identical content) ---
    seen = {}
    for col in df.columns:
        try:
            key = tuple(df[col].astype(str).fillna("NA"))
        except Exception:
            continue
        if key in seen:
            issues.append(Issue(
                category="Duplicate Columns", severity="medium", column=col,
                description=f"Column '{col}' is identical to column '{seen[key]}'.",
                affected_count=n_rows, affected_pct=100, penalty=5,
            ))
        else:
            seen[key] = col

    # --- Per-column checks ---
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = 100 * n_missing / max(n_rows, 1)
        dtype = str(series.dtype)

        col_info = {
            "dtype": dtype,
            "missing": n_missing,
            "missing_pct": round(pct_missing, 1),
            "unique": int(series.nunique(dropna=True)),
        }

        # Missing values
        if n_missing > 0:
            severity = "high" if pct_missing > 30 else ("medium" if pct_missing > 5 else "low")
            issues.append(Issue(
                category="Missing Values", severity=severity, column=col,
                description=f"{n_missing} missing value(s) ({pct_missing:.1f}% of rows).",
                affected_count=n_missing, affected_pct=pct_missing,
                penalty=min(10, pct_missing * 0.3),
            ))

        # Mixed types within an "object" column (e.g. numbers stored as strings + actual numbers)
        if dtype == "object":
            non_null = series.dropna()
            type_set = set(type(v).__name__ for v in non_null.sample(min(len(non_null), 500), random_state=0)) if len(non_null) else set()
            numeric_like = 0
            for v in non_null:
                s = str(v).strip()
                try:
                    float(s.replace(",", ""))
                    numeric_like += 1
                except ValueError:
                    pass
            if 0 < numeric_like < len(non_null):
                mixed_pct = 100 * min(numeric_like, len(non_null) - numeric_like) / max(len(non_null), 1)
                if mixed_pct > 1:
                    issues.append(Issue(
                        category="Mixed Data Types", severity="high", column=col,
                        description=f"Column mixes numeric-looking values with text (e.g. 'N/A' inside a numeric column).",
                        affected_count=numeric_like, affected_pct=mixed_pct, penalty=8,
                    ))

            # Whitespace issues
            has_ws = non_null.astype(str).str.strip().ne(non_null.astype(str)).sum()
            if has_ws > 0:
                issues.append(Issue(
                    category="Whitespace", severity="low", column=col,
                    description=f"{has_ws} value(s) have leading/trailing whitespace.",
                    affected_count=int(has_ws), affected_pct=100 * has_ws / max(len(non_null), 1),
                    penalty=min(3, 100 * has_ws / max(len(non_null), 1) * 0.05),
                ))

            # Categorical inconsistency (casing / stray spaces creating fake distinct categories)
            if non_null.nunique() <= max(20, 0.2 * len(non_null)):
                normalized = non_null.astype(str).str.strip().str.lower()
                if normalized.nunique() < non_null.nunique():
                    issues.append(Issue(
                        category="Categorical Inconsistency", severity="medium", column=col,
                        description=f"{non_null.nunique()} raw categories collapse to {normalized.nunique()} after normalizing case/whitespace (e.g. 'North' vs 'north' vs ' North ').",
                        affected_count=non_null.nunique() - normalized.nunique(), affected_pct=0, penalty=5,
                    ))

            # High cardinality check (mostly informative)
            if non_null.nunique() > 0.9 * len(non_null) and len(non_null) > 20 and not looks_like_email_column(series):
                issues.append(Issue(
                    category="High Cardinality", severity="low", column=col,
                    description=f"Nearly every value is unique ({non_null.nunique()} unique / {len(non_null)} rows) — likely an identifier, not a useful feature.",
                    affected_count=non_null.nunique(), affected_pct=100, penalty=1,
                ))

            # Email validation
            if looks_like_email_column(series):
                invalid = non_null[~non_null.astype(str).str.match(EMAIL_RE)]
                if len(invalid) > 0:
                    pct = 100 * len(invalid) / max(len(non_null), 1)
                    issues.append(Issue(
                        category="Invalid Emails", severity="high" if pct > 10 else "medium", column=col,
                        description=f"{len(invalid)} value(s) do not look like valid email addresses.",
                        affected_count=len(invalid), affected_pct=pct, penalty=min(8, pct * 0.3),
                    ))

            # Phone validation
            if looks_like_phone_column(series):
                invalid = non_null[~non_null.astype(str).str.match(PHONE_RE)]
                if len(invalid) > 0:
                    pct = 100 * len(invalid) / max(len(non_null), 1)
                    issues.append(Issue(
                        category="Invalid Phone Numbers", severity="medium", column=col,
                        description=f"{len(invalid)} value(s) do not match a plausible phone number pattern.",
                        affected_count=len(invalid), affected_pct=pct, penalty=min(6, pct * 0.2),
                    ))

            # Date validation
            if looks_like_date_column(series):
                parsed = pd.to_datetime(non_null, errors="coerce")
                invalid = parsed.isna().sum()
                if invalid > 0:
                    pct = 100 * invalid / max(len(non_null), 1)
                    issues.append(Issue(
                        category="Invalid Dates", severity="medium", column=col,
                        description=f"{invalid} value(s) could not be parsed as dates.",
                        affected_count=int(invalid), affected_pct=pct, penalty=min(6, pct * 0.2),
                    ))

        # Numeric-only checks (skip identifier-like columns: IDs, phone numbers — not real measurements)
        is_identifier_like = "id" in str(col).lower() or looks_like_phone_column(series)
        numeric_series = pd.to_numeric(series, errors="coerce")
        n_numeric_valid = numeric_series.notna().sum()
        if not is_identifier_like and (dtype != "object" or n_numeric_valid > 0.5 * len(series.dropna() if len(series.dropna()) else [1])):
            clean_numeric = numeric_series.dropna()
            if len(clean_numeric) > 10:
                q1, q3 = clean_numeric.quantile(0.25), clean_numeric.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = clean_numeric[(clean_numeric < lower) | (clean_numeric > upper)]
                    if len(outliers) > 0:
                        pct = 100 * len(outliers) / len(clean_numeric)
                        issues.append(Issue(
                            category="Outliers", severity="medium" if pct < 5 else "high", column=col,
                            description=f"{len(outliers)} value(s) fall outside the IQR-based normal range ({lower:.1f} to {upper:.1f}).",
                            affected_count=len(outliers), affected_pct=pct, penalty=min(6, pct * 0.5),
                        ))

                # Skewness
                if clean_numeric.std() > 0:
                    skew = clean_numeric.skew()
                    if abs(skew) > 1.5:
                        issues.append(Issue(
                            category="Skewness", severity="low", column=col,
                            description=f"Distribution is {'right' if skew > 0 else 'left'}-skewed (skew={skew:.2f}). Consider a log transform.",
                            affected_count=0, affected_pct=0, penalty=1,
                        ))

        column_summary[col] = col_info

    # --- Correlation / potential leakage (numeric columns only) ---
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs().copy()
        vals = corr.values.copy()
        np.fill_diagonal(vals, 0)
        corr = pd.DataFrame(vals, index=corr.index, columns=corr.columns)
        max_corr = corr.max().max()
        if max_corr > 0.95:
            pair = corr.stack().idxmax()
            issues.append(Issue(
                category="High Correlation / Possible Leakage", severity="medium", column=f"{pair[0]} & {pair[1]}",
                description=f"Columns '{pair[0]}' and '{pair[1]}' are correlated at {max_corr:.2f} — check for duplicated or leaked information.",
                affected_count=0, affected_pct=0, penalty=3,
            ))

    # --- Class imbalance heuristic on low-cardinality object/int columns ---
    for col in df.columns:
        series = df[col]
        if series.nunique(dropna=True) in range(2, 8) and len(series.dropna()) > 20:
            counts = series.value_counts(normalize=True)
            if counts.iloc[0] > 0.9:
                issues.append(Issue(
                    category="Class Imbalance", severity="low", column=col,
                    description=f"'{counts.index[0]}' makes up {counts.iloc[0]*100:.1f}% of values in this column.",
                    affected_count=0, affected_pct=counts.iloc[0]*100, penalty=1,
                ))

    total_penalty = sum(i.penalty for i in issues)
    score = max(0.0, 100.0 - total_penalty)

    return QualityReport(
        n_rows=n_rows, n_cols=n_cols, score=round(score, 1), grade=_grade(score),
        issues=sorted(issues, key=lambda i: {"high": 0, "medium": 1, "low": 2}[i.severity]),
        column_summary=column_summary,
    )
