"""
AI Data Quality Studio — Cleaning Engine
==========================================
Given a DataFrame and a QualityReport (from profiler.py), recommend concrete
cleaning actions, and apply them either one-by-one or via a single
"auto-clean" pipeline. Every action records what it did so we can produce a
before/after comparison and an audit trail.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CleaningAction:
    column: str | None
    action: str
    detail: str
    rows_affected: int = 0


@dataclass
class CleaningResult:
    df: pd.DataFrame
    actions: list[CleaningAction] = field(default_factory=list)

    def log(self) -> pd.DataFrame:
        if not self.actions:
            return pd.DataFrame(columns=["Column", "Action", "Detail", "Rows Affected"])
        return pd.DataFrame([{
            "Column": a.column or "(table-wide)",
            "Action": a.action,
            "Detail": a.detail,
            "Rows Affected": a.rows_affected,
        } for a in self.actions])


def _numeric_fill_strategy(series: pd.Series) -> str:
    """Pick mean vs median based on skew — median is more robust to outliers."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return "median"
    return "median" if abs(clean.skew()) > 1 else "mean"


def auto_clean(df: pd.DataFrame, options: dict | None = None) -> CleaningResult:
    """
    Run the default recommended cleaning pipeline.

    options: optional dict to toggle steps, e.g.
        {"drop_duplicates": True, "impute_missing": True, "clean_text": True,
         "fix_dtypes": True, "cap_outliers": True}
    """
    opts = {
        "drop_duplicates": True,
        "impute_missing": True,
        "clean_text": True,
        "fix_dtypes": True,
        "cap_outliers": True,
        "drop_duplicate_columns": True,
    }
    if options:
        opts.update(options)

    work = df.copy()
    actions: list[CleaningAction] = []

    # 1. Strip whitespace from column names
    renamed = {c: str(c).strip() for c in work.columns if str(c).strip() != c}
    if renamed:
        work = work.rename(columns=renamed)
        actions.append(CleaningAction(None, "Rename columns", f"Trimmed whitespace from {len(renamed)} column name(s).", len(renamed)))

    # 2. Drop duplicate columns (identical content)
    if opts["drop_duplicate_columns"]:
        seen = {}
        drop_cols = []
        for col in work.columns:
            key = tuple(work[col].astype(str).fillna("NA"))
            if key in seen:
                drop_cols.append(col)
            else:
                seen[key] = col
        if drop_cols:
            work = work.drop(columns=drop_cols)
            actions.append(CleaningAction(None, "Drop duplicate columns", f"Removed {len(drop_cols)} column(s) identical to another: {', '.join(drop_cols)}.", 0))

    # 3. Drop duplicate rows
    if opts["drop_duplicates"]:
        n_before = len(work)
        work = work.drop_duplicates(keep="first").reset_index(drop=True)
        n_removed = n_before - len(work)
        if n_removed > 0:
            actions.append(CleaningAction(None, "Drop duplicate rows", f"Removed {n_removed} exact duplicate row(s).", n_removed))

    # 4. Clean text columns: strip whitespace, normalize casing on low-cardinality categoricals
    if opts["clean_text"]:
        for col in work.select_dtypes(include="object").columns:
            non_null = work[col].dropna()
            if len(non_null) == 0:
                continue
            stripped = work[col].astype(str).str.strip()
            n_changed = (stripped != work[col].astype(str)).sum()

            # Normalize casing only for likely categorical columns (low cardinality, not email/id-like)
            if non_null.nunique() <= max(20, 0.2 * len(non_null)) and "email" not in col.lower() and "name" not in col.lower():
                normalized = stripped.where(work[col].isna(), stripped.str.title())
                work[col] = normalized
                if n_changed > 0:
                    actions.append(CleaningAction(col, "Clean text", f"Trimmed whitespace and standardized casing for {n_changed} value(s).", int(n_changed)))
            else:
                work[col] = stripped.where(~work[col].isna(), work[col])
                if n_changed > 0:
                    actions.append(CleaningAction(col, "Trim whitespace", f"Trimmed whitespace on {n_changed} value(s).", int(n_changed)))

    # 5. Fix mixed-type numeric columns (coerce "N/A"-style strings to NaN, then to numeric)
    if opts["fix_dtypes"]:
        for col in work.columns:
            if work[col].dtype == "object":
                non_null = work[col].dropna()
                if len(non_null) == 0:
                    continue
                numeric_like = 0
                for v in non_null:
                    try:
                        float(str(v).replace(",", ""))
                        numeric_like += 1
                    except ValueError:
                        pass
                if numeric_like > 0.6 * len(non_null):
                    coerced = pd.to_numeric(work[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
                    n_new_na = coerced.isna().sum() - work[col].isna().sum()
                    work[col] = coerced
                    actions.append(CleaningAction(col, "Fix data type", f"Converted to numeric; {max(n_new_na,0)} non-numeric value(s) became missing.", int(max(n_new_na, 0))))

            # Parse likely date columns
            if "date" in col.lower() or "dob" in col.lower():
                try:
                    parsed = pd.to_datetime(work[col], errors="coerce")
                    if parsed.notna().sum() > 0.5 * len(work[col].dropna()):
                        work[col] = parsed
                        actions.append(CleaningAction(col, "Parse dates", "Converted column to proper datetime type.", int(parsed.notna().sum())))
                except Exception:
                    pass

    # 6. Impute missing values
    if opts["impute_missing"]:
        for col in work.columns:
            n_missing = int(work[col].isna().sum())
            if n_missing == 0:
                continue
            if pd.api.types.is_numeric_dtype(work[col]):
                strategy = _numeric_fill_strategy(work[col])
                fill_value = work[col].median() if strategy == "median" else work[col].mean()
                work[col] = work[col].fillna(fill_value)
                actions.append(CleaningAction(col, f"Impute missing ({strategy})", f"Filled {n_missing} missing value(s) with column {strategy} ({fill_value:.2f}).", n_missing))
            elif pd.api.types.is_datetime64_any_dtype(work[col]):
                work[col] = work[col].ffill().bfill()
                actions.append(CleaningAction(col, "Impute missing (forward/backward fill)", f"Filled {n_missing} missing date value(s).", n_missing))
            else:
                mode = work[col].mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) else "Unknown"
                work[col] = work[col].fillna(fill_value)
                actions.append(CleaningAction(col, "Impute missing (mode)", f"Filled {n_missing} missing value(s) with most frequent value ('{fill_value}').", n_missing))

    # 7. Cap outliers (winsorize numeric columns using IQR bounds) — never applied to ID-like columns
    if opts["cap_outliers"]:
        for col in work.select_dtypes(include=[np.number]).columns:
            if "id" in col.lower():
                continue
            series = work[col].dropna()
            if len(series) < 10:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_capped = int(((work[col] < lower) | (work[col] > upper)).sum())
            if n_capped > 0:
                work[col] = work[col].clip(lower=lower, upper=upper)
                actions.append(CleaningAction(col, "Cap outliers", f"Capped {n_capped} outlier value(s) to the range [{lower:.1f}, {upper:.1f}].", n_capped))

    return CleaningResult(df=work, actions=actions)


def before_after_summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Simple side-by-side stats comparison for the UI."""
    rows = []
    rows.append({"Metric": "Rows", "Before": len(before), "After": len(after)})
    rows.append({"Metric": "Columns", "Before": before.shape[1], "After": after.shape[1]})
    rows.append({"Metric": "Missing values (total)", "Before": int(before.isna().sum().sum()), "After": int(after.isna().sum().sum())})
    rows.append({"Metric": "Duplicate rows", "Before": int(before.duplicated().sum()), "After": int(after.duplicated().sum())})
    return pd.DataFrame(rows)
