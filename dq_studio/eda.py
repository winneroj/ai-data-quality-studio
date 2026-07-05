"""
AI Data Quality Studio — Lightweight EDA helpers
==================================================
Kept intentionally simple: summary stats + a handful of Plotly figures.
This is the part designed to be extended next (Phase 10 in the original
spec) once the quality/cleaning core is solid.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="object").columns.tolist()


def histogram(df: pd.DataFrame, column: str):
    return px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")


def correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    return px.imshow(corr, text_auto=".2f", title="Correlation Matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)


def bar_chart(df: pd.DataFrame, column: str, top_n: int = 15):
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    return px.bar(counts, x=column, y="count", title=f"Top values in {column}")


def box_plot(df: pd.DataFrame, column: str):
    return px.box(df, y=column, title=f"Box plot of {column} (outlier check)")
