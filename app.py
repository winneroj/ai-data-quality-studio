"""
AI Data Quality Studio
======================
Run with:  streamlit run app.py

Single-file Streamlit app for the competition demo. Deliberately no
auth/database/microservices — everything runs locally, in-memory, from
one process, so it's fast to demo and has nothing to misconfigure.
"""

import io
import streamlit as st
import pandas as pd

from dq_studio.profiler import profile_dataframe
from dq_studio.cleaner import auto_clean, before_after_summary
from dq_studio import eda
from dq_studio import ml

st.set_page_config(page_title="AI Data Quality Studio", page_icon="🧪", layout="wide")

# ---------- Session state ----------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = None

# ---------- Sidebar ----------
st.sidebar.title("🧪 AI Data Quality Studio")
st.sidebar.caption("Upload → Detect → Clean → Explore → Export")

uploaded = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "tsv", "json"])
use_sample = st.sidebar.button("Or load the sample messy dataset")

if uploaded is not None:
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv") or name.endswith(".tsv"):
            sep = "\t" if name.endswith(".tsv") else ","
            df = pd.read_csv(uploaded, sep=sep)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded)
        elif name.endswith(".json"):
            df = pd.read_json(uploaded)
        else:
            st.sidebar.error("Unsupported file type.")
            df = None
        if df is not None:
            st.session_state.raw_df = df
            st.session_state.cleaned_df = None
            st.session_state.cleaning_log = None
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

if use_sample:
    st.session_state.raw_df = pd.read_csv("sample_data/messy_retail_data.csv")
    st.session_state.cleaned_df = None
    st.session_state.cleaning_log = None

st.title("AI Data Quality Studio")

if st.session_state.raw_df is None:
    st.info("👈 Upload a CSV/Excel/TSV/JSON file, or click **'Or load the sample messy dataset'** in the sidebar to try it out instantly.")
    st.stop()

raw_df = st.session_state.raw_df

tab_quality, tab_cleaning, tab_eda, tab_ml, tab_export = st.tabs(
    ["📊 Data Quality Report", "🧹 Cleaning Engine", "🔍 Exploratory Analysis", "🤖 Machine Learning", "📤 Export"]
)

# ---------- TAB 1: Data Quality Report ----------
with tab_quality:
    report = profile_dataframe(raw_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Quality Score", f"{report.score}/100", report.grade.split("—")[0].strip())
    col2.metric("Rows", f"{report.n_rows:,}")
    col3.metric("Columns", f"{report.n_cols:,}")
    col4.metric("Issues Found", len(report.issues))

    st.progress(min(report.score / 100, 1.0))
    st.caption(report.grade)

    st.subheader("Detected Issues")
    issues_df = report.as_dataframe()
    if issues_df.empty:
        st.success("No issues detected — this dataset is clean! ✨")
    else:
        severity_filter = st.multiselect("Filter by severity", ["high", "medium", "low"], default=["high", "medium", "low"])
        st.dataframe(issues_df[issues_df["Severity"].isin(severity_filter)], use_container_width=True, hide_index=True)

    with st.expander("Column-level summary"):
        st.dataframe(pd.DataFrame(report.column_summary).T, use_container_width=True)

    with st.expander("Raw data preview"):
        st.dataframe(raw_df.head(50), use_container_width=True)

# ---------- TAB 2: Cleaning Engine ----------
with tab_cleaning:
    st.subheader("Recommended cleaning pipeline")
    st.caption("Toggle steps on/off, then run. Every action is logged so you can show judges exactly what changed and why.")

    c1, c2, c3 = st.columns(3)
    with c1:
        drop_dupes = st.checkbox("Remove duplicate rows", value=True)
        drop_dupe_cols = st.checkbox("Remove duplicate columns", value=True)
    with c2:
        impute = st.checkbox("Impute missing values (mean/median/mode)", value=True)
        clean_text = st.checkbox("Clean text (whitespace, casing)", value=True)
    with c3:
        fix_dtypes = st.checkbox("Fix data types & parse dates", value=True)
        cap_outliers = st.checkbox("Cap outliers (IQR winsorize)", value=True)

    if st.button("🚀 Run one-click auto-clean", type="primary"):
        options = {
            "drop_duplicates": drop_dupes,
            "drop_duplicate_columns": drop_dupe_cols,
            "impute_missing": impute,
            "clean_text": clean_text,
            "fix_dtypes": fix_dtypes,
            "cap_outliers": cap_outliers,
        }
        result = auto_clean(raw_df, options)
        st.session_state.cleaned_df = result.df
        st.session_state.cleaning_log = result.log()

    if st.session_state.cleaned_df is not None:
        st.success(f"Cleaning complete — {len(st.session_state.cleaning_log)} action(s) applied.")

        st.subheader("Before / After")
        summary = before_after_summary(raw_df, st.session_state.cleaned_df)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        new_score = profile_dataframe(st.session_state.cleaned_df).score
        old_score = profile_dataframe(raw_df).score
        st.metric("Quality Score", f"{new_score}/100", delta=round(new_score - old_score, 1))

        st.subheader("What changed")
        st.dataframe(st.session_state.cleaning_log, use_container_width=True, hide_index=True)

        with st.expander("Cleaned data preview"):
            st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)
    else:
        st.warning("Run the auto-clean pipeline above to see before/after results.")

# ---------- TAB 3: EDA ----------
with tab_eda:
    active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else raw_df
    st.caption("Analyzing " + ("the cleaned dataset" if st.session_state.cleaned_df is not None else "the raw dataset (clean it first for better results)"))

    st.subheader("Summary statistics")
    st.dataframe(eda.summary_stats(active_df), use_container_width=True)

    num_cols = eda.numeric_columns(active_df)
    cat_cols = eda.categorical_columns(active_df)

    colA, colB = st.columns(2)
    with colA:
        if num_cols:
            chosen_num = st.selectbox("Numeric column to visualize", num_cols)
            st.plotly_chart(eda.histogram(active_df, chosen_num), use_container_width=True)
            st.plotly_chart(eda.box_plot(active_df, chosen_num), use_container_width=True)
    with colB:
        if cat_cols:
            chosen_cat = st.selectbox("Categorical column to visualize", cat_cols)
            st.plotly_chart(eda.bar_chart(active_df, chosen_cat), use_container_width=True)

    corr_fig = eda.correlation_heatmap(active_df)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)

# ---------- TAB 4: Machine Learning ----------
with tab_ml:
    active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else raw_df
    st.caption("Training on " + ("the cleaned dataset (recommended)." if st.session_state.cleaned_df is not None else "the raw dataset — clean it first for better results."))

    target = st.selectbox("Choose a target column to predict", active_df.columns.tolist())

    if target:
        problem_type = ml.detect_problem_type(active_df[target].dropna())
        st.info(f"Detected problem type: **{problem_type.title()}** "
                f"(based on the target column's data type and number of unique values).")

        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

        if st.button("🎯 Train models", type="primary"):
            with st.spinner("Training Linear/Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting..."):
                try:
                    report = ml.train_and_evaluate(active_df, target, test_size=test_size)
                    st.session_state.ml_report = report
                except Exception as e:
                    st.error(f"Couldn't train models: {e}")
                    st.session_state.ml_report = None

    if st.session_state.get("ml_report") is not None:
        report = st.session_state.ml_report
        st.success(f"Best model: **{report.best_model}**")
        st.caption(f"Features used: {', '.join(report.features_used)}")

        st.subheader("Model comparison")
        results_df = report.as_dataframe()
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        best = report.results[0]
        if report.problem_type == "classification" and best.confusion is not None:
            st.subheader(f"Confusion matrix — {best.name}")
            import plotly.express as px
            labels = report.class_labels if report.class_labels else [str(i) for i in range(len(best.confusion))]
            fig = px.imshow(best.confusion, text_auto=True, x=labels, y=labels,
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        elif report.problem_type == "regression":
            st.subheader(f"Predicted vs Actual — {best.name}")
            import plotly.express as px
            import pandas as pd
            plot_df = pd.DataFrame({"Actual": active_df.dropna(subset=[report.target])[report.target].iloc[-len(best.predictions):].values, "Predicted": best.predictions})
            fig = px.scatter(plot_df, x="Actual", y="Predicted", trendline="ols", title="Predicted vs Actual")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Pick a target column and click **Train models** to see results here.")

# ---------- TAB 5: Export ----------
with tab_export:
    active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else raw_df
    st.caption("Exporting " + ("the cleaned dataset." if st.session_state.cleaned_df is not None else "the raw dataset — run cleaning first if you want the cleaned version."))

    csv_bytes = active_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download as CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

    excel_buffer = io.BytesIO()
    active_df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button("⬇️ Download as Excel", excel_buffer.getvalue(), file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    json_bytes = active_df.to_json(orient="records", indent=2).encode("utf-8")
    st.download_button("⬇️ Download as JSON", json_bytes, file_name="cleaned_data.json", mime="application/json")
