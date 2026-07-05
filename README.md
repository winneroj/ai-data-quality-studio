# AI Data Quality Studio

An app that automatically profiles an uploaded dataset, scores its data
quality out of 100, explains exactly what's wrong, cleans it with one click,
and lets you explore and export the result — all running locally, no
account, no server, no internet required after install.

Built for a competition demo: fast to run, nothing to misconfigure.

## What it does

1. **Upload** a CSV, Excel, TSV, or JSON file (or click "load the sample
   messy dataset" to try it instantly).
2. **Data Quality Report** — detects missing values, duplicate rows/columns,
   mixed data types, outliers, invalid emails/phones/dates, whitespace and
   casing inconsistencies, high cardinality, class imbalance, skewness, and
   correlated/leaky columns. Produces a single 0–100 quality score.
3. **Cleaning Engine** — one-click "auto-clean" pipeline (toggle any step
   off if you want manual control): removes duplicates, imputes missing
   values (median for skewed numeric columns, mean otherwise, mode for
   categoricals), fixes mixed types, parses dates, standardizes text, and
   caps outliers using IQR winsorization. Every change is logged in a
   before/after audit trail.
4. **Exploratory Analysis** — summary stats, histograms, box plots, bar
   charts, and a correlation heatmap, generated on whichever dataset
   (raw or cleaned) you're currently viewing.
5. **Export** — download the result as CSV, Excel, or JSON.

## Setup

Requires Python 3.10+.

```bash
cd ai_dq_studio
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Nothing is
sent anywhere — the app never touches the network.

## Project structure

```
ai_dq_studio/
├── app.py                    # Streamlit UI — the whole app in one file
├── dq_studio/
│   ├── profiler.py           # Data quality detection engine + scoring
│   ├── cleaner.py            # Auto-clean pipeline + audit log
│   └── eda.py                # Summary stats + Plotly charts
├── sample_data/
│   └── messy_retail_data.csv # Realistic messy dataset for demos
├── requirements.txt
└── README.md
```

## Demo script (suggested)

1. Click "load the sample messy dataset" — instant, no upload needed.
2. Show the **Data Quality Report** tab: point at the score (starts around
   80/100 on the sample), scroll the issues table, expand a column's
   details.
3. Switch to **Cleaning Engine**, click "Run one-click auto-clean" live.
   Score jumps to 100/100 on the sample dataset — show the before/after
   table and the full action log as proof it's not a black box.
4. Switch to **Exploratory Analysis** to show a chart or two on the
   now-clean data.
5. Export the cleaned file as CSV or Excel to close the loop.

## What's intentionally NOT in this version

This was scoped down from a much larger 24-phase spec (auth, Postgres,
FastAPI backend, ML training, PDF/DOCX report generation, Docker, CI/CD,
deployment) to something that's **actually solid** for a local competition
demo under time pressure. Good next additions once the deadline passes:

- ML tab: auto-detect regression/classification, train a couple of models,
  show metrics.
- PDF/DOCX report export summarizing the quality report + cleaning log.
- A "manual fix" mode next to auto-clean for column-by-column control.

## Known limitations

- Correlation/leakage check only looks at numeric columns.
- Categorical casing normalization is skipped for columns that look like
  names or emails (by design, to avoid mangling real data).
- Large files (>~200MB) may be slow since everything runs in memory.
"# ai-data-quality-studio" 
