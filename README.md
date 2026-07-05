# 🧪 AI Data Quality Studio

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

**Upload a messy dataset. Get a quality score, a plain-English list of
what's wrong, a one-click fix, exploratory charts, and a trained ML model —
all running locally in your browser, no account, no server, no internet
required after install.**

## 🎯 The problem

Across Nigerian primary health centres, agricultural extension offices,
microfinance branches, and local government data units, frontline staff
collect data in Excel and paper-to-digital forms every day — patient
visits, farmer surveys, loan applications, census records. That data is
almost never clean: duplicate patient entries from re-registration,
inconsistent diagnosis spellings ("malaria" / "MALARIA" / "Malaria "),
missing phone numbers, impossible ages, invalid dates. Most of these
teams have no data scientist on staff, so the errors flow straight into
reports that inform real decisions — health resource allocation, loan
approvals, agricultural planning, government budgeting.

**AI Data Quality Studio puts a data quality engine in front of anyone who
can open a spreadsheet, no coding required.** The included demo dataset
models messy primary health centre (PHC) patient records — but the same
engine works unmodified on agriculture, finance, or governance data; data
quality problems (missing values, duplicates, inconsistent categories,
invalid contact info) look the same shape regardless of the domain.

**One deliberate design choice:** when data truly can't be recovered — a
patient's phone number was simply never recorded — the tool marks it
"Not Provided" rather than fabricating a realistic-looking fake value.
Silently guessing a phone number or copying someone else's real email
onto hundreds of other records would be actively dangerous for health
data. The tool is honest about what still needs a human to follow up on.

---

## ✨ Features

- ✅ **Automated Data Quality Scoring** — a single 0–100 score, explained
- ✅ **12+ issue detectors** — missing values, duplicate rows/columns,
  mixed data types, outliers, invalid emails/phones/dates, whitespace,
  inconsistent casing, high cardinality, class imbalance, skewness,
  correlated/leaky columns
- ✅ **One-click AI cleaning pipeline** — imputation, deduplication,
  type fixing, date parsing, text normalization, outlier capping —
  with a full before/after audit log (nothing is a black box)
- ✅ **Exploratory Data Analysis** — summary stats, histograms, box plots,
  bar charts, correlation heatmap
- ✅ **Machine Learning** — auto-detects regression vs. classification,
  trains 4 models (Linear/Logistic Regression, Decision Tree, Random
  Forest, Gradient Boosting), ranks them, shows metrics + confusion matrix
  or predicted-vs-actual plot
- ✅ **Multi-format import/export** — CSV, Excel, TSV, JSON in; CSV, Excel,
  JSON out
- ✅ **Zero setup friction** — one Python process, no database, no auth,
  nothing to deploy

## 📸 What it looks like

| Tab | What you get |
|---|---|
| 📊 Data Quality Report | Score, grade, filterable issue list, column-level breakdown |
| 🧹 Cleaning Engine | Toggle-able auto-clean pipeline + before/after comparison |
| 🔍 Exploratory Analysis | Stats + interactive Plotly charts |
| 🤖 Machine Learning | Auto model selection + evaluation metrics |
| 📤 Export | Download the cleaned dataset in your format of choice |

## 🚀 Quick start

```bash
git clone https://github.com/winneroj/ai-data-quality-studio.git
cd ai-data-quality-studio
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Click **"load the sample messy dataset"**
in the sidebar for an instant demo — no file needed.

## 🗂 Project structure

```
ai_dq_studio/
├── app.py                    # Streamlit UI — the whole app in one file
├── dq_studio/
│   ├── profiler.py           # Data quality detection engine + scoring
│   ├── cleaner.py            # Auto-clean pipeline + audit log
│   ├── eda.py                # Summary stats + Plotly charts
│   └── ml.py                 # Model training, evaluation, auto-selection
├── sample_data/
│   └── messy_retail_data.csv # Realistic messy dataset for demos
├── requirements.txt
└── README.md
```

## 🎤 Suggested demo script

1. In the sidebar, pick **"🏥 Messy clinic patient records"** and click
   **"Load sample"** — instant, no upload needed.
2. **Data Quality Report** tab: point at the score, scroll the issues
   table — call out the inconsistent diagnosis spellings and missing
   contact info as the kind of thing that quietly breaks health reporting.
3. **Cleaning Engine** tab: click "Run one-click auto-clean" live. Show
   the before/after table and the full action log — especially the
   "Flag as missing" entries for phone/email, to make the honesty point.
4. **Exploratory Analysis** tab: show a chart or two on the cleaned data.
5. **Machine Learning** tab: pick a target column (e.g. Diagnosis), click
   "Train models," show the model comparison table.
6. **Export** tab: download the cleaned file to close the loop.

## 🛣 Roadmap

- [ ] PDF/DOCX report export (quality report + cleaning log + charts)
- [ ] Manual column-by-column cleaning mode alongside auto-clean
- [ ] Date-based feature engineering for the ML module
- [ ] Optional user accounts / saved history (currently intentionally
      out of scope — this app runs fully in-memory, single-user, by design)

## ⚠️ Known limitations

- Correlation/leakage detection only considers numeric columns.
- Categorical casing normalization is skipped for columns that look like
  names or emails, by design, to avoid mangling real data.
- Large files (200MB+) may be slow since everything runs in memory.
- ML module uses scikit-learn only (no XGBoost/LightGBM) to keep install
  lightweight and dependency-free of compiled binaries.

## 📄 License

MIT — see [LICENSE](LICENSE).
