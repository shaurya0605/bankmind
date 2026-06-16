# BankMind Challenge - Track A (Data Analyst)

A small EDA + interactive Streamlit dashboard exploring the UCI Bank
Marketing Dataset, built for the VITB AI Innovators Hub screening task.

## What's in here

- `eda.py` – loads the data, checks shape/dtypes/missing values/class
  distribution, and saves the four required plots into `plots/`.
- `app.py` – interactive Streamlit dashboard with filters (job, age
  range, housing loan) showing the same four insights live.
- `EXPLANATION.md` – answers to the reflection questions.
- `requirements.txt` – Python dependencies.

## Setup

1. Download `bank-full.csv` from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
   and place it in this folder (same folder as `app.py`).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the EDA script

```bash
python eda.py
```

This prints the data checks to the console and saves four plots into
the `plots/` folder.

## Running the dashboard

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).
Use the sidebar to filter by job, age range, and housing loan status —
the metrics and all four charts update live.

## Notes

- The dataset uses `;` as the column separator (not a comma) — already
  handled in both scripts via `sep=";"`.
- `bank-full.csv` is not included in this repo (per dataset license /
  size); download it separately as described above.
