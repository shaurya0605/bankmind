"""
Streamlit dashboard for the BankMind Challenge - Track A.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BankMind - Customer Insights", layout="wide")
sns.set_style("whitegrid")

DATA_PATH = "bank-full.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    bins = [18, 30, 45, 60, 100]
    labels = ["18-30", "31-45", "46-60", "60+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


df = load_data()

st.title("BankMind: Customer Cross-Sell Insights")
st.markdown(
    "Explore which customers are most likely to subscribe to a new "
    "financial product, based on the UCI Bank Marketing Dataset."
)

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.header("Filters")

job_options = ["All"] + sorted(df["job"].unique().tolist())
selected_job = st.sidebar.selectbox("Job", job_options)

age_range = st.sidebar.slider(
    "Age range", int(df["age"].min()), int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)

housing_options = ["All", "yes", "no"]
selected_housing = st.sidebar.selectbox("Has Housing Loan", housing_options)

# Apply filters
filtered = df.copy()
if selected_job != "All":
    filtered = filtered[filtered["job"] == selected_job]
filtered = filtered[
    (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])
]
if selected_housing != "All":
    filtered = filtered[filtered["housing"] == selected_housing]

st.sidebar.markdown(f"**Filtered rows:** {len(filtered)}")

# ---------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total customers (filtered)", len(filtered))
sub_rate = (filtered["y"] == "yes").mean() * 100 if len(filtered) else 0
col2.metric("Subscription rate", f"{sub_rate:.1f}%")
col3.metric("Avg. balance (€)", f"{filtered['balance'].mean():,.0f}" if len(filtered) else "N/A")

st.divider()

# ---------------------------------------------------------------
# Q1: Subscription rate by job
# ---------------------------------------------------------------
st.subheader("1. Subscription Rate by Job Type")
job_rate = (
    filtered.groupby("job")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .sort_values(ascending=False)
)
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x=job_rate.values, y=job_rate.index, hue=job_rate.index,
            palette="viridis", legend=False, ax=ax1)
ax1.set_xlabel("Subscription rate (%)")
ax1.set_ylabel("Job")
st.pyplot(fig1)

# ---------------------------------------------------------------
# Q2: Balance vs subscription
# ---------------------------------------------------------------
st.subheader("2. Account Balance vs Subscription")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(x="y", y="balance", data=filtered, showfliers=False, ax=ax2)
ax2.set_xlabel("Subscribed")
ax2.set_ylabel("Balance (€)")
st.pyplot(fig2)

# ---------------------------------------------------------------
# Q3: Subscription rate by age group
# ---------------------------------------------------------------
st.subheader("3. Subscription Rate by Age Group")
age_rate = (
    filtered.groupby("age_group")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
)
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(x=age_rate.index, y=age_rate.values, hue=age_rate.index,
            palette="mako", legend=False, ax=ax3)
ax3.set_xlabel("Age Group")
ax3.set_ylabel("Subscription rate (%)")
st.pyplot(fig3)

# ---------------------------------------------------------------
# Q4: Housing loan vs subscription
# ---------------------------------------------------------------
st.subheader("4. Housing Loan vs Subscription Rate")
housing_rate = (
    filtered.groupby("housing")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
)
fig4, ax4 = plt.subplots(figsize=(6, 5))
sns.barplot(x=housing_rate.index, y=housing_rate.values, hue=housing_rate.index,
            palette="rocket", legend=False, ax=ax4)
ax4.set_xlabel("Has Housing Loan")
ax4.set_ylabel("Subscription rate (%)")
st.pyplot(fig4)

st.divider()
st.subheader("Raw Data (filtered)")
st.dataframe(filtered.head(100))
