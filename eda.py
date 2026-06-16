"""
EDA script for the UCI Bank Marketing Dataset.
Loads bank-full.csv, does a basic data check, then answers the four
business questions with plots (saved to the plots/ folder).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "bank-full.csv"  # change this if your file has a different name
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. Load data and basic checks
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution (y):\n", df["y"].value_counts())
print("\nClass distribution (%):\n", df["y"].value_counts(normalize=True) * 100)

# ---------------------------------------------------------------
# 2. Business Question 1: Which job types have the highest subscription rate?
# ---------------------------------------------------------------
job_rate = (
    df.groupby("job")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(x=job_rate.values, y=job_rate.index, hue=job_rate.index, palette="viridis", legend=False)
plt.xlabel("Subscription rate (%)")
plt.ylabel("Job")
plt.title("Subscription Rate by Job Type")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/subscription_by_job.png")
plt.close()

# ---------------------------------------------------------------
# 3. Business Question 2: Balance vs likelihood to subscribe
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="y", y="balance", data=df, showfliers=False)
plt.xlabel("Subscribed (y)")
plt.ylabel("Account Balance (€)")
plt.title("Account Balance vs Subscription")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/balance_vs_subscription.png")
plt.close()

avg_balance = df.groupby("y")["balance"].mean()
print("\nAverage balance by subscription status:\n", avg_balance)

# ---------------------------------------------------------------
# 4. Business Question 3: Subscription rate by age group
# ---------------------------------------------------------------
bins = [18, 30, 45, 60, 100]
labels = ["18-30", "31-45", "46-60", "60+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

age_rate = (
    df.groupby("age_group")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
)

plt.figure(figsize=(8, 6))
sns.barplot(x=age_rate.index, y=age_rate.values, hue=age_rate.index, palette="mako", legend=False)
plt.xlabel("Age Group")
plt.ylabel("Subscription rate (%)")
plt.title("Subscription Rate by Age Group")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/subscription_by_age_group.png")
plt.close()

# ---------------------------------------------------------------
# 5. Business Question 4: Does an existing housing loan reduce likelihood?
# ---------------------------------------------------------------
housing_rate = (
    df.groupby("housing")["y"]
    .apply(lambda x: (x == "yes").mean() * 100)
)

plt.figure(figsize=(6, 6))
sns.barplot(x=housing_rate.index, y=housing_rate.values, hue=housing_rate.index, palette="rocket", legend=False)
plt.xlabel("Has Housing Loan")
plt.ylabel("Subscription rate (%)")
plt.title("Subscription Rate: Housing Loan vs No Housing Loan")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/subscription_by_housing.png")
plt.close()

print("\nSubscription rate by housing loan status:\n", housing_rate)
print("\nAll plots saved to the 'plots/' folder.")
