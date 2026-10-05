import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.load_data import get_base_dir, load_data

base_dir = get_base_dir()
charts_dir = os.path.join(base_dir, "app", "static", "charts")
results_dir = os.path.join(base_dir, "results")

os.makedirs(charts_dir, exist_ok=True)

if os.path.isdir(results_dir):
    for fname in os.listdir(results_dir):
        src = os.path.join(results_dir, fname)
        dst = os.path.join(charts_dir, fname)
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        except Exception:
            pass


def basic_eda(df):
    print("First five rows")
    print(df.head())
    print("Last five rows")
    print(df.tail())
    print("Column names")
    print(df.columns.tolist())
    print("Complete information of the dataset:")
    print(df.info())
    print("Description of the dataset:")
    print(df.describe(include="all"))
    print("Missing values:")
    print(df.isnull().sum())
    print("Duplicate values:")
    print(df.duplicated().sum())

    if "click" in df.columns:
        print("Target variable (click) distribution:")
        print(df["click"].value_counts())

        count = df["click"].value_counts()
        plt.figure(figsize=(6, 5))
        colors = ["#ef4444", "#22c55e"] if len(count) > 1 else ["#4f46e5"]
        plt.bar(count.index.astype(str), count.values, color=colors)
        plt.title("Distribution of Click (Target Variable)")
        plt.xlabel("Click")
        plt.ylabel("Count")
        plt.savefig(os.path.join(charts_dir, "click_distribution.png"))
        plt.close()


def univariate(df):
    if "device_type" in df.columns:
        device_count = df["device_type"].value_counts()
        plt.figure(figsize=(6, 5))
        plt.pie(
            device_count,
            labels=device_count.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Distribution of Device Type")
        plt.savefig(os.path.join(charts_dir, "device_type_distribution.png"))
        plt.close()

    if "hour" in df.columns:
        plt.figure(figsize=(6, 5))
        plt.hist(df["hour"], bins=24, color="#4f46e5", edgecolor="white")
        plt.title("Histogram of Hour")
        plt.xlabel("Hour")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(charts_dir, "hour_distribution.png"))
        plt.close()


def bivariate(df):
    if "banner_pos" in df.columns and "click" in df.columns:
        plt.figure(figsize=(6, 5))
        banner_click = pd.crosstab(df["banner_pos"], df["click"], normalize="index")
        banner_click.plot(kind="bar", rot=0, color=["#ef4444", "#22c55e"])
        plt.title("Banner Position vs Click Rate")
        plt.xlabel("Banner Position")
        plt.ylabel("Proportion")
        plt.legend(["No Click", "Click"])
        plt.savefig(os.path.join(charts_dir, "banner_pos_click.png"))
        plt.close()

    if "device_type" in df.columns and "click" in df.columns:
        plt.figure(figsize=(6, 5))
        device_click = pd.crosstab(df["device_type"], df["click"])
        device_click.plot(kind="bar", stacked=True, rot=0)
        plt.title("Device Type vs Click")
        plt.xlabel("Device Type")
        plt.ylabel("Count")
        plt.savefig(os.path.join(charts_dir, "device_type_click.png"))
        plt.close()


def multivariate(df):
    numerical_cols = [
        "hour", "C1", "banner_pos", "device_type",
        "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "click",
    ]
    available = [col for col in numerical_cols if col in df.columns]
    if len(available) > 1:
        data = df[available]
        correlation = data.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Numerical Features)")
        plt.savefig(os.path.join(charts_dir, "correlation_heatmap.png"))
        plt.close()


if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)
    print("EDA charts saved to:", charts_dir)
