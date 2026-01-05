"""Exploratory Data Analysis for the Heart Disease dataset.

This script loads the dataset from the `data/` directory, performs basic
cleaning and exploratory analysis, and saves visualizations to the
`images/` directory. The figures include histograms for numerical
variables, a correlation heatmap, and the class distribution. Run this
script with Python to produce the plots before generating the final report.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart.csv"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # Basic summary statistics
    summary_file = IMAGE_DIR / "summary_stats.csv"
    df.describe(include="all").transpose().to_csv(summary_file)

    # Histograms for numerical features
    num_cols = df.select_dtypes(exclude=["object", "category"]).columns.drop("HeartDisease")
    plt.figure(figsize=(12, 8))
    df[num_cols].hist(bins=20, figsize=(12, 8), layout=(len(num_cols) // 3 + 1, 3))
    plt.suptitle("Distribution of Numerical Features")
    hist_file = IMAGE_DIR / "histograms.png"
    plt.tight_layout()
    plt.savefig(hist_file)
    plt.close()

    # Correlation heatmap
    corr = df[num_cols.tolist() + ["HeartDisease"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    heatmap_file = IMAGE_DIR / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_file)
    plt.close()

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="HeartDisease", data=df)
    plt.title("Class Distribution")
    plt.xlabel("Heart Disease")
    plt.ylabel("Count")
    class_file = IMAGE_DIR / "class_balance.png"
    plt.tight_layout()
    plt.savefig(class_file)
    plt.close()

    print(f"EDA completed. Summary stats saved to {summary_file}. Plots saved in {IMAGE_DIR}.")


if __name__ == "__main__":
    main()