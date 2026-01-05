"""Data loading and preprocessing utilities for the heart disease project.

This module provides functions to load the Heart Disease UCI dataset from
disk, perform basic preprocessing (handling missing values, encoding
categorical features, scaling numerical features), and split the data
into training and test sets. It relies on scikit‑learn transformers to
construct a reproducible preprocessing pipeline that can be saved and
reused during inference.

Note: The dataset shipped with this repository (`data/heart.csv`) is a
processed version of the Cleveland heart disease dataset with 918
observations and 11 predictive features plus the binary target. See
scripts/download_data.sh for download instructions.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Import SimpleImputer directly from sklearn.impute. Using the top‑level
# sklearn module to access SimpleImputer (e.g. via __import__("sklearn"))
# can lead to AttributeError because the impute submodule isn't exposed at
# the top level in recent versions. Direct import ensures correct usage.
from sklearn.impute import SimpleImputer

def load_dataset(path: str) -> pd.DataFrame:
    """Load the heart disease dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please run scripts/download_data.sh")
    df = pd.read_csv(path)
    return df


def preprocess_features(df: pd.DataFrame, target_col: str = "HeartDisease") -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Split the dataset into features and target and build a preprocessing pipeline.

    The preprocessing pipeline performs the following steps:

    - Numerical features: impute missing values with the median and scale using StandardScaler.
    - Categorical features: impute missing values with the most frequent value and
      apply one‑hot encoding.

    Args:
        df: The full dataset as a pandas DataFrame.
        target_col: Name of the target column.

    Returns:
        X_processed: Transformed feature matrix ready for modelling.
        y: Target vector.
        preprocessor: Fitted scikit‑learn pipeline that performs the transformations.
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine transformers into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def train_test_split_data(df: pd.DataFrame, target_col: str = "HeartDisease", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and test sets while preserving the target distribution.

    Args:
        df: The full dataset as a pandas DataFrame.
        target_col: Name of the target column.
        test_size: Fraction of the dataset to allocate to the test set.
        random_state: Random seed for reproducibility.

    Returns:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        y_train: Training target Series.
        y_test: Test target Series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)