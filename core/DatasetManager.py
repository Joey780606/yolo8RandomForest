import os
import pandas as pd
import config


def loadDataset():
    """Load WhoDataset.csv and return as DataFrame. Returns empty DataFrame if file doesn't exist."""
    try:
        if os.path.exists(config.DatasetPath):
            df = pd.read_csv(config.DatasetPath)
            return df
    except Exception as e:
        print(f"[DatasetManager] Error loading dataset: {e}")
    return pd.DataFrame(columns=["Name"] + config.AllFeatureNames)


def saveDataset(df):
    """Save DataFrame to WhoDataset.csv."""
    try:
        os.makedirs(os.path.dirname(config.DatasetPath), exist_ok=True)
        df.to_csv(config.DatasetPath, index=False)
    except Exception as e:
        print(f"[DatasetManager] Error saving dataset: {e}")
        raise


def appendRecord(name, features):
    """
    Append a single record (name + 25 features) to WhoDataset.csv.

    Args:
        name: Person's name string.
        features: List of 25 float feature values.
    """
    df = loadDataset()
    newRow = {"Name": name}
    for i, featureName in enumerate(config.AllFeatureNames):
        newRow[featureName] = features[i]
    newDf = pd.DataFrame([newRow])
    df = pd.concat([df, newDf], ignore_index=True)
    saveDataset(df)


def appendRecords(name, featuresList):
    """
    Append multiple records for the same person to WhoDataset.csv.

    Args:
        name: Person's name string.
        featuresList: List of feature lists, each containing 25 float values.
    """
    df = loadDataset()
    rows = []
    for features in featuresList:
        row = {"Name": name}
        for i, featureName in enumerate(config.AllFeatureNames):
            row[featureName] = features[i]
        rows.append(row)
    newDf = pd.DataFrame(rows)
    df = pd.concat([df, newDf], ignore_index=True)
    saveDataset(df)


def getSampleCountsPerPerson():
    """
    Get sample counts per person.

    Returns:
        Dictionary mapping person name to sample count.
    """
    df = loadDataset()
    if df.empty or "Name" not in df.columns:
        return {}
    return df["Name"].value_counts().to_dict()


def getTrainingData():
    """
    Get feature matrix X and label vector y from dataset.

    Returns:
        Tuple (X, y) where X is numpy array of shape (n, 25) and y is numpy array of name strings.
        Returns (None, None) if dataset is empty.
    """
    df = loadDataset()
    if df.empty or "Name" not in df.columns:
        return None, None

    featureCols = [col for col in config.AllFeatureNames if col in df.columns]
    if len(featureCols) != len(config.AllFeatureNames):
        print(f"[DatasetManager] Warning: expected {len(config.AllFeatureNames)} feature columns, found {len(featureCols)}")
        return None, None

    X = df[featureCols].values
    y = df["Name"].values
    return X, y
