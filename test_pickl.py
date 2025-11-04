import os
import pickle
import traceback
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import sys


MODEL_PATH = "models/logistic_regression_model.pkl"

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def prepare_data():
    # Fetch dataset from UCI ML Repository
    heart_disease = fetch_ucirepo(id=45)

    # Extract features and targets
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine into single DataFrame
    data = pd.concat([X, y], axis=1)

    # Separate features and target
    X = data.drop("num", axis=1)
    y = data["num"].apply(lambda x: 0 if x == 0 else 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #Preprocessing
    X_train_no_nan = X_train.dropna()
    y_train_no_nan = y_train[X_train_no_nan.index]
    X_test_no_nan = X_test.dropna()
    y_test_no_nan = y_test[X_test_no_nan.index]

    # return numpy arrays to avoid pandas truth-value ambiguity
    return X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan

def test_model():
    model = load_model(MODEL_PATH)
    _, X_test_scaled, _, y_test = prepare_data()

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.3f}\n")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print("Error during testing:", e)
        traceback.print_exc()