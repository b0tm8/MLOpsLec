# Imports
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import sys
import mlflow
import argparse

def parse_unknown_args(unknown_args):
    """Parst Liste wie ['--param1', 'val1', '--param2', 'val2'] nach dict"""
    params = {}
    key = None
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("-")
            params[key] = True  # falls kein Wert folgt
        else:
            if key:
                # Versuche Wert zu casten (int, float)
                try:
                    val = int(arg)
                except:
                    try:
                        val = float(arg)
                    except:
                        val = arg
                params[key] = val
                key = None
    return params

def read_args():
    parser = argparse.ArgumentParser(description="Trainingsskript")

    parser.add_argument("--train_load", type=int, choices=[0,1], default=0)
    parser.add_argument("--model", type=str, required=True, 
                        choices=["logistic_regression", "decision_tree"])

    args, unknown = parser.parse_known_args()

    model_params = parse_unknown_args(unknown)

    return args.train_load, args.model, model_params

def load_data():
    # Fetch dataset from UCI ML Repository
    heart_disease = fetch_ucirepo(id=45)

    # Extract features and targets
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine into single DataFrame
    data = pd.concat([X, y], axis=1)
    return data

def preprocess_data(data):
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

    return X_train_no_nan, X_test_no_nan, y_train_no_nan, y_test_no_nan


def train_model(model_name, model_params, X, y):
    # model selection
    if model_name == "logistic_regression":
        model = LogisticRegression(**model_params, random_state=42)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(**model_params ,random_state=42)

    print(f"Training {model_name} with params: {model_params}")
    mlflow.log_param("model_name", model_name)
    for param, value in model_params.items():
        mlflow.log_param(param, value)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    # Fit pipeline on cleaned training data (rows without NaNs)
    pipeline.fit(X, y)

    scores = evaluate_model(pipeline, X, y)
    for metric, value in scores.items():
        mlflow.log_metric("training " + metric, value)
        print(f"training {metric}: {value}")

    return pipeline

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    pre = np.sum(y == y_pred) / len(y)
    rec = np.sum((y == 1) & (y_pred == 1)) / np.sum(y == 1)
    f1 = 2 * (pre * rec) / (pre + rec)
    return {"acuracy": acc,"precision": pre,"recall": rec,"f1 score": f1}


def safe_model(model_name, pipeline):
    # safe model to pickle file
    os.makedirs("models", exist_ok=True)
    file_name = f"models/{model_name}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(pipeline, f)
    mlflow.log_artifact(file_name)

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    train_load, model_name, model_params = read_args()

    try:
        mlflow.set_experiment("Heart Disease Prediction")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_name=f"{model_name}{model_params}"):
            #load data
            data = load_data()
            X_train, X_test, y_train, y_test = preprocess_data(data)

            #train or load model    
            if train_load == 0:
                model = train_model(model_name, model_params, X_train, y_train)
            elif train_load == 1:
                model_path = f"models/{model_name}.pkl"
                model = load_model(model_path)

            test_scores = evaluate_model(model, X_test, y_test)
            for metric, value in test_scores.items():
                mlflow.log_metric("test " + metric, value)
                print(f"test {metric}: {value}")
            safe_model(model_name, model)


    except Exception as e:
        print("An error occurred:", e.with_traceback(sys.exc_info()[2]))