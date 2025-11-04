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

def read_argv():
    print("argv:", sys.argv)
    
    # Default Werte
    train_load = 0
    model_name = "logistic_regression"
    model_params = {}
    
    # Argumente als Key-Value-Paare mit --key value erwartet
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            # Wert holen, wenn möglich
            if i + 1 < len(args) and not args[i+1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                # Flag ohne Wert, setze True
                value = True
                i += 1
            
            # Spezielle Keys behandeln
            if key == "train_load":
                # Annahme: 0 oder 1 als int
                try:
                    train_load = int(value)
                except:
                    pass
            elif key == "model":
                model_name = str(value)
            else:
                # Für generische Params: Try int, float else str
                try:
                    value_converted = int(value)
                except ValueError:
                    try:
                        value_converted = float(value)
                    except ValueError:
                        value_converted = value
                model_params[key] = value_converted
        else:
            # Ungültiges Argument, überspringen oder abbrechen
            print(f"Warnung: Unerwartetes Argument ohne '--': {arg}")
            i += 1
    
    return train_load, model_name, model_params

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

    train_load, model_name, model_params = read_argv()#
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
