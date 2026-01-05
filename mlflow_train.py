
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

mlflow.set_experiment("Heart_Disease_Experiment")

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

    mlflow.sklearn.log_model(model, "model")

    with open("models/logistic_mlflow.pkl", "wb") as f:
        pickle.dump(model, f)
