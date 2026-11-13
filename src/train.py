import pandas as pd
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def create_models(preprocessor):
    models_config = [
        ("Logistic Regression", LogisticRegression,
         {"max_iter": 1000, "random_state": 42}),
        ("Random Forest", RandomForestClassifier,
         {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
        ("Linear SVM", SVC,
         {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42, "probability": True})
    ]
    models = {}
    for name, model_class, params in models_config:
        models[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_class(**params))
        ])
    return models

def train_models(models, x_train, y_train):
    print("Entrainement des models: ")
    for name, model in models.items():
        print(f"Entrainement: {name}")
        model.fit(x_train, y_train)
        print("=="*10)
    return models

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    y_score = model.predict_proba(x_test)
    labels = model.named_steps["model"].classes_
    roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    res = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Roc-Auc": roc_auc
    }
    return res, cm

def evaluate_all_models(models, x_test, y_test):
    print("Evaluation des models: ")
    all_res = {}
    con_mat = {}
    for name, model in models.items():
        res, cm = evaluate_model(model, x_test, y_test)
        all_res[name] = res
        con_mat[name] = cm
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"Precision: {res['Precision']:.4f}")
        print(f"Recall: {res['Recall']:.4f}")
        print(f"F1-score: {res['F1-score']:.4f}")
        print(f"Roc-Auc: {res['Roc-Auc']:.4f}")
        print("\n matrice de confusion: ")
        print(cm)
    return all_res, con_mat

def create_resultats_dataframe(all_res):
    res_df = pd.DataFrame(all_res).T
    res_df = res_df.round(4)
    print("Comparaison des models: ")
    print(res_df)
    return res_df

def plot_confusion_matrices(models, x_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(x_test)
        labels = model.named_steps["model"].classes_
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Prediction")
        plt.ylabel("Valeur reelle")
        plt.title(f"Matrice de confusion - {name}")
        plt.tight_layout()
        plt.show()

def save_best_model(models, res_df, path="../models"):
    os.makedirs(path, exist_ok=True)
    best_name = res_df["F1-score"].idxmax()
    best_pipeline = models[best_name]
    pipeline_path = os.path.join(path, "best_model.pkl")
    name_path = os.path.join(path, "best_model_name.pkl")
    joblib.dump(best_pipeline, pipeline_path)
    joblib.dump(best_name,     name_path)
    print(f"meilleur modele: {best_name}")
    print(f"F1-score CV: {res_df.loc[best_name, 'F1-score']:.4f}")
    print(f"sauvegarde dans: {pipeline_path}")
    return best_name, pipeline_path


def load_best_model(path="models"):
    pipeline_path = os.path.join(path, "best_model.pkl")
    name_path = os.path.join(path, "best_model_name.pkl")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Aucun modele trouve a {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    name = joblib.load(name_path) if os.path.exists(name_path) else "Best Model"
    print(f"modele charge: {name}")
    return pipeline, name