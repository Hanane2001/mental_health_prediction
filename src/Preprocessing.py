import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exploration import SOURCE, exporter_data

def definir_xy(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def separer_train_test(X, y):
    x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return x_train, y_train, x_test, y_test

def separer_types(x_train):
    num_col = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_col = x_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print(f"variable numeriques: {num_col}")
    print(f"variable categorielles: {cat_col}")
    return num_col, cat_col

def cree_preprocessing(num_col, cat_col):
    num_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    onehot_col = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]
    ordinal_col = [
        "Physical_Activity",
        "Diet_Quality"
    ]
    onehot_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    ordinal_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[
            ["Low", "Moderate", "High"],
            ["Poor", "Average", "Good"]
        ], handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipline, num_col),
        ("onehot", onehot_pipline, onehot_col),
        ("ordinal", ordinal_pipline, ordinal_col)
    ])
    return preprocessor

def applique_preprocessing(preprocessor, x_train, x_test):
    x_train_prepared = preprocessor.fit_transform(x_train)
    x_test_prepared = preprocessor.transform(x_test)
    return x_train_prepared, x_test_prepared

# if __name__ == "__main__":
#     df = exporter_data(SOURCE)
#     X, y = definir_xy(df, "Mental_Health_Risk")
#     X_train, X_test, y_train, y_test = separer_train_test(X, y)
#     print("train:", X_train.shape)
#     print("test:", X_test.shape)
#     num_col, cat_col = separer_types(X_train)
#     preprocessor = cree_preprocessing(num_col, cat_col)
#     X_train_prepared, X_test_prepared = applique_preprocessing(preprocessor, X_train, X_test)
#     print("resultats:")
#     print("X_train:", X_train_prepared.shape)
#     print("X_test:", X_test_prepared.shape)
#     print("y_train:", y_train.shape)
#     print("y_test:", y_test.shape)