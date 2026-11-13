import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold


def cross_validate_models(models, x, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "roc_auc": "roc_auc_ovr"
    }
    res1 = []
    for name, model in models.items():
        scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        res = {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1-score": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean()
        }
        res1.append(res)
    res_df = pd.DataFrame(res1)
    res_df = res_df.set_index("Model").round(4)
    print(f"\n comparaison des models: {res_df}")
    return res_df

def grid_search_models(models, x, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"]
        },
        "Linear SVM": [
            {
                "model__kernel": ["linear"],
                "model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"]
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [0.01, 0.1, 1, 10],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"]
            }
        ]
    }
    best_models = {}
    results = []
    print("Grid Search des modeles:")
    for name, model in models.items():
        print(f"\nOptimisation: {name}")
        grid = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring="f1_macro", cv=cv, n_jobs=-1, refit=True, verbose=1, error_score="raise")
        grid.fit(x, y)
        best_models[name] = grid.best_estimator_
        res = {
            "Model": name,
            "Best Parameters": grid.best_params_,
            "Best F1-score": grid.best_score_
        }
        results.append(res)
        print("meilleurs parametres:", grid.best_params_)
        print("meilleur f1-score:", round(grid.best_score_, 4))
    results_df = pd.DataFrame(results).set_index("Model")
    print("\nresultats Grid Search:")
    print(results_df)
    return best_models, results_df