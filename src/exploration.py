import pandas as pd
import os
from pathlib import Path

SOURCE = Path("data/raw")
def exporter_data(path):
    if not os.path.exists(path):
        return []
    else:
        data = pd.read_csv(path / "student_mental_health.csv")
        return data

def analyse_data(df):
    print(f"shape: {df.shape}")
    print(f"nombre dimension: {df.ndim}")
    print("les types de donnees: ")
    print(df.dtypes)
    print("info: ")
    df.info()
    print("description: ")
    print(df.describe())

def compter_valeurs_manquants(df):
    res = df.isna().sum()
    return res

def compter_valeurs_doublons(df):
    res = df.duplicated().sum()
    return res

def manquants_percent(df):
    somme = compter_valeurs_manquants(df)
    res = (somme / len(df)) * 100
    return res

def eviter_doublons(df):
    df = df.drop_duplicates()
    return df

def verifier_types_col(df, col):
    types = df[col].dropna().apply(type).unique()
    print(col, types)

def verifier_types(df):
    for col in df.columns:
        types = df[col].apply(type).nunique()
        if types > 1:
            print(col, types)

def verifier_categories(df, col, valeurs_autorisees):
    valeurs = df[col].dropna().unique()
    incoherences = []
    for val in valeurs:
        if val not in valeurs_autorisees:
            incoherences.append(val)
    print(f"colonne: {col}")
    print(f"valeurs autorisees: {valeurs_autorisees}")
    print(f"valeurs incorrectes: {incoherences}")
    return incoherences

def verifier_limites(df, col, min_val, max_val):
    incoherences = df[(df[col] < min_val) | (df[col] > max_val)]
    print(f"colonne: {col}")
    print(f"nombre de valeurs incorrectes: {len(incoherences)}")
    return incoherences

def afficher_valeurs_uniques(df, col):
    print(f"valeurs uniques de {col}:")
    print(df[col].unique())

def analyser_target(df, target):
    print("nombre d'etudiants par classe:")
    print(df[target].value_counts())
    print("pourcentage par classe:")
    print(df[target].value_counts(normalize=True) * 100)


# if __name__ == "__main__":
#     data = exporter_data(SOURCE)
#     df = pd.DataFrame(data=data)
#     analyse_data(df)
#     nb_manq = compter_valeurs_manquants(df)
#     print(f"nombre manquants est: {nb_manq}")
#     percent_manq = manquants_percent(df)
#     print(f"porcentage des valeurs manquants est: {percent_manq}%")
#     nb_doub = compter_valeurs_doublons(df)
#     print(f"nombre doublons est: {nb_doub}")
#     df = eviter_doublons(df)
#     verifier_types(df)
#     verifier_types_col(df, "Diet_Quality")
#     verifier_categories(df, "Gender", ["Male", "Female", "Non-binary"])
#     verifier_categories(df, "Diet_Quality", ["Poor", "Average", "Good"])
#     verifier_categories(df, "Mental_Health_Risk", ["Low", "Medium", "High"])
#     verifier_limites(df, "Age", 18, 30)
#     verifier_limites(df, "GPA", 1.5, 4.0)
#     afficher_valeurs_uniques(df, "Gender")
#     afficher_valeurs_uniques(df, "Diet_Quality")
#     afficher_valeurs_uniques(df, "Mental_Health_Risk")
#     analyser_target(df, "Mental_Health_Risk")
