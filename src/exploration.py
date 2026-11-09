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
    print(df.info())
    print("description: ")
    print(df.describe)

def compter_valeurs_manquants(df):
    res = df.isna().sum()
    return res

def compter_valeurs_doublons(df):
    res = df.duplicated().sum()
    return res

def manquants_percent(df):
    somme = compter_valeurs_manquants(df)
    res = (somme.sum() / len(df)) * 100
    return res


if __name__ == "__main__":
    data = exporter_data(SOURCE)
    df = pd.DataFrame(data=data)
    # analyse_data(df)
    # nb_manq = compter_valeurs_manquants(df)
    # print(f"nombre manquants est: {nb_manq}")
    # nb_doub = compter_valeurs_doublons(df)
    # print(f"nombre doublons est: {nb_doub}")
    percent_manq = manquants_percent(df)
    print(f"porcentage des valeurs manquants est: {percent_manq}%")
