import matplotlib.pyplot as plt
import pandas as pd

def visualisation_boxplot(df, col):
    plt.figure(figsize=(10,8))
    df.boxplot(column=col)
    plt.title(f"boxplot de {col}")
    plt.ylabel(col)
    plt.show()

def visualisation_histogramme(df, col):
    plt.figure(figsize=(10,8))
    df[col].dropna().hist(bins=30)
    plt.title(f"distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("frequence")
    plt.show()

def visualisation_barres(df, col):
    plt.figure(figsize=(10, 8))
    df[col].values_counts().plot(kind="bar")
    plt.title(f"repartition de {col}")
    plt.xlabel(col)
    plt.ylabel("nombre")
    plt.show()

def detection_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    min_val = Q1 - 1.5 * IQR
    max_val = Q3 + 1.5 * IQR
    outliers = df[(df[col] < min_val) | (df[col] > max_val)]
    print(f"colonne: {col}")
    print(f"nombre des outliers: {len(outliers)}")
    return outliers