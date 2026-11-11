import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.exploration import SOURCE, exporter_data

def visualisation_boxplot(df, col):
    plt.figure(figsize=(8,5))
    df.boxplot(column=col)
    plt.title(f"boxplot de {col}")
    plt.ylabel(col)
    plt.show()

def visualisation_histogramme(df, col):
    plt.figure(figsize=(8,5))
    df[col].dropna().hist(bins=30)
    plt.title(f"distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("frequence")
    plt.show()

def visualisation_barres(df, col):
    plt.figure(figsize=(8, 5))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"repartition de {col}")
    plt.xlabel(col)
    plt.ylabel("nombre")
    plt.show()

def visualiser_feature_target(df, feature, target):
    plt.figure(figsize=(8, 5))
    df.boxplot(column=feature, by=target)
    plt.title(f"{feature} selon {target}")
    plt.suptitle("")
    plt.xlabel(target)
    plt.ylabel(feature)
    plt.show()

def visualiser_correlations(df):
    correlation = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(12, 8))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title("matrice de correlation")
    plt.tight_layout()
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
    df[col] = np.clip(df[col], a_min = min_val, a_max=max_val)
    return outliers


# if __name__ == "__main__":
#     data = exporter_data(SOURCE)
#     df = pd.DataFrame(data)
#     visualisation_histogramme(df, "Study_Hours_Per_Week")
#     visualisation_boxplot(df, "Study_Hours_Per_Week")
#     visualisation_barres(df, "Gender")
#     visualisation_barres(df, "Mental_Health_Risk")
#     visualiser_feature_target(df, "Sleep_Hours", "Mental_Health_Risk")
#     outliers = detection_outliers(df, "Study_Hours_Per_Week")
#     print(outliers)
#     visualiser_correlations(df)