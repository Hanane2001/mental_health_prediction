
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_DIR / "data" / "raw" / "student_mental_health.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

default_args = {
    "owner": "hanane",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_raw_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {RAW_DATA}"
        )
    print("Dataset trouve avec succes")


def run_preprocessing():
    script = PROJECT_DIR / "src" / "Preprocessing.py"
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_DIR),
        check=True
    )


def check_processed_data():
    required_files = [
        "x_train.csv",
        "x_test.csv",
        "y_train.csv",
        "y_test.csv"
    ]
    for file in required_files:
        file_path = PROCESSED_DIR / file
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier preprocessing manquant: {file}")
    print("Les donnees pretraitees sont disponibles")


def run_training():
    script = PROJECT_DIR / "src" / "train.py"
    subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_DIR), check=True)


def check_trained_models():
    required_models = [
        "logistic_regression_tuned.pkl",
        "random_forest_tuned.pkl",
        "linear_svm_tuned.pkl"
    ]

    for model in required_models:
        model_path = MODELS_DIR / model
        if not model_path.exists():
            raise FileNotFoundError(f"Modele manquant: {model}")
    print("Tous les modeles sont entraines et sauvegardes")


def run_validation():
    script = PROJECT_DIR / "src" / "validation.py"
    subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_DIR), check=True)


def check_best_model():
    required_files = ["best_model.pkl", "best_model_name.pkl"]
    for file in required_files:
        file_path = MODELS_DIR / file
        if not file_path.exists():
            raise FileNotFoundError(f"Resultat de validation manquant: {file}")
    print("Le meilleur modèle est disponible")

with DAG(
    dag_id="mental_health_prediction_pipeline",
    description="Automatisation du projet Mental Health Prediction",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["machine-learning", "mental-health"]
) as dag:

    task_check_data = PythonOperator(
        task_id="check_raw_data",
        python_callable=check_raw_data
    )

    task_preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=run_preprocessing
    )

    task_check_processed = PythonOperator(
        task_id="check_processed_data",
        python_callable=check_processed_data
    )

    task_training = PythonOperator(
        task_id="model_training",
        python_callable=run_training
    )

    task_check_models = PythonOperator(
        task_id="check_trained_models",
        python_callable=check_trained_models
    )

    task_validation = PythonOperator(
        task_id="model_validation",
        python_callable=run_validation
    )

    task_check_best_model = PythonOperator(
        task_id="check_best_model",
        python_callable=check_best_model
    )

    (
        task_check_data
        >> task_preprocessing
        >> task_check_processed
        >> task_training
        >> task_check_models
        >> task_validation
        >> task_check_best_model
    )