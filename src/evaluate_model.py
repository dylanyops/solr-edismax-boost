import mlflow
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score
import yaml

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

import os

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------
# Load config
# -------------------------------

CONFIG_FILE = Path("configs/config.yaml")

with CONFIG_FILE.open() as f:
    config = yaml.safe_load(f)

DATASET_PATH = Path(config["data"]["parquet_file"])
TRAINING_DATASET_PATH = Path(config["data"]["training_parquet_file"])
REPORT_PATH = Path(config["data"]["report_file"])

# -------------------------------
# Load dataset
# -------------------------------

def load_data(evaluate):
    path = DATASET_PATH if evaluate else TRAINING_DATASET_PATH
    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError(f"Directory does not exist: {os.path.dirname(path)}")

    return pd.read_parquet(path)

# -------------------------------
# Evaluate ranking quality
# -------------------------------

def evaluate_ranking(df):
    scores = []
    for _, group in df.groupby("query_id"):
        y_true = group["label"].values
        y_pred = group["prediction_score"].values

        if y_true.sum() == 0:
            continue  # skip queries with no clicked documents
        
        scores.append(ndcg_score([y_true], [y_pred]))
    ndcg = sum(scores) / len(scores)
    return ndcg

# -------------------------------
# Run Evidently
# -------------------------------

def run_evidently(df_evaluation, df_training):
    # Drop IDs for data drift
    df_evaluation = df_evaluation.drop(columns=["query_id", "person_id"])
    df_training = df_training.drop(columns=["query_id", "person_id"])
    
    # Create Evidently report
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])

    print("Report object:", report)
    
    result = report.run(reference_data=df_training, current_data=df_evaluation)

    print("Result: ", result)
    
    # Save HTML
    #REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.save_html(str(REPORT_PATH))
        print("save_html executed")
    except Exception as e:
        print("ERROR during save_html:", e)

    if not REPORT_PATH.exists():
        raise RuntimeError("Report was NOT created!")

# -------------------------------
# Main
# -------------------------------

def main():
    df_evaluation = load_data(True)
    df_training = load_data(False)
    ndcg = evaluate_ranking(df_evaluation)
    run_evidently(df_evaluation, df_training)

    mlflow.set_experiment("ridge_feature_weights")

    import os

    print("REPORT_PATH:", REPORT_PATH)
    print("Exists:", os.path.exists(REPORT_PATH))

    if os.path.exists("/mnt/data"):
        print("Files in /mnt/data:", os.listdir("/mnt/data"))
    
    # Log metrics and report to MLflow
    with mlflow.start_run():
        mlflow.log_metric("ndcg", ndcg)
        mlflow.log_artifact(str(REPORT_PATH))
    
    print("NDCG:", ndcg)

if __name__ == "__main__":
    main()