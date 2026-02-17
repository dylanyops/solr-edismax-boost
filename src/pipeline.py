import mlflow
import yaml
import time
import pandas as pd
from src import generate_features_wide as gen_wide
from src import feature_engineering as feat_eng
from src import ridge_normalize as ridge

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(config_path="configs/config.yaml"):
    config = load_config(config_path)

    mlflow.set_experiment("bm25_ranking_pipeline")

    with mlflow.start_run(run_name="ridge_weight_training"):
        start_time = time.time()

        # Log full config
        mlflow.log_params({
            "ridge_alpha": config["ridge"]["alpha"],
            "bm25_fields": ",".join(config["features"]["bm25_fields"])
        })
        mlflow.log_artifact(config_path, artifact_path="config")

        # Step 1: Generate wide features
        gen_wide.run(
            debugqueries_json=config["data"]["bm25_input"],
            clicks_json=config["data"]["clicks_json"],
            output_file=config["features"]["output_wide"]
        )

        mlflow.log_artifact(config["features"]["output_wide"], artifact_path="wide_features")

        # Step 2: Feature engineering + pairwise deltas
        feat_eng.run(
            features_file=config["features"]["output_wide"],
            clicks_file=config["data"]["clicks_json"],
            bm25_fields=config["features"]["bm25_fields"],
            output_file=config["feature_engineering"]["output_pairwise"]
        )

        mlflow.log_artifact(
            config["feature_engineering"]["output_pairwise"],
            artifact_path="pairwise_features"
        )

        # Step 3: Ridge regression + normalized coefficients
        coef_df, metrics = ridge.run(
            input_file=config["feature_engineering"]["output_pairwise"],
            alpha=config["ridge"]["alpha"],
            output_file=config["ridge"]["output_weights"]
        )

        # Log ridge outputs
        mlflow.log_artifact(config["ridge"]["output_weights"], artifact_path="ridge_weights")

        # Log ridge metrics (R2, RMSE, feature count)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log normalized coefficients
        for _, row in coef_df.iterrows():
            mlflow.log_metric(f"coef_{row['field']}", row["normalized_weight"])

        # Runtime metric
        runtime = time.time() - start_time
        mlflow.log_metric("pipeline_runtime_sec", runtime)

if __name__ == "__main__":
    run_pipeline()