import yaml
import pandas as pd
from src import generate_features_wide as gen_wide
from src import feature_engineering as feat_eng
from src import ridge_normalize as ridge

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(config_path="configs/config.yaml"):
    config = load_config(config_path)

    # Step 1: Generate wide features
    gen_wide.run(
        bm25_input=config["data"]["bm25_input"],
        clicks_json=config["data"]["clicks_json"],
        output_file=config["features"]["output_wide"]
    )

    # Step 2: Feature engineering + pairwise deltas
    feat_eng.run(
        features_file=config["features"]["output_wide"],
        clicks_file=config["data"]["clicks_json"],
        bm25_fields=config["features"]["bm25_fields"],
        output_file=config["feature_engineering"]["output_pairwise"]
    )

    # Step 3: Ridge regression + normalized coefficients
    ridge.run(
        input_file=config["feature_engineering"]["output_pairwise"],
        alpha=config["ridge"]["alpha"],
        output_file=config["ridge"]["output_weights"]
    )

if __name__ == "__main__":
    run_pipeline()