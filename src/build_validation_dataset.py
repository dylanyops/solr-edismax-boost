import json
import pandas as pd
from pathlib import Path
import yaml
import re

import logging

# -------------------------------
# Load config
# -------------------------------

CONFIG_FILE = Path("configs/config.yaml")

with CONFIG_FILE.open() as f:
    config = yaml.safe_load(f)

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SOLR_LOGS = Path(config["data"]["validation_solr_logs"])
TRAINING_LOGS = Path(config["data"]["bm25_input"])
CLICK_LOGS = Path(config["data"]["validation_clicks"])
TRAINING_CLICK_LOGS = Path(config["data"]["clicks_json"])
OUTPUT_DATASET = Path(config["data"]["parquet_file"])
OUTPUT_DATASET_TRAINING = Path(config["data"]["training_parquet_file"])

FIELDS = [
    "first_name",
    "middle_name",
    "last_name",
    "institution",
    "city",
    "state",
    "topics"
]

# -------------------------------
# Extract BM25 scores
# -------------------------------

def extract_bm25(details):
    scores = {f"bm25_{f}": 0.0 for f in FIELDS}
    for d in details:
        desc = d.get("description", "")
        for f in FIELDS:
            if f"weight({f}:" in desc:
                scores[f"bm25_{f}"] = d.get("value", 0.0)
    return scores

# -------------------------------
# Parse Solr logs
# -------------------------------

def parse_solr_logs(solr_logs):
    rows = []
    for query_id, entry in enumerate(solr_logs):
        docs = entry.get("response", {}).get("docs", [])
        explain = entry.get("debug", {}).get("explain", {})

        for doc in docs:
            doc_id = doc["id"]
            score = explain.get(doc_id, {}).get("value", 0.0)
            details = explain.get(doc_id, {}).get("details", [])
            bm25 = extract_bm25(details)

            row = {
                "query_id": str(query_id),
                "person_id": doc_id,
                "prediction_score": score
            }
            row.update(bm25)
            rows.append(row)

    return pd.DataFrame(rows)

# -------------------------------
# Build dataset
# -------------------------------

def build_dataset(evaluate):
    # Load Solr logs
    with open(SOLR_LOGS if evaluate else TRAINING_LOGS) as f:
        solr_logs = json.load(f)

    # Load click logs
    with open(CLICK_LOGS if evaluate else TRAINING_CLICK_LOGS) as f:
        click_logs = json.load(f)

    df_solr = parse_solr_logs(solr_logs)
    df_click = pd.DataFrame(click_logs)

    # Ensure clicked_person_id exists even if some queries have no clicks
    if "clicked_person_id" not in df_click.columns:
        df_click["clicked_person_id"] = None

    # Merge
    df = df_solr.merge(df_click, on="query_id", how="left")

    # Fill missing clicks with placeholder
    df["clicked_person_id"] = df["clicked_person_id"].fillna("none")

    # Create label
    df["label"] = (df["person_id"] == df["clicked_person_id"]).astype(int)

    # Optional: drop columns no longer needed
    for col in ["clicked_person_id", "clicked_rank"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

# -------------------------------
# Main
# -------------------------------

def main():
    df_evaluation = build_dataset(True)
    df_training = build_dataset(False)

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)

    # Save Parquet
    df_evaluation.to_parquet(OUTPUT_DATASET)
    logger.info(f"Validation dataset parquet written to {OUTPUT_DATASET}")
    df_training.to_parquet(OUTPUT_DATASET_TRAINING)
    logger.info(f"Validation dataset parquet written to {OUTPUT_DATASET_TRAINING}")

if __name__ == "__main__":
    main()