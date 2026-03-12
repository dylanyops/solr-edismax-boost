import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import mlflow
import yaml

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Base directory and data paths
# ------------------------------------------------------------------------------
DATA_DIR = Path("artifacts")
DATA_DIR.mkdir(exist_ok=True)

CONFIG_FILE = Path("configs/config.yaml")
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")

with CONFIG_FILE.open() as f:
    config = yaml.safe_load(f)

BM25_INPUT_FILE = Path(config["data"]["bm25_input"])
CLICKS_FILE = Path(config["data"]["clicks_json"])

BM25_FIELDS = [
    "first_name",
    "middle_name",
    "last_name",
    "institution",
    "city",
    "state",
    "topics"
]

# Required schema columns
REQUIRED_COLUMNS = {
    "query_id",
    "doc_id",
    "total_doc_score",
    *{f"bm25_{f}" for f in BM25_FIELDS},
    "clicked",
    "event_timestamp"
}

FIELD_REGEX = re.compile(r"weight\(([^:]+):")

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def extract_field_from_description(description: str) -> str | None:
    match = FIELD_REGEX.search(description)
    return match.group(1) if match else None


def traverse_details(details: List[Dict[str, Any]], features: Dict[str, float]):
    """
    Recursively traverse Solr debug details to extract BM25 contributions
    """
    for detail in details:
        description = detail.get("description", "")
        field_name = extract_field_from_description(description)

        if field_name in BM25_FIELDS:
            features[f"bm25_{field_name}"] = float(detail.get("value", 0.0))

        if "details" in detail:
            traverse_details(detail["details"], features)


def parse_features(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        logger.error(f"BM25 input file not found: {file_path}")
        return []

    with file_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        logger.error("Unexpected JSON structure")
        return []

    results = []

    for entry in entries:

        query_id = entry.get("query_id")

        debug_explain = entry.get("debug", {}).get("explain", {})
        docs = entry.get("response", {}).get("docs", [])

        logger.info(f"Processing {len(docs)} documents for query {query_id}")

        for rank, doc in enumerate(docs, start=1):

            doc_id = doc.get("id")
            if not doc_id:
                continue

            features = {
                "query_id": query_id,
                "doc_id": doc_id,
                "original_rank": rank
            }

            explain = debug_explain.get(doc_id, {})
            features["total_doc_score"] = float(
                explain.get("value", doc.get("score", 0.0))
            )

            for f in BM25_FIELDS:
                features[f"bm25_{f}"] = 0.0

            if explain:
                traverse_details(explain.get("details", []), features)

            results.append(features)

    return results


def load_clicks(click_file: Path) -> pd.DataFrame:

    if not click_file.exists():
        logger.error(f"Clicks file not found: {click_file}")
        return pd.DataFrame()

    clicks_df = pd.read_json(click_file)

    clicks_df = clicks_df.rename(columns={
        "clicked_person_id": "doc_id"
    })

    clicks_df["clicked"] = 1

    return clicks_df[["query_id", "doc_id", "clicked"]]


def validate_schema(df: pd.DataFrame):

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Schema validation passed.")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():

    logger.info(f"Starting Solr feature generation job using {BM25_INPUT_FILE}")

    mlflow.set_experiment("solr_feature_generation")

    with mlflow.start_run():

        all_features = parse_features(BM25_INPUT_FILE)

        if not all_features:
            logger.error("No features generated. Check input JSON structure!")
            return

        df = pd.DataFrame(all_features)

        # ----------------------------------------------------------
        # Load and merge click labels
        # ----------------------------------------------------------

        clicks_df = load_clicks(CLICKS_FILE)

        if not clicks_df.empty:

            df = df.merge(
                clicks_df,
                on=["query_id", "doc_id"],
                how="left"
            )

            df["clicked"] = df["clicked"].fillna(0)

        else:
            logger.warning("Clicks dataset empty — labels default to 0")
            df["clicked"] = 0

        df["event_timestamp"] = pd.Timestamp.now(tz="UTC")

        validate_schema(df)

        # ----------------------------------------------------------
        # Save outputs
        # ----------------------------------------------------------

        csv_file = DATA_DIR / "bm25_features_wide.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV features to {csv_file}")

        mlflow_parquet = DATA_DIR / "bm25_features_wide.parquet"
        df.to_parquet(mlflow_parquet, index=False)
        logger.info(f"Saved Parquet features to {mlflow_parquet}")

        feast_parquet = Path("feature_repo/data/documents.parquet")
        feast_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feast_parquet, index=False)
        logger.info(f"Saved Parquet features to {feast_parquet}")

        mlflow.log_param("num_documents", len(df))
        mlflow.log_param("bm25_fields", ",".join(BM25_FIELDS))

        mlflow.log_artifact(str(csv_file))
        mlflow.log_artifact(str(mlflow_parquet))

        logger.info(f"✅ Total rows processed: {len(df)}")
        logger.info("Feature generation job completed successfully.")


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()