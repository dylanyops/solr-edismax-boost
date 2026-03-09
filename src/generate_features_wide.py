import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import mlflow

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

INPUT_FILE = Path("/Users/shared/data/person_search_debugQueries.json")

BM25_FIELDS = [
    "first_name",
    "middle_name",
    "last_name",
    "institution",
    "city",
    "state",
    "topics"
]

# Required schema columns for validation
REQUIRED_COLUMNS = {
    "doc_id",
    "total_doc_score",
    *{f"bm25_{f}" for f in BM25_FIELDS},
    "event_timestamp"
}

# Precompiled regex for field extraction from Solr explain description
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

    # Ensure we always have a list of entries
    if isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        logger.error("Unexpected JSON structure")
        return []

    results = []
    for entry in entries:
        debug_explain = entry.get("debug", {}).get("explain", {})
        docs = entry.get("response", {}).get("docs", [])

        logger.info(f"Processing {len(docs)} documents")

        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            features = {"doc_id": doc_id}

            explain = debug_explain.get(doc_id, {})
            features["total_doc_score"] = float(explain.get("value", doc.get("score", 0.0)))

            # Initialize BM25 fields
            for f in BM25_FIELDS:
                features[f"bm25_{f}"] = 0.0

            # Fill BM25 fields from explain details
            if explain:
                traverse_details(explain.get("details", []), features)

            results.append(features)

    return results

def validate_schema(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.")

# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main():
    logger.info(f"Starting Solr feature generation job using {INPUT_FILE}")
    mlflow.set_experiment("solr_feature_generation")

    with mlflow.start_run():
        all_features = parse_features(INPUT_FILE)

        if not all_features:
            logger.error("No features generated. Check input JSON structure!")
            return

        df = pd.DataFrame(all_features)
        df["event_timestamp"] = pd.Timestamp.now(tz="UTC")

        validate_schema(df)

        # Save CSV and Parquet locally
        csv_file = DATA_DIR / "bm25_features_wide.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV features to {csv_file}")

        mlflow_parquet = Path("artifacts/bm25_features_wide.parquet")
        mlflow_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(mlflow_parquet, index=False)
        logger.info(f"Saved Parquet features to artifacts/bm25_features_wide.parquet")

        feast_parquet = Path("feature_repo/data/documents.parquet")
        feast_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feast_parquet, index=False)
        logger.info(f"Saved Parquet features to feature_repo/data/documents.parquet")

        # MLflow logging
        mlflow.log_param("num_documents", len(df))
        mlflow.log_param("bm25_fields", ",".join(BM25_FIELDS))
        mlflow.log_artifact(str(csv_file))
        mlflow.log_artifact("artifacts/bm25_features_wide.parquet")

        logger.info(f"✅ Total docs processed: {len(df)}")
        logger.info("Feature generation job completed successfully.")

# ------------------------------------------------------------------------------
# Entry Point Guard
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()