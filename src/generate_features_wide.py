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
DATA_DIR = Path("/mnt/data")

# Paths to Solr debug query logs
SOLR_LOG_FILES = [
    DATA_DIR / "person_search_debugQueries.json",
]

# Fields for which we want BM25 contributions
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
    """
    Extract the field name from Solr explain description.
    Example:
        'weight(first_name:John in doc) ...' -> 'first_name'
    """
    match = FIELD_REGEX.search(description)
    return match.group(1) if match else None


def parse_and_compute(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a Solr debug log file and compute BM25 features for each document.
    
    Args:
        file_path (Path): Path to Solr debug JSON log.
    
    Returns:
        List[Dict[str, Any]]: List of per-document feature dictionaries.
    """
    results = []

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return results

    try:
        with file_path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.exception(f"Invalid JSON in file: {file_path}")
        return results
    except Exception:
        logger.exception(f"Unexpected error reading file: {file_path}")
        return results

    # Determine documents
    docs = data if isinstance(data, list) else data.get("response", {}).get("docs", [])
    logger.info(f"Processing {len(docs)} documents from {file_path.name}")

    for doc in docs:
        try:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            # Initialize feature dictionary
            features = {
                "doc_id": doc_id,
                "total_doc_score": 0.0
            }

            # Initialize BM25 fields to 0.0
            for field in BM25_FIELDS:
                features[f"bm25_{field}"] = 0.0

            # Extract debug explain block for per-field BM25
            explain = doc.get("debug", {}).get("explain", {}).get(doc_id, {})
            features["total_doc_score"] = float(explain.get("value", 0.0))

            for detail in explain.get("details", []):
                description = detail.get("description", "")
                field_name = extract_field_from_description(description)
                if field_name in BM25_FIELDS:
                    features[f"bm25_{field_name}"] = float(detail.get("value", 0.0))

            results.append(features)

        except Exception:
            logger.exception(f"Failed processing document {doc.get('id')}")
            continue

    return results


def validate_schema(df: pd.DataFrame):
    """
    Basic schema validation to ensure required columns exist.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Raises:
        ValueError: If required columns are missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.")


# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main():
    """
    Main entry point for the feature generation pipeline.
    Processes Solr debug logs, extracts BM25 features, saves offline artifacts,
    and logs metadata to MLflow.
    """
    logger.info("Starting Solr feature generation job")
    mlflow.set_experiment("solr_feature_generation")

    with mlflow.start_run():

        all_features = []

        # Parse each Solr log file
        for file_path in SOLR_LOG_FILES:
            file_results = parse_and_compute(file_path)
            all_features.extend(file_results)

        if not all_features:
            logger.warning("No features generated. Exiting.")
            return

        # Convert all features to a DataFrame (efficient bulk creation)
        df = pd.DataFrame.from_records(all_features)

        # Add timestamp required by Feast
        df["event_timestamp"] = pd.Timestamp.now(tz="UTC")

        # Validate schema before writing
        validate_schema(df)

        # ---------------- Save Artifacts ----------------
        # Save JSON for debugging purposes
        json_file = DATA_DIR / "features.json"
        df.to_json(json_file, orient="records", indent=2)
        logger.info(f"Saved JSON features to {json_file}")

        # Save Parquet for Feast offline store
        parquet_file = DATA_DIR / "feature_repo" / "data" / "documents.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved Parquet features to {parquet_file}")

        # ---------------- MLflow Logging ----------------
        mlflow.log_param("num_input_files", len(SOLR_LOG_FILES))
        mlflow.log_param("bm25_fields", ",".join(BM25_FIELDS))
        mlflow.log_metric("num_documents_processed", len(df))
        mlflow.log_metric("avg_total_doc_score", df["total_doc_score"].mean())

        mlflow.log_artifact(str(json_file))
        mlflow.log_artifact(str(parquet_file))

        logger.info(f"✅ Total docs processed: {len(df)}")
        logger.info("Feature generation job completed successfully.")


# ------------------------------------------------------------------------------
# Entry Point Guard
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()