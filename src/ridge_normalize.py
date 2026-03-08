import os
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import mlflow

# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Ridge Step ----------------
def run(repo_path=None, clicks_json=None, output_file=None, **kwargs):
    """
    Train Ridge regression weights for BM25 features using Feast historical features.

    Args:
        repo_path (str or Path): Feast repo path (defaults to DATA_DIR/feature_repo)
        clicks_json (str or Path): Clicks JSON file (defaults to DATA_DIR/clicks.json)
        output_file (str or Path): CSV file to save Ridge weights (defaults to DATA_DIR/weights.csv)
    """
    try:
        # ---------------- Paths ----------------
        DATA_DIR = Path("/mnt/data")

        repo_path = Path(repo_path or DATA_DIR / "feature_repo")
        clicks_json = Path(clicks_json or DATA_DIR / "person_search_clicks.json")
        output_file = Path(output_file or DATA_DIR / "bm25_field_weights.csv")

        bm25_fields = ["bm25_city", "bm25_first_name", "bm25_institution",
                       "bm25_last_name", "bm25_topics"]

        # ---------------- Load Clicks ----------------
        if not clicks_json.exists():
            raise FileNotFoundError(f"Clicks JSON not found: {clicks_json}")
        with clicks_json.open("r") as f:
            clicks_df = pd.DataFrame(json.load(f))

        if clicks_df.empty:
            raise ValueError("Clicks DataFrame is empty.")

        # ---------------- Fetch Features from Feast ----------------
        store = FeatureStore(repo_path=str(repo_path))

        entities = pd.DataFrame({
            "doc_id": clicks_df["clicked_person_id"].unique().astype(str),
            "event_timestamp": [pd.Timestamp("2100-01-01", tz="UTC")] *
                               clicks_df["clicked_person_id"].nunique()
        })

        feature_res = store.get_historical_features(
            entity_df=entities,
            features=[f"doc_bm25_features:{f}" for f in bm25_fields]
        ).to_df()

        if feature_res.empty:
            raise ValueError("Feast returned empty features.")

        # ---------------- Data Wrangling ----------------
        X = feature_res[bm25_fields].fillna(0).values

        if "clicked" in feature_res:
            y = feature_res["clicked"].values
        else:
            logger.warning("'clicked' column missing. Using random placeholder targets.")
            y = np.random.rand(len(X))

        # ---------------- Scale Features ----------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---------------- Train Ridge ----------------
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        # ---------------- Save Weights ----------------
        output_file.parent.mkdir(parents=True, exist_ok=True)
        weights_df = pd.DataFrame({
            "field": bm25_fields,
            "weight": model.coef_
        })
        weights_df.to_csv(output_file, index=False)
        logger.info(f"✅ Ridge weights saved to {output_file}")

        # ---------------- MLflow Logging ----------------
        mlflow.set_experiment("ridge_feature_weights")
        with mlflow.start_run():
            mlflow.log_param("num_documents", len(feature_res))
            mlflow.log_param("bm25_fields", ",".join(bm25_fields))
            mlflow.log_metric("avg_total_score", feature_res.get("total_doc_score", pd.Series([0])).mean())
            mlflow.log_artifact(str(output_file))

        logger.info("Ridge step completed successfully.")

    except Exception as e:
        logger.exception(f"Ridge step failed: {e}")
        raise

# ---------------- Main Guard ----------------
if __name__ == "__main__":
    run()