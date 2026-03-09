import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml
import mlflow

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Load config ----------------
CONFIG_FILE = Path("configs/config.yaml")
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")

with CONFIG_FILE.open() as f:
    config = yaml.safe_load(f)

BM25_FIELDS = config["features"]["bm25_fields"]

OUTPUT_FILE = Path(__file__).parent / config["ridge"]["output_weights"]
PARQUET_FILE = (Path(__file__).parent / "artifacts" / "bm25_features_wide.parquet").resolve()

logger.info(f"Looking for Parquet file at: {PARQUET_FILE}")

# ---------------- Main ----------------
def main():
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"Feature Parquet not found: {PARQUET_FILE}")

    # ---------------- Load data ----------------
    df = pd.read_parquet(PARQUET_FILE)
    logger.info(f"Loaded {len(df)} rows from {PARQUET_FILE}")

    X = df[BM25_FIELDS].fillna(0).values

    # Placeholder clicks — replace with real signals
    y = np.random.rand(len(X))

    # ---------------- Feature scaling ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- Train Ridge ----------------
    model = Ridge(alpha=config["ridge"]["alpha"])
    model.fit(X_scaled, y)

    raw_coefs = model.coef_

    # ---------------- Convert to ranking weights ----------------
    # Step 1: remove negative coefficients
    positive_coefs = np.maximum(raw_coefs, 0)

    # Step 2: normalize to sum = 1
    if positive_coefs.sum() > 0:
        normalized_weights = positive_coefs / positive_coefs.sum()
    else:
        logger.warning("All coefficients were negative — falling back to uniform weights")
        normalized_weights = np.ones_like(positive_coefs) / len(positive_coefs)

    # ---------------- Save weights ----------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    weights_df = pd.DataFrame({
        "field": BM25_FIELDS,
        "ridge_coef_raw": raw_coefs,
        "ridge_coef_positive": positive_coefs,
        "normalized_weight": normalized_weights
    })

    weights_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("✅ Ridge weights saved")
    logger.info("\n" + weights_df.to_string(index=False))

    # ---------------- MLflow logging ----------------
    mlflow.set_experiment("ridge_feature_weights")

    with mlflow.start_run():
        mlflow.log_param("num_documents", len(df))
        mlflow.log_param("bm25_fields", ",".join(BM25_FIELDS))
        mlflow.log_param("alpha", config["ridge"]["alpha"])

        mlflow.log_metric("avg_total_score", df.get("total_doc_score", pd.Series([0])).mean())

        mlflow.log_artifact(str(OUTPUT_FILE))

        logger.info("✅ Ridge step logged to MLflow")


if __name__ == "__main__":
    main()