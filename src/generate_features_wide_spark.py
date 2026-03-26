from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, lit, regexp_extract,
    coalesce, current_timestamp,
    monotonically_increasing_id, when
)
import logging
import yaml
from pathlib import Path
import sys

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # ------------------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------------------
    CONFIG_FILE = Path("configs/config.yaml")

    with CONFIG_FILE.open() as f:
        config = yaml.safe_load(f)

    BM25_INPUT_FILE = config["data"]["bm25_input"]
    CLICKS_FILE = config["data"]["clicks_json"]

    BM25_FIELDS = [
        "first_name",
        "middle_name",
        "last_name",
        "institution",
        "city",
        "state",
        "topics"
    ]

    # ------------------------------------------------------------------------------
    # Spark Session
    # ------------------------------------------------------------------------------
    spark = SparkSession.builder \
        .appName("SolrFeatureEngineering") \
        .config("spark.driver.memory", "3g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()

    logger.info("Spark session created")

    # ------------------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------------------
    bm25_df = spark.read.option("multiline", "true").json(BM25_INPUT_FILE)
    clicks_df = spark.read.option("multiline", "true").json(CLICKS_FILE)

    logger.info("Data loaded")

    # ------------------------------------------------------------------------------
    # Extract correct fields
    # ------------------------------------------------------------------------------
    bm25_df = bm25_df.select(
        col("response.docs").alias("docs"),
        col("debug.explain.details").alias("explain")
    )

    # Optional debug
    # bm25_df = bm25_df.limit(100)

    bm25_df = bm25_df.repartition(8)

    # ------------------------------------------------------------------------------
    # Add query_id
    # ------------------------------------------------------------------------------
    bm25_df = bm25_df.withColumn(
        "query_id",
        monotonically_increasing_id().cast("string")
    )

    # ------------------------------------------------------------------------------
    # Explode docs
    # ------------------------------------------------------------------------------
    docs_df = bm25_df.select(
        col("query_id"),
        explode(col("docs")).alias("doc"),
        col("explain")
    )

    docs_df = docs_df.select(
        col("query_id"),
        col("doc.id").alias("doc_id"),
        col("doc.score").alias("total_doc_score"),
        col("explain")
    )

    # ------------------------------------------------------------------------------
    # Explode explain
    # ------------------------------------------------------------------------------
    exploded = docs_df.select(
        "query_id",
        "doc_id",
        "total_doc_score",
        explode(col("explain")).alias("exp")
    )

    # ------------------------------------------------------------------------------
    # Filter relevant entries
    # ------------------------------------------------------------------------------
    exploded = exploded.filter(
        col("exp.description").startswith("weight(")
    )

    # ------------------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------------------
    exploded = exploded.withColumn(
        "field",
        regexp_extract(col("exp.description"), r"weight\(([^:]+):", 1)
    ).withColumn(
        "value",
        col("exp.value").cast("double")
    )

    exploded = exploded.filter(col("field").isin(BM25_FIELDS))

    # ------------------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------------------
    from pyspark.sql.functions import sum as _sum

    bm25_features = exploded.groupBy(
        "query_id", "doc_id", "total_doc_score"
    ).agg(
        *[
            _sum(
                when(col("field") == f, col("value")).otherwise(0.0)
            ).alias(f"bm25_{f}")
            for f in BM25_FIELDS
        ]
    )

    # ------------------------------------------------------------------------------
    # Clicks
    # ------------------------------------------------------------------------------
    clicks_df = clicks_df \
        .withColumnRenamed("clicked_person_id", "doc_id") \
        .withColumn("clicked", lit(1))

    # ------------------------------------------------------------------------------
    # Join
    # ------------------------------------------------------------------------------
    final_df = bm25_features.join(
        clicks_df.select("query_id", "doc_id", "clicked"),
        on=["query_id", "doc_id"],
        how="left"
    )

    final_df = final_df.withColumn(
        "clicked",
        coalesce(col("clicked"), lit(0))
    )

    # ------------------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------------------
    final_df = final_df.withColumn(
        "event_timestamp",
        current_timestamp()
    )

    # ------------------------------------------------------------------------------
    # Write Output (FIXED PATH)
    # ------------------------------------------------------------------------------
    output_path = "/mnt/data/features.parquet"

    final_df.write.mode("overwrite").parquet(output_path)

    logger.info(f"✅ Features written to {output_path}")

    # ------------------------------------------------------------------------------
    # Stop Spark cleanly
    # ------------------------------------------------------------------------------
    spark.stop()

    logger.info("✅ Job completed successfully")

    sys.exit(0)

# ------------------------------------------------------------------------------
# ERROR HANDLING (CRITICAL)
# ------------------------------------------------------------------------------
except Exception as e:
    logger.error("❌ Job failed", exc_info=True)
    sys.exit(1)