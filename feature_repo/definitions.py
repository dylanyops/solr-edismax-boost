import os
from pathlib import Path
from feast import Entity, FeatureView, Field, PushSource, FileSource
from feast.types import Float32, String

# --- DYNAMIC PATH LOGIC ---
# If the /mnt/data directory exists, we assume we are running in the cluster.
# Otherwise, we use the local path on your Mac.
if os.path.exists("/mnt/data"):
    PARQUET_PATH = "/mnt/data/feature_repo/data/documents.parquet"
else:
    # Gets the directory of this definitions.py file and looks for data/documents.parquet
    PARQUET_PATH = str(Path(__file__).parent / "data" / "documents.parquet")

# 1. The primary key (Entity)
document = Entity(
    name="document", 
    join_keys=["doc_id"], 
    description="The document being ranked"
)

# 2. Define the Batch Source
# Note: event_timestamp_column is the standard parameter for FileSource.
batch_source = FileSource(
    name="doc_batch_source",
    path=PARQUET_PATH,
    event_timestamp_column="event_timestamp",
)

# 3. Define the Push Source 
# This allows us to "push" data into the offline store (Parquet) and online store
doc_push_source = PushSource(
    name="doc_push_source",
    batch_source=batch_source
)

# 4. Define the Feature View
doc_bm25_view = FeatureView(
    name="doc_bm25_features",
    entities=[document],
    ttl=None, # Keep features indefinitely for this use case
    schema=[
        Field(name="bm25_city", dtype=Float32),
        Field(name="bm25_first_name", dtype=Float32),
        Field(name="bm25_institution", dtype=Float32),
        Field(name="bm25_last_name", dtype=Float32),
        Field(name="bm25_middle_name", dtype=Float32),
        Field(name="bm25_state", dtype=Float32),
        Field(name="bm25_topics", dtype=Float32),
        Field(name="total_doc_score", dtype=Float32),
    ],
    online=True,
    source=doc_push_source,
)