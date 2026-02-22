from feast import Entity, FeatureView, Field, PushSource, FileSource # Added FileSource
from feast.types import Float32, String

# The primary key
document = Entity(name="document", join_keys=["doc_id"])

# 1. Define a Batch Source for the PushSource to write to
# In your Argo environment, this path must be on your Persistent Volume
batch_source = FileSource(
    name="doc_batch_source",
    path="/mnt/data/feature_repo/data/documents.parquet",
    event_timestamp_column="event_timestamp",
    timestamp_field="event_timestamp",  # <--- ADD THIS EXPLICIT MAPPING
)

# 2. Update the Push Source
doc_push_source = PushSource(
    name="doc_push_source",
    batch_source=batch_source # Link the FileSource here
)

# 3. Define the View (Updated source to doc_push_source)
doc_bm25_view = FeatureView(
    name="doc_bm25_features",
    entities=[document],
    ttl=None,
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