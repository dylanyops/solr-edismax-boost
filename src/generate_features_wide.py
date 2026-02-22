import pandas as pd
import datetime
import shutil
import os
from pathlib import Path
from feast import FeatureStore
from feast.data_source import PushMode

def bootstrap_feast_repo(repo_path):
    repo_dir = Path(repo_path)
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "data").mkdir(exist_ok=True)
    parquet_path = repo_dir / "data" / "documents.parquet"
    
    if parquet_path.exists():
        parquet_path.unlink()
    
    all_columns = [
        "doc_id", "event_timestamp", "total_doc_score",
        "bm25_city", "bm25_first_name", "bm25_institution", 
        "bm25_last_name", "bm25_middle_name", "bm25_state", "bm25_topics"
    ]
    
    dummy_df = pd.DataFrame({col: [0.0] for col in all_columns})
    dummy_df["doc_id"] = ["initial_stub"]
    dummy_df["event_timestamp"] = pd.to_datetime([datetime.datetime.now(datetime.timezone.utc)])
    
    dummy_df.to_parquet(parquet_path, engine='pyarrow', index=False)
    print(f"✅ Bootstrap complete: Fresh stub created at {parquet_path}")

def run(repo_path='/mnt/data/feature_repo', solr_url=None, **kwargs):
    # Accept **kwargs so Argo won't crash if it sends extra params like 'debugqueries_json'
    bootstrap_feast_repo(repo_path)
    
    # --- YOUR SOLR FETCH LOGIC HERE ---
    # df = fetch_from_solr(solr_url) 
    # For this example, I assume 'df' is prepared and has the 10 columns above.
    
    store = FeatureStore(repo_path=repo_path)
    
    if 'df' in locals() or 'df' in globals():
        print(f"🚀 Pushing {len(df)} docs to OFFLINE store...")
        # FIX: Positional argument for source name, PushMode.OFFLINE for Parquet update
        store.push("doc_push_source", df, to=PushMode.OFFLINE)
    else:
        print("⚠️ No dataframe 'df' found to push. Check Solr logic.")

    print(f"✅ Ingestion step finished.")