import pandas as pd
import os
from src import feature_engineering as feat_eng

def test_feature_engineering(tmp_path):
    # Synthetic wide features
    features_file = tmp_path / "wide.csv"
    df = pd.DataFrame({
        "query_id": ["1", "1"],
        "doc_id": ["a","b"],
        "bm25_first_name": [0.5,0.3],
        "bm25_last_name": [0.7,0.2],
        "clicked": [1,0]
    })
    df.to_csv(features_file, index=False)

    # Clicks JSON
    clicks_file = tmp_path / "clicks.json"
    import json
    clicks = [{"query_id": "1", "clicked_person_id": "a"}]
    with open(clicks_file, "w") as f:
        json.dump(clicks, f)

    output_file = tmp_path / "pairwise.csv"
    bm25_fields = ["bm25_first_name", "bm25_last_name"]

    feat_eng.run(str(features_file), str(clicks_file), bm25_fields, str(output_file))

    df_out = pd.read_csv(output_file)
    assert "delta_bm25_first_name" in df_out.columns
    assert "delta_bm25_last_name" in df_out.columns
    assert df_out["label"].iloc[0] == 1
