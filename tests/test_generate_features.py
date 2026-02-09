import pandas as pd
import os
from src import generate_features_wide as gen_wide

def test_generate_features(tmp_path):
    # Prepare small BM25 CSV
    bm25_csv = tmp_path / "bm25.csv"
    df_bm25 = pd.DataFrame({
        "query_id": ["1", "1"],
        "doc_id": ["a", "b"],
        "rank": [1,2],
        "field": ["first_name", "last_name"],
        "bm25_score": [0.5, 0.8]
    })
    df_bm25.to_csv(bm25_csv, index=False)

    # Prepare clicks JSON
    clicks_json = tmp_path / "clicks.json"
    clicks = [{"query_id": "1", "clicked_person_id": "a"}]
    import json
    with open(clicks_json, "w") as f:
        json.dump(clicks, f)

    output_file = tmp_path / "wide.csv"

    gen_wide.run(str(bm25_csv), str(clicks_json), str(output_file))

    # Check output exists
    assert os.path.exists(output_file)

    df_out = pd.read_csv(output_file)
    assert "clicked" in df_out.columns
    assert len(df_out) == 2