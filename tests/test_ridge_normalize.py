import pandas as pd
import os
from src import ridge_normalize as ridge

def test_ridge_normalize(tmp_path):
    input_file = tmp_path / "pairwise.csv"
    df = pd.DataFrame({
        "delta_bm25_first_name": [0.2, -0.1],
        "delta_bm25_last_name": [0.3, -0.2],
        "label": [1, 1]
    })
    df.to_csv(input_file, index=False)

    output_file = tmp_path / "weights.csv"
    ridge.run(str(input_file), alpha=1.0, output_file=str(output_file))

    assert os.path.exists(output_file)
    df_out = pd.read_csv(output_file)
    assert "normalized_weight" in df_out.columns
    assert abs(df_out["normalized_weight"].sum() - 1) < 1e-6