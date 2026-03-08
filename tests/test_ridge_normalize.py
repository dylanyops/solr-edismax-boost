# tests/test_ridge_normalize.py

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from src import ridge_normalize as ridge

# -------------------------------
# Helper function to create mock clicks JSON
# -------------------------------
def create_mock_clicks(file_path: Path, num_rows: int = 3):
    data = [{"clicked_person_id": f"person_{i}"} for i in range(num_rows)]
    file_path.write_text(json.dumps(data))
    return file_path

# -------------------------------
# Unit Test: run with placeholder y
# -------------------------------
def test_ridge_run_placeholder_y(tmp_path):
    """Test Ridge run when 'clicked' column is missing."""
    clicks_file = create_mock_clicks(tmp_path / "clicks.json")
    output_file = tmp_path / "weights.csv"
    
    # Patch FeatureStore to return only bm25_fields without 'clicked'
    mock_fs = MagicMock()
    mock_fs.get_historical_features.return_value.to_df.return_value = pd.DataFrame({
        "bm25_city": [0.1, 0.2, 0.3],
        "bm25_first_name": [0.0, 0.1, 0.2],
        "bm25_institution": [0.3, 0.1, 0.0],
        "bm25_last_name": [0.2, 0.2, 0.1],
        "bm25_topics": [0.0, 0.0, 0.1],
        "total_doc_score": [1, 2, 3],
        "doc_id": ["person_0", "person_1", "person_2"]
    })

    with patch("src.ridge_normalize.FeatureStore", return_value=mock_fs):
        with patch("mlflow.start_run") as mock_mlflow_run:
            ridge.run(
                repo_path=tmp_path,
                clicks_json=clicks_file,
                output_file=output_file
            )
    
    # Check output CSV exists
    assert output_file.exists()
    df_weights = pd.read_csv(output_file)
    assert set(df_weights["field"]) == {"bm25_city", "bm25_first_name", "bm25_institution", "bm25_last_name", "bm25_topics"}
    assert len(df_weights) == 5

    # Check MLflow was called
    assert mock_mlflow_run.called

# -------------------------------
# Unit Test: run with 'clicked' present
# -------------------------------
def test_ridge_run_with_clicked(tmp_path):
    clicks_file = create_mock_clicks(tmp_path / "clicks.json")
    output_file = tmp_path / "weights.csv"

    # Mock FeatureStore to return 'clicked' column
    mock_fs = MagicMock()
    mock_fs.get_historical_features.return_value.to_df.return_value = pd.DataFrame({
        "bm25_city": [0.1, 0.2],
        "bm25_first_name": [0.0, 0.1],
        "bm25_institution": [0.3, 0.1],
        "bm25_last_name": [0.2, 0.2],
        "bm25_topics": [0.0, 0.0],
        "total_doc_score": [1, 2],
        "clicked": [1, 0],
        "doc_id": ["person_0", "person_1"]
    })

    with patch("src.ridge_normalize.FeatureStore", return_value=mock_fs):
        with patch("mlflow.start_run") as mock_mlflow_run:
            ridge.run(
                repo_path=tmp_path,
                clicks_json=clicks_file,
                output_file=output_file
            )

    df_weights = pd.read_csv(output_file)
    assert not df_weights.empty
    assert "weight" in df_weights.columns
    assert mock_mlflow_run.called

# -------------------------------
# Unit Test: missing clicks file
# -------------------------------
def test_ridge_missing_clicks_file(tmp_path):
    missing_file = tmp_path / "missing_clicks.json"
    output_file = tmp_path / "weights.csv"
    with pytest.raises(FileNotFoundError):
        ridge.run(clicks_json=missing_file, output_file=output_file)

# -------------------------------
# Unit Test: empty clicks file
# -------------------------------
def test_ridge_empty_clicks(tmp_path):
    empty_file = tmp_path / "empty_clicks.json"
    empty_file.write_text("[]")
    output_file = tmp_path / "weights.csv"
    with pytest.raises(ValueError):
        ridge.run(clicks_json=empty_file, output_file=output_file)