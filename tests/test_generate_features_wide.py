import json
import pandas as pd
import pytest
from pathlib import Path

from src.generate_features_wide import (
    extract_field_from_description,
    traverse_details,
    parse_features,
    validate_schema,
    BM25_FIELDS
)


def test_extract_field_from_description():
    desc = "weight(first_name:john in 123) [BM25]"
    field = extract_field_from_description(desc)
    assert field == "first_name"


def test_traverse_details_extracts_features():
    details = [
        {
            "description": "weight(first_name:john)",
            "value": 1.5,
            "details": []
        },
        {
            "description": "weight(last_name:doe)",
            "value": 2.0,
            "details": []
        }
    ]

    features = {f"bm25_{f}": 0.0 for f in BM25_FIELDS}
    traverse_details(details, features)

    assert features["bm25_first_name"] == 1.5
    assert features["bm25_last_name"] == 2.0


def test_parse_features(tmp_path):
    test_json = {
        "response": {
            "docs": [
                {"id": "doc1", "score": 5.0}
            ]
        },
        "debug": {
            "explain": {
                "doc1": {
                    "value": 5.0,
                    "details": [
                        {
                            "description": "weight(first_name:john)",
                            "value": 1.2,
                            "details": []
                        }
                    ]
                }
            }
        }
    }

    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(test_json, f)

    results = parse_features(file_path)

    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    assert results[0]["total_doc_score"] == 5.0
    assert results[0]["bm25_first_name"] == 1.2


def test_validate_schema_pass():
    data = {
        "doc_id": ["1"],
        "total_doc_score": [1.0],
        "event_timestamp": [pd.Timestamp.utcnow()]
    }

    for f in BM25_FIELDS:
        data[f"bm25_{f}"] = [0.0]

    df = pd.DataFrame(data)

    # Should not raise
    validate_schema(df)


def test_validate_schema_fail():
    df = pd.DataFrame({"doc_id": ["1"]})

    with pytest.raises(ValueError):
        validate_schema(df)