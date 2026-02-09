from src import pipeline
import os

def test_pipeline_runs(tmp_path):
    # Copy or generate small data inside tmp_path
    # For simplicity, reuse synthetic data setup here
    # Ensure pipeline runs end-to-end without exceptions
    pipeline.run(config_path="configs/config.yaml")