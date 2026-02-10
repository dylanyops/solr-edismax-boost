import pandas as pd
import json
import re
from pathlib import Path
import yaml

FIELD_PATTERN = re.compile(r"weight\(([^:]+):")

def extract_field_scores(explain_node, field_scores):
    """Recursively walk Solr explain tree and extract BM25 scores per field"""
    if not isinstance(explain_node, dict):
        return

    desc = explain_node.get("description", "")
    value = explain_node.get("value", 0.0)

    match = FIELD_PATTERN.search(desc)
    if match:
        field = match.group(1)
        field_scores[field] = field_scores.get(field, 0.0) + value

    for child in explain_node.get("details", []):
        extract_field_scores(child, field_scores)

def run(debugqueries_json, clicks_json, output_file):
    # -----------------------------
    # 1️⃣ Generate BM25 rows from debugQueries
    # -----------------------------
    input_path = Path(debugqueries_json)
    with input_path.open() as f:
        data = json.load(f)

    rows = []
    for query_idx, solr_resp in enumerate(data):
        docs = solr_resp.get("response", {}).get("docs", [])
        explain = solr_resp.get("debug", {}).get("explain", {})

        for rank, doc in enumerate(docs, start=1):
            doc_id = str(doc.get("id"))
            total_score = doc.get("score", 0.0)
            field_scores = {}
            explain_tree = explain.get(doc_id)
            if explain_tree:
                extract_field_scores(explain_tree, field_scores)
            for field, bm25_score in field_scores.items():
                rows.append({
                    "query_id": query_idx,
                    "doc_id": doc_id,
                    "rank": rank,
                    "field": field,
                    "bm25_score": round(bm25_score, 6),
                    "total_doc_score": round(total_score, 6)
                })

    bm25 = pd.DataFrame(rows)

    # -----------------------------
    # 2️⃣ Pivot BM25 into wide/tabulated features
    # -----------------------------
    features = bm25.pivot_table(
        index=["query_id", "doc_id", "rank"],
        columns="field",
        values="bm25_score",
        fill_value=0.0
    ).reset_index()

    features.columns = [
        f"bm25_{c}" if c not in ("query_id", "doc_id", "rank") else c
        for c in features.columns
    ]

    # -----------------------------
    # 3️⃣ Merge with click logs
    # -----------------------------
    with open(clicks_json, "r") as f:
        click_logs = pd.DataFrame(json.load(f))
    
    # Ensure query_id types match
    features["query_id"] = features["query_id"].astype(str)
    features["doc_id"] = features["doc_id"].astype(str)

    click_logs["query_id"] = click_logs["query_id"].astype(str)
    if "clicked_person_id" in click_logs.columns:
        click_logs["clicked_person_id"] = click_logs["clicked_person_id"].astype(str)
    df = features.merge(click_logs, on="query_id", how="left")
    df["clicked"] = (df["doc_id"] == df["clicked_person_id"]).astype(int)

    # -----------------------------
    # 4️⃣ Merge total_doc_score if present
    # -----------------------------
    if "total_doc_score" in bm25.columns:
        totals = bm25[["query_id", "doc_id", "total_doc_score"]].drop_duplicates()

        # Normalize merge key types
        totals["query_id"] = totals["query_id"].astype(str)
        totals["doc_id"] = totals["doc_id"].astype(str)

        df = df.merge(totals, on=["query_id", "doc_id"], how="left")

    # -----------------------------
    # 5️⃣ Clean up
    # -----------------------------
    df.drop(columns=["clicked_rank", "clicked_person_id"], inplace=True, errors="ignore")

    # -----------------------------
    # 6️⃣ Save wide features
    # -----------------------------
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Wrote {len(df)} rows → {output_file}")

# -----------------------------
# Optional: run from config
# -----------------------------
def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    run(
        debugqueries_json=config["data"]["bm25_input"],  # now points to JSON
        clicks_json=config["data"]["clicks_json"],
        output_file=config["features"]["output_wide"]
    )