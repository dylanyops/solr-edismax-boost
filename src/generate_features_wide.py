import pandas as pd
import json

def run(bm25_input, clicks_json, output_file):
    bm25 = pd.read_csv(bm25_input)
    bm25["query_id"] = bm25["query_id"].astype(str)

    with open(clicks_json, "r") as f:
        click_logs = pd.DataFrame(json.load(f))
    click_logs["query_id"] = click_logs["query_id"].astype(str)

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

    df = features.merge(click_logs, on="query_id", how="left")
    df["clicked"] = (df["doc_id"] == df["clicked_person_id"]).astype(int)

    if "total_doc_score" in bm25.columns:
        totals = bm25[["query_id", "doc_id", "total_doc_score"]].drop_duplicates()
        df = df.merge(totals, on=["query_id", "doc_id"], how="left")

    df.drop(columns=["clicked_rank", "clicked_person_id"], inplace=True, errors="ignore")
    df.to_csv(output_file, index=False)
    print(f"✅ Wrote {len(df)} rows → {output_file}")