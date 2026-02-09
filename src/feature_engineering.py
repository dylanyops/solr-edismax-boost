import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

def run(features_file, clicks_file, bm25_fields, output_file):
    features_df = pd.read_csv(features_file)

    with open(clicks_file, "r") as f:
        clicks = json.load(f)
    clicks_df = pd.DataFrame(clicks)

    features_df["query_id"] = features_df["query_id"].astype(str)
    clicks_df["query_id"] = clicks_df["query_id"].astype(str)

    rows = []

    for _, click in clicks_df.iterrows():
        query_id = click["query_id"]
        clicked_doc = click["clicked_person_id"]

        query_docs = features_df[features_df["query_id"] == query_id]
        if query_docs.empty:
            continue

        clicked_row = query_docs[query_docs["doc_id"] == clicked_doc]
        if clicked_row.empty:
            continue
        clicked_row = clicked_row.iloc[0]

        for _, doc in query_docs.iterrows():
            if doc["doc_id"] == clicked_doc:
                continue

            delta_row = {
                "query_id": query_id,
                "clicked_doc_id": clicked_doc,
                "unclicked_doc_id": doc["doc_id"],
                "label": 1
            }

            for field in bm25_fields:
                delta_row[f"delta_{field}"] = clicked_row[field] - doc[field]

            rows.append(delta_row)

    pairwise_df = pd.DataFrame(rows)

    delta_cols = [c for c in pairwise_df.columns if c.startswith("delta_")]
    scaler = StandardScaler()
    pairwise_df[delta_cols] = scaler.fit_transform(pairwise_df[delta_cols])

    pairwise_df.to_csv(output_file, index=False)
    print(f"✅ Generated {output_file} ({len(pairwise_df)} rows)")