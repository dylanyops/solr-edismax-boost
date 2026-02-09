import pandas as pd
from sklearn.linear_model import Ridge

def run(input_file, alpha, output_file):
    df = pd.read_csv(input_file)
    y = df["label"].values
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[feature_cols].values

    model = Ridge(alpha=alpha, fit_intercept=False, random_state=42)
    model.fit(X, y)

    raw_coefs = model.coef_
    fields = [c.replace("delta_", "") for c in feature_cols]

    coef_df = pd.DataFrame({
        "field": fields,
        "raw_weight": raw_coefs
    })
    coef_df["clamped_weight"] = coef_df["raw_weight"].clip(lower=0.0)

    total = coef_df["clamped_weight"].sum()
    coef_df["normalized_weight"] = coef_df["clamped_weight"] / total if total > 0 else 0.0

    coef_df = coef_df.sort_values("normalized_weight", ascending=False)
    coef_df.to_csv(output_file, index=False)

    print("✅ Learned BM25 field weights:")
    print(coef_df)