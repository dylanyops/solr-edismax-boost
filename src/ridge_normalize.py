import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


def run(input_file, alpha, output_file):
    df = pd.read_csv(input_file)

    y = df["label"].values
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[feature_cols].values

    model = Ridge(alpha=alpha, fit_intercept=False, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    # Training metrics
    r2 = r2_score(y, preds)
    rmse = mean_squared_error(y, preds)

    raw_coefs = model.coef_
    fields = [c.replace("delta_", "") for c in feature_cols]

    coef_df = pd.DataFrame({
        "field": fields,
        "raw_weight": raw_coefs
    })

    # Clamp negatives
    coef_df["clamped_weight"] = coef_df["raw_weight"].clip(lower=0.0)

    # Normalize safely
    total = coef_df["clamped_weight"].sum()
    if total > 0:
        coef_df["normalized_weight"] = coef_df["clamped_weight"] / total
    else:
        coef_df["normalized_weight"] = 0.0

    coef_df = coef_df.sort_values("normalized_weight", ascending=False)
    coef_df.to_csv(output_file, index=False)

    print("✅ Learned BM25 field weights:")
    print(coef_df)

    # Return both weights + metrics for MLflow logging
    return coef_df, {
        "ridge_r2": r2,
        "ridge_rmse": rmse,
        "num_features": len(feature_cols)
    }