# Click-Model Pipeline

This repository contains a simple **3-step ML pipeline** for ranking/search using BM25 features and pairwise learning.

## Pipeline Steps

1. **Generate wide features** from BM25 scores and click logs
2. **Feature engineering** to compute pairwise deltas and normalize features
3. **Train normalized Ridge regression** to learn BM25 field weights