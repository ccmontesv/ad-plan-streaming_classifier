import pytest
import pandas as pd
from src import preprocess, heuristic, clustering, hybrid

def test_combine_labels():
    df = preprocess.load_and_clean_data()
    df_heur, df_full = heuristic.compute_heuristics(df)
    df_cluster, feature_cols, _, _ = clustering.perform_clustering(df_full)

    cluster_means = df_cluster.groupby("kmeans_cluster")[feature_cols].mean()
    c0, c1 = cluster_means.loc[0], cluster_means.loc[1]
    ad_supported_cluster = 0 if c0["gap_session_ratio"] > c1["gap_session_ratio"] else 1
    df_cluster["customer_plan_cluster"] = df_cluster["kmeans_cluster"].apply(
        lambda x: "ad-supported" if x == ad_supported_cluster else "ad-free"
    )

    df_hybrid = hybrid.combine_labels(df_heur, df_cluster)

    assert isinstance(df_hybrid, pd.DataFrame), "Output is not a DataFrame"
    assert "hybrid_label" in df_hybrid.columns, "Missing 'hybrid_label' column"
