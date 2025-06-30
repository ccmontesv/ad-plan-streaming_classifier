import pandas as pd

def hybrid_label(row):
    """
    Defines hybrid rule:
    - If both heuristic and cluster say 'ad-supported' or 'ad-free' → 'very likely'
    - Otherwise, fall back to heuristic label
    """
    if row["customer_plan_gap_ratio"] == "ad-supported" and row["customer_plan_cluster"] == "ad-supported":
        return "very likely ad-supported"
    elif row["customer_plan_gap_ratio"] == "ad-free" and row["customer_plan_cluster"] == "ad-free":
        return "very likely ad-free"
    return row["customer_plan_gap_ratio"]

def combine_labels(heur_df, cluster_df):
    """
    Merges heuristic and cluster labels, then applies hybrid logic.
    """
    df = pd.merge(heur_df, cluster_df, on=["tv_id", "service"], suffixes=("_heur", "_clust"))
    df["hybrid_label"] = df.apply(hybrid_label, axis=1)
    return df