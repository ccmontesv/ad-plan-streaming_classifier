from src import preprocess, heuristic, clustering, hybrid, report
from src.config import PROCESSED_DATA_PATH, PDF_OUTPUT_PATH, CSV_OUTPUT_PATH
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # STEP 1: Load and preprocess
    df = preprocess.load_and_clean_data()

    # STEP 2: Apply heuristic rules
    df_heur, df_full = heuristic.compute_heuristics(df)
    df_full.to_csv(PROCESSED_DATA_PATH, index=False)

    # STEP 3: Perform clustering
    df_cluster, feature_cols, pca_result, tsne_result = clustering.perform_clustering(df_full)

    # STEP 4: Label clusters (you would compare cluster means here manually or automatically)
    cluster_means = df_cluster.groupby("kmeans_cluster")[feature_cols].mean()
    c0, c1 = cluster_means.loc[0], cluster_means.loc[1]
    ad_supported_cluster = 0 if c0["gap_session_ratio"] > c1["gap_session_ratio"] else 1
    df_cluster["customer_plan_cluster"] = df_cluster["kmeans_cluster"].apply(
        lambda x: "ad-supported" if x == ad_supported_cluster else "ad-free"
    )

    # STEP 5: computing feature importance for the clustering model
    X_scaled = StandardScaler().fit_transform(df_cluster[feature_cols])
    user_features_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    user_features_scaled["cluster"] = df_cluster["kmeans_cluster"]

    # Feature importance = how much each feature separates the clusters
    importance_scores = {
        f: user_features_scaled.groupby("cluster")[f].mean().var() / user_features_scaled[f].var()
        for f in feature_cols
    }
    importances_df = (
        pd.DataFrame.from_dict(importance_scores, orient="index", columns=["importance_ratio"])
        .sort_values(by="importance_ratio", ascending=False)
    )

    # STEP 6: Combine into hybrid
    df_hybrid = hybrid.combine_labels(df_heur, df_cluster)

    # STEP 7: Summarize and visualize
    heu_summary = df_heur.groupby(["service", "customer_plan_gap_ratio"]).size().unstack(fill_value=0)
    cluster_summary = df_cluster.groupby(["service", "customer_plan_cluster"]).size().unstack(fill_value=0)
    hybrid_summary = df_hybrid.groupby(["service", "hybrid_label"]).size().unstack(fill_value=0)

    report.generate_pdf(
        df_cluster=df_cluster,
        heu_summary=heu_summary,
        cluster_summary_table=cluster_summary,
        hybrid_summary=hybrid_summary,
        pca_result=pca_result,
        tsne_result=tsne_result,
        feature_cols=feature_cols,
        importances_df=importances_df
    )

    # Save hybrid classification results
    df_hybrid.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"One-page PDF report saved to: {PDF_OUTPUT_PATH}")