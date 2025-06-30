from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def perform_clustering(df: pd.DataFrame):
    """
    Performs clustering on user behavior features.
    - Aggregates per-user features like session duration and gap patterns
    - Applies KMeans clustering
    - Computes PCA and t-SNE for 2D projection
    """
    features = df.groupby(["tv_id", "service"]).agg(
        total_sessions=("duration_min", "count"),
        avg_session_duration=("duration_min", "mean"),
        std_session_duration=("duration_min", "std"),
        gap_session_ratio=("gap_flag", "mean"),
        gap_eligible_ratio=("gap_flag_eligible", "mean"),
        avg_gap_length=("gap_to_next", "mean"),
        sessions_with_gap=("gap_flag", "sum"),
        sessions_with_gap_eligible=("gap_flag_eligible", "sum")
    ).fillna(0).reset_index()

    feature_cols = features.columns.drop(["tv_id", "service"]).tolist()
    X_scaled = StandardScaler().fit_transform(features[feature_cols])

    kmeans = KMeans(n_clusters=2, random_state=42)
    features["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    pca_result = PCA(n_components=2).fit_transform(X_scaled)
    tsne_result = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)

    return features, feature_cols, pca_result, tsne_result
