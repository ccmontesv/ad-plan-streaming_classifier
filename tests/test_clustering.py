import pytest
from src import preprocess, heuristic, clustering

def test_perform_clustering():
    df = preprocess.load_and_clean_data()
    _, df_full = heuristic.compute_heuristics(df)
    df_cluster, feature_cols, pca_result, tsne_result = clustering.perform_clustering(df_full)

    assert "kmeans_cluster" in df_cluster.columns, "Missing 'kmeans_cluster' label"
    assert len(feature_cols) > 0, "No feature columns identified"
    assert pca_result.shape[1] == 2, "PCA result should have 2 components"
    assert tsne_result.shape[1] == 2, "t-SNE result should have 2 components"
