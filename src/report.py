import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import table
from src.config import PDF_OUTPUT_PATH

def generate_pdf(df_cluster, heu_summary, cluster_summary_table, hybrid_summary,
                 pca_result, tsne_result, feature_cols, importances_df):
    """
    Generates a one-page PDF report that includes:
    - PCA and t-SNE 2D projections
    - Average feature bar charts per cluster
    - Feature importance ranking
    - Summary tables (heuristic, cluster, hybrid)
    """
    fontsize = 10

    with PdfPages(PDF_OUTPUT_PATH) as pdf:
        fig = plt.figure(figsize=(18, 24))  # Taller layout
        gs = fig.add_gridspec(7, 4, hspace=0.9)

        # --- 1. PCA Projection ---
        ax1 = fig.add_subplot(gs[0, 0:2])
        for cluster_id in sorted(df_cluster["kmeans_cluster"].unique()):
            mask = df_cluster["kmeans_cluster"] == cluster_id
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1],
                        label=f"Cluster {cluster_id}", alpha=0.7)
        ax1.set_title("PCA Projection", fontsize=11)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.legend()
        ax1.grid(True)

        # --- 2. t-SNE Projection ---
        ax2 = fig.add_subplot(gs[0, 2:4])
        for cluster_id in sorted(df_cluster["kmeans_cluster"].unique()):
            mask = df_cluster["kmeans_cluster"] == cluster_id
            ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                        label=f"Cluster {cluster_id}", alpha=0.7)
        ax2.set_title("t-SNE Projection", fontsize=11)
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.legend()
        ax2.grid(True)

        # --- 3. Feature Value Bar Charts per Cluster ---
        cluster_summary = df_cluster.groupby("kmeans_cluster")[feature_cols].mean()
        colors = ["#64A5AD", "#783A40"]  # Consistent palette

        for i, feature in enumerate(feature_cols):
            row = 1 + i // 4  # Row 1 or 2
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            cluster_summary[feature].plot(kind="bar", color=colors, ax=ax)
            ax.set_title(f"Avg {feature}", fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Cluster 0", "Cluster 1"], rotation=0)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        # --- 4. Feature Importance ---
        ax_imp = fig.add_subplot(gs[3, 0:4])
        importances_df.plot(kind="bar", ax=ax_imp, legend=False, color="gray")
        ax_imp.set_title("Feature Importance on Clustering", fontsize=11)
        ax_imp.set_ylabel("Variance Explained Ratio")
        ax_imp.tick_params(axis='x', labelrotation=90)
        ax_imp.grid(axis="y", linestyle="--", alpha=0.5)

        # --- 5. Summary Tables ---
        # Heuristic
        ax10 = fig.add_subplot(gs[5, 0])
        ax10.axis("off")
        ax10.set_title("Heuristic Summary", fontsize=14)
        tab_heu = table(ax10, heu_summary, loc="center", colWidths=[0.2]*heu_summary.shape[1])
        for _, cell in tab_heu.get_celld().items():
            cell.set_fontsize(fontsize)

        # Cluster
        ax11 = fig.add_subplot(gs[5, 1])
        ax11.axis("off")
        ax11.set_title("Cluster Summary", fontsize=14)
        tab_clust = table(ax11, cluster_summary_table, loc="center", colWidths=[0.2]*cluster_summary_table.shape[1])
        for _, cell in tab_clust.get_celld().items():
            cell.set_fontsize(fontsize)

        # Hybrid
        ax12 = fig.add_subplot(gs[5, 2:4])
        ax12.axis("off")
        ax12.set_title("Hybrid Summary", fontsize=14)
        tab_hyb = table(ax12, hybrid_summary, loc="center", colWidths=[0.2]*hybrid_summary.shape[1])
        for _, cell in tab_hyb.get_celld().items():
            cell.set_fontsize(fontsize)

        # --- Global Title ---
        fig.suptitle("Ad Plan Inference - Clustering & Heuristics Summary", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

