import numpy as np
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import mahalanobis
import argparse
import pandas as pd


def intra_class_sparse_guidance(
        data,
        alpha=0.05,
        p=100,
        cluster_method='ward',
        cluster_threshold=0.5
):
    n_samples, n_features = data.shape
    corr_matrix, _ = spearmanr(data.T)
    Z = linkage(corr_matrix, method=cluster_method)
    cluster_labels = fcluster(Z, t=cluster_threshold, criterion='distance')
    unique_clusters = np.unique(cluster_labels)

    guided_samples = []

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_data = data[:, cluster_indices]
        mu = np.mean(cluster_data, axis=0)
        cov = np.cov(cluster_data.T)

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov += np.eye(cov.shape[0]) * 1e-8
            inv_cov = np.linalg.inv(cov)

        distances = np.array([mahalanobis(x, mu, inv_cov) for x in cluster_data])
        p_values = np.array([np.sum(distances >= d) / len(distances) for d in distances])
        non_outliers = np.where(p_values >= alpha)[0]
        sorted_indices = non_outliers[np.argsort(distances[non_outliers])]
        cluster_size = len(cluster_indices)
        pj = max(1, int(round(p * cluster_size / n_features)))
        selected = sorted_indices[:pj]
        guided_samples.extend(selected)

    guided_samples = np.unique(guided_samples)
    if len(guided_samples) > p:
        np.random.shuffle(guided_samples)
        guided_samples = guided_samples[:p]
    elif len(guided_samples) < p:
        remaining = np.setdiff1d(np.arange(n_samples), guided_samples)
        np.random.shuffle(remaining)
        guided_samples = np.concatenate([guided_samples, remaining[:p - len(guided_samples)]])

    return data[guided_samples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Intra-class Sparse Space Guidance Algorithm')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset file')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for outlier detection')
    parser.add_argument('--p', type=int, default=100, help='Number of guided samples to select')
    parser.add_argument('--cluster_method', type=str, default='ward', help='Hierarchical clustering method')
    parser.add_argument('--cluster_threshold', type=float, default=0.5,
                        help='Distance threshold for feature clustering')

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data_path)
        data = df.values
    except FileNotFoundError:
        print(f"Error: The file {args.data_path} was not found.")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
    else:
        guided_samples = intra_class_sparse_guidance(
            data,
            alpha=args.alpha,
            p=args.p,
            cluster_method=args.cluster_method,
            cluster_threshold=args.cluster_threshold
        )
        print(f"Selected {len(guided_samples)} guided samples")
