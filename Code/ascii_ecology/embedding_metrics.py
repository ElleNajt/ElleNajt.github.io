"""Embedding-based metrics for tracking soup evolution.

Computes effective rank (participation ratio of SVD singular values),
mean pairwise cosine similarity, and differential entropy of a
Gaussian fit to the embedding distribution.

Usage:
    python embedding_metrics.py snapshots/ embedding_metrics.jsonl
"""

import json
import glob
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist
from scipy.stats import differential_entropy


def compute_embedding_metrics(cells: list[str], model: SentenceTransformer) -> dict:
    """Compute embedding-based metrics for a list of cell strings."""
    embeddings = model.encode(cells, show_progress_bar=False)
    n, d = embeddings.shape

    # Effective rank: participation ratio of SVD singular values
    # Center the embeddings first
    centered = embeddings - embeddings.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = 1.0 / np.sum(S_norm ** 2)

    # Mean pairwise cosine similarity
    # Normalize embeddings for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-8, None)
    cos_sim_matrix = normed @ normed.T
    # Mean of upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    mean_cosine = float(np.mean(cos_sim_matrix[triu_idx]))

    # Differential entropy of Gaussian fit
    # Fit a Gaussian to the embedding distribution via covariance
    cov = np.cov(centered.T)
    # Differential entropy of multivariate Gaussian: 0.5 * ln((2*pi*e)^d * det(cov))
    # Use log of eigenvalues for numerical stability
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]  # drop near-zero
    log_det = np.sum(np.log(eigvals))
    d_eff = len(eigvals)
    diff_entropy = 0.5 * (d_eff * np.log(2 * np.pi * np.e) + log_det)

    return {
        "eff_rank": round(float(eff_rank), 2),
        "mean_cosine": round(float(mean_cosine), 4),
        "diff_entropy": round(float(diff_entropy), 2),
    }


def main(snapshot_dir, output_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    snaps = sorted(glob.glob(os.path.join(snapshot_dir, "epoch_*.json")))

    with open(output_path, "w") as out:
        for snap in snaps:
            epoch = int(snap.split("epoch_")[1].split(".")[0])
            with open(snap) as f:
                cells = json.load(f)
            metrics = compute_embedding_metrics(cells, model)
            metrics["epoch"] = epoch
            out.write(json.dumps(metrics) + "\n")
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: eff_rank={metrics['eff_rank']}, "
                      f"mean_cos={metrics['mean_cosine']}, "
                      f"diff_ent={metrics['diff_entropy']}")

    print(f"Done: {len(snaps)} epochs written to {output_path}")


if __name__ == "__main__":
    snapshot_dir = sys.argv[1] if len(sys.argv) > 1 else "results/snapshots"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "embedding_metrics.jsonl"
    main(snapshot_dir, output_path)
