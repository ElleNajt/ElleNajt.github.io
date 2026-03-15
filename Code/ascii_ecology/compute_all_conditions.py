"""Compute string + embedding metrics across all conditions, save to one jsonl.

Usage:
    python compute_all_conditions.py

Reads snapshot directories listed in CONDITIONS below and writes
all_conditions.jsonl with per-epoch metrics for each condition.
"""

import json
import glob
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from metrics import compute_string_metrics

CONDITIONS = [
    ("Plain (16x16)", "results/llm_soup_grid_16x16_plain_v4/snapshots/"),
    ("Pretty (16x16)", "results/llm_soup_grid_16x16_pretty/snapshots/"),
    ("Interesting (16x16)", "results/llm_soup_grid_16x16_interesting_v2/snapshots/"),
    ("Personality (32x32)", "results/llm_soup_grid_32x32_personality_v2/snapshots/"),
    ("Wikipedia (1D)", "results/llm_soup_prefill_wiki_short/snapshots/"),
    ("ASCII art (1D)", "results/llm_soup_prefill_ascii_200/snapshots/"),
]

OUT_PATH = "all_conditions.jsonl"


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(OUT_PATH, "w") as out:
        for name, snap_dir in CONDITIONS:
            snaps = sorted(glob.glob(os.path.join(snap_dir, "epoch_*.json")))
            print(f"\n{name}: {len(snaps)} snapshots")
            for snap in snaps:
                epoch = int(snap.split("epoch_")[1].split(".")[0])
                with open(snap) as f:
                    cells = json.load(f)

                sm = compute_string_metrics(cells)

                embeddings = model.encode(cells, show_progress_bar=False)
                n, d = embeddings.shape
                centered = embeddings - embeddings.mean(axis=0)
                _, S, _ = np.linalg.svd(centered, full_matrices=False)
                S_norm = S / S.sum()
                eff_rank = float(1.0 / np.sum(S_norm ** 2))

                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normed = embeddings / np.clip(norms, 1e-8, None)
                cos_sim = normed @ normed.T
                triu = np.triu_indices(n, k=1)
                mean_cos = float(np.mean(cos_sim[triu]))

                # Differential entropy of Gaussian fit, dropping near-zero eigenvalues
                cov = np.cov(centered.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > 1e-12]
                d_eff = len(eigvals)
                log_det = np.sum(np.log(eigvals))
                diff_entropy = 0.5 * (d_eff * np.log(2 * np.pi * np.e) + log_det)

                row = {
                    "condition": name,
                    "epoch": epoch,
                    "unique": sm["unique"],
                    "entropy": round(sm["entropy"], 3),
                    "compression_ratio": round(sm["compression_ratio"], 4),
                    "eff_rank": round(eff_rank, 2),
                    "mean_cosine": round(mean_cos, 4),
                    "diff_entropy": round(float(diff_entropy), 2),
                }
                out.write(json.dumps(row) + "\n")
                print(f"  epoch {epoch}: eff_rank={eff_rank:.1f}, mean_cos={mean_cos:.3f}")

    print(f"\nDone: {OUT_PATH}")


if __name__ == "__main__":
    main()
