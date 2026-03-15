"""Metrics for tracking soup evolution.

Usage:
    python metrics.py snapshots/ string_metrics.jsonl
    python metrics.py snapshots/ species_counts.jsonl --species
"""

import glob
import json
import math
import os
import sys
import zlib
from collections import Counter


def shannon_entropy(soup: list[str]) -> float:
    """Shannon entropy of the byte distribution across all strings."""
    all_bytes = "".join(soup).encode("utf-8", errors="replace")
    if not all_bytes:
        return 0.0
    counts = Counter(all_bytes)
    total = len(all_bytes)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def compression_ratio(soup: list[str]) -> float:
    """zlib compressed size / raw size. Lower = more structure."""
    raw = "".join(soup).encode("utf-8", errors="replace")
    if not raw:
        return 1.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)


def high_order_entropy(soup: list[str]) -> float:
    """Shannon entropy minus normalized compressed size.
    From Agüera y Arcas et al. 2024: zero for random noise,
    positive for structured repetition (self-replicators)."""
    h = shannon_entropy(soup)
    raw = "".join(soup).encode("utf-8", errors="replace")
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    normalized_compressed = (len(compressed) * 8) / len(raw)
    return h - normalized_compressed


def compute_string_metrics(cells: list[str]) -> dict:
    """All string-level metrics for one snapshot."""
    return {
        "unique": len(set(cells)),
        "entropy": shannon_entropy(cells),
        "compression_ratio": compression_ratio(cells),
        "high_order_entropy": high_order_entropy(cells),
    }


def compute_species_counts(cells: list[str]) -> dict:
    """Species counts and unique strings for one snapshot."""
    dominants = []
    for c in cells:
        counts = Counter(ch for ch in c if not ch.isspace())
        dominants.append(counts.most_common(1)[0][0] if counts else " ")
    return {
        "species": dict(Counter(dominants)),
        "unique_strings": len(set(cells)),
    }


def main(snapshot_dir, output_path, species=False):
    snaps = sorted(glob.glob(os.path.join(snapshot_dir, "epoch_*.json")))
    compute = compute_species_counts if species else compute_string_metrics

    with open(output_path, "w") as out:
        for snap in snaps:
            epoch = int(snap.split("epoch_")[1].split(".")[0])
            with open(snap) as f:
                cells = json.load(f)
            metrics = compute(cells)
            metrics["epoch"] = epoch
            out.write(json.dumps(metrics) + "\n")
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: {metrics}")

    print(f"Done: {len(snaps)} epochs written to {output_path}")


if __name__ == "__main__":
    snapshot_dir = sys.argv[1] if len(sys.argv) > 1 else "results/snapshots"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "string_metrics.jsonl"
    species = "--species" in sys.argv
    main(snapshot_dir, output_path, species=species)
