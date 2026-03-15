"""Compute species counts (char, 3x3 block, 4-gram) across all conditions per epoch.

Usage:
    python compute_species_dynamics.py

Reads snapshot directories and writes species_dynamics_all.jsonl with
per-epoch species counts for each condition at three classification levels.
"""

import json
import glob
import os
from collections import Counter

CONDITIONS = [
    ("Plain (16x16)", "results/llm_soup_grid_16x16_plain_v4/snapshots/", "2d"),
    ("Pretty (16x16)", "results/llm_soup_grid_16x16_pretty/snapshots/", "2d"),
    ("Interesting (16x16)", "results/llm_soup_grid_16x16_interesting_v2/snapshots/", "2d"),
    ("Personality (32x32)", "results/llm_soup_grid_32x32_personality_v2/snapshots/", "2d"),
    ("Wikipedia (1D)", "results/llm_soup_prefill_wiki_short/snapshots/", "1d"),
    ("ASCII art (1D)", "results/llm_soup_prefill_ascii_200/snapshots/", "1d"),
]

OUT_PATH = "species_dynamics_all.jsonl"


def dominant_char(s):
    counts = Counter(ch for ch in s if not ch.isspace())
    return counts.most_common(1)[0][0] if counts else " "


def top_4gram(s):
    flat = s.replace("\n", "")
    if len(flat) < 4:
        return flat
    grams = [flat[i : i + 4] for i in range(len(flat) - 3)]
    return Counter(grams).most_common(1)[0][0]


def top_3x3(s):
    rows = [r for r in s.split("\n") if r]
    if len(rows) < 3:
        return top_4gram(s)
    blocks = []
    for i in range(len(rows) - 2):
        ml = min(len(rows[i]), len(rows[i + 1]), len(rows[i + 2]))
        for j in range(ml - 2):
            blocks.append(
                rows[i][j : j + 3]
                + "|"
                + rows[i + 1][j : j + 3]
                + "|"
                + rows[i + 2][j : j + 3]
            )
    if not blocks:
        return top_4gram(s)
    return Counter(blocks).most_common(1)[0][0]


def main():
    with open(OUT_PATH, "w") as out:
        for name, snap_dir, mode in CONDITIONS:
            snaps = sorted(glob.glob(os.path.join(snap_dir, "epoch_*.json")))
            print(f"\n{name}: {len(snaps)} snapshots")
            for snap in snaps:
                epoch = int(snap.split("epoch_")[1].split(".")[0])
                with open(snap) as f:
                    cells = json.load(f)

                char_species = Counter(dominant_char(c) for c in cells)
                if mode == "2d":
                    pattern_species = Counter(top_3x3(c) for c in cells)
                    pattern_label = "3x3"
                else:
                    pattern_species = Counter(top_4gram(c) for c in cells)
                    pattern_label = "4gram"

                row = {
                    "condition": name,
                    "epoch": epoch,
                    "char_species": dict(char_species),
                    "pattern_species": dict(pattern_species),
                    "pattern_type": pattern_label,
                    "n_char_species": len(char_species),
                    "n_pattern_species": len(pattern_species),
                }
                out.write(json.dumps(row) + "\n")

            print(f"  done")

    print(f"\nDone: {OUT_PATH}")


if __name__ == "__main__":
    main()
