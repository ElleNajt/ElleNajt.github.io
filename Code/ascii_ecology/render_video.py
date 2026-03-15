"""Render soup snapshots as video frames.

Cells are sorted by cosine distance to the mean character-frequency vector
of the final snapshot, so convergent cells cluster top-left and positions
stay stable across frames.

Usage:
    python render_video.py snapshots/ frames/
    ffmpeg -framerate 10 -i frames/frame_%04d.png \
        -c:v libx264 -pix_fmt yuv420p -crf 18 ascii_ecology.mp4
"""

import json
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cosine


FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_SIZE = 14
GRID_N = 10  # 10x10 grid of cells
GRID_K = 32  # each cell is 32x32 chars
BORDER = 4
HEADER_H = 80
GREEN = (0, 220, 80)


def cells_to_features(cells, all_chars, char_to_idx):
    """L2-normalized character frequency vector per cell."""
    features = np.zeros((len(cells), len(all_chars)))
    for i, cell in enumerate(cells):
        for c in cell:
            if c in char_to_idx:
                features[i, char_to_idx[c]] += 1
        norm = np.linalg.norm(features[i])
        if norm > 0:
            features[i] /= norm
    return features


def main(snapshot_dir, frame_dir, max_epochs=None):
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    header_font = ImageFont.truetype(FONT_PATH, 36)

    # Measure character dimensions
    bbox = ImageDraw.Draw(Image.new("RGB", (200, 200))).textbbox((0, 0), "█", font=font)
    char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    cell_w, cell_h = char_w * GRID_K, char_h * GRID_K
    total_w = (cell_w + BORDER) * GRID_N + BORDER
    total_h = (cell_h + BORDER) * GRID_N + BORDER + HEADER_H
    total_w += total_w % 2  # even for h264
    total_h += total_h % 2

    # Collect all unique chars across snapshots
    snaps = sorted(glob.glob(os.path.join(snapshot_dir, "epoch_*.json")))
    if max_epochs:
        snaps = snaps[:max_epochs]

    all_chars = set()
    for snap in snaps:
        with open(snap) as f:
            for cell in json.load(f):
                all_chars.update(cell)
    all_chars = sorted(all_chars)
    char_to_idx = {c: i for i, c in enumerate(all_chars)}

    # Reference vector: mean of final snapshot
    with open(snaps[-1]) as f:
        final_features = cells_to_features(json.load(f), all_chars, char_to_idx)
    reference = final_features.mean(axis=0)
    reference /= np.linalg.norm(reference)

    os.makedirs(frame_dir, exist_ok=True)

    frame_idx = 0
    for i, snap in enumerate(snaps):
        epoch = int(snap.split("epoch_")[1].split(".")[0])
        with open(snap) as f:
            cells = json.load(f)

        features = cells_to_features(cells, all_chars, char_to_idx)
        dists = [
            cosine(features[j], reference) if np.linalg.norm(features[j]) > 0 else 2.0
            for j in range(len(cells))
        ]
        order = np.argsort(dists).tolist()

        img = Image.new("RGB", (total_w, total_h), (15, 15, 15))
        draw = ImageDraw.Draw(img)
        draw.text(
            (15, 20),
            f"Epoch {epoch:3d}  —  32×32 ASCII Ecology",
            fill=(180, 200, 180),
            font=header_font,
        )

        for grid_pos, cell_idx in enumerate(order):
            row, col = divmod(grid_pos, GRID_N)
            x0 = BORDER + col * (cell_w + BORDER)
            y0 = HEADER_H + BORDER + row * (cell_h + BORDER)
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=(5, 5, 5))
            for r, line in enumerate(cells[cell_idx].split("\n")[:GRID_K]):
                draw.text((x0, y0 + r * char_h), line[:GRID_K], fill=GREEN, font=font)

        # Hold first 10 epochs longer
        dupes = 10 if epoch <= 10 else 3
        for _ in range(dupes):
            img.save(os.path.join(frame_dir, f"frame_{frame_idx:04d}.png"))
            frame_idx += 1

        if i % 20 == 0:
            print(f"  Rendered epoch {epoch}")

    print(f"Done: {len(snaps)} epochs, {frame_idx} frames in {frame_dir}/")


if __name__ == "__main__":
    snapshot_dir = sys.argv[1] if len(sys.argv) > 1 else "results/snapshots"
    frame_dir = sys.argv[2] if len(sys.argv) > 2 else "frames"
    main(snapshot_dir, frame_dir)
