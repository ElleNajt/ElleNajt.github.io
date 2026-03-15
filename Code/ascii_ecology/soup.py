"""Primordial soup simulation with LLM substrate.

Replicating Computational Life (Agüera y Arcas et al., 2024) but replacing
the BFF interpreter with Claude Haiku at temperature 0.
"""

import json
import random
from pathlib import Path

from client import call_haiku

ASCII_ART_CHARS = list(
    "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬░▒▓█▄▀■□▪▫●○◆◇★☆♠♣♥♦←→↑↓↔↕╱╲╳/\\|+-*#@~^=<>()[]{}.:;!?_"
)


def random_grid(k, chars):
    return "\n".join(
        "".join(random.choice(chars) for _ in range(k)) for _ in range(k)
    )


def interact_grid(a, b, k, system=None):
    """Concatenate two grids as prefill, split continuation into two new grids."""
    output = call_haiku(
        a + "\n" + b,
        max_tokens=k * k,
        system=system,
        prefill=True,
        user_prompt="Continue the pattern:",
    )
    if not output:
        return a, b
    lines = output.split("\n")[: 2 * k]
    if len(lines) < 2:
        return a, b
    mid = len(lines) // 2
    return "\n".join(lines[:mid]), "\n".join(lines[mid:])


def run(n=100, k=32, epochs=200, pairs_per_epoch=50, system=None, seed=42,
        output_dir="results"):
    random.seed(seed)
    out = Path(output_dir)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)

    soup = [random_grid(k, ASCII_ART_CHARS) for _ in range(n)]
    with open(out / "snapshots" / "epoch_0000.json", "w") as f:
        json.dump(soup, f)

    for epoch in range(1, epochs + 1):
        for _ in range(pairs_per_epoch):
            i, j = random.sample(range(n), 2)
            soup[i], soup[j] = interact_grid(soup[i], soup[j], k, system=system)

        with open(out / "snapshots" / f"epoch_{epoch:04d}.json", "w") as f:
            json.dump(soup, f)


if __name__ == "__main__":
    run(
        system="Output two 32x32 grids of characters stacked vertically (64 rows total). "
               "No explanations, just the grids. Imbue your personality in it.",
    )
