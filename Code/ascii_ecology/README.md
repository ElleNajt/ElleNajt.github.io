# ASCII Ecology

Replicating [Agüera y Arcas et al. 2024](https://arxiv.org/abs/2406.19108) with an LLM (Claude Haiku 4.5) instead of a BFF interpreter.

## Usage

```bash
# 1. Run the soup (requires ANTHROPIC_API_KEY)
python soup.py                      # → results/snapshots/epoch_*.json

# 2. Compute metrics from snapshots
python metrics.py snapshots/ string_metrics.jsonl
python metrics.py snapshots/ species_counts.jsonl --species
python embedding_metrics.py snapshots/ embedding_metrics.jsonl

# 3. Aggregate metrics across conditions (for cross-condition plot)
python compute_all_conditions.py    # → all_conditions.jsonl

# 4. Render video
python render_video.py snapshots/ frames/
ffmpeg -framerate 10 -i frames/frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 ascii_ecology.mp4
```
