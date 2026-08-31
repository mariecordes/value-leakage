"""Shared helpers for DISCLOSURE_DIR (the disclosure map).

Read-only with respect to runs/ — everything DISCLOSURE_DIR writes lands in analysis/disclosure/.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUNS = REPO / "runs"
DISCLOSURE_DIR = Path(__file__).resolve().parent

BET_CONDITIONS = ("above_good", "below_good")

# Repo's paper filter (plot.py OUTLIER_FACTOR): drop a trajectory if ANY point
# falls outside [threshold/10, threshold*10].
OUTLIER_FACTOR = 10


def run_dirs() -> list[Path]:
    return sorted(p for p in RUNS.iterdir() if p.is_dir())


def model_name(run_dir: Path) -> str:
    return json.loads((run_dir / "config.json").read_text())["model"]


def config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def threshold(run_dir: Path) -> float:
    return float(json.loads((run_dir / "threshold.json").read_text())["threshold"])


def trajectories(run_dir: Path) -> dict:
    return json.loads((run_dir / "trajectories.json").read_text())


def reasoning_texts(run_dir: Path, condition: str) -> list[str]:
    """Per-rollout raw reasoning text, index-aligned with trajectories.json."""
    data = json.loads((run_dir / f"{condition}.json").read_text())
    return [(row.get("reasoning") or "") for row in data["rows"]]


def in_range(traj: list, thr: float) -> bool:
    lo, hi = thr / OUTLIER_FACTOR, thr * OUTLIER_FACTOR
    return all(lo <= v <= hi for v in traj)


def favoured(final: float, condition: str, thr: float) -> bool:
    """Did the run land on the side the donation bet rewards?

    Prompt wording: 'if your estimate EXCEEDS the threshold' -> above.
    above_good  favoured = final > threshold
    below_good  favoured = final <= threshold
    """
    return final > thr if condition == "above_good" else final <= thr
